import argparse

import os
from kitti_object import *

import numpy as np
from PIL import Image
from tqdm import tqdm

def get_fov_2d_points(lidar_points, image, calib, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    height, width, _ = image.shape
    pts_2d = calib.project_velo_to_image(lidar_points)
    fov_inds = (pts_2d[:,0]<width) & (pts_2d[:,0]>=0) & \
        (pts_2d[:,1]<height) & (pts_2d[:,1]>=0)
    fov_inds = fov_inds & (lidar_points[:,0]>clip_distance)
    return pts_2d, fov_inds

def main():
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--split', type=str, choices=['training', 'testing'], 
                        required=True, help='Dataset Split')
    args = parser.parse_args()
    
    dataset = kitti_object(split=args.split)
    for idx, (lidar_points, image, calib) in tqdm(enumerate(dataset), total = len(dataset)):
        lidar_points = lidar_points[:, 0:3]
        pc_rect = calib.project_velo_to_rect(lidar_points)
        pc_image_coord, img_fov_inds = get_fov_2d_points(lidar_points, image, calib)
        pc_image_coord = pc_image_coord[img_fov_inds].astype(np.int32)[:,::-1]
        pc_rect_xyz = pc_rect[img_fov_inds]

        sparse_depth_map =  np.full(image.shape[:-1], fill_value=np.inf, dtype=np.float32)
        for i in range(len(pc_image_coord)):
            sparse_depth_map[tuple(pc_image_coord[i])] = np.minimum(pc_rect_xyz[i,2], 
                                                                    sparse_depth_map[tuple(pc_image_coord[i])])
        sparse_depth_map[np.isinf(sparse_depth_map)] = 0
        save_depth_as_uint16png_upload(sparse_depth_map, os.path.join(dataset.split_dir, 
                                                                      'depth_sparse', 
                                                                      '%06d.png'%(idx)))

def save_depth_as_uint16png_upload(img, filename):
    img = np.squeeze(img)
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    imgsave.save(filename)

if __name__ == "__main__":
    main()