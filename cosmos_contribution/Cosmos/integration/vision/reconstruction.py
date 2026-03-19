"""
cosmos 3D Scene Reconstruction - Structure from Motion (SfM) Core.

"I can see the depth, and the depth sees me!"

This module implements:
1. Feature Matching across keyframes (multi-view geometry).
2. Fundamental Matrix estimation.
3. Sparse Point Cloud generation (simulated or via OpenCV Triangulation).
4. Depth Estimation from motion.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
from loguru import logger
from dataclasses import dataclass

@dataclass
class Point3D:
    x: float
    y: float
    z: float
    color: Tuple[int, int, int] = (255, 255, 255)

@dataclass
class ReconstructedScene:
    points: List[Point3D]
    cameras: List[np.ndarray] # Projection matrices
    point_cloud_json: str # For visualization

class ReconstructionEngine:
    def __init__(self):
        # ORB for feature detection (FAST + robust)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_frames(self, frames: List[Image.Image]) -> ReconstructedScene:
        """
        Build a sparse 3D point cloud from a sequence of images.
        """
        if len(frames) < 2:
            return ReconstructedScene([], [], "{}")

        all_points = []
        cameras = []
        
        # 1. Feature Extraction for first frame
        img1 = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2GRAY)
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        
        for i in range(1, len(frames)):
            img2 = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_RGB2GRAY)
            kp2, des2 = self.orb.detectAndCompute(img2, None)
            
            # 2. Matching
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 3. Essential Matrix & Triangulation (Simplified)
            # Assuming pinhole camera with identity K
            K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
            
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            
            # Projection matrices
            P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = K @ np.hstack((R, t))
            
            # Triangulate
            pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
            pts3D = pts4D / pts4D[3]
            
            # Add to cloud
            for j in range(pts3D.shape[1]):
                if mask[j]:
                    p = pts3D[:, j]
                    # Filter points behind camera or too far
                    if p[2] > 0 and p[2] < 100:
                        all_points.append(Point3D(float(p[0]), float(p[1]), float(p[2])))
            
            # Move to next pair
            kp1, des1 = kp2, des2
            cameras.append(P2)

        logger.info(f"Reconstruction: Generated sparse cloud with {len(all_points)} points.")
        
        return ReconstructedScene(
            points=all_points,
            cameras=cameras,
            point_cloud_json=self._serialize_cloud(all_points)
        )

    def _serialize_cloud(self, points: List[Point3D]) -> str:
        import json
        data = [{"x": p.x, "y": p.y, "z": p.z} for p in points]
        return json.dumps(data)

# Factory
def create_reconstruction_engine():
    return ReconstructionEngine()
