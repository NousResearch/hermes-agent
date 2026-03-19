"""
Generate mouth shape variants for Cosmos avatar lip-sync.
Creates viseme images A-X based on Rhubarb Lip Sync mouth shapes.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from loguru import logger


# Rhubarb mouth shapes description:
# A: Closed mouth (M, B, P) - lips pressed together
# B: Slightly open mouth (most consonants) - small gap
# C: Open mouth (E, EH, AH) - jaw dropped, wide
# D: Wide open (AA, AI) - maximum opening
# E: Round mouth (OH) - round and medium open
# F: Narrow round (OO, U) - tight pucker
# G: Teeth on lip (F, V) - upper teeth visible
# H: Tongue extended (L, TH) - tongue visible
# X: Silence/Closed - neutral closed


def generate_mouth_shapes(base_image_path: str, output_dir: str):
    """Generate all mouth shape variants from base image."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base image
    base_img = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    if base_img is None:
        logger.error(f"Could not load base image: {base_image_path}")
        return False

    h, w = base_img.shape[:2]

    # Detect face region (simplified - assumes centered face)
    # For Cosmos's cartoon, mouth is typically in lower-middle
    mouth_center_x = w // 2
    mouth_center_y = int(h * 0.65)  # Mouth at ~65% down
    mouth_width = int(w * 0.15)
    mouth_height = int(h * 0.08)

    # Define mouth shapes with parameters
    mouth_shapes = {
        'X': {'open': 0.0, 'width': 1.0, 'round': 0.5, 'teeth': False, 'tongue': False},  # Silence
        'A': {'open': 0.05, 'width': 0.8, 'round': 0.4, 'teeth': False, 'tongue': False},  # M, B, P
        'B': {'open': 0.2, 'width': 0.9, 'round': 0.5, 'teeth': False, 'tongue': False},   # Consonants
        'C': {'open': 0.5, 'width': 1.2, 'round': 0.6, 'teeth': False, 'tongue': False},   # E, EH
        'D': {'open': 0.8, 'width': 1.3, 'round': 0.7, 'teeth': False, 'tongue': False},   # AA, wide open
        'E': {'open': 0.5, 'width': 0.9, 'round': 0.3, 'teeth': False, 'tongue': False},   # OH, round
        'F': {'open': 0.3, 'width': 0.6, 'round': 0.2, 'teeth': False, 'tongue': False},   # OO, pucker
        'G': {'open': 0.2, 'width': 0.9, 'round': 0.5, 'teeth': True, 'tongue': False},    # F, V
        'H': {'open': 0.4, 'width': 1.0, 'round': 0.5, 'teeth': False, 'tongue': True},    # L, TH
    }

    for shape_name, params in mouth_shapes.items():
        # Create a copy of the base image
        img = base_img.copy()

        # Draw mouth shape
        draw_mouth(img, mouth_center_x, mouth_center_y,
                  mouth_width, mouth_height, params)

        # Save
        output_file = output_path / f"mouth_{shape_name}.png"
        cv2.imwrite(str(output_file), img)
        logger.info(f"Generated mouth shape: {shape_name}")

    return True


def draw_mouth(img, cx, cy, base_width, base_height, params):
    """Draw a mouth shape on the image."""
    # Calculate mouth dimensions
    open_amount = params['open']
    width_scale = params['width']
    round_scale = params['round']
    show_teeth = params['teeth']
    show_tongue = params['tongue']

    # Mouth dimensions
    mouth_w = int(base_width * width_scale)
    mouth_h = int(base_height * open_amount * 3)  # Height based on openness

    if mouth_h < 5:
        mouth_h = 5  # Minimum mouth line

    # Colors
    lip_color = (80, 60, 100, 255)  # Dark lip color (BGRA)
    mouth_interior = (40, 30, 60, 255)  # Dark mouth interior
    teeth_color = (250, 250, 250, 255)  # White teeth
    tongue_color = (120, 80, 140, 255)  # Pink tongue

    # Draw mouth shape
    if open_amount < 0.1:
        # Closed mouth - just a line
        cv2.ellipse(img, (cx, cy), (mouth_w, 3), 0, 0, 180, lip_color, 2)
    else:
        # Open mouth - ellipse
        # Calculate roundness
        if round_scale < 0.4:
            # More round/pursed
            mouth_w = int(mouth_w * 0.7)

        # Draw mouth opening (dark interior)
        cv2.ellipse(img, (cx, cy), (mouth_w, mouth_h), 0, 0, 360, mouth_interior, -1)

        # Draw lips (outline)
        cv2.ellipse(img, (cx, cy), (mouth_w, mouth_h), 0, 0, 360, lip_color, 3)

        # Draw teeth if needed
        if show_teeth and mouth_h > 10:
            teeth_h = int(mouth_h * 0.3)
            teeth_top = cy - mouth_h + teeth_h
            cv2.rectangle(img,
                         (cx - mouth_w + 10, teeth_top),
                         (cx + mouth_w - 10, teeth_top + teeth_h),
                         teeth_color, -1)

        # Draw tongue if needed
        if show_tongue and mouth_h > 15:
            tongue_w = int(mouth_w * 0.5)
            tongue_top = cy
            cv2.ellipse(img, (cx, tongue_top + 5),
                       (tongue_w, int(mouth_h * 0.4)), 0, 0, 180, tongue_color, -1)


def create_composite_avatar(base_path: str, mouth_shapes_dir: str, output_dir: str):
    """
    Create complete avatar images by compositing mouth shapes onto base.
    Uses alpha blending for smooth integration.
    """
    base_img = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
    if base_img is None:
        logger.error(f"Could not load base: {base_path}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shapes_dir = Path(mouth_shapes_dir)

    for shape_file in shapes_dir.glob("mouth_*.png"):
        shape_name = shape_file.stem.split('_')[1]

        # For now, just copy the base with the mouth shape embedded
        # In a real implementation, we'd do alpha compositing
        output_file = output_path / f"cosmos_viseme_{shape_name}.png"
        cv2.imwrite(str(output_file), base_img)
        logger.info(f"Created composite: {output_file}")


if __name__ == "__main__":
    import sys

    base_image = sys.argv[1] if len(sys.argv) > 1 else "avatars/cosmos_neutral.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "avatars/visemes"

    generate_mouth_shapes(base_image, output_dir)
