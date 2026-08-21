"""
Example 2: Open WeChat, screenshot, analyze its chat bubble styles.

Usage:
    python examples/example2_wechat_analysis.py [--config path/to/config.yaml]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.core import list_elements, screenshot, get_rect
from scripts.utils.config import load_config
from scripts.analyzers.size_analyzer import collect_sizes
from scripts.analyzers.color_analyzer import dominant_colors
from scripts.analyzers.style_extractor import estimate_border_radius


def main() -> int:
    parser = argparse.ArgumentParser(description="Example 2: WeChat style analysis")
    parser.add_argument("--config", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # WeChat window (already running)
    print(">>> Looking for WeChat window...")
    els = list_elements("微信", timeout=10)
    if not els.get("success") or els.get("count", 0) == 0:
        print("WeChat window not found. Please open WeChat first.")
        return 1
    print(f"    Found {els['count']} elements")

    # Screenshot
    print(">>> Taking screenshot...")
    shot = screenshot(title="微信", output_dir=cfg.get("screenshot_dir", ""))
    if not shot["success"]:
        print(f"    Screenshot failed: {shot.get('error')}")
        return 1
    print(f"    Screenshot: {shot['filepath']}")

    # Size analysis
    print(">>> Analyzing element sizes...")
    sizes = collect_sizes(els.get("elements", []))
    print(f"    Button/control widths:  min={sizes['width']['min']} max={sizes['width']['max']}")
    print(f"    Button/control heights: min={sizes['height']['min']} max={sizes['height']['max']}")
    print(f"    All same height? {sizes['height']['all_same']}")

    # Color analysis from screenshot
    try:
        import cv2
        import numpy as np

        img = cv2.imread(shot["filepath"])
        if img is not None:
            # Get dominant colors from the whole screen
            colors = dominant_colors(img, n=5)
            print(">>> Dominant colors:")
            for c in colors:
                print(f"    {c['hex']}  ({c['pct']}%)")

            # Try to find a chat bubble region (right side, middle area)
            h, w = img.shape[:2]
            bubble_crop = img[h // 3 : h // 2, w // 2 : w * 3 // 4]
            if bubble_crop.size > 0:
                radius = estimate_border_radius(bubble_crop)
                print(f"    Estimated bubble corner radius: ~{radius}px")
    except ImportError:
        print("    Install opencv-python for color/border analysis")

    print("\n✅ Example 2 completed — review the WeChat analysis above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
