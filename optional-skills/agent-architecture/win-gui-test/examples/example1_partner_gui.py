"""
Example 1: Launch Partner GUI, screenshot, verify button presence.

Usage:
    python examples/example1_partner_gui.py [--config path/to/config.yaml]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.core import launch, list_elements, screenshot, get_rect
from scripts.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Example 1: Partner GUI test")
    parser.add_argument("--config", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. Launch Partner
    print(">>> Launching Partner GUI...")
    result = launch(r"C:\Users\zty12\Desktop\Partner.exe", wait=6)
    print(f"    Launch: {result}")
    if not result["success"]:
        print("FAILED: Could not launch Partner")
        return 1

    # 2. List elements to verify window loaded
    print(">>> Listing elements...")
    els = list_elements("Partner", timeout=15)
    print(f"    Found {els.get('count', 0)} elements")
    if els["count"] == 0:
        print("FAILED: No elements found")
        return 1

    # 3. Check for key buttons
    key_names = ["发送", "文件", "对话", "收起"]
    found = []
    for e in els.get("elements", []):
        name = e.get("name", "")
        for kn in key_names:
            if kn in name:
                found.append(kn)
    print(f"    Key buttons detected: {set(found)}")

    # 4. Screenshot
    print(">>> Taking screenshot...")
    shot = screenshot(title="Partner", output_dir=cfg.get("screenshot_dir", ""))
    if shot["success"]:
        print(f"    Screenshot saved: {shot['filepath']}")
    else:
        print(f"    Screenshot failed: {shot.get('error')}")

    # 5. Get precise rect of the send button
    print(">>> Getting send button rectangle...")
    rect = get_rect("Partner", "发送")
    if rect["success"]:
        print(f"    Send button: {rect['rect']}  center=({rect['center_x']},{rect['center_y']})")
    else:
        print(f"    Send button not found: {rect.get('error')}")

    print("\n✅ Example 1 completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
