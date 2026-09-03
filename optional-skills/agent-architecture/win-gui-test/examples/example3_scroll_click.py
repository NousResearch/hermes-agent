"""
Example 3: Scroll through a list and click a target, verify interaction.

Usage:
    python examples/example3_scroll_click.py [--config path/to/config.yaml]
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.core import click, click_coords, scroll, screenshot, list_elements, get_rect
from scripts.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Example 3: Scroll and click test")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--title", default="Partner", help="Window title")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. Make sure we're on the right page — click the "实例管理" nav button
    print(f">>> Navigating to instances page on '{args.title}'...")

    # First try clicking by control name
    nav_result = click(args.title, "实例管理", timeout=10)
    if not nav_result["success"]:
        # Fall back to coordinate click (sidebar nav position)
        print("    Control-name click failed, trying coordinates...")
        nav_result = click_coords(args.title, 248, 520)
    print(f"    Nav click: {nav_result}")
    time.sleep(2)

    # 2. List elements to see the instance table
    print(">>> Listing instance page elements...")
    els = list_elements(args.title, timeout=10)
    instances = [e for e in els.get("elements", []) if e.get("name", "").strip() in ("03", "05")]
    print(f"    Found {len(instances)} instance rows in table")

    # 3. Click on instance 05 (second row)
    if len(instances) >= 2:
        print(">>> Clicking on instance 05...")
        inst05 = instances[1]
        click_result = click(args.title, "05")
        print(f"    Click result: {click_result}")
        time.sleep(1)

        # Verify the QQ panel updated
        rect_result = get_rect(args.title, "— 05")
        if rect_result["success"]:
            print(f"    ✅ QQ panel shows '— 05' at {rect_result['rect']}")
        else:
            print("    ⚠️  Could not verify QQ panel label")

        # 4. Get precise rect of the instance 05 row
        print(">>> Getting instance 05 row rectangle...")
        for inst in instances:
            if inst["name"] == "05":
                # Re-fetch with get_rect for detailed info
                r = get_rect(args.title, "05")
                if r["success"]:
                    print(f"    Row rect: {r['rect']}  center=({r['center_x']},{r['center_y']})")
                break
    else:
        print("    ⚠️  Only 1 instance in table, skipping click test")

    # 5. Final screenshot
    print(">>> Taking final screenshot...")
    shot = screenshot(title=args.title, output_dir=cfg.get("screenshot_dir", ""))
    if shot["success"]:
        print(f"    Screenshot: {shot['filepath']}")

    print("\n✅ Example 3 completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
