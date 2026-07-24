#!/usr/bin/env python3
"""
CLI entry point for win-gui-test-skill.

Usage
-----
    python scripts/cli.py list-all
    python scripts/cli.py list-elements "Partner"
    python scripts/cli.py screenshot "Partner" --out /tmp
    python scripts/cli.py click "Partner" "发送"
    python scripts/cli.py click-coords "Partner" 500 300
    python scripts/cli.py sendkeys "Partner" "hello"
    python scripts/cli.py scroll "Partner" --target "列表" --dy -5
    python scripts/cli.py get-rect "Partner" "发送"
    python scripts/cli.py launch "calc.exe"
    python scripts/cli.py analyze "Partner" --out-dir ./reports
    python scripts/cli.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.core import (
    list_windows,
    list_elements,
    screenshot,
    click,
    click_coords,
    send_keys,
    scroll,
    get_rect,
    launch,
)
from scripts.utils.config import load_config
from scripts.utils.logger import setup_logger
from scripts.analyzers.size_analyzer import collect_sizes, detect_inconsistencies
from scripts.analyzers.color_analyzer import dominant_colors, detect_edge_colors
from scripts.analyzers.style_extractor import estimate_border_radius, extract_font_hint


def _out(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, default=str))


def _load_elements_from_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    start = raw.find("{")
    data = json.loads(raw[start:])
    return data.get("elements", [])


def cmd_list_all(_args) -> None:
    _out(list_windows())


def cmd_list_elements(args) -> None:
    _out(list_elements(args.title, timeout=args.timeout))


def cmd_screenshot(args) -> None:
    _out(
        screenshot(
            title=args.title,
            output_dir=args.out_dir or "",
            fallback=not args.no_fallback,
        )
    )


def cmd_click(args) -> None:
    _out(click(args.title, args.target, timeout=args.timeout))


def cmd_click_coords(args) -> None:
    _out(click_coords(args.title, args.x, args.y, timeout=args.timeout))


def cmd_sendkeys(args) -> None:
    _out(send_keys(args.title, args.keys, timeout=args.timeout))


def cmd_scroll(args) -> None:
    _out(scroll(args.title, target=args.target or "", dy=args.dy, timeout=args.timeout))


def cmd_get_rect(args) -> None:
    _out(get_rect(args.title, args.target, timeout=args.timeout))


def cmd_launch(args) -> None:
    _out(launch(args.app, wait=args.wait))


def cmd_analyze(args) -> None:
    """Run size & color analysis on a window's elements."""
    els_result = list_elements(args.title, timeout=args.timeout)
    if not els_result.get("success"):
        _out(els_result)
        return
    elements = els_result.get("elements", [])

    # Size analysis
    size_stats = collect_sizes(elements)
    inconsistencies = detect_inconsistencies(elements)

    # Color analysis (requires screenshot)
    shot = screenshot(title=args.title, output_dir=args.out_dir or "")
    color_data = {"dominant": [], "edges": {}}
    if shot.get("success"):
        try:
            import cv2
            import numpy as np

            img = cv2.imread(shot["filepath"])
            if img is not None:
                color_data["dominant"] = dominant_colors(img, n=8)
                color_data["edges"] = detect_edge_colors(img)
        except ImportError:
            color_data["note"] = "Install opencv-python for color analysis"

    result = {
        "success": True,
        "window": args.title,
        "element_count": len(elements),
        "size_statistics": size_stats,
        "size_inconsistencies": inconsistencies,
        "colors": color_data,
    }

    # Optional: save report
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        rpath = os.path.join(args.out_dir, f"analysis_{args.title}.json")
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["report_file"] = rpath

    _out(result)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WinGUI Test Skill — CLI")
    p.add_argument("--config", help="Path to YAML/JSON config file")
    p.add_argument("--log-dir", help="Override log directory")
    p.add_argument("--timeout", type=float, default=10, help="Window lookup timeout (s)")

    sub = p.add_subparsers(dest="command", required=True)

    # list-all
    sub.add_parser("list-all", help="List all visible windows")

    # list-elements
    le = sub.add_parser("list-elements", help="List controls in a window")
    le.add_argument("title", help="Window title fragment")

    # screenshot
    ss = sub.add_parser("screenshot", help="Take a screenshot")
    ss.add_argument("title", nargs="?", default=None, help="Window title to focus")
    ss.add_argument("--out-dir", help="Output directory (default: config screenshot_dir)")
    ss.add_argument("--no-fallback", action="store_true", help="Skip PIL fallback")

    # click
    ck = sub.add_parser("click", help="Click a control by name")
    ck.add_argument("title")
    ck.add_argument("target")

    # click-coords
    cc = sub.add_parser("click-coords", help="Click at screen coordinates")
    cc.add_argument("title")
    cc.add_argument("x", type=int)
    cc.add_argument("y", type=int)

    # sendkeys
    sk = sub.add_parser("sendkeys", help="Send keystrokes")
    sk.add_argument("title")
    sk.add_argument("keys")

    # scroll
    sc = sub.add_parser("scroll", help="Scroll a control/window")
    sc.add_argument("title")
    sc.add_argument("--target", help="Specific control name")
    sc.add_argument("--dy", type=int, default=-3, help="Wheel clicks (negative=down)")

    # get-rect
    gr = sub.add_parser("get-rect", help="Get precise control rectangle")
    gr.add_argument("title")
    gr.add_argument("target")

    # launch
    ln = sub.add_parser("launch", help="Launch an application")
    ln.add_argument("app")
    ln.add_argument("--wait", type=float, default=2.0, help="Seconds to wait after launch")

    # analyze
    an = sub.add_parser("analyze", help="Run full visual analysis on a window")
    an.add_argument("title")
    an.add_argument("--out-dir", help="Directory for report files")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Config + logger
    cfg = load_config(args.config)
    log_dir = args.log_dir or cfg.get("log_dir", "")
    setup_logger(log_dir=log_dir)

    handlers = {
        "list-all": cmd_list_all,
        "list-elements": cmd_list_elements,
        "screenshot": cmd_screenshot,
        "click": cmd_click,
        "click-coords": cmd_click_coords,
        "sendkeys": cmd_sendkeys,
        "scroll": cmd_scroll,
        "get-rect": cmd_get_rect,
        "launch": cmd_launch,
        "analyze": cmd_analyze,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
