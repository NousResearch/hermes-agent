#!/usr/bin/env python3
"""Record a HAR file while driving a website with Playwright.

Usage:
  python3 har_capture.py <url> <output.har> [--wait SECONDS] \
      [--action "fill:SELECTOR:TEXT"] [--action "press:SELECTOR:KEY"] \
      [--action "click:SELECTOR"] [--action "goto:URL"] [--action "sleep:SECONDS"]

Actions run in order after page load. The HAR embeds request/response bodies
(record_har_content='embed') so derived clients can see payload shapes.

If an action fails, everything captured up to that point is flushed to the
HAR anyway. Fix the selector (try --headed to watch) and rerun.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from har_actions import run_action  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("har_path")
    ap.add_argument("--wait", type=float, default=3.0,
                    help="seconds to idle at the end so late XHRs land in the HAR")
    ap.add_argument("--action", action="append", default=[],
                    help="fill:SEL:TEXT | press:SEL:KEY | click:SEL | goto:URL | sleep:SECS")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = None
        try:
            context = browser.new_context(
                record_har_path=args.har_path,
                record_har_content="embed",  # keep response bodies in the HAR
            )
            page = context.new_page()
            page.goto(args.url, wait_until="domcontentloaded")
            for spec in args.action:
                run_action(page, spec)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass  # some pages never fully idle, the trailing --wait covers it
            time.sleep(args.wait)
        finally:
            if context is not None:
                context.close()  # flushes the HAR even when an action failed
            browser.close()
    print(f"HAR written: {args.har_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
