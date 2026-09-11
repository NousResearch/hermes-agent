#!/usr/bin/env python3
"""Hold one headed Playwright page so a VNC/Xvfb mirror is not blank.

Also serves HERMES_INSPECT_SOCK so inspect.py evaluates this same tab.
"""
import os
import signal
import sys
from pathlib import Path

WS = os.environ.get("HERMES_BROWSER_WS", "ws://127.0.0.1:9377/camoufox")
URL = os.environ.get("HERMES_HOLD_URL", "https://example.com/")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from browser_inspect import serve_loop  # noqa: E402


def main() -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.firefox.connect(WS, timeout=20000)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        print("HOLDING", page.title(), page.url, flush=True)
        try:
            serve_loop(page)
        finally:
            try:
                page.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())
