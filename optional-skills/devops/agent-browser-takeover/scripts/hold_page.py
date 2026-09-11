#!/usr/bin/env python3
"""Hold one headed Playwright page so Xvfb / noVNC is not a black screen.

Playwright destroys pages when the client disconnects. This process stays
connected. Default page is example.com (gray), which is easy to verify on VNC.
"""
import os
import signal
import sys
import time

WS = os.environ.get("HERMES_BROWSER_WS", "ws://127.0.0.1:9377/camoufox")
URL = os.environ.get("HERMES_HOLD_URL", "https://example.com/")


def main() -> int:
    from playwright.sync_api import sync_playwright

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    with sync_playwright() as p:
        browser = p.firefox.connect(WS, timeout=20000)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        print("HOLDING", page.title(), page.url, flush=True)
        while not stop:
            time.sleep(1)
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
    sys.exit(main())
