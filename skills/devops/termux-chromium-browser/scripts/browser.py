#!/usr/bin/env python3
"""browser.py — Native Chromium browser driver over CDP for Termux.

Credits: @pjy010218

The browser runs as a persistent headless runit service (port 9222).
This driver communicates via raw Chrome DevTools Protocol (CDP) over WebSocket
without requiring Selenium, Playwright, or Node.js.

Usage:
  python3 browser.py navigate <url>       # Navigate active tab
  python3 browser.py read [n]             # Read visible page text (first n chars)
  python3 browser.py eval "<js>"          # Evaluate JavaScript expression
  python3 browser.py shot [out.png]       # Capture screenshot to file
  python3 browser.py find "<text>"        # Search for text in page body
  python3 browser.py tabs                 # List open browser tabs
  python3 browser.py newtab <url>         # Open a new tab and attach to it
  python3 browser.py version              # Print browser version / CDP status
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import websocket
except ImportError:
    websocket = None

CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


def http_json(path: str, method: str = "GET") -> Any:
    c = http.client.HTTPConnection(CDP_HOST, CDP_PORT, timeout=15)
    c.request(method, path)
    r = c.getresponse()
    body = r.read().decode("utf-8", errors="replace")
    c.close()
    try:
        return json.loads(body)
    except Exception:
        return body


class CDPBrowser:
    """Thin CDP-over-WebSocket driver for headless Chromium on Termux."""

    STATEFILE = os.path.join(os.path.expanduser("~"), ".cdp_last_tab")

    def __init__(self, host: str = CDP_HOST, port: int = CDP_PORT):
        self.host = host
        self.port = port
        self.ws: Any = None
        self._id = 0
        self.tab_url: Optional[str] = None

    @staticmethod
    def alive() -> str:
        try:
            v = http_json("/json/version")
            if isinstance(v, dict):
                return v.get("Browser", "")
            return ""
        except Exception:
            return ""

    def tabs(self) -> List[Dict[str, Any]]:
        try:
            res = http_json("/json")
            if isinstance(res, list):
                return [t for t in res if t.get("type") == "page"]
            return []
        except Exception:
            return []

    def attach(self, url_substr: Optional[str] = None) -> Dict[str, Any]:
        """Attach to the last-used tab, matching url_substr, or latest open page tab."""
        if websocket is None:
            raise RuntimeError("websocket-client is required. Run: pip install websocket-client")

        pages = self.tabs()
        if not pages:
            raise RuntimeError("No page tabs open in Chromium.")

        t = None
        if os.path.exists(self.STATEFILE):
            try:
                with open(self.STATEFILE, "r", encoding="utf-8") as f:
                    wanted = f.read().strip()
                t = next((p for p in pages if p.get("id") == wanted), None)
            except Exception:
                pass

        if t is None and url_substr:
            t = next((p for p in pages if url_substr in p.get("url", "")), None)

        if t is None:
            t = pages[-1]

        ws_url = t.get("webSocketDebuggerUrl")
        if not ws_url:
            raise RuntimeError(f"Tab {t.get('id')} has no webSocketDebuggerUrl.")

        self.ws = websocket.create_connection(ws_url, timeout=45)
        self.tab_url = t.get("url")
        return t

    def _cmd(self, method: str, **params: Any) -> Any:
        if not self.ws:
            self.attach()
        self._id += 1
        payload = json.dumps({"id": self._id, "method": method, "params": params})
        self.ws.send(payload)
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                return msg.get("result", msg)

    def newtab(self, url: str) -> Dict[str, Any]:
        c = http.client.HTTPConnection(self.host, self.port, timeout=15)
        c.request("PUT", "/json/new?" + url)
        res = c.getresponse().read().decode("utf-8", errors="replace")
        tab = json.loads(res)
        c.close()

        try:
            with open(self.STATEFILE, "w", encoding="utf-8") as f:
                f.write(tab.get("id", ""))
        except Exception:
            pass

        self.attach()
        self.settle()
        return tab

    def navigate(self, url: str) -> None:
        if not self.ws:
            self.attach()
        self._cmd("Page.navigate", url=url)
        self.settle()

    def settle(self, seconds: Optional[int] = None) -> None:
        deadline = time.time() + (seconds if seconds is not None else 15)
        while time.time() < deadline:
            try:
                if (self.text() or "").strip():
                    return
            except Exception:
                pass
            time.sleep(1)
        time.sleep(1)

    def js(self, expr: str) -> Any:
        r = self._cmd("Runtime.evaluate", expression=expr, returnByValue=True)
        res = r.get("result", {})
        if res.get("subtype") == "error":
            raise RuntimeError(res.get("description", "JavaScript evaluation error"))
        return res.get("value")

    def text(self) -> str:
        return self.js("document.body ? document.body.innerText : ''") or ""

    def find(self, needle: str) -> bool:
        return bool(self.js(f"document.body ? document.body.innerText.includes({needle!r}) : false"))

    def shot(self, out_path: str) -> str:
        r = self._cmd("Page.captureScreenshot", format="png")
        data = r.get("data")
        if not data:
            raise RuntimeError(f"Screenshot failed: {str(r)[:120]}")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(data))
        return out_path


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tabs"

    if cmd == "tabs":
        for t in CDPBrowser().tabs():
            print((t.get("title") or "")[:50], "|", t.get("url"))
        return

    if cmd == "version":
        ver = CDPBrowser.alive()
        print(ver if ver else "CDP DOWN")
        return

    b = CDPBrowser()
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "newtab":
        if not arg:
            print("Error: newtab requires a URL", file=sys.stderr)
            sys.exit(1)
        b.newtab(arg)
        print("opened:", arg)
    elif cmd == "navigate":
        if not arg:
            print("Error: navigate requires a URL", file=sys.stderr)
            sys.exit(1)
        b.navigate(arg)
        print("navigated:", arg)
    elif cmd == "read":
        n = int(arg) if arg else 4000
        print(b.text()[:n])
    elif cmd == "eval":
        if not arg:
            print("Error: eval requires a JS expression", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(b.js(arg), default=str)[:3000])
    elif cmd == "find":
        if not arg:
            print("Error: find requires search text", file=sys.stderr)
            sys.exit(1)
        print("FOUND" if b.find(arg) else "NOT FOUND")
    elif cmd == "shot":
        out = arg or os.path.join(os.path.expanduser("~"), "page_screenshot.png")
        print("saved:", b.shot(out))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
