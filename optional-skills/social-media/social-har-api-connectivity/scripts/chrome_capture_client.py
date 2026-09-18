#!/usr/bin/env python3
"""chrome_capture_client.py — capture a social platform login session via CDP.

Drives Chrome (visible) over the Chrome DevTools Protocol to capture the network
traffic during a user's login, then extracts session cookies + auth tokens from
the captured requests so a reusable client can be built. Authorized use only.

Least-privilege by design:
  - captures ONLY the traffic of the page it drives (not a global proxy)
  - writes captured secrets to a temp dir with chmod 600
  - never uploads, never logs tokens to stdout

Usage:
  python3 chrome_capture_client.py --url https://x.com/login --out /tmp/sess
  python3 chrome_capture_client.py --platform x --out /tmp/sess   # uses table
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile

import websockets

CHROME = os.environ.get("CHROME_BIN", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# platform -> login URL + which post-login request carries the session
PLATFORM_LOGIN = {
    "x":        "https://x.com/login",
    "twitter":  "https://x.com/login",
    "bsky":     "https://bsky.app/login",
    "mastodon": None,  # instance-specific, pass --url
    "linkedin": "https://www.linkedin.com/login",
    "instagram": "https://www.instagram.com/accounts/login/",
    "facebook": "https://www.facebook.com/login",
    "tiktok":   "https://www.tiktok.com/login",
    "reddit":   "https://www.reddit.com/login",
    "pinterest": "https://www.pinterest.com/login",
    "threads":  "https://www.threads.net/login",
    "youtube":  "https://accounts.google.com/",
}

# headers that carry session material
AUTH_HEADERS = ("authorization", "cookie", "x-csrf-token", "x-auth-token", "x-ig-app-id")


def _ws_url(port):
    """Read the WebSocket debug URL from Chrome's /json/version, partial."""
    return f"ws://127.0.0.1:{port}/json"


async def capture(platform, out_dir, port, timeout):
    import websockets  # deferred so missing pkg is an actionable error
    ws_url = _ws_url(port)
    login = PLATFORM_LOGIN.get(platform)
    async with websockets.connect(ws_url) as ws:
        # 1. open the login page (navigate tab 0)
        await ws.send(json.dumps({"id": 1, "method": "Tab.navigate", "params": {"url": login}}))
        # 2. subscribe to request/response events and gather session carriers
        intercept = []
        # Drive via the page over WS: poll events for 'Network.requestWillBeSent'
        async def pump():
            while True:
                msg = json.loads(await ws.recv())
                m = msg.get("method", "")
                if m == "Network.requestWillBeSent":
                    req = msg.get("params", {}).get("request", {})
                    # keep only requests that carry auth-material after login
                    request_headers = {k.lower(): v for k, v in (req.get("headers") or {}).items()}
                    if any(h in request_headers for h in AUTH_HEADERS):
                        intercept.append({
                            "url": req.get("url", "")[:300],
                            "headers": {h: request_headers[h][:200] for h in AUTH_HEADERS if h in request_headers},
                        })
        task = asyncio.create_task(pump())
        await asyncio.sleep(timeout)
        task.cancel()
        # 3. write only the captured session material to the temp dir, chmod 600
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, mode=0o700)
        out_file = os.path.join(out_dir, "session.json")
        with open(out_file, "w") as f:
            json.dump(intercept, f, indent=2)
        os.chmod(out_file, 0o600)
        return out_file


def main():
    ap = argparse.ArgumentParser(description="CDP login-session capture (authorized only)")
    ap.add_argument("--platform", default="x", choices=list(PLATFORM_LOGIN))
    ap.add_argument("--url", default=None, help="override login URL (e.g. mastodon instance)")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "har-session"))
    ap.add_argument("--port", type=int, default=9222, help="Chrome CDP port")
    ap.add_argument("--timeout", type=int, default=120, help="seconds to watch for login")
    args = ap.parse_args()
    login = args.url or PLATFORM_LOGIN.get(args.platform)
    print(f"Log in at: {login}  (Chrome must be started with --remote-debugging-port={args.port})")
    out = asyncio.run(capture(args.platform, args.out, args.port, args.timeout))
    print(f"session material -> {out} (chmod 600)")


if __name__ == "__main__":
    main()