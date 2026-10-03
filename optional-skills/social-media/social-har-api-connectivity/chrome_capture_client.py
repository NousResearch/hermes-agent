#!/usr/bin/env python3
"""Chrome CDP HAR capture — drive Chrome to login, capture session tokens.

Usage:
  python3 chrome_capture_client.py --url https://example.com/login

Opens Chrome in visible mode, captures all network traffic, extracts
session cookies and API endpoints.
"""

import argparse
import asyncio
import copy
import json
import os
import ssl
from pathlib import Path

try:
    import websockets
except ImportError:
    websockets = None  # optional dep


HAR_TEMPLATE = {
    "log": {
        "version": "1.2",
        "creator": {"name": "chrome-capture-client", "version": "1.0.0"},
        "entries": [],
    }
}


async def capture_har(cdp_url: str, output: str, timeout: int = 120,
                      navigate_url: str | None = None) -> dict:
    """Connect to Chrome DevTools, capture network, write HAR."""
    if websockets is None:
        raise ImportError("websockets library is required — install with: pip install websockets")

    har = copy.deepcopy(HAR_TEMPLATE)

    # CDP connections via ws:// on localhost need no SSL context
    ctx = None
    if cdp_url.startswith("wss://"):
        ctx = ssl.create_default_context()

    async with websockets.connect(cdp_url, ssl=ctx) as ws:
        # Enable network tracking
        await ws.send(json.dumps({"id": 1, "method": "Network.enable"}))
        await ws.recv()  # result

        # Navigate if URL provided
        if navigate_url:
            await ws.send(json.dumps({
                "id": 2,
                "method": "Page.navigate",
                "params": {"url": navigate_url},
            }))
            await ws.recv()  # result

        entries = []

        async def listen_until_timeout():
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("method") == "Network.requestWillBeSent":
                    req = msg["params"]["request"]
                    entries.append({
                        "request": {
                            "url": req.get("url"),
                            "method": req.get("method"),
                            "headers": dict(req.get("headers", {})),
                        }
                    })
                elif msg.get("method") == "Network.responseReceived":
                    resp = msg["params"]["response"]
                    if entries:
                        entries[-1]["response"] = {
                            "status": resp.get("status"),
                            "headers": dict(resp.get("headers", {})),
                        }
            # never returns — cancelled by wait_for

        try:
            await asyncio.wait_for(listen_until_timeout(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # expected — timeout means capture window is over

    har["log"]["entries"] = entries
    Path(output).write_text(json.dumps(har, indent=2))
    print(f"HAR saved to {output} ({len(entries)} requests)")
    return har


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chrome CDP HAR capture")
    parser.add_argument("--cdp", default="http://127.0.0.1:9222",
                        help="Chrome DevTools URL")
    parser.add_argument("--output", "-o", default="capture.har",
                        help="Output HAR file")
    parser.add_argument("--url", help="URL to navigate to first")
    args = parser.parse_args()

    asyncio.run(capture_har(args.cdp, args.output, navigate_url=args.url))