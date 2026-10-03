#!/usr/bin/env python3
"""Capture a HAR from a browser you connect to over CDP (not one you launch).

Use this when the browser is owned by someone else and only reachable over the
Chrome DevTools Protocol: Hermes cloud backends (Browserbase, Browser-Use,
Firecrawl), a Camofox session exposing CDP, or anything wired via
`/browser connect <url>` / BROWSER_CDP_URL / browser.cdp_url in config.

Why this exists: Playwright's record_har_path only works on a context you
launched locally. connect_over_cdp() attaches to an existing browser, so
record_har is unavailable — we assemble the HAR from CDP Network.* events
ourselves via page.on("request"/"response").

Usage:
  python3 har_capture_cdp.py <cdp_url> <output.har> [--wait S] \
      [--goto URL] [--action "fill:SEL:TEXT"] [--action "click:SEL"] ...

<cdp_url> is the ws:// or http:// CDP endpoint. For Hermes: run
`/browser connect` to see the active endpoint, or read BROWSER_CDP_URL.
"""
import argparse
import base64
import json
import sys
import time

from playwright.sync_api import sync_playwright


def run_action(page, spec: str) -> None:
    parts = spec.split(":", 2)
    kind = parts[0]
    if kind == "fill":
        page.fill(parts[1], parts[2])
    elif kind == "press":
        page.press(parts[1], parts[2])
    elif kind == "click":
        page.click(parts[1])
    elif kind == "goto":
        page.goto(parts[1] + (":" + parts[2] if len(parts) > 2 else ""))
    elif kind == "sleep":
        time.sleep(float(parts[1]))
    else:
        raise ValueError(f"unknown action: {spec}")


def _har_entry(req, resp):
    """Build a minimal HAR entry from a Playwright request/response pair."""
    body_text, encoding = "", ""
    if resp is not None:
        try:
            raw = resp.body()
            try:
                body_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                body_text = base64.b64encode(raw).decode("ascii")
                encoding = "base64"
        except Exception:
            pass
    post = req.post_data
    return {
        "_resourceType": req.resource_type,
        "request": {
            "method": req.method,
            "url": req.url,
            "headers": [{"name": k, "value": v} for k, v in req.headers.items()],
            "queryString": [],  # har_to_client.py re-parses the URL, so leave empty
            "postData": {"mimeType": req.headers.get("content-type", ""),
                         "text": post} if post else {},
        },
        "response": {
            "status": resp.status if resp else 0,
            "headers": [{"name": k, "value": v} for k, v in (resp.headers.items() if resp else [])],
            "content": {
                "mimeType": (resp.headers.get("content-type", "") if resp else ""),
                "text": body_text,
                **({"encoding": encoding} if encoding else {}),
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cdp_url")
    ap.add_argument("har_path")
    ap.add_argument("--goto", default=None, help="URL to navigate to after attaching")
    ap.add_argument("--wait", type=float, default=3.0)
    ap.add_argument("--action", action="append", default=[])
    args = ap.parse_args()

    entries = []
    pending = {}  # id(request) -> request

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(args.cdp_url)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.pages[0] if context.pages else context.new_page()

        def on_request(req):
            pending[id(req)] = req

        def on_response(resp):
            req = resp.request
            pending.pop(id(req), None)
            entries.append(_har_entry(req, resp))

        page.on("request", on_request)
        page.on("response", on_response)

        if args.goto:
            page.goto(args.goto, wait_until="domcontentloaded")
        for spec in args.action:
            run_action(page, spec)
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
        time.sleep(args.wait)

        page.remove_listener("request", on_request)
        page.remove_listener("response", on_response)
        # Do NOT close: we connected to someone else's browser.

    har = {"log": {"version": "1.2",
                   "creator": {"name": "har_capture_cdp", "version": "0.1"},
                   "entries": entries}}
    with open(args.har_path, "w", encoding="utf-8") as f:
        json.dump(har, f)
    print(f"HAR written: {args.har_path} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
