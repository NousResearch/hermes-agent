#!/usr/bin/env python3
"""Capture a HAR from a browser you connect to over CDP (not one you launch).

Use this when the browser is owned by someone else and only reachable over the
Chrome DevTools Protocol: Hermes cloud backends (Browserbase, Browser-Use,
Firecrawl), a Camofox session exposing CDP, or anything wired via
`/browser connect <url>` / BROWSER_CDP_URL / browser.cdp_url in config.

Why this exists: Playwright's record_har_path only works on a context you
launched locally. connect_over_cdp() attaches to an existing browser, so
record_har is unavailable, so we assemble the HAR from context
request/response events instead.

Usage:
  python3 har_capture_cdp.py <cdp_url> <output.har> [--wait S] \
      [--goto URL] [--action "fill:SEL:TEXT"] [--action "click:SEL"] ...

<cdp_url> is the ws:// or http:// CDP endpoint. For Hermes: run
`/browser connect` to see the active endpoint, or read BROWSER_CDP_URL.

--goto opens a NEW tab so it does not navigate a page Hermes is already using.
Listeners are attached to every existing context (not just pages[0]).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from har_actions import choose_drive_page, flush_pending, run_action  # noqa: E402


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


def _attach_network(contexts, on_request, on_response) -> list:
    attached = []
    for ctx in contexts:
        ctx.on("request", on_request)
        ctx.on("response", on_response)
        attached.append(ctx)
    return attached


def _detach_network(contexts, on_request, on_response) -> None:
    for ctx in contexts:
        try:
            ctx.remove_listener("request", on_request)
            ctx.remove_listener("response", on_response)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cdp_url")
    ap.add_argument("har_path")
    ap.add_argument("--goto", default=None, help="URL to open in a new tab after attaching")
    ap.add_argument("--wait", type=float, default=3.0)
    ap.add_argument("--action", action="append", default=[])
    args = ap.parse_args()

    entries = []
    pending = {}  # id(request) -> request
    drive_error = None

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(args.cdp_url)
        contexts = list(browser.contexts) or [browser.new_context()]

        def on_request(req):
            pending[id(req)] = req

        def on_response(resp):
            req = resp.request
            pending.pop(id(req), None)
            entries.append(_har_entry(req, resp))

        attached = _attach_network(contexts, on_request, on_response)
        try:
            page = choose_drive_page(contexts[0], new_page=bool(args.goto))
            if args.goto:
                page.goto(args.goto, wait_until="domcontentloaded")
            for spec in args.action:
                run_action(page, spec)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
            time.sleep(args.wait)
        except Exception as exc:
            drive_error = exc
        finally:
            # Detach before flushing so a late response event can't append a
            # duplicate of an entry the flush is about to write.
            _detach_network(attached, on_request, on_response)
            flush_pending(pending, entries, _har_entry)
        # Do NOT close: we connected to someone else's browser.

    har = {
        "log": {
            "version": "1.2",
            "creator": {"name": "har_capture_cdp", "version": "0.1"},
            "entries": entries,
        }
    }
    with open(args.har_path, "w", encoding="utf-8") as f:
        json.dump(har, f)
    print(f"HAR written: {args.har_path} ({len(entries)} entries)")
    if drive_error is not None:
        raise drive_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
