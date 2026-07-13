#!/usr/bin/env python3
"""
Shared HTTP handler + lifecycle logic for the captcha relay scripts.

Both `captcha_relay.py` (arbitrary sitekey) and `captcha_test.py` (Google test
sitekey) wire the same HTTP server, the same /token capture, and the same
auto-shutdown. Extracting the shared logic here keeps the two entry-point
scripts honest: a bug fix only needs to land in one place.

Lifecycle invariants this module enforces:

  * The token file is **cleared** when the relay starts, so a stale token from
    a prior run cannot leak into a new run.
  * The 2-minute timeout fires based on a **request-local** flag
    (`server.token_received`), not on the file's existence, so a stale file
    cannot disable the timeout.
  * The token is read from the file **after** `serve_forever()` returns and
    the file has just been written by the request handler, so the value we
    return is always the one the human just produced.
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
import time
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

DEFAULT_TOKEN_FILE = "/tmp/captcha_token.txt"
DEFAULT_TIMEOUT_SECONDS = 120


def clear_token_file(path: str = DEFAULT_TOKEN_FILE) -> None:
    """Remove a stale token file if present. Idempotent."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def make_handler(html_factory: Callable[[], str], token_file: str):
    """Build an HTTP handler bound to a per-page HTML factory.

    The factory is called on every GET / so that any future dynamic state
    (e.g. live status from the server) could be reflected without restarting.
    Static pages work fine too — the factory just returns the same string.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (stdlib naming)
            if self.path == "/" or self.path.startswith("/?"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(html_factory().encode("utf-8"))
            elif self.path.startswith("/token"):
                qs = parse_qs(urlparse(self.path).query)
                t = qs.get("t", [None])[0]
                if t:
                    with open(token_file, "w") as f:
                        f.write(t)
                    print(f"\n✓ Token received ({len(t)} chars)", flush=True)
                    print(json.dumps({"token": t}), flush=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode())
                if t:
                    # Mark request-local state BEFORE scheduling shutdown so the
                    # timeout thread sees it on its next check.
                    self.server.token_received = True
                    threading.Thread(
                        target=self.server.shutdown, daemon=True
                    ).start()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args, **kwargs):  # silence default access log
            pass

    return Handler


def run_relay(
    *,
    html_factory: Callable[[], str],
    banner: str,
    port: int,
    token_file: str = DEFAULT_TOKEN_FILE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Start a captcha relay server. Returns the captured token, or None.

    Steps performed, in order:
      1. Clear any stale token file left by a prior run.
      2. Bind the HTTP server with allow_reuse_address.
      3. Spawn a daemon timeout thread that fires after `timeout_seconds`
         iff `server.token_received` is still False at that moment.
      4. serve_forever() blocks until /token or the timeout shuts us down.
      5. After shutdown, read the token from disk (always written by the
         request handler if a solve happened) and return it.
    """
    clear_token_file(token_file)

    socketserver.TCPServer.allow_reuse_address = True
    handler_cls = make_handler(html_factory, token_file)
    server = socketserver.TCPServer(("0.0.0.0", port), handler_cls)
    # Request-local flag — never read from disk for timeout decisions.
    # setattr keeps strict type-checkers (pyright/mypy) happy; stdlib's
    # socketserver.TCPServer accepts arbitrary attributes at runtime.
    setattr(server, "token_received", False)

    print(banner, flush=True)
    print(f"Tunnel: cloudflared tunnel --url http://localhost:{port}", flush=True)
    print("Waiting for human to solve...", flush=True)

    def timeout():
        time.sleep(timeout_seconds)
        if not getattr(server, "token_received", False):
            print("\nTimeout. No token received.", flush=True)
            server.shutdown()

    threading.Thread(target=timeout, daemon=True).start()

    server.serve_forever()

    # After serve_forever returns, either /token was hit or the timeout fired.
    # We only read the token if it was actually written this run.
    if getattr(server, "token_received", False) and os.path.exists(token_file):
        with open(token_file) as f:
            token = f.read().strip()
        print(f"\nResult: {json.dumps({'token': token})}", flush=True)
        return token
    return None