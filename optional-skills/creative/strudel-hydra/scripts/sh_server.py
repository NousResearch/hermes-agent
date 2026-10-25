#!/usr/bin/env python3
"""strudel-hydra liveset server.

Serves the host page and a Server-Sent Events (SSE) stream, and relays pushed
audio/visual "sets" to every connected browser. This is the thin protocol that
lets an agent hot-swap a running liveset, the same way pd-patching drives a
`[netreceive]` socket and supercollider drives `scsynth` over OSC — here the
transport is SSE over plain HTTP, so it needs nothing beyond the standard
library.

Run it in the background, open http://127.0.0.1:8765 in a browser, then push
sets with sh_client.py / sh_examples.py.

Security: the write endpoints (`/push`, `/telemetry`) deliver code that the page
evaluates, so they are gated — same-origin only (no wildcard CORS), JSON body
required, and a per-run capability token. The token is generated at startup,
injected into the served page, and written to a per-port file that the client
scripts read automatically. The server refuses to bind a non-loopback address
unless `--allow-remote` is passed.
"""
import argparse
import hmac
import json
import os
import queue
import secrets
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "page.html"
TOKEN_PLACEHOLDER = "__SH_TOKEN__"

# Hosts that keep the code-push transport on the local machine.
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "[::1]"}


def is_loopback_host(host):
    return host in LOOPBACK_HOSTS


def validate_bind_host(host, allow_remote):
    """Loopback binds freely; anything else needs an explicit opt-in because the
    push transport is unauthenticated at the network layer beyond the token."""
    if is_loopback_host(host) or allow_remote:
        return host
    raise SystemExit(
        f"refusing to bind non-loopback host {host!r} without --allow-remote "
        "(the code-push transport would be exposed on the network)"
    )


def token_file_path(port):
    """Per-port file the client scripts read to discover the capability token."""
    return Path(tempfile.gettempdir()) / f"strudel-hydra-{port}.token"


def origin_allowed(origin, host_header):
    """A missing Origin means a non-browser client (curl, urllib) — allowed; the
    token still gates the write. A present Origin must match our own host, which
    rejects cross-site requests a malicious page would carry."""
    if not origin:
        return True
    return origin in ("http://" + host_header, "https://" + host_header)


def content_type_is_json(ctype):
    return ctype is not None and ctype.split(";")[0].strip().lower() == "application/json"


def inject_token(html, token):
    return html.replace(TOKEN_PLACEHOLDER, token)


class Telemetry:
    """Latest measured features reported by the page (the perception channel
    that closes the live-coding loop)."""

    def __init__(self):
        self._data = None
        self._ts = None
        self._lock = threading.Lock()

    def set(self, data):
        with self._lock:
            self._data = data
            self._ts = time.time()

    def get(self):
        with self._lock:
            if self._data is None:
                return {"data": None, "age": None}
            return {"data": self._data, "age": round(time.time() - self._ts, 3)}


telemetry = Telemetry()


class Broker:
    """Fan-out of the latest set to all open SSE streams.

    The most recent set is retained so a browser that connects *after* a push
    still receives the current liveset instead of a blank page.
    """

    def __init__(self):
        self._subs = set()
        self._lock = threading.Lock()
        self._last = None

    def subscribe(self):
        q = queue.Queue()
        with self._lock:
            self._subs.add(q)
            if self._last is not None:
                q.put(self._last)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subs.discard(q)

    def publish(self, data):
        payload = json.dumps(data)
        with self._lock:
            self._last = payload
            for q in list(self._subs):
                q.put(payload)
            return len(self._subs)

    def count(self):
        with self._lock:
            return len(self._subs)


broker = Broker()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    token = ""  # set per-process in main()

    def log_message(self, *_a):  # keep the terminal quiet
        pass

    def _send(self, code, ctype, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _authorize_write(self):
        """Guard the endpoints that deliver evaluable code. Returns True only for
        a same-origin JSON request bearing the capability token; otherwise it
        writes the rejection and returns False."""
        if not origin_allowed(self.headers.get("Origin"), self.headers.get("Host", "")):
            self._send(403, "application/json", b'{"error":"cross-origin forbidden"}')
            return False
        if not content_type_is_json(self.headers.get("Content-Type")):
            self._send(415, "application/json", b'{"error":"content-type must be application/json"}')
            return False
        supplied = self.headers.get("X-SH-Token", "")
        if not self.token or not hmac.compare_digest(supplied, self.token):
            self._send(401, "application/json", b'{"error":"missing or invalid token"}')
            return False
        return True

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                html = TEMPLATE.read_text(encoding="utf-8")
            except OSError as e:
                self._send(500, "text/plain", f"template missing: {e}".encode())
                return
            body = inject_token(html, self.token).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
            return

        if self.path == "/status":
            body = json.dumps({"subscribers": broker.count()}).encode()
            self._send(200, "application/json", body)
            return

        if self.path == "/telemetry":
            self._send(200, "application/json", json.dumps(telemetry.get()).encode())
            return

        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = broker.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        payload = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")  # keep proxies from timing out
                        self.wfile.flush()
                        continue
                    # SSE frames data line-by-line; prefix every line.
                    frame = "data: " + payload.replace("\n", "\ndata: ") + "\n\n"
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                broker.unsubscribe(q)
            return

        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/push":
            if not self._authorize_write():
                return
            try:
                data = self._read_json()
            except json.JSONDecodeError:
                self._send(400, "application/json", b'{"error":"bad json"}')
                return
            if not isinstance(data, dict):
                self._send(400, "application/json", b'{"error":"expected object"}')
                return
            subs = broker.publish(data)
            body = json.dumps({"ok": True, "subscribers": subs}).encode()
            self._send(200, "application/json", body)
            return

        if self.path == "/telemetry":
            if not self._authorize_write():
                return
            try:
                data = self._read_json()
            except json.JSONDecodeError:
                self._send(400, "application/json", b'{"error":"bad json"}')
                return
            telemetry.set(data)
            self._send(200, "application/json", b'{"ok":true}')
            return

        self._send(404, "text/plain", b"not found")


def main():
    ap = argparse.ArgumentParser(description="strudel-hydra liveset server")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (keep on loopback)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit binding a non-loopback --host (exposes the token-gated push transport)",
    )
    args = ap.parse_args()

    validate_bind_host(args.host, args.allow_remote)

    token = secrets.token_urlsafe(24)
    Handler.token = token
    tf = token_file_path(args.port)
    tf.write_text(token, encoding="utf-8")
    try:
        os.chmod(tf, 0o600)
    except OSError:
        pass

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"strudel-hydra server on http://{args.host}:{args.port}  (open it in a browser)")
    print(f"push token: {token}  (clients read it from {tf})")
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        try:
            tf.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
