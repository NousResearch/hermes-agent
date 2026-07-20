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
"""
import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "page.html"


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

    def log_message(self, *_a):  # keep the terminal quiet
        pass

    def _send(self, code, ctype, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                body = TEMPLATE.read_bytes()
            except OSError as e:
                self._send(500, "text/plain", f"template missing: {e}".encode())
                return
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
            self.send_header("Access-Control-Allow-Origin", "*")
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
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
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
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
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
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"strudel-hydra server on http://{args.host}:{args.port}  (open it in a browser)")
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
