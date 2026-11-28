"""Tests for bounded reads of streaming HTTP error response bodies.

Exercises the real ``httpx`` streaming path against an in-process socket server
(no mocks) so the byte-cap and hard-deadline contracts are validated end to end,
the way they behave against a real misbehaving provider.

Covers the bug class ported from openclaw/openclaw#95108: an unbounded
``response.read()`` on a non-OK streaming response can balloon memory (huge
body) or hang forever (body opens then stalls).
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from agent.bounded_response import read_streaming_error_body


class _ThreadingServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler():
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            if self.path == "/oversize":
                # ~128 MiB if read unbounded; no Content-Length.
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                try:
                    for _ in range(2000):
                        self.wfile.write(b"x" * 65536)
                        self.wfile.flush()
                except Exception:
                    pass
            elif self.path == "/stall":
                # Send a little, then stall forever (no further bytes).
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"partial failure detail")
                self.wfile.flush()
                time.sleep(60)
            elif self.path == "/normal":
                body = json.dumps(
                    {
                        "error": {
                            "code": 429,
                            "message": "quota exceeded",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    }
                ).encode()
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/empty":
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()

    return _Handler


@pytest.fixture()
def server_base():
    httpd = _ThreadingServer(("127.0.0.1", 0), _make_handler())
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()


@pytest.fixture()
def client():
    # Generous read timeout so the bounding is provably done by our helper,
    # not by httpx's own timeout.
    c = httpx.Client(
        timeout=httpx.Timeout(connect=5.0, read=45.0, write=5.0, pool=5.0)
    )
    try:
        yield c
    finally:
        c.close()


def test_oversize_body_is_capped(server_base, client):
    start = time.monotonic()
    with client.stream("POST", server_base + "/oversize") as response:
        text = read_streaming_error_body(
            response, max_bytes=64 * 1024, timeout_s=10.0
        )
    elapsed = time.monotonic() - start
    assert 0 < len(text) <= 64 * 1024
    # Capping must return promptly, not after draining the whole body.
    assert elapsed < 9.0


def test_stalled_body_releases_the_drain_thread(server_base, client):
    """After the deadline the worker must unwind, not stay parked in ``recv`` for as long as the
    server holds the connection open (on Linux, ``close()`` from another thread does not wake it)."""
    with client.stream("POST", server_base + "/stall") as response:
        text = read_streaming_error_body(response, timeout_s=0.5)
    assert text == "partial failure detail"
    drains = [t for t in threading.enumerate() if t.name == "bounded-error-body-read"]
    for thread in drains:
        thread.join(timeout=5.0)
    assert not any(t.is_alive() for t in drains)


class _BlockedReadResponse:
    """Response whose body read blocks until its socket is shut down, recording whether
    ``close()`` arrived while the reader was still inside that read."""

    def __init__(self):
        self.woken = threading.Event()
        self.in_read = False
        self.close_while_reading: list[bool] = []
        sock = SimpleNamespace(settimeout=lambda _t: None, shutdown=lambda _how: self.woken.set())
        self.extensions = {"network_stream": SimpleNamespace(get_extra_info=lambda key: sock if key == "socket" else None)}

    def iter_bytes(self):
        yield b"partial"
        self.in_read = True
        self.woken.wait(10.0)
        self.in_read = False

    def close(self):
        self.close_while_reading.append(self.in_read)


def test_deadline_never_closes_under_a_blocked_read():
    """close() from the caller's thread while the worker is still in the socket read releases an
    FD the reader's SSL BIO caches (the TLS FD-recycle class): the deadline shuts the socket down
    first and closes only after the reader has unwound."""
    response = _BlockedReadResponse()
    try:
        text = read_streaming_error_body(response, timeout_s=0.3)
    finally:
        response.woken.set()
    assert text == "partial"
    assert response.close_while_reading == [False]
