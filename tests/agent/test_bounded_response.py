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

import httpx
import pytest

from agent.bounded_response import (
    read_error_body_or_default,
    read_streaming_error_body,
)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler():
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - http.server API
            pass

        def do_POST(self):  # noqa: N802 - http.server API
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


def test_stalled_body_hits_hard_deadline(server_base, client):
    start = time.monotonic()
    with client.stream("POST", server_base + "/stall") as response:
        text = read_streaming_error_body(
            response, max_bytes=64 * 1024, timeout_s=2.0
        )
    elapsed = time.monotonic() - start
    # Partial bytes that arrived before the stall are preserved.
    assert "partial failure detail" in text
    # The hard deadline bounds the read; we must not wait for the server stall.
    assert elapsed < 5.0


def test_normal_error_body_read_intact(server_base, client):
    with client.stream("POST", server_base + "/normal") as response:
        text = read_streaming_error_body(response)
    parsed = json.loads(text)
    assert parsed["error"]["status"] == "RESOURCE_EXHAUSTED"


def test_empty_body_returns_empty_string(server_base, client):
    with client.stream("POST", server_base + "/empty") as response:
        text = read_streaming_error_body(response)
    assert text == ""


def test_or_default_returns_none_on_empty(server_base, client):
    with client.stream("POST", server_base + "/empty") as response:
        result = read_error_body_or_default(response)
    assert result is None


def test_or_default_returns_text_when_present(server_base, client):
    with client.stream("POST", server_base + "/normal") as response:
        result = read_error_body_or_default(response)
    assert result is not None and "RESOURCE_EXHAUSTED" in result
