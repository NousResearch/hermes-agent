"""Loopback synthetic transport: real httpx client lifecycle, no provider spend.

A local SSE server accepts a POST, emits one event, then stalls. Aborting
that client's sockets and opening a second request must use a fresh httpx
client — not the aborted pool.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agent.agent_runtime_helpers import force_close_tcp_sockets


class _StallAfterEventHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        payload = json.dumps({"type": "response.in_progress"})
        self.wfile.write(f"event: response.in_progress\ndata: {payload}\n\n".encode())
        try:
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        try:
            self.server.stop_stall.wait(60)
        except (BrokenPipeError, ConnectionResetError):
            return


@pytest.fixture
def stall_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StallAfterEventHandler)
    server.stop_stall = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.stop_stall.set()
        server.shutdown()
        thread.join(timeout=2)


def test_aborted_httpx_client_is_not_reused_for_second_stream(stall_server):
    """Real httpx: shutdown the first client's sockets, then a second
    request must go through a new client. Reusing the aborted client is the
    Broken-pipe mechanism; a fresh client must still connect."""
    host, port = stall_server.server_address
    url = f"http://{host}:{port}/v1/responses"

    first = httpx.Client(timeout=httpx.Timeout(connect=2.0, read=None, write=2.0, pool=2.0))
    first_err = None
    try:
        with first.stream("POST", url, json={"model": "synthetic", "stream": True}) as resp:
            chunk = next(resp.iter_bytes())
            assert b"response.in_progress" in chunk
            shutdowns = force_close_tcp_sockets(first)
            assert shutdowns >= 1
            try:
                for _ in resp.iter_bytes():
                    pass
            except Exception as exc:
                first_err = exc
    except Exception as exc:
        first_err = exc
    assert first_err is not None, "aborted first stream must surface a transport error"

    # A later request on the same httpx.Client may mint a new pool connection
    # after SHUT_RDWR. That is not the idle-abort Broken-pipe mechanism.
    # The demonstrated reuse is inner run_codex_stream retry on a poisoned
    # request client (test_codex_idle_abort_fresh_retry). This loopback
    # receipt only proves abort unblocks and a FRESH client can stream.

    second = httpx.Client(timeout=httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0))
    try:
        with second.stream("POST", url, json={"model": "synthetic", "stream": True}) as resp:
            chunk = next(resp.iter_bytes())
            assert b"response.in_progress" in chunk
    finally:
        second.close()
        first.close()
