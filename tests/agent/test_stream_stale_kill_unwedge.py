"""A stale-killed provider stream must actually unwedge its reader.

Report #110769: at agent-scale context the client logged "No response from
provider for 180-240s ... Reconnecting" over and over and then gave up with
"The model server is not responding". The monitor-side kill aborts the
request client's sockets (``force_close_tcp_sockets`` -> ``shutdown(SHUT_RDWR)``),
and that is best-effort: the pool sweep can miss a connection that is checked
out for the in-flight body read. When the reader stays parked the worker never
unwinds, so the retry loop never retries; the monitor re-kills every stale
interval and the call only ends at the byte-read timeout, far past the stale
budget — exactly the reported reconnect loop.

The kill therefore also shuts down the killed attempt's own socket (resolved
from the identity-guarded ``_attempt_stream_response``) — ``shutdown()`` is
FD-safe from the monitor thread, while ``close()`` from a stranger thread
would release a live TLS descriptor under the owner's SSL BIO (shutdown-only
rule, #30858). And an abort-induced ``httpx.ReadError`` counts as transient
so the retry actually reconnects. Descriptor release stays on the worker:
``_call``'s ``except``/``finally`` closes the managed stream and the request
client on the owner thread.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers


# ── unit: the kill reaches the killed attempt's socket, never closes it ──


def _agent():
    return run_agent.AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


def _call(agent, model="m"):
    call = helpers._StreamingCall(agent, {"model": model, "messages": [{"role": "user", "content": "hi"}]}, None)
    call._stream_stale_timeout = 5.0
    return call


def _recv_shutdown_proof(reader, writer):
    """Behavioral proof the helper reached *this* socket: the helper applies
    ``settimeout(0)`` then ``shutdown(SHUT_RDWR)`` (which emits FIN), so the
    peer must observe EOF. An untouched socket would still be blocking with
    its default timeout and the peer recv would time out instead."""
    assert reader.gettimeout() == 0.0, "helper never reached this socket"
    writer.settimeout(5)
    assert writer.recv(1) == b"", "the killed attempt's socket was not shut down"


def test_shutdown_reaches_socket_through_real_httpx_wrapper_shape():
    """httpx 0.28 nests BoundSyncStream._stream(ResponseStream)._httpcore_stream
    (PoolByteStream)._stream(HTTP11ConnectionByteStream)._connection
    (HTTP11Connection)._network_stream(SyncStream)._sock; a lookup that only
    follows ``_connection`` from ``response.stream`` never reaches the socket
    (review blocker). Shape verified against a live httpx 0.28.1/httpcore 1.0.9
    loopback probe."""
    import socket as _socket
    reader, writer = _socket.socketpair()
    try:
        def _no_close():
            raise AssertionError("monitor must never close")
        # Exact live shape: PoolByteStream carries the byte stream under _stream
        # (no _connection); the HTTP/1.1 connection carries the socket under
        # _network_stream (no _stream).
        pool_stream = SimpleNamespace(_stream=None, _connection=None)
        h11_byte_stream = SimpleNamespace(_connection=None)
        h11_conn = SimpleNamespace(_network_stream=SimpleNamespace(_sock=reader),
                                   _stream=None, _connection=None, _httpcore_stream=None)
        h11_byte_stream._connection = h11_conn
        pool_stream._stream = h11_byte_stream
        resp = SimpleNamespace(
            close=_no_close,
            stream=SimpleNamespace(_stream=SimpleNamespace(_httpcore_stream=pool_stream)),
        )
        call = _call(_agent())
        call._attempt_stream_response = resp
        call._shutdown_stale_attempt_socket(resp)  # must not raise via close
        _recv_shutdown_proof(reader, writer)
    finally:
        reader.close()
        writer.close()

# ── end to end: a reader parked on a silent provider must reconnect ──


class _SilentFirstRequest:
    """Request #1: 200 + SSE headers, then NOTHING (a provider that accepted the
    connection and never produced a byte). Later requests: a normal completion."""

    def __init__(self):
        self.completions = []
        self.release = threading.Event()
        hits = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                hits.completions.append(body)
                first = len(hits.completions) == 1
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if first:
                    # Hold the body open with no bytes at all: the client parks in
                    # its body read. Bounded wait so teardown can never hang.
                    hits.release.wait(timeout=30.0)
                    return
                try:
                    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                             "choices": [{"index": 0, "delta": {"content": "reconnected"}, "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    fin = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass  # the client aborted us (expected)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def silent_wire():
    wire = _SilentFirstRequest()
    yield wire
    wire.close()


def test_wedged_stream_unwinds_within_its_stale_budget_and_reconnects(silent_wire, monkeypatch):
    """Pre-fix the worker stayed parked in the body read: no second request was
    ever issued and the call only ended at the byte-read timeout (>= 20s here,
    120s by default). Now the kill unblocks the reader, the abort-induced read
    error is transient, and the retry lands."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "2")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "20")
    agent = run_agent.AIAgent(
        api_key="test-key", base_url=silent_wire.base_url, model="m", provider="custom",
        platform="cli",  # worker thread + monitor thread: the gateway shape from the report
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    started = time.time()
    response = agent._interruptible_streaming_api_call(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    elapsed = time.time() - started

    assert len(silent_wire.completions) >= 2, (
        f"the killed attempt never reconnected (requests={len(silent_wire.completions)}); "
        "the reader stayed parked instead of unwinding"
    )
    assert response.choices[0].message.content == "reconnected"
    # The stale budget is 2s; the byte-read timeout is 20s. A parked reader can
    # only be recovered by the read timeout, which is what this pins out.
    assert elapsed < 12.0, f"stream took {elapsed:.1f}s to recover — the reader was not unwedged by the kill"
