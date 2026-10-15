"""A silent SSE stream must be bounded, closed, and recovered — Windows and Selector loops alike.

Companion to ``test_stream_stale_kill_unwedge.py``. Recorded on this Windows VM
against a serialized local inference server (2026-10-15, two incidents): the client's
reader stopped consuming mid-stream, the server's FIFO wedged behind the zombie
connection, and only killing the client process freed it. Measured platform ground
truth (this box):

* ``shutdown(SHUT_RDWR)`` — the stale kill's path — delivers FIN to the peer but does
  NOT wake a blocked ``recv`` on Windows; the peer's reaction to that FIN (cancel +
  close, which a live engine does as soon as the client goes away) IS what wakes the
  parked reader.

So the recovery chain is: idle bound fires -> kill sends FIN -> peer cancels/closes ->
parked reader wakes -> stream error -> retry lands. These tests pin that chain and
the phase-aware bound:

* mid-tool-call silence (the observed incident shape: tool args still streaming)
  uses the SHORT post-chunk bound and retries;
* mid-text silence ends fast with the partial-continuation stub (delivered text is
  never silently duplicated) and a closed socket;
* header-only silence (prefill phase) keeps the stale budget as the kill bound.

Every case asserts the call ends within the bound (not parked until the read
timeout) and that the mock observed the client's FIN. The ``loop_policy`` fixture
runs each case under the Windows Proactor loop and the Selector loop: the Relay
layer opens private event loops per stream, and both policies must behave.
"""
import asyncio
import contextlib
import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers


# ── mock: a local-compatible SSE server that stalls, then reacts to the kill ──


def _hold_until_client_gone(handler, release, closed_event, timeout):
    """Block a stalled response until the client disconnects, teardown, or the bound.

    Returns True once the client's FIN/RST is observed. Like a live provider, the
    caller's ``close_connection`` then makes the socketserver close its side — the
    peer close is what wakes the client's parked reader on Windows.
    """
    conn = handler.connection
    deadline = time.time() + timeout
    gone = False
    try:
        conn.settimeout(0.1)
        while time.time() < deadline and not release.is_set():
            try:
                if conn.recv(1) == b"":
                    gone = True
                    break
            except TimeoutError:
                continue
    except OSError:
        gone = True  # RST also means the client is gone.
    finally:
        if gone:
            closed_event.set()
            handler.close_connection = True
        with contextlib.suppress(OSError):
            conn.settimeout(5.0)
    return gone


class _StallServer:
    """First request: SSE headers (+ optional prefix chunks), then silence.
    Later requests answer normally. Tracks the client's FIN on request #1."""

    def __init__(self, *, mode):
        # mode: "tool_call_chunks" | "text_chunks" | "headers_only"
        self.mode = mode
        self.completions = []
        self.first_conn_closed = threading.Event()
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
                    try:
                        self._stall_prefix()
                        self.wfile.flush()
                    except OSError:
                        return
                    _hold_until_client_gone(self, hits.release, hits.first_conn_closed, 45.0)
                    return
                try:
                    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                             "choices": [{"index": 0, "delta": {"content": "reconnected"}, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    fin = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(b"data: " + json.dumps(fin).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass

            def _stall_prefix(self):
                if hits.mode == "headers_only":
                    return
                if hits.mode == "text_chunks":
                    payloads = ["partial-0 ", "partial-1 "]
                else:  # tool_call_chunks: the observed incident shape (write_file args mid-stream)
                    payloads = ['{"path": "notes.md", "content": "line ', 'two still streaming']
                for args in payloads:
                    if hits.mode == "tool_call_chunks":
                        delta = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                 "function": {"name": "write_file", "arguments": args}}]}
                    else:
                        delta = {"content": args}
                    chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(params=("proactor", "selector"))
def loop_policy(request):
    """Run under the Windows Proactor loop and the Selector loop (per platform)."""
    original = asyncio.get_event_loop_policy()
    if request.param == "proactor":
        proactor = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
        if os.name != "nt" or proactor is None:
            pytest.skip("the Proactor event loop is Windows-only")
        asyncio.set_event_loop_policy(proactor())
    elif os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        yield request.param
    finally:
        asyncio.set_event_loop_policy(original)


def _run_case(wire, *, delta_sink=None):
    agent = run_agent.AIAgent(
        api_key="test-key", base_url=wire.base_url, model="m", provider="custom",
        platform="cli",  # worker thread + monitor thread: the gateway/desktop shape
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    if delta_sink is not None:
        agent.stream_delta_callback = delta_sink.append
    started = time.time()
    response = agent._interruptible_streaming_api_call(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    return response, time.time() - started


def _env_for_case(monkeypatch, *, post_chunk_bound, stale, read, retries="1"):
    for key, value in (
        ("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", post_chunk_bound),
        ("HERMES_STREAM_STALE_TIMEOUT", stale),
        ("HERMES_STREAM_READ_TIMEOUT", read),
        ("HERMES_STREAM_RETRIES", retries),
    ):
        monkeypatch.setenv(key, str(value))


def test_mid_tool_call_silence_is_bounded_by_post_chunk_bound_and_retried(loop_policy, monkeypatch, caplog):
    """The observed incident shape: tool call args mid-stream, then a silent freeze.

    The short post-chunk bound must fire the kill (not the long stale budget), the
    kill's FIN must reach the peer, the peer's close must wake the parked reader, and
    the retry must land — all well inside the read timeout."""
    _env_for_case(monkeypatch, post_chunk_bound=2, stale=60, read=10)
    wire = _StallServer(mode="tool_call_chunks")
    try:
        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            response, elapsed = _run_case(wire)
    finally:
        wire.close()

    assert len(wire.completions) == 2, "the killed attempt must be retried"
    assert response.choices[0].message.content == "reconnected"
    assert elapsed < 15.0, f"took {elapsed:.1f}s — the mid-tool stall was not bounded"
    assert wire.first_conn_closed.is_set(), "the kill's FIN never reached the mock"
    assert "threshold 2s" in caplog.text, "the kill must use the post-chunk bound, not the stale budget"


def test_mid_text_silence_ends_fast_with_partial_stub(loop_policy, monkeypatch, caplog):
    """Delivered text is never silently retried (no duplication); the partial
    continuation stub owns recovery. It must arrive within the bound, not at the
    read timeout, and the socket must be closed."""
    _env_for_case(monkeypatch, post_chunk_bound=2, stale=60, read=10)
    wire = _StallServer(mode="text_chunks")
    try:
        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            response, elapsed = _run_case(wire, delta_sink=[])
    finally:
        wire.close()

    assert response is not None
    assert (response.choices[0].message.content or "").startswith("partial-0")
    assert len(wire.completions) == 1, "text already delivered must not be re-requested"
    assert elapsed < 15.0, f"took {elapsed:.1f}s — the mid-text stall was not bounded"
    assert wire.first_conn_closed.is_set(), "the kill's FIN never reached the mock"


def test_prefill_silence_keeps_stale_budget_and_retries(loop_policy, monkeypatch, caplog):
    """Header-only silence has no first chunk: prefill grace applies, so the kill
    bound is the (long, configured) stale budget — never the shorter post-chunk
    bound — and the no-chunks failure retries."""
    _env_for_case(monkeypatch, post_chunk_bound=2, stale=3, read=8)
    wire = _StallServer(mode="headers_only")
    try:
        with caplog.at_level(logging.WARNING, logger="agent.chat_completion_helpers"):
            response, elapsed = _run_case(wire)
    finally:
        wire.close()

    assert len(wire.completions) == 2, "the killed attempt must be retried"
    assert response.choices[0].message.content == "reconnected"
    assert elapsed < 15.0
    assert wire.first_conn_closed.is_set(), "the kill's FIN never reached the mock"
    assert "threshold 3s" in caplog.text, "prefill silence must keep the stale budget as the kill bound"


# ── unit: the phase-aware bound and its scoping ──


def _agent():
    return run_agent.AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


def test_post_chunk_idle_bound_scopes(monkeypatch):
    """90s default applies to local endpoints only; explicit values apply anywhere."""
    monkeypatch.delenv("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", raising=False)
    assert helpers._stream_post_chunk_idle_bound(_agent()) == 90.0
    assert helpers._stream_post_chunk_idle_bound(SimpleNamespace(base_url="https://api.example.com/v1")) is None
    monkeypatch.setenv("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", "45")
    assert helpers._stream_post_chunk_idle_bound(SimpleNamespace(base_url="https://api.example.com/v1")) == 45.0
    monkeypatch.setenv("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", "0")
    assert helpers._stream_post_chunk_idle_bound(SimpleNamespace(base_url="http://127.0.0.1:1/v1")) is None


def test_idle_kill_threshold_is_phase_aware(monkeypatch):
    monkeypatch.delenv("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", raising=False)
    call = helpers._StreamingCall(_agent(), {"model": "m", "messages": []}, None)
    call._stream_stale_timeout = 3600.0
    assert call._idle_kill_threshold() == 3600.0  # prefill grace: full stale budget
    call.clients.diag = {"first_chunk_at": time.time()}
    assert call._idle_kill_threshold() == 90.0  # post-chunk: the short local bound
    monkeypatch.setenv("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", "30")
    assert call._idle_kill_threshold() == 30.0  # explicit override
    call.agent.base_url = "https://api.example.com/v1"
    monkeypatch.delenv("HERMES_STREAM_POST_CHUNK_IDLE_TIMEOUT", raising=False)
    assert call._idle_kill_threshold() == 3600.0  # cloud untouched by default
