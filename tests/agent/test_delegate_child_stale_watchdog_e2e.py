"""End-to-end repro/gate tests for P-0097 (kanban t_a227df9e).

Incident: delegate_task children (platform="subagent") that ended up in
NON-streaming mode (streaming disabled by a provider signal or model config)
hit the inline ``direct_api_call`` path with the implicit 90s stale-call
watchdog (run_agent._resolved_api_call_stale_timeout_base). Sitting behind a
Cloudflare-fronted aggregator whose Proxy Read Timeout is 120s, the local
watchdog killed every doomed call before the gateway's own 524 arrived, so
retries re-entered the same >120s window with growing backoff: the child died
silently after ~12 minutes of guaranteed-fatal retries. The fix (t_5806fd2b
RCA §4.2) floors the implicit stale timeout at 150s for delegated children so
the provider's actionable 524 wins the race.

The resolver-level unit tests live in test_non_stream_stale_timeout.py. This
file proves the behavior END-TO-END through the real conversation loop, the
real inline dispatch, the real watchdog timers, and a real OpenAI-wire client
talking to an in-process mock provider:

1. test_child_nonstream_hang_retries_sequentially_and_ends_gracefully —
   the incident shape (silent hang): the child must not wedge, must run its
   retry attempts sequentially on the caller thread, and must terminate in a
   graceful failed turn result.
2. test_child_streaming_transient_524s_then_recovers — children stream by
   default; transient 524s must be retried and a healthy SSE stream must
   complete the turn with the recovered answer.
3. test_child_nonstream_provider_524_arrives_before_local_kill — the RCA
   race: with the provider error arriving well inside the (floored) local
   watchdog window, the turn must surface the provider's 524, never a local
   "Non-streaming API call timed out" kill.

The mock provider is a real HTTP server; the agent's ``base_url`` stays
cloud-shaped (non-local, so the local-endpoint stale-disable short-circuit
cannot mask the watchdog) while a per-request OpenAI client is injected
pointing at the mock — the same seam production uses
(``agent._create_request_openai_client``).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.delegation_context import delegated_child_context
from hermes_state import SessionDB

CLOUD_BASE_URL = "https://zai-relay.example.com/v1"


# ---------------------------------------------------------------------------
# In-process mock OpenAI-wire provider
# ---------------------------------------------------------------------------


class _MockProvider:
    """Thread-safe scripted OpenAI-wire provider.

    ``script`` is a callable ``(request_count, body_dict) -> response`` where
    response is one of:
      ("hang",)                      — accept the request and never respond
      ("status", code, payload_dict) — respond with an HTTP status + JSON body
      ("sse", chunks)                — respond 200 with an SSE chat stream
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []  # list of parsed request bodies, in arrival order
        self.script = lambda count, body: ("hang",)
        self._server = None
        self._thread = None
        self.port = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        provider = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {"stream": False}
                with provider.lock:
                    provider.requests.append(body)
                    count = len(provider.requests)
                    outcome = provider.script(count, body)
                if outcome[0] == "hang":
                    # Hold the connection open; the watchdog abort shuts the
                    # socket from the client side. Thread stays parked (daemon).
                    threading.Event().wait(120)
                    return
                if outcome[0] == "status":
                    _, code, payload = outcome
                    payload_bytes = json.dumps(payload).encode("utf-8")
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload_bytes)))
                    self.end_headers()
                    self.wfile.write(payload_bytes)
                    return
                if outcome[0] == "sse":
                    _, chunks = outcome
                    events = []
                    for piece in chunks:
                        events.append(
                            "data: "
                            + json.dumps(
                                {
                                    "id": "chatcmpl-test",
                                    "object": "chat.completion.chunk",
                                    "created": 1,
                                    "model": "test-model",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": piece},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            + "\n\n"
                        )
                    events.append(
                        "data: "
                        + json.dumps(
                            {
                                "id": "chatcmpl-test",
                                "object": "chat.completion.chunk",
                                "created": 1,
                                "model": "test-model",
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            }
                        )
                        + "\n\n"
                    )
                    events.append("data: [DONE]\n\n")
                    payload_bytes = "".join(events).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(payload_bytes)))
                    self.end_headers()
                    self.wfile.write(payload_bytes)
                    return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # -- helpers -----------------------------------------------------------
    @property
    def request_count(self) -> int:
        with self.lock:
            return len(self.requests)

    def streamed_requests(self) -> int:
        with self.lock:
            return sum(1 for body in self.requests if body.get("stream"))


@pytest.fixture
def mock_provider():
    provider = _MockProvider()
    provider.start()
    try:
        yield provider
    finally:
        provider.stop()


# ---------------------------------------------------------------------------
# Environment + backoff isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_timeout_env(monkeypatch):
    """Timeout knobs must come from production defaults, not the ambient env
    of whichever machine runs the suite."""
    for var in (
        "HERMES_API_CALL_STALE_TIMEOUT",
        "HERMES_STREAM_STALE_TIMEOUT",
        "HERMES_API_TIMEOUT",
        "HERMES_STREAM_STALE_GIVEUP",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch):
    """Collapse inter-attempt backoff so tests exercise the watchdog timing,
    not the sleep."""
    from agent import retry_utils as _retry_utils

    monkeypatch.setattr(_retry_utils, "jittered_backoff", lambda *a, **k: 0.0)
    monkeypatch.setattr(_retry_utils, "adaptive_rate_limit_backoff", lambda *a, **k: 0.0)


# ---------------------------------------------------------------------------
# Child agent factory (mirrors delegate_tool._build_child_agent)
# ---------------------------------------------------------------------------


def _make_child_agent(tmp_path, mock_provider: _MockProvider, *, retries: int = 3):
    from openai import OpenAI
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-repro-child"
    db.create_session(session_id=sid, source="cli")

    agent = AIAgent(
        api_key="test-key",
        base_url=CLOUD_BASE_URL,  # cloud-shaped: keeps the local-endpoint stale-disable out of the picture
        provider="openai-compat",
        model="test-model",
        max_iterations=10,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        platform="subagent",
        session_db=db,
        session_id=sid,
    )
    # Production children get _api_max_retries from config (default 3); pin it
    # so attempt accounting is exact regardless of host config.
    agent._api_max_retries = retries

    # Per-request client injection (the same seam production uses), pointed at
    # the mock. max_retries=0 keeps SDK-internal retries out of the accounting.
    def _fake_create_request_client(*, reason, api_kwargs=None):
        return OpenAI(
            api_key="test-key",
            base_url=f"http://127.0.0.1:{mock_provider.port}/v1",
            max_retries=0,
        )

    agent._create_request_openai_client = _fake_create_request_client
    agent._close_request_openai_client = lambda *a, **k: None
    return agent


def _run_turn_with_deadline(agent, user_message, *, timeout_s: float = 90.0):
    """Run one child turn on a worker thread with a hard join deadline so a
    regression that wedges the child fails the test instead of hanging CI."""
    holder = {}

    def _target():
        try:
            with delegated_child_context("sess-repro-child"):
                holder["result"] = agent.run_conversation(user_message)
        except BaseException as exc:  # pragma: no cover - surfaced below
            holder["error"] = exc

    thread = threading.Thread(target=_target, daemon=True, name="child-turn")
    started = time.monotonic()
    thread.start()
    thread.join(timeout=timeout_s)
    elapsed = time.monotonic() - started
    if thread.is_alive():
        import sys as _sys
        import traceback as _tb

        frames = _sys._current_frames()
        for tid, frame in frames.items():
            if tid == threading.main_thread().ident:
                continue
            stack = "".join(_tb.format_stack(frame)).strip()
            print(f"=== WEDGED THREAD {tid} STACK ===\n{stack}\n=== END STACK ===", flush=True)
    assert not thread.is_alive(), (
        f"child turn wedged: run_conversation did not return within {timeout_s}s "
        "(the P-0097 death-spiral regression)"
    )
    if "error" in holder:
        raise holder["error"]
    return holder["result"], elapsed


# ---------------------------------------------------------------------------
# 1. The incident shape: non-stream child vs a silently hanging provider
# ---------------------------------------------------------------------------


def test_child_nonstream_hang_retries_sequentially_and_ends_gracefully(
    tmp_path, mock_provider, monkeypatch
):
    """A delegated child forced into non-streaming mode against a silently
    hanging provider must run its inline watchdog, retry sequentially on the
    caller thread (no nested-pool wedge), and end in a graceful failed turn —
    not a 12-minute silent death spiral and not a hung tool call."""
    monkeypatch.setattr("agent.chat_completion_helpers.env_int", lambda *a, **k: 0)

    class _ScaledTimer(threading.Timer):
        """Compress production watchdog windows (150s) to 0.5s so the retry
        machinery is exercised end-to-end in seconds."""

        def __init__(self, interval, function, args=None, kwargs=None):
            super().__init__(min(float(interval), 0.5), function, args, kwargs)

    monkeypatch.setattr(threading, "Timer", _ScaledTimer)

    attempts = {"now": 0}
    lock = threading.Lock()

    def _script(count, body):
        with lock:
            attempts["now"] = count
        assert body.get("stream") is not True, (
            "the incident child is in non-streaming mode; a streaming request "
            "here means the fixture lost the non-stream state"
        )
        return ("hang",)

    mock_provider.script = _script

    agent = _make_child_agent(tmp_path, mock_provider)
    agent._disable_streaming = True  # the incident state (streaming disabled downstream)

    # Windows reality check (#85252): the watchdog's shutdown(SHUT_RDWR) does NOT
    # wake a recv() parked on a silent socket — production bounds the read with
    # the per-call timeout knob (HERMES_API_TIMEOUT-derived). Compress it so the
    # parked read unblocks promptly after the watchdog kill instead of waiting
    # out the 1800s default; the stale watchdog remains the actual killer.
    agent._resolved_api_call_timeout = lambda: 2.0

    result, elapsed = _run_turn_with_deadline(agent, "Summarize the incident", timeout_s=60)

    # Exactly one request per retry attempt, run sequentially to completion.
    assert mock_provider.request_count == 3, (
        f"expected exactly 3 sequential attempts, saw {mock_provider.request_count}"
    )

    # The turn terminates as a graceful failure with the watchdog's actionable
    # timeout surfaced in the response text — the child is never silently lost.
    assert isinstance(result, dict), f"run_conversation returned {type(result)}"
    final_text = str(result.get("final_response") or "")
    failed = result.get("failed")
    assert failed or "timed out" in final_text.lower() or "timeout" in final_text.lower(), (
        f"expected a graceful failed-turn result, got: { {k: result.get(k) for k in ('failed', 'completed', 'final_response')} }"
    )
    assert "timed out" in final_text.lower() or (result.get("failed") is True), final_text[:400]

    # Three compressed watchdog windows must actually elapse: the calls were
    # bounded by the stale watchdog, not failed instantly.
    assert elapsed >= 1.2, f"3 watchdog windows should take >=1.2s, took {elapsed:.2f}s"
    assert elapsed < 45, f"retry loop ran away ({elapsed:.1f}s)"


# ---------------------------------------------------------------------------
# 2. Children stream by default; transient 524s retry, healthy SSE completes
# ---------------------------------------------------------------------------


def test_child_streaming_transient_524s_then_recovers(tmp_path, mock_provider, monkeypatch):
    """Default (streaming) children hit the streaming watchdog path, not the
    non-stream one: two transient gateway 524s must be retried and the healthy
    SSE stream must complete the turn with the recovered answer."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "10")

    def _script(count, body):
        assert body.get("stream") is True, "children stream by default; expected stream=true"
        if count <= 2:
            return (
                "status",
                524,
                {
                    "error": {
                        "message": "A timeout occurred: origin did not respond within the proxy read timeout",
                        "code": 524,
                    }
                },
            )
        return ("sse", ["Gateway relay "])

    mock_provider.script = _script

    agent = _make_child_agent(tmp_path, mock_provider)

    result, elapsed = _run_turn_with_deadline(agent, "Summarize the incident", timeout_s=60)

    assert mock_provider.request_count == 3, mock_provider.request_count
    assert mock_provider.streamed_requests() == 3
    final_text = str(result.get("final_response") or "")
    assert "Gateway relay" in final_text, (
        f"streamed answer did not reach the final response: {final_text[:300]!r}"
    )
    assert not result.get("failed"), result.get("failed")
    assert elapsed < 45


# ---------------------------------------------------------------------------
# 3. The RCA race: the provider's 524 must arrive before any local kill
# ---------------------------------------------------------------------------


def test_child_nonstream_provider_524_arrives_before_local_kill(tmp_path, mock_provider):
    """With the fix, the child's implicit non-stream watchdog (150s) is longer
    than the gateway's time-to-524, so the turn fails with the provider's
    actionable 524 — never the local 'Non-streaming API call timed out' kill
    that made the incident unrecoverable."""
    delay_s = 12.0

    def _script(count, body):
        assert body.get("stream") is not True
        time.sleep(delay_s)
        return (
            "status",
            524,
            {
                "error": {
                    "message": "A timeout occurred: origin did not respond within the proxy read timeout",
                    "code": 524,
                }
            },
        )

    mock_provider.script = _script

    agent = _make_child_agent(tmp_path, mock_provider, retries=2)
    agent._disable_streaming = True

    # The floored child budget must cover the REAL gateway window the RCA
    # measured (Cloudflare proxy read timeout ~120s), with room to spare.
    # Pre-fix the implicit budget was 90s < 120s — this assertion is the
    # fix-discriminating tripwire; fail cheaply here if it is re-tightened.
    effective_budget = agent._compute_non_stream_stale_timeout({})
    assert effective_budget >= 120.0, (
        f"child stale budget {effective_budget}s no longer covers the ~120s "
        "gateway proxy window — the P-0097 race is re-armed"
    )

    result, elapsed = _run_turn_with_deadline(agent, "Summarize the incident", timeout_s=120)

    assert mock_provider.request_count == 2
    final_text = str(result.get("final_response") or "")
    assert "524" in final_text or "origin did not respond" in final_text or "proxy read timeout" in final_text, (
        f"expected the provider's 524 to be surfaced, got: {final_text[:400]!r}"
    )
    assert "non-streaming api call timed out" not in final_text.lower(), (
        "the local stale watchdog fired before the provider's 524 — pre-fix behavior"
    )
    # Two full provider-error cycles happened before the terminal result.
    assert elapsed >= 2 * delay_s - 1, f"expected >= {2 * delay_s}s, took {elapsed:.1f}s"
