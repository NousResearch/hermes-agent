"""``fallback.no_first_byte_budget_seconds``: one wall budget over a request that produces no byte.

A provider that accepts the connection and never streams a byte is retried in place, each attempt
on its own stale timer, before ``fallback_providers`` is consulted. With the budget set, the
streaming monitor cuts the attempt once the budget is spent and a fallback entry is pending, and
the error path moves to the next entry.

The end-to-end tests drive the real ``AIAgent`` streaming path against local HTTP servers: a
primary that answers 200 + SSE headers and then holds the body open with no bytes (or a variant),
and a fallback that answers normally. The stale timer is scaled down so a run takes seconds.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from agent import first_byte_budget as fbb


def _chunk(delta, finish=None):
    return {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class _Server:
    """``plan``: one mode per request, the last repeating. ``silent`` = 200 + SSE headers then no
    bytes until released; ``ok`` = a one-chunk reply; ``error500`` = an immediate HTTP 500;
    ``slow`` = ten chunks 0.3 s apart (a healthy stream that outlasts a short budget)."""

    def __init__(self, *plan: str, text: str = "ok") -> None:
        self.hits: list[float] = []
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def _send(self, *chunks):
                for c in chunks:
                    self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                    self.wfile.flush()

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                self.rfile.read(n)
                if not self.path.endswith("/chat/completions"):
                    self.send_response(404)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                outer.hits.append(time.monotonic())
                mode = plan[min(len(outer.hits), len(plan)) - 1]
                if mode == "error500":
                    body = b'{"error": {"message": "internal error", "type": "server_error"}}'
                    self.send_response(500)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                try:
                    if mode == "silent":
                        outer.release.wait(timeout=60.0)
                        return
                    if mode == "slow":
                        for i in range(10):
                            self._send(_chunk({"content": f"{i} "}))
                            outer.release.wait(timeout=0.3)
                        self._send(_chunk({}, "stop"))
                    else:
                        self._send(_chunk({"content": text}), _chunk({}, "stop"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def close(self) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def make_server():
    made: list[_Server] = []

    def _make(*plan: str, text: str = "ok") -> _Server:
        made.append(_Server(*plan, text=text))
        return made[-1]

    yield _make
    for s in made:
        s.close()


class _FakeBackoffClock:
    """Stands in for ``agent.turn_recovery.time`` so the retry backoff takes no real time."""

    def __init__(self):
        self._offset = 0.0

    def time(self):
        return time.time() + self._offset

    def sleep(self, seconds):
        self._offset += seconds

    def __getattr__(self, name):
        return getattr(time, name)


def _run(primary, fallback, monkeypatch, *, budget, with_fallback=True, stale=1.0, backoff=0.0,
         real_backoff=False, setup=None):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", str(stale))
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "0")
    chain = [{"provider": "custom", "model": "fallback-model", "base_url": fallback.url, "api_key": "k2"}]
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}):
        agent = AIAgent(api_key="k", base_url=primary.url, provider="custom", model="primary-model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
                        fallback_model=chain if with_fallback else None)
    agent._api_max_retries = 2
    agent._auto_recovery_cycles = 0  # the post-exhaustion wait ladder is not under test here
    if setup is not None:
        setup(agent)
    from hermes_cli import config as cfg_mod
    real = cfg_mod.load_config_readonly

    def _cfg(*a, **k):
        cfg = dict(real(*a, **k) or {})
        cfg["fallback"] = {**(cfg.get("fallback") or {}), "no_first_byte_budget_seconds": budget}
        return cfg

    statuses: list[str] = []
    real_buffer = agent._buffer_diagnostic_status

    def _buffer(message, *a, **k):
        statuses.append(str(message))
        return real_buffer(message, *a, **k)

    with patch("hermes_cli.config.load_config_readonly", _cfg), \
         patch("agent.model_metadata.get_model_context_length", return_value=200000), \
         patch("agent.retry_utils.jittered_backoff", return_value=backoff), \
         patch("agent.turn_recovery.time", time if real_backoff else _FakeBackoffClock()), \
         patch.object(agent, "_buffer_diagnostic_status", _buffer), \
         patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"):
        result = agent.run_conversation("hello")
    return result, statuses


def test_budget_moves_a_silent_primary_to_the_fallback_before_the_retries_run_out(make_server, monkeypatch):
    primary, fallback = make_server("silent"), make_server("ok", text="served by fallback")
    result, statuses = _run(primary, fallback, monkeypatch, budget=1.5)
    assert result["final_response"] == "served by fallback"
    # The cut lands in the second stale window: the remaining in-place stream retries and the
    # outer API retries (6 primary requests with the knob off, see the control) never go out.
    assert 1 <= len(primary.hits) <= 3, primary.hits
    assert len(fallback.hits) == 1
    # The switch is attributed to the budget, not to a generic timeout.
    assert any("No first byte from primary-model" in s for s in statuses), statuses


def test_budget_off_keeps_the_per_attempt_retries(make_server, monkeypatch):
    """Control: with the knob at 0 every stream retry and API retry runs before the chain moves."""
    primary, fallback = make_server("silent"), make_server("ok", text="served by fallback")
    result, _ = _run(primary, fallback, monkeypatch, budget=0)
    assert result["final_response"] == "served by fallback"
    assert len(primary.hits) >= 4, primary.hits


def test_without_a_pending_fallback_the_budget_never_cuts(make_server, monkeypatch):
    """Nothing to move to: the stale timer and retry budget decide, exactly as with the knob off."""
    primary, fallback = make_server("silent"), make_server("ok")
    result, _ = _run(primary, fallback, monkeypatch, budget=0.5, with_fallback=False)
    assert result.get("failed") is True
    assert len(primary.hits) >= 4, primary.hits
    assert not fallback.hits


def test_a_stream_that_is_delivering_is_never_cut(make_server, monkeypatch):
    """The first byte ends the wait: a healthy answer that streams for longer than the budget
    finishes on the primary."""
    primary, fallback = make_server("slow"), make_server("ok", text="served by fallback")
    result, _ = _run(primary, fallback, monkeypatch, budget=1.0, stale=30.0)
    assert result["final_response"].startswith("0 1 2"), result["final_response"]
    assert len(primary.hits) == 1 and not fallback.hits


def test_our_own_retry_backoff_does_not_count_against_the_budget(make_server, monkeypatch):
    """A 500, then 3 s of our own retry backoff, then a silent attempt. The budget measures provider
    silence only, so the cut waits for 2 s of silence after the silent attempt opens. Without the
    credit the clock already reads ~3 s there and the cut fires on the monitor's first tick."""
    primary, fallback = make_server("error500", "silent"), make_server("ok", text="served by fallback")
    # Spend the one-per-minute non-streaming 5xx probe up front so the 500 reaches the retry loop.
    result, _ = _run(primary, fallback, monkeypatch, budget=2.0, stale=60.0, backoff=3.0, real_backoff=True,
                     setup=lambda agent: setattr(agent, "_stream_5xx_probe_ts", time.monotonic()))
    assert result["final_response"] == "served by fallback"
    assert len(primary.hits) == 2, primary.hits
    silence = fallback.hits[0] - primary.hits[1]
    assert silence >= 1.5, silence


# ── clock semantics ──


class _Agent:
    def __init__(self, pending=True):
        self._first_byte_budget = None
        self._pending = pending

    def _has_pending_fallback(self):
        return self._pending


@pytest.fixture
def budget(monkeypatch):
    def _set(seconds):
        monkeypatch.setattr(fbb, "budget_seconds", lambda: float(seconds))
    return _set


def test_clock_spans_retries_and_cuts_once(budget, monkeypatch):
    budget(10)
    now = [100.0]
    monkeypatch.setattr(fbb.time, "monotonic", lambda: now[0])
    agent = _Agent()
    fbb.arm(agent)
    now[0] = 106.0
    fbb.arm(agent)  # a retry re-arms: the clock keeps running
    assert fbb.elapsed(agent) == 6.0 and not fbb.should_cut(agent)
    now[0] = 110.0
    assert fbb.should_cut(agent) is True
    assert fbb.should_cut(agent) is False  # once per wait


def test_credit_and_disarm(budget, monkeypatch):
    budget(10)
    now = [0.0]
    monkeypatch.setattr(fbb.time, "monotonic", lambda: now[0])
    agent = _Agent()
    fbb.arm(agent)
    now[0] = 12.0
    fbb.note_excused_wait(agent, 5.0)
    assert fbb.elapsed(agent) == 7.0 and not fbb.should_cut(agent)
    fbb.disarm(agent)
    assert agent._first_byte_budget is None and fbb.elapsed(agent) is None
    fbb.note_excused_wait(agent, 5.0)  # unarmed: no-op


def test_off_creates_no_state_and_no_cut_without_a_pending_fallback(budget, monkeypatch):
    budget(0)
    agent = _Agent()
    fbb.arm(agent)
    assert agent._first_byte_budget is None
    budget(1)
    now = [0.0]
    monkeypatch.setattr(fbb.time, "monotonic", lambda: now[0])
    agent = _Agent(pending=False)
    fbb.arm(agent)
    now[0] = 50.0
    assert fbb.should_cut(agent) is False


def test_shipped_default_is_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["fallback"]["no_first_byte_budget_seconds"] == 0


def test_cut_is_classified_as_a_timeout():
    from agent.error_classifier import FailoverReason, classify_api_error
    classified = classify_api_error(fbb.FirstByteBudgetExceeded("cut", provider="p", elapsed_s=1.0),
                                    provider="p", model="m")
    assert classified.reason == FailoverReason.timeout
