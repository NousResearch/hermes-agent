"""Regression tests for the Codex TTFB (first parsed stream event) watchdog.

The chatgpt.com/backend-api/codex endpoint has an intermittent failure mode
where it accepts the connection but never emits a single stream event. The
watchdog in ``interruptible_api_call`` kills such a connection at a short TTFB
cutoff (instead of waiting out the much longer wall-clock stale timeout) so the
retry loop can reconnect promptly. Once any stream event arrives, the TTFB
watchdog is satisfied and a separate idle watchdog handles streams that stop
emitting SSE events.

Parsed-event activity is recorded in ``agent._codex_stream_last_event_ts``;
substantive model progress is recorded separately. For the implicit official
OpenAI Codex policy on large contexts, lifecycle frames satisfy TTFB without
arming the short post-progress idle budget. Small requests, explicit overrides,
and compatible backends retain their first-parsed-event semantics. Raw SSE
comments are outside this layer.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(
    tmp_path,
    monkeypatch,
    *,
    provider="openai-codex",
    base_url="https://chatgpt.com/backend-api/codex",
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider=provider,
        api_key="sk-dummy",
        base_url=base_url,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # The watchdog is gated on the codex_responses api_mode; assert/force it so
    # the test is robust to detection-logic changes elsewhere.
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    # Keep the wall-clock stale timeout high so any early kill is unambiguously
    # the TTFB path, not the stale-call path.
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0
    )
    return agent


def _shorten_implicit_idle_watchdog(monkeypatch, helpers, timeout=2.0):
    """Keep the resolver on its implicit branch while scaling time for tests."""
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    original = helpers._resolve_nonstream_watchdogs

    def resolve(agent, api_kwargs):
        watchdogs = original(agent, api_kwargs)
        watchdogs.idle_timeout = timeout
        return watchdogs

    monkeypatch.setattr(helpers, "_resolve_nonstream_watchdogs", resolve)


def _install_codex_event_stream(agent, monkeypatch, event_factory, closes):
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: event_factory())
    )
    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **_kwargs: client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda _client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda _client, reason=None: closes.append(reason),
    )


def test_ttfb_includes_silent_hang_hint_for_gpt_5_5(tmp_path, monkeypatch):
    """The no-first-event watchdog should surface the same actionable hint as the
    stale-call timeout path when the model matches the silent-hang heuristic."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.4")

    closes: list = []
    statuses: list[str] = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(agent, "_buffer_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(agent, "_emit_status", lambda msg: statuses.append(msg))
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        message = str(excinfo.value)
        assert "gpt-5.4" in message
        assert "gpt-5.3-codex" in message
        assert "gpt-5.4-codex" in message
        assert "codex_ttfb_kill" in closes
        assert statuses, "expected a user-facing watchdog status"
        assert any("gpt-5.4" in s and "gpt-5.3-codex" in s for s in statuses)
    finally:
        stop["flag"] = True


def test_ttfb_installs_and_retires_the_codex_request_token(tmp_path, monkeypatch):
    """The watchdog must publish a per-request token and clear it on the kill.

    ``run_codex_stream`` reads ``agent._active_codex_stream_request_token`` to
    tell whether it is still the owning attempt. Without an install here the
    whole retirement guard would be inert, and without the clear on kill a
    retired worker would keep normalizing partial deltas into a "completed"
    response.

    The worker also unwinds with its own local error after the force-close;
    that error must not replace the watchdog's retryable ``TimeoutError``.
    """
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "1")

    closes: list = []
    seen = {"token_while_running": None}
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        seen["token_while_running"] = getattr(
            agent, "_active_codex_stream_request_token", None
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            if getattr(agent, "_active_codex_stream_request_token", None) is None:
                # Retired by the watchdog — mimic the transport unwinding.
                raise RuntimeError("retired worker stream ended without terminal")
            time.sleep(0.02)
        raise RuntimeError("test timed out waiting for retirement")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    with pytest.raises(TimeoutError) as excinfo:
        h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})

    assert seen["token_while_running"] is not None, (
        "interruptible_api_call must install a request token before the worker runs"
    )
    assert "TTFB" in str(excinfo.value)
    assert "retired worker" not in str(excinfo.value)
    assert "codex_ttfb_kill" in closes
    assert getattr(agent, "_active_codex_stream_request_token", None) is None


def test_non_codex_api_mode_installs_no_request_token(tmp_path, monkeypatch):
    """The token is codex_responses-only — other api_modes stay untouched."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    agent.api_mode = "chat_completions"

    seen = {"token": "unset"}
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)

    def fake_dispatch(_agent, _api_kwargs, *, make_client):
        make_client("test")
        seen["token"] = getattr(
            _agent, "_active_codex_stream_request_token", "absent"
        )
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(h, "_dispatch_nonstreaming_api_request", fake_dispatch)

    h.interruptible_api_call(agent, {"model": "gpt-5.5", "messages": []})

    assert seen["token"] in (None, "absent")




def test_ttfb_does_not_kill_when_events_flow(tmp_path, monkeypatch):
    """Once a stream event has arrived, a generation that runs past the TTFB
    cutoff is NOT killed by the watchdog — it completes normally."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.4")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        # A parsed event marks stream activity right away; then keep generating
        # past the 0.4s TTFB cutoff before returning a real response.
        from agent.codex_runtime import _codex_watchdog_state_var

        now = time.time()
        state = _codex_watchdog_state_var.get()
        with state.lock:
            state.last_event_ts = now
        agent._codex_stream_last_event_ts = now
        if on_first_delta:
            on_first_delta()
        time.sleep(0.9)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    resp = h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
    assert resp is sentinel
    assert "codex_ttfb_kill" not in closes


@pytest.mark.parametrize(
    ("provider", "base_url", "input_chars", "idle_env", "idle_enabled", "requires_progress"),
    [
        ("openai-codex", "https://chatgpt.com/backend-api/codex", 40_004, None, True, True),
        ("openai-codex", "https://chatgpt.com/backend-api/codex", 40_004, "2", True, False),
        ("openai-codex", "https://chatgpt.com/backend-api/codex", 40_000, None, True, False),
        ("xai-oauth", "https://api.x.ai/v1", 40_004, None, True, False),
        ("openai-codex", "https://chatgpt.com/backend-api/codex", 40_004, "", True, True),
        ("openai-codex", "https://chatgpt.com/backend-api/codex", 40_004, "invalid", True, True),
        ("openai-codex", "https://chatgpt.com/backend-api/codex", 40_004, "0", False, False),
    ],
)
def test_idle_phase_policy_is_narrow_and_preserves_operator_overrides(
    tmp_path,
    monkeypatch,
    provider,
    base_url,
    input_chars,
    idle_env,
    idle_enabled,
    requires_progress,
):
    """Only an implicit, large, official request uses progress-phase arming."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(
        tmp_path, monkeypatch, provider=provider, base_url=base_url
    )
    if idle_env is None:
        monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", idle_env)

    watchdogs = h._resolve_nonstream_watchdogs(
        agent, {"model": "gpt-5.6-sol", "input": "x" * input_chars}
    )

    assert watchdogs.est_tokens == input_chars // 4
    assert watchdogs.idle_enabled is idle_enabled
    assert watchdogs.idle_requires_progress is requires_progress


@pytest.mark.parametrize(
    "mode", ["initial_gap", "stall", "retry_gap", "retry_no_event"]
)
def test_event_stale_phase_is_scoped_to_physical_stream_attempt(
    tmp_path, monkeypatch, mode
):
    """Retry lifecycle resets phase without hiding a zero-event reconnect hang."""
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    _shorten_implicit_idle_watchdog(monkeypatch, h)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("HERMES_CODEX_TTFB_STRICT", "1")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "5")

    closes: list = []
    attempts = {"count": 0}

    def stream_attempt():
        attempts["count"] += 1
        if mode == "initial_gap":
            yield SimpleNamespace(type="response.created")
            yield SimpleNamespace(type="response.in_progress")
            time.sleep(3.0)
            yield SimpleNamespace(type="response.reasoning_text.delta", delta="working")
            yield SimpleNamespace(type="response.output_text.delta", delta="done")
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="resp-1", usage=None),
            )
            return
        if mode == "retry_no_event" and attempts["count"] == 2:
            while getattr(agent, "_active_codex_stream_request_token", None) is not None:
                time.sleep(0.02)
            raise RuntimeError("retired zero-event retry")
        if mode == "retry_gap" and attempts["count"] == 2:
            yield SimpleNamespace(type="response.created")
            yield SimpleNamespace(type="response.in_progress")
            time.sleep(3.0)
            yield SimpleNamespace(type="response.reasoning_text.delta", delta="retry step")
            yield SimpleNamespace(type="response.output_text.delta", delta="done")
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", id="resp-2", usage=None),
            )
            return

        yield SimpleNamespace(type="response.created")
        if mode != "retry_no_event":
            yield SimpleNamespace(type="response.reasoning_text.delta", delta="first step")
        if mode in {"retry_gap", "retry_no_event"}:
            raise ConnectionError("retry physical stream")
        while getattr(agent, "_active_codex_stream_request_token", None) is not None:
            time.sleep(0.02)
        raise ConnectionError("retired stalled stream")

    _install_codex_event_stream(agent, monkeypatch, stream_attempt, closes)

    if mode in {"initial_gap", "retry_gap"}:
        response = h.interruptible_api_call(
            agent, {"model": "gpt-5.6-sol", "input": "x" * 40_004}
        )
        assert response.output_text == "done"
        assert attempts["count"] == (1 if mode == "initial_gap" else 2)
    else:
        error_match = "no parsed stream event" if mode == "retry_no_event" else "no SSE events"
        with pytest.raises(TimeoutError, match=error_match):
            h.interruptible_api_call(
                agent, {"model": "gpt-5.6-sol", "input": "x" * 40_004}
            )

    assert ("codex_stream_idle_kill" in closes) is (mode == "stall")
    assert ("codex_ttfb_kill" in closes) is (mode == "retry_no_event")
    if mode == "stall":
        assert attempts["count"] == 1
    if mode == "retry_no_event":
        assert attempts["count"] == 2


@pytest.mark.parametrize(
    "stale_timeout",
    [float("inf"), float("-inf"), float("nan")],
)
def test_wait_notice_omits_reconnect_when_all_deadlines_are_non_finite(
    stale_timeout,
):
    """A disabled watchdog must not be advertised as a future reconnect."""
    from agent import chat_completion_helpers as h

    recovery = h._codex_wait_notice_recovery(
        stale_timeout=stale_timeout,
        ttfb_enabled=False,
        ttfb_timeout=float("nan"),
        last_event_ts=None,
        last_progress_ts=None,
        retry_started_ts=None,
        call_start=100.0,
        idle_enabled=False,
        idle_timeout=float("nan"),
        idle_requires_progress=False,
        elapsed=30.0,
    )

    assert recovery == ""






def test_moa_heartbeat_survives_infinite_stale_timeout(monkeypatch):
    """The full 100-poll MoA heartbeat must leave a healthy call running."""
    from agent import chat_completion_helpers as h

    notices: list[str] = []
    response = SimpleNamespace(ok=True)
    agent = SimpleNamespace(
        platform="desktop",
        api_mode="chat_completions",
        provider="moa",
        _consecutive_stale_streams=0,
        _interrupt_requested=False,
        _compute_non_stream_stale_timeout=lambda _kwargs: float("inf"),
        _touch_activity=lambda _message: None,
        _emit_wait_notice=notices.append,
    )

    class HeartbeatThread:
        """Keep the synthetic worker alive through one heartbeat."""

        def __init__(self, *, target, daemon):
            self._polls = 0
            self._target = target

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            self._polls += 1
            if self._polls == 101:
                self._target()
                return False
            return True

    monkeypatch.setattr(h.threading, "Thread", HeartbeatThread)
    monkeypatch.setattr(
        h,
        "_dispatch_nonstreaming_api_request",
        lambda *_args, **_kwargs: response,
    )

    result = h.interruptible_api_call(agent, {"model": "openai-xai-wide"})

    assert result is response
    assert len(notices) == 1
    assert "waiting on openai-xai-wide" in notices[0]
    assert "auto-reconnect" not in notices[0]


def test_wait_notice_formatting_error_does_not_abort_request(monkeypatch):
    """Status construction is fail-open even if its formatter breaks."""
    from agent import chat_completion_helpers as h

    response = SimpleNamespace(ok=True)
    agent = SimpleNamespace(
        platform="desktop",
        api_mode="chat_completions",
        provider="moa",
        _consecutive_stale_streams=0,
        _interrupt_requested=False,
        _compute_non_stream_stale_timeout=lambda _kwargs: float("inf"),
        _touch_activity=lambda _message: None,
        _emit_wait_notice=lambda _message: None,
    )

    class HeartbeatThread:
        def __init__(self, *, target, daemon):
            self._polls = 0
            self._target = target

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            self._polls += 1
            if self._polls == 101:
                self._target()
                return False
            return True

    monkeypatch.setattr(h.threading, "Thread", HeartbeatThread)
    monkeypatch.setattr(
        h,
        "_dispatch_nonstreaming_api_request",
        lambda *_args, **_kwargs: response,
    )
    monkeypatch.setattr(
        h,
        "_codex_wait_notice_recovery",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad display state")),
    )

    result = h.interruptible_api_call(agent, {"model": "openai-xai-wide"})

    assert result is response










def test_large_codex_request_hard_ceiling_reclaims_silent_stall(tmp_path, monkeypatch):
    """#64507 regression: a large Codex request (TTFB watchdog disabled by the
    size gate, stale floor *raised*) that never emits a parsed event must still
    be reclaimed at a finite hard ceiling — not hang for 13+ minutes while the
    worker stays idle and the session shows as active.

    Uses the real default TTFB threshold (120s) and asserts the request dies at
    the hard ceiling regardless of the size-based TTFB disable.
    """
    from agent import chat_completion_helpers as h

    agent = _make_codex_agent(tmp_path, monkeypatch)
    # Real default TTFB threshold (no HERMES_CODEX_TTFB_* override) → for a
    # >10k-token request the no-event TTFB watchdog is auto-disabled.
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "3")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_hang(api_kwargs, client=None, on_first_delta=None):
        # No event marker AND no event ever: the exact issue-64507 stall.
        deadline = time.time() + 120
        while time.time() < deadline and not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_hang)

    large_input = "x" * 44_000  # ~11k estimated tokens → TTFB disabled, stale raised
    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": large_input})
        elapsed = time.time() - t0
        # Must die at the hard ceiling (3s), nowhere near the raised stale floor.
        assert elapsed < 30, f"hard ceiling took {elapsed:.1f}s — stall not reclaimed"
        assert "stale_call_kill" in closes, f"stale kill expected, got {closes}"
        assert "timed out after" in str(excinfo.value)
        assert "with no response" in str(excinfo.value)
    finally:
        stop["flag"] = True
