"""The generic stale watchdog on a streamed ``codex_responses`` request measures time
since the request's last substantive model progress, not time since the request started.

``codex_responses`` requests run through ``interruptible_api_call`` but are streamed
underneath (``run_codex_stream``), and every substantive event (reasoning-summary,
reasoning, output-text, tool-call deltas) is recorded on the request-local watchdog
state. A stream that keeps making progress is alive up to an absolute ceiling (hosted
Codex hard ceiling / run budget); a stream that stops making progress (silence, or
lifecycle/keepalive frames only) is still stale after ``stale_timeout``. Before the first
progress event, and whenever the extension is off, nothing changes.

Decisions are unit-tested deterministically; a few end-to-end tests drive the real
``run_codex_stream`` with scripted events and scaled-down timers (wide margins).
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace
from typing import Optional

import pytest

STALE = 1.0  # scaled-down generic stale timeout for the end-to-end tests (poll interval is 0.3 s)


def _make_agent(tmp_path, monkeypatch, *, provider="xai-oauth", base_url="https://api.x.ai/v1",
                model="grok-4.7"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "HERMES_CODEX_TTFB_MAX_SECONDS",
                 "HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", "HERMES_CODEX_TTFB_STRICT",
                 "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "HERMES_CODEX_HARD_TIMEOUT_SECONDS",
                 "HERMES_API_CALL_STALE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    from run_agent import AIAgent

    agent = AIAgent(model=model, provider=provider, api_key="sk-dummy", base_url=base_url,
                    quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def _scale_watchdogs(monkeypatch, captured=None, **fields):
    """Keep the real resolver, then scale its thresholds (stale defaults to ``STALE``)."""
    from agent import chat_completion_helpers as h

    original = h._resolve_nonstream_watchdogs

    def resolve(agent, api_kwargs):
        wd = original(agent, api_kwargs)
        if captured is not None:
            captured.append(SimpleNamespace(**vars(wd)))  # as resolved, before scaling
        wd.stale_timeout = STALE
        for name, value in fields.items():
            setattr(wd, name, value)
        return wd

    monkeypatch.setattr(h, "_resolve_nonstream_watchdogs", resolve)


def _install_stream(agent, monkeypatch, attempt_factory, closes):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kw: attempt_factory()))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kw: client)
    monkeypatch.setattr(agent, "_abort_request_openai_client",
                        lambda _c, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client",
                        lambda _c, reason=None: closes.append(reason))


def _pace(seconds: float = 0.1) -> None:
    """Gap between two scripted stream events: wall time passing while the stream stays (or stops
    being) productive is exactly what the stale watchdog measures."""
    time.sleep(seconds)  # health: allow HX013 -- the event cadence against a wall-clock watchdog is the behaviour under test


def _current(agent) -> bool:
    return getattr(agent, "_active_codex_stream_request_token", None) is not None


def _reasoning(text="thinking"):
    return SimpleNamespace(type="response.reasoning_summary_text.delta", delta=text)


def _finish(text="done"):
    yield SimpleNamespace(type="response.output_text.delta", delta=text)
    yield SimpleNamespace(type="response.completed",
                          response=SimpleNamespace(status="completed", id="resp-1", usage=None))


def _wait_until_retired(agent, limit=10.0):
    deadline = time.monotonic() + limit
    while _current(agent) and time.monotonic() < deadline:
        _pace(0.02)


def _endless(agent, event_factory, limit=10.0):
    """An attempt that keeps emitting ``event_factory()`` until retired (bounded, so a broken
    watchdog fails the test instead of hanging it)."""
    def attempt():
        yield SimpleNamespace(type="response.created")
        deadline = time.monotonic() + limit
        while _current(agent) and time.monotonic() < deadline:
            yield event_factory()
            _pace()
        raise ConnectionError("retired while still streaming")
    return attempt


# --- Decisions (deterministic) ------------------------------------------------------------

@pytest.mark.parametrize(("now", "elapsed", "anchor", "expected"), [
    (150.0, 50.0, None, None),             # before progress, inside the wall-clock window
    (191.0, 91.0, None, "stale"),          # before progress: wall clock, exactly as before
    (500.0, 400.0, 450.0, None),           # progressing, 50 s since the last progress
    (500.0, 400.0, 405.0, "no_progress"),  # 95 s since the last progress
    (1700.0, 1600.0, 1690.0, "ceiling"),   # still progressing at the 1500 s ceiling
    (1700.0, 1600.0, 1500.0, "no_progress"),  # quiet AND past the ceiling: reported as quiet
])
def test_stale_verdict(now, elapsed, anchor, expected):
    from agent.chat_completion_nonstream import _stale_verdict

    assert _stale_verdict(now=now, elapsed=elapsed, anchor=anchor, stale_timeout=90.0,
                          ceiling=1500.0) == expected


def test_stale_verdict_with_a_ceiling_close_to_the_stale_window():
    """Ceiling 100 s, stale 90 s: quiet for 86 s at the ceiling is a ceiling stop (the kill text
    reports the 86 s), quiet for 91 s before it is a no-progress kill."""
    from agent.chat_completion_nonstream import _stale_verdict

    assert _stale_verdict(now=1101.0, elapsed=101.0, anchor=1015.0, stale_timeout=90.0,
                          ceiling=100.0) == "ceiling"
    assert _stale_verdict(now=1099.0, elapsed=99.0, anchor=1008.0, stale_timeout=90.0,
                          ceiling=100.0) == "no_progress"


def _set_run_budget(monkeypatch, agent, seconds: float, started_ago: float = 0.0) -> None:
    monkeypatch.setattr(agent, "run_budget_seconds", seconds, raising=False)
    monkeypatch.setattr(agent, "_run_budget_started_at", time.time() - started_ago, raising=False)


def _request(agent, **kwargs):
    from agent.chat_completion_nonstream import _NonStreamRequest

    return _NonStreamRequest(agent, {"model": agent.model, "input": "hi", **kwargs})


def test_progress_ceiling_defaults_to_the_hosted_codex_hard_ceiling(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    assert _request(agent).progress_ceiling == 1500.0
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "600")
    assert _request(agent).progress_ceiling == 600.0


def test_progress_ceiling_is_capped_by_the_run_budget(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    _set_run_budget(monkeypatch, agent, 300.0, started_ago=100.0)  # 200 s left -> half of it
    assert 99.0 < _request(agent).progress_ceiling <= 100.0


@pytest.mark.parametrize("variant", ["explicit_env", "explicit_config", "hard_ceiling_disabled",
                                     "local_endpoint", "non_codex"])
def test_progress_extension_is_off(tmp_path, monkeypatch, variant):
    """An explicit stale timeout keeps its wall-clock meaning (#115024); without a finite ceiling,
    or outside Codex Responses, the stale clock is never extended."""
    if variant == "local_endpoint":
        agent = _make_agent(tmp_path, monkeypatch, provider="custom",
                            base_url="http://127.0.0.1:1234/v1", model="local-model")
    else:
        agent = _make_agent(tmp_path, monkeypatch)
    if variant == "explicit_env":
        monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "90")
    if variant == "explicit_config":
        monkeypatch.setattr(agent, "_stale_timeout_is_explicit", lambda: True)
    if variant == "hard_ceiling_disabled":
        monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0")
    if variant == "non_codex":
        agent.api_mode = "chat_completions"
    request = _request(agent)
    if variant == "local_endpoint":  # an implicit local stale is already inf; pin a finite one
        request.wd.stale_timeout = 90.0
        request.progress_ceiling = request._resolve_progress_ceiling()
    assert request.progress_ceiling == math.inf
    if request.codex_watchdog_state is not None:
        request.codex_watchdog_state.request_progress_ts = time.time()
    assert request._codex_watchdog_snapshot().progress_anchor_ts is None


def test_progress_extension_is_off_when_the_ceiling_would_not_exceed_the_stale_timeout(
        tmp_path, monkeypatch):
    """A large request floors the stale timeout (600 s) after the run-budget cap; with 300 s of
    budget left the ceiling would be 150 s, so following progress could only cut it short."""
    agent = _make_agent(tmp_path, monkeypatch)
    _set_run_budget(monkeypatch, agent, 300.0)
    request = _request(agent, input="x" * 40_004)
    assert request.wd.stale_timeout >= 600.0
    assert request.progress_ceiling == math.inf


def test_progress_extension_is_off_when_the_run_budget_binds_the_stale_timeout(tmp_path, monkeypatch):
    """120 s of budget left caps the stale timeout at 60 s; the ceiling is the same cap, so there
    is no room to extend and the stale timeout stays wall-clock."""
    agent = _make_agent(tmp_path, monkeypatch)
    _set_run_budget(monkeypatch, agent, 120.0)
    request = _request(agent)
    assert request.wd.stale_timeout <= 60.0
    assert request.progress_ceiling == math.inf


def test_local_endpoint_with_a_run_budget_uses_the_budget_as_ceiling(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch, provider="custom",
                        base_url="http://127.0.0.1:1234/v1", model="local-model")
    _set_run_budget(monkeypatch, agent, 300.0)
    request = _request(agent)
    request.wd.stale_timeout = 90.0
    assert 149.0 < request._resolve_progress_ceiling() <= 150.0


def test_anchor_survives_the_phase_aware_retry_reset(tmp_path, monkeypatch):
    """The physical retry clears ``last_progress_ts``; the request-wide anchor stays, so the
    reconnect is judged from the last progress, not from ``call_start``."""
    from agent.chat_completion_nonstream import _stale_verdict

    agent = _make_agent(tmp_path, monkeypatch)
    request = _request(agent)
    state = request.codex_watchdog_state
    assert state is not None
    request.call_start = 1000.0
    state.request_progress_ts = 1150.0      # attempt 1 progressed until t=150 s
    state.last_progress_ts = None           # ... then the retry reset cleared the per-attempt marker
    state.last_event_ts = None
    state.retry_started_ts = 1160.0
    snap = request._codex_watchdog_snapshot()
    assert snap.progress_anchor_ts == 1150.0
    assert _stale_verdict(now=1200.0, elapsed=200.0, anchor=snap.progress_anchor_ts,
                          stale_timeout=90.0, ceiling=request.progress_ceiling) is None


def test_wait_notice_deadline_follows_the_progress_anchor():
    from agent import chat_completion_wait_notice as wn

    def deadline(*, elapsed: float, progress_ts: Optional[float] = None,
                 ceiling: float = math.inf):
        # call_start is 100.0; progress_ts doubles as the last event and the anchor.
        return wn.codex_watchdog_deadline(stale_timeout=90.0, ttfb_enabled=False, ttfb_timeout=120.0,
            last_event_ts=progress_ts, last_progress_ts=progress_ts, retry_started_ts=None,
            call_start=100.0, idle_enabled=False, idle_timeout=12.0, idle_requires_progress=False,
            elapsed=elapsed, progress_anchor_ts=progress_ts, progress_ceiling=ceiling)

    assert deadline(elapsed=60.0) == ("wall-clock stale", 30.0)
    # Progress at 150 s into the call, now 200 s: the stale deadline is 150 + 90 = 240 s.
    assert deadline(elapsed=200.0, progress_ts=250.0, ceiling=1500.0) == ("no-progress stale", 40.0)
    # Progress at 1440 s, now 1450 s: the 1500 s ceiling comes before 1440 + 90.
    assert deadline(elapsed=1450.0, progress_ts=1540.0, ceiling=1500.0) == ("absolute ceiling", 50.0)
    # Extension off (infinite ceiling): the anchor is ignored.
    assert deadline(elapsed=60.0, progress_ts=150.0) == ("wall-clock stale", 30.0)


# --- End to end (real run_codex_stream, scripted events) ------------------------------------

@pytest.mark.parametrize("progress_type", [
    "response.reasoning_summary_text.delta",
    "response.output_text.delta",
])
def test_progressing_stream_outlives_the_stale_timeout(tmp_path, monkeypatch, progress_type):
    """Substantive deltas every 0.1 s for 2.5x the stale timeout: the call completes."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    _scale_watchdogs(monkeypatch, idle_timeout=5.0)
    closes: list = []

    def attempt():
        yield SimpleNamespace(type="response.created")
        end = time.monotonic() + STALE * 2.5
        while time.monotonic() < end:
            yield SimpleNamespace(type=progress_type, delta="x")
            _pace()
        yield from _finish()

    _install_stream(agent, monkeypatch, attempt, closes)
    response = h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})

    assert response.output_text.endswith("done")
    assert "stale_call_kill" not in closes
    assert h._stale_streak(agent) == 0


def test_progress_then_silence_is_stale_killed_from_last_progress_with_idle_disabled(
        tmp_path, monkeypatch):
    """Event-idle disabled: generic stale recovery still owns a stream that goes quiet,
    measured from the last progress (not from the start, and not deferred to the ceiling)."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    _scale_watchdogs(monkeypatch, idle_enabled=False)
    closes: list = []
    progress_for = STALE * 1.5

    def attempt():
        yield SimpleNamespace(type="response.created")
        end = time.monotonic() + progress_for
        while time.monotonic() < end:
            yield _reasoning()
            _pace()
        _wait_until_retired(agent)
        raise ConnectionError("retired after going quiet")

    _install_stream(agent, monkeypatch, attempt, closes)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="made no model progress"):
        h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})
    elapsed = time.monotonic() - started

    assert "stale_call_kill" in closes
    assert progress_for + STALE * 0.7 < elapsed < progress_for + STALE + 2.0
    assert h._stale_streak(agent) == 1


def test_progress_then_lifecycle_frames_only_is_a_no_progress_kill(tmp_path, monkeypatch):
    """Default shape of a wedged stream: progress, then only lifecycle/keepalive frames. Those
    keep the event-idle watchdog fed; the stale clock still fires from the last progress."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    _scale_watchdogs(monkeypatch, idle_timeout=5.0)
    closes: list = []
    progress_for = STALE * 0.5

    def attempt():
        yield SimpleNamespace(type="response.created")
        end = time.monotonic() + progress_for
        while time.monotonic() < end:
            yield _reasoning()
            _pace()
        deadline = time.monotonic() + 10.0
        while _current(agent) and time.monotonic() < deadline:
            yield SimpleNamespace(type="response.in_progress")
            _pace()
        raise ConnectionError("retired lifecycle-only stream")

    _install_stream(agent, monkeypatch, attempt, closes)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="made no model progress"):
        h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})

    assert progress_for + STALE * 0.7 < time.monotonic() - started < progress_for + STALE + 2.0
    assert "stale_call_kill" in closes
    assert "codex_stream_idle_kill" not in closes


def test_lifecycle_and_empty_deltas_do_not_extend_the_stale_clock(tmp_path, monkeypatch):
    """Frames that prove only transport liveness never move the progress anchor."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    _scale_watchdogs(monkeypatch, idle_timeout=5.0)
    closes: list = []
    frames = iter([SimpleNamespace(type="response.in_progress"),
                   SimpleNamespace(type="response.reasoning_summary_text.delta", delta="")] * 200)
    _install_stream(agent, monkeypatch, _endless(agent, lambda: next(frames)), closes)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out after"):
        h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})

    assert time.monotonic() - started < STALE + 1.5
    assert "stale_call_kill" in closes


def test_reconnect_after_progress_is_judged_from_the_last_progress(tmp_path, monkeypatch):
    """Phase-aware path: the retry clears the per-attempt progress marker. Attempt 2 waits
    0.4x the stale timeout before its first progress, by which time the request is 1.6x the
    stale timeout old: anchored on call_start (or on the cleared marker) it would be killed."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch, provider="openai-codex",
                        base_url="https://chatgpt.com/backend-api/codex", model="gpt-5.6-sol")
    captured: list = []
    stale = 2.0  # wider than the module default: attempt 2's quiet start needs margin
    _scale_watchdogs(monkeypatch, captured, stale_timeout=stale, idle_timeout=10.0, progress_timeout=10.0,
                     ttfb_timeout=10.0)
    closes: list = []
    attempts = {"n": 0}

    def attempt():
        attempts["n"] += 1
        yield SimpleNamespace(type="response.created")
        if attempts["n"] == 1:
            end = time.monotonic() + stale * 1.2
            while time.monotonic() < end:
                yield _reasoning("first attempt")
                _pace()
            raise ConnectionError("transport dropped mid-stream")
        _pace(stale * 0.4)
        yield _reasoning("second attempt")
        yield from _finish()

    _install_stream(agent, monkeypatch, attempt, closes)
    response = h.interruptible_api_call(agent, {"model": "gpt-5.6-sol", "input": "x" * 40_004})

    assert captured[0].idle_requires_progress  # the phase-aware branch that resets on retry
    assert response.output_text == "done"
    assert attempts["n"] == 2
    assert "stale_call_kill" not in closes


def test_progressing_stream_is_stopped_at_the_absolute_ceiling(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "2.5")
    _scale_watchdogs(monkeypatch, idle_timeout=5.0)
    closes: list = []
    _install_stream(agent, monkeypatch, _endless(agent, _reasoning), closes)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="absolute ceiling"):
        h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})

    assert 2.4 < time.monotonic() - started < 4.5
    assert "codex_progress_ceiling_kill" in closes
    assert "stale_call_kill" not in closes
    assert h._stale_streak(agent) == 1  # counted like the stale kill it replaces


def test_explicit_stale_timeout_still_cuts_a_progressing_stream(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", str(STALE))
    _scale_watchdogs(monkeypatch, idle_timeout=5.0)
    closes: list = []
    _install_stream(agent, monkeypatch, _endless(agent, _reasoning), closes)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out after"):
        h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})
    assert time.monotonic() - started < STALE + 1.5
    assert "stale_call_kill" in closes


def test_non_codex_request_keeps_the_wall_clock_stale_timeout(tmp_path, monkeypatch):
    """No stream state exists outside codex_responses; behaviour there is unchanged."""
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch, provider="openrouter",
                        base_url="https://openrouter.ai/api/v1", model="some/model")
    agent.api_mode = "chat_completions"
    _scale_watchdogs(monkeypatch)
    closes: list = []
    aborted = threading.Event()

    def slow_request(*_a, **_k):
        aborted.wait(timeout=10.0)
        raise RuntimeError("closed")

    monkeypatch.setattr(h, "_dispatch_nonstreaming_api_request",
                        lambda agent_, kwargs, make_client: (make_client("t"), slow_request())[1])
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_kw: SimpleNamespace())
    monkeypatch.setattr(agent, "_abort_request_openai_client",
                        lambda _c, reason=None: (closes.append(reason), aborted.set()))
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda _c, reason=None: None)

    started = time.monotonic()
    # The worker's own transport error surfaces here (non-codex workers are not retired).
    with pytest.raises((RuntimeError, TimeoutError)):
        h.interruptible_api_call(agent, {"model": "some/model", "messages": []})
    assert time.monotonic() - started < STALE + 1.5
    assert "stale_call_kill" in closes


def test_interrupt_during_progress_phase_is_not_counted_as_stale(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent = _make_agent(tmp_path, monkeypatch)
    _scale_watchdogs(monkeypatch, idle_timeout=5.0)
    closes: list = []
    _install_stream(agent, monkeypatch, _endless(agent, _reasoning), closes)
    timer = threading.Timer(STALE * 1.5, lambda: setattr(agent, "_interrupt_requested", True))
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            h.interruptible_api_call(agent, {"model": "grok-4.7", "input": "hi"})
    finally:
        timer.cancel()
        agent._interrupt_requested = False

    # Either side may notice the interrupt first (the stream loop checks it per event); what
    # matters is that a stop during the progress phase is never a stale or ceiling kill.
    assert "stale_call_kill" not in closes
    assert "codex_progress_ceiling_kill" not in closes
    assert h._stale_streak(agent) == 0
