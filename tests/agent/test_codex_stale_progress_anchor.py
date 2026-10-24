"""Regression tests: the non-streaming stale watchdog is anchored to model progress.

A long Codex generation that keeps streaming output (e.g. a ~12 minute planning
draft at high reasoning effort) used to be killed at the wall-clock stale timeout
(600s) while text was still arriving, then retried from scratch until the run
budget ran out. Once substantive model progress flows, only silence since the
latest progress may trip the stale kill. Before any progress, and for lifecycle
frames alone, the deadline stays the wall clock since call start.
"""

from __future__ import annotations

import time

import pytest

from tests.agent.test_codex_ttfb_watchdog import _make_codex_agent


def _setup(tmp_path, monkeypatch, stale_timeout):
    agent = _make_codex_agent(tmp_path, monkeypatch)
    # Only the stale watchdog may fire: no TTFB cutoff, no event-idle watchdog.
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: stale_timeout)
    closes: list = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: object())
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason))
    return agent, closes


def _mark(*, progress: bool) -> None:
    from agent.codex_runtime import _codex_watchdog_state_var

    state = _codex_watchdog_state_var.get()
    now = time.time()
    with state.lock:
        state.last_event_ts = now
        if progress:
            state.last_progress_ts = now


def test_streaming_progress_outlives_wall_clock_stale_timeout(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent, closes = _setup(tmp_path, monkeypatch, stale_timeout=1.0)
    sentinel = object()

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        end = time.time() + 2.5  # well past the 1s stale timeout
        while time.time() < end:
            _mark(progress=True)
            time.sleep(0.1)
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    assert h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"}) is sentinel
    assert "stale_call_kill" not in closes


def test_silence_after_progress_is_still_killed(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent, closes = _setup(tmp_path, monkeypatch, stale_timeout=1.0)
    stop = {"flag": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        end = time.time() + 1.5  # progress past the stale timeout, then wedge
        while time.time() < end:
            _mark(progress=True)
            time.sleep(0.1)
        while not stop["flag"] and not agent._interrupt_requested:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)
    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        elapsed = time.time() - t0
        assert 2.3 < elapsed < 15, f"stale kill fired at {elapsed:.1f}s"
        assert "stale_call_kill" in closes
        assert "with no new model output" in str(excinfo.value)
    finally:
        stop["flag"] = True


def test_lifecycle_frames_alone_keep_wall_clock_deadline(tmp_path, monkeypatch):
    from agent import chat_completion_helpers as h

    agent, closes = _setup(tmp_path, monkeypatch, stale_timeout=1.0)
    stop = {"flag": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        while not stop["flag"] and not agent._interrupt_requested:
            _mark(progress=False)  # transport chatter, never model output
            time.sleep(0.1)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)
    t0 = time.time()
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-5.5", "input": "hi"})
        assert time.time() - t0 < 15
        assert "stale_call_kill" in closes
        assert "with no response" in str(excinfo.value)
    finally:
        stop["flag"] = True


def test_wait_notice_stale_deadline_follows_latest_progress():
    from agent import chat_completion_wait_notice as wn

    kwargs = dict(stale_timeout=600.0, ttfb_enabled=False, ttfb_timeout=0.0,
        last_event_ts=None, retry_started_ts=None, call_start=100.0,
        idle_enabled=False, idle_timeout=0.0, idle_requires_progress=False)
    # No progress yet: the wall clock since call start.
    assert wn.codex_watchdog_deadline(last_progress_ts=None, elapsed=550.0, **kwargs) == ("stale", 50.0)
    # Progress at t=800 (elapsed 700): the deadline moves to 800 + 600.
    kwargs["last_event_ts"] = 800.0
    assert wn.codex_watchdog_deadline(last_progress_ts=800.0, elapsed=750.0, **kwargs) == ("stale", 550.0)
