"""Tests for the cron WALL-CLOCK max-runtime cap (HERMES_CRON_MAX_RUNTIME).

The inactivity watchdog (HERMES_CRON_TIMEOUT) only fires when the agent goes quiet. A job
that keeps making API calls never goes idle, so it ran unbounded: kanban t_b3e3f2cb, job
``d03b9c4f77ff`` ran 44 API calls over 16 minutes while the gateway event loop was wedged
and the inactivity limit never tripped.

These tests pin the new wall-clock cap: it fires on a BUSY agent (activity keeps
ticking), it does not fire on a fast run, and it can be disabled.
"""

from __future__ import annotations

import threading
import time

import pytest


class _BusyAgent:
    """An agent that never goes idle: it keeps reporting activity, so only a wall-clock
    cap can stop it (the inactivity watchdog cannot)."""

    def __init__(self, *, run_seconds: float = 30.0):
        self._run_seconds = run_seconds
        self._started = threading.Event()
        self._stop = threading.Event()
        self.interrupted: list[str] = []
        self.activity_polls = 0

    def run_conversation(self, *args, **kwargs):
        self._started.set()
        self._stop.wait(self._run_seconds)
        return {"final_response": "finished"}

    def get_activity_summary(self):
        self.activity_polls += 1
        # Always fresh activity -> seconds_since_activity stays ~0 -> inactivity never fires.
        return {
            "last_activity_desc": "API call",
            "seconds_since_activity": 0.0,
            "api_call_count": self.activity_polls,
            "max_iterations": 100,
            "current_tool": "terminal",
        }

    def interrupt(self, message, **kwargs):
        self.interrupted.append(message)
        self._stop.set()


def test_max_runtime_fires_on_a_busy_agent(monkeypatch):
    """The cap must fire even though the agent never goes idle."""
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 1.0)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)

    agent = _BusyAgent(run_seconds=30.0)
    try:
        with pytest.raises(TimeoutError) as exc:
            scheduler._run_agent_with_watchdog(
                agent, "probe", {"schedule": {"kind": "every"}}, "probe", "busy-job", "probe", None)
    finally:
        agent._stop.set()

    assert "wall-clock cap" in str(exc.value)
    assert agent.interrupted, "the agent must be hard-interrupted on a wall-clock overrun"


def test_max_runtime_does_not_fire_on_a_fast_run(monkeypatch):
    """A run that finishes inside the cap must complete normally."""
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 60.0)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)

    agent = _BusyAgent(run_seconds=0.2)
    result = scheduler._run_agent_with_watchdog(
        agent, "probe", {"schedule": {"kind": "every"}}, "probe", "fast-job", "probe", None)

    assert result["final_response"] == "finished"
    assert not agent.interrupted


def test_max_runtime_can_be_disabled(monkeypatch):
    """0 = unlimited: the cap must not fire at all."""
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 0.0)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)

    agent = _BusyAgent(run_seconds=0.3)
    result = scheduler._run_agent_with_watchdog(
        agent, "probe", {"schedule": {"kind": "every"}}, "probe", "unlimited-job", "probe", None)

    assert result["final_response"] == "finished"
    assert not agent.interrupted


def test_max_runtime_env_parsing(monkeypatch):
    """HERMES_CRON_MAX_RUNTIME parsing: default, explicit, invalid, zero-disables."""
    from cron import scheduler

    monkeypatch.delenv("HERMES_CRON_MAX_RUNTIME", raising=False)
    assert scheduler._cron_max_runtime_seconds() == float(scheduler._DEFAULT_CRON_MAX_RUNTIME_SECONDS)

    monkeypatch.setenv("HERMES_CRON_MAX_RUNTIME", "600")
    assert scheduler._cron_max_runtime_seconds() == 600.0

    monkeypatch.setenv("HERMES_CRON_MAX_RUNTIME", "0")
    assert scheduler._cron_max_runtime_seconds() == 0.0

    monkeypatch.setenv("HERMES_CRON_MAX_RUNTIME", "not-a-number")
    assert scheduler._cron_max_runtime_seconds() == float(scheduler._DEFAULT_CRON_MAX_RUNTIME_SECONDS)

    monkeypatch.setenv("HERMES_CRON_MAX_RUNTIME", "-5")
    assert scheduler._cron_max_runtime_seconds() == float(scheduler._DEFAULT_CRON_MAX_RUNTIME_SECONDS)


def test_runtime_watcher_thread_name(monkeypatch):
    """The wall-clock watcher runs on its own daemon thread, like the inactivity one."""
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 0.5)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)

    names: list[str] = []
    original_start = threading.Thread.start

    def _start(self):
        names.append(self.name)
        return original_start(self)

    agent = _BusyAgent(run_seconds=30.0)
    monkeypatch.setattr(threading.Thread, "start", _start)
    try:
        with pytest.raises(TimeoutError):
            scheduler._run_agent_with_watchdog(
                agent, "probe", {"schedule": {"kind": "every"}}, "probe", "thread-job", "probe", None)
    finally:
        agent._stop.set()

    assert any(n.startswith("cron-maxruntime-") for n in names)
