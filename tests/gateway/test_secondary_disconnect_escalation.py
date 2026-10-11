"""Secondary adapter reconnect escalation on Disconnected.

Regression for #130797 (a secondary Discord transport close whose in-place
reconnect ends in Disconnected left the profile offline indefinitely instead
of reaching background reconnection).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_adapters import GatewayAdapterLifecycleMixin


class _StubRunner(GatewayAdapterLifecycleMixin):
    def __init__(self):
        self._profile_adapters: dict = {}
        self._profile_failed_platforms: dict = {}
        self._running = True
        self.watcher_ensured = 0
        self.scheduled: list = []

    def _ensure_reconnect_watcher_running(self) -> None:
        self.watcher_ensured += 1

    def _schedule_secondary_profile_reconnect(self, profile_name, platform, adapter) -> None:
        self.scheduled.append((profile_name, platform, adapter))
        pending = self._profile_failed_platforms.setdefault(profile_name, {})
        pending[platform] = adapter


def _dead_adapter(*, retryable=True, has_fatal=True):
    state = {"fatal": has_fatal, "retryable": retryable, "running": False}

    def _set_fatal_error(code, message, *, retryable):
        state["fatal"] = True
        state["retryable"] = retryable

    return SimpleNamespace(
        _running=False,
        has_fatal_error=state["fatal"],
        fatal_error_retryable=state["retryable"],
        _set_fatal_error=_set_fatal_error,
        _state=state,
    )


def test_disconnected_secondary_escalates_to_background_reconnect():
    runner = _StubRunner()
    adapter = _dead_adapter()
    runner._profile_adapters = {"worker": {Platform.DISCORD: adapter}}

    assert runner._escalate_secondary_disconnect_to_watcher("worker", Platform.DISCORD) is True
    assert Platform.DISCORD not in runner._profile_adapters["worker"]
    assert runner.watcher_ensured == 1
    assert runner.scheduled and runner.scheduled[0][0] == "worker"


def test_running_secondary_needs_no_escalation():
    runner = _StubRunner()
    adapter = SimpleNamespace(_running=True, has_fatal_error=False,
                              fatal_error_retryable=True)
    runner._profile_adapters = {"worker": {Platform.DISCORD: adapter}}

    assert runner._escalate_secondary_disconnect_to_watcher("worker", Platform.DISCORD) is False
    assert runner._profile_adapters["worker"][Platform.DISCORD] is adapter
    assert runner.scheduled == []


def test_non_retryable_disconnect_does_not_reconnect():
    runner = _StubRunner()
    adapter = SimpleNamespace(_running=False, has_fatal_error=True,
                              fatal_error_retryable=False)
    runner._profile_adapters = {"worker": {Platform.DISCORD: adapter}}

    assert runner._escalate_secondary_disconnect_to_watcher("worker", Platform.DISCORD) is False
    assert runner.scheduled == []
