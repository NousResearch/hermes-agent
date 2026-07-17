"""Gateway lifecycle contract for the opt-in systemd watchdog."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner


class _FakeWatchdog:
    instances: list["_FakeWatchdog"] = []

    def __init__(self, *, config_enabled: bool = True):
        self.config_enabled = config_enabled
        self.calls: list[str] = []
        self.__class__.instances.append(self)

    def start(self) -> bool:
        self.calls.append("start")
        return self.config_enabled

    def ready(self, status: str) -> bool:
        self.calls.append(f"ready:{status}")
        return True

    async def stop(self) -> None:
        self.calls.append("stop")


def _bare_runner(*, seconds: int, running: bool = True) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(systemd_watchdog_seconds=seconds)
    runner._running = running
    runner._systemd_watchdog = None
    return runner


def test_runner_starts_watchdog_only_after_running(monkeypatch):
    _FakeWatchdog.instances.clear()
    monkeypatch.setattr("gateway.systemd_notify.SystemdWatchdog", _FakeWatchdog)
    runner = _bare_runner(seconds=120, running=True)

    assert runner._start_systemd_watchdog() is True

    watchdog = _FakeWatchdog.instances[-1]
    assert watchdog.config_enabled is True
    assert watchdog.calls == ["start", "ready:Hermes Gateway running"]


def test_runner_does_not_start_watchdog_when_config_disabled_or_not_running(monkeypatch):
    _FakeWatchdog.instances.clear()
    monkeypatch.setattr("gateway.systemd_notify.SystemdWatchdog", _FakeWatchdog)

    assert _bare_runner(seconds=0)._start_systemd_watchdog() is False
    assert _bare_runner(seconds=120, running=False)._start_systemd_watchdog() is False
    assert _FakeWatchdog.instances == []


@pytest.mark.asyncio
async def test_gateway_stop_stops_watchdog_before_session_drain():
    runner, _adapter = make_restart_runner()
    order: list[str] = []

    class _OrderingWatchdog:
        async def stop(self) -> None:
            order.append("watchdog_stop")

    async def _notify_sessions() -> None:
        order.append("notify_sessions")

    runner._systemd_watchdog = _OrderingWatchdog()
    runner._notify_active_sessions_of_shutdown = _notify_sessions

    async def _cancel_secondary_reconnects() -> None:
        order.append("secondary_reconnect_cancel")

    runner._cancel_secondary_profile_reconnect_tasks = _cancel_secondary_reconnects

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    assert order[:3] == [
        "watchdog_stop",
        "secondary_reconnect_cancel",
        "notify_sessions",
    ]
