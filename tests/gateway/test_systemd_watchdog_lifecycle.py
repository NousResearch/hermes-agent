"""Gateway lifecycle contract for the opt-in systemd watchdog."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, start_gateway
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


