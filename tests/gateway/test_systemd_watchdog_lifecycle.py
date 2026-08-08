"""Gateway lifecycle contract for the opt-in systemd watchdog."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import (
    GatewayRunner,
    _complete_systemd_startup,
    _discover_mcp_and_start_runner,
    start_gateway,
)
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


@pytest.mark.asyncio
async def test_startup_deadline_stays_active_until_ready_boundary(monkeypatch):
    order: list[str] = []

    class _FakeDeadline:
        def __init__(self, *, config_enabled):
            order.append(f"deadline.init:{config_enabled}")

        def start(self):
            order.append("deadline.start")
            return True

        async def stop(self):
            order.append("deadline.stop")

    class _Runner:
        config = GatewayConfig(systemd_watchdog_seconds=120)

        async def start(self):
            order.append("runner.start")
            return True

        def _start_systemd_watchdog(self):
            order.append("watchdog.ready")
            return True

    monkeypatch.setattr("gateway.run.SystemdStartupDeadline", _FakeDeadline)
    monkeypatch.setattr(
        "tools.mcp_tool.discover_mcp_tools",
        lambda: order.append("mcp.discover"),
    )

    runner = _Runner()
    success, deadline = await _discover_mcp_and_start_runner(runner)
    assert success is True
    assert order == [
        "deadline.init:True",
        "deadline.start",
        "mcp.discover",
        "runner.start",
    ]

    await _complete_systemd_startup(deadline, runner)
    assert order[-2:] == ["deadline.stop", "watchdog.ready"]


@pytest.mark.asyncio
async def test_startup_deadline_stops_when_runner_returns_false(monkeypatch):
    order: list[str] = []

    class _FakeDeadline:
        def __init__(self, *, config_enabled):
            order.append(f"deadline.init:{config_enabled}")

        def start(self):
            order.append("deadline.start")
            return True

        async def stop(self):
            order.append("deadline.stop")

    class _Runner:
        config = GatewayConfig(systemd_watchdog_seconds=120)

        async def start(self):
            order.append("runner.start:false")
            return False

    monkeypatch.setattr("gateway.run.SystemdStartupDeadline", _FakeDeadline)
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: None)

    success, deadline = await _discover_mcp_and_start_runner(_Runner())

    assert success is False
    assert deadline is not None
    assert order[-2:] == ["runner.start:false", "deadline.stop"]


@pytest.mark.asyncio
async def test_startup_deadline_stops_when_runner_start_fails(monkeypatch):
    order: list[str] = []

    class _FakeDeadline:
        def __init__(self, *, config_enabled):
            order.append(f"deadline.init:{config_enabled}")

        def start(self):
            order.append("deadline.start")
            return True

        async def stop(self):
            order.append("deadline.stop")

    class _Runner:
        config = GatewayConfig(systemd_watchdog_seconds=120)

        async def start(self):
            order.append("runner.start")
            raise RuntimeError("startup failed")

    monkeypatch.setattr("gateway.run.SystemdStartupDeadline", _FakeDeadline)
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: None)

    with pytest.raises(RuntimeError, match="startup failed"):
        await _discover_mcp_and_start_runner(_Runner())

    assert order[-1] == "deadline.stop"


