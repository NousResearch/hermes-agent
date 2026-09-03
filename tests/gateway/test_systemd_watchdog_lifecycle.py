"""Gateway lifecycle contract for the opt-in systemd watchdog."""

from __future__ import annotations

import asyncio
import inspect
import os
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.run import (
    GatewayRunner,
    _complete_systemd_startup,
    _construct_runner_with_startup_deadline,
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


class _FakeStartupDeadline:
    def __init__(self):
        self.stop_calls = 0

    async def stop(self):
        self.stop_calls += 1


class _RunningStartupRunner:
    def __init__(self):
        self.config = GatewayConfig(systemd_watchdog_seconds=120)
        self.adapters = {}
        self.should_exit_cleanly = False
        self.exit_reason = None
        self.exit_code = None
        self._running = True
        self._draining = False
        self._external_drain_active = False

    def request_restart(self, *, detached, via_service):
        return False

    async def stop(self):
        return None


def _patch_start_gateway_shell(monkeypatch, tmp_path, runner, deadline):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.resource_limits.apply_nofile_soft_limit", lambda: None)
    monkeypatch.setattr("gateway.code_skew.record_boot_fingerprint", lambda: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr(
        "hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path
    )
    monkeypatch.setattr(
        "hermes_logging._add_rotating_handler", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "hermes_cli.security_audit_startup.log_startup_security_warnings",
        lambda **kwargs: None,
    )

    async def construct(_config):
        return runner, deadline

    monkeypatch.setattr(
        "gateway.run._construct_runner_with_startup_deadline", construct
    )
    monkeypatch.setattr("gateway.run._run_planned_stop_watcher", lambda *args: None)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *args: None)
    monkeypatch.setattr(loop, "set_exception_handler", lambda *args: None)


@pytest.mark.asyncio
async def test_start_gateway_stops_startup_deadline_on_early_false_return(
    monkeypatch, tmp_path
):
    runner = _RunningStartupRunner()
    deadline = _FakeStartupDeadline()
    _patch_start_gateway_shell(monkeypatch, tmp_path, runner, deadline)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: False)

    assert await start_gateway(config=runner.config, verbosity=None) is False
    assert deadline.stop_calls == 1


@pytest.mark.asyncio
async def test_start_gateway_stops_deadline_when_post_construction_setup_fails(
    monkeypatch, tmp_path
):
    runner = _RunningStartupRunner()
    deadline = _FakeStartupDeadline()
    _patch_start_gateway_shell(monkeypatch, tmp_path, runner, deadline)

    def fail_log_routing(_config):
        raise RuntimeError("log routing failed")

    monkeypatch.setattr(
        "gateway.run._enable_multiplex_log_routing", fail_log_routing
    )

    with pytest.raises(RuntimeError, match="log routing failed"):
        await start_gateway(config=runner.config, verbosity=None)

    assert deadline.stop_calls == 1


@pytest.mark.asyncio
async def test_start_gateway_stops_startup_deadline_on_post_runner_exception(
    monkeypatch, tmp_path
):
    runner = _RunningStartupRunner()
    deadline = _FakeStartupDeadline()
    _patch_start_gateway_shell(monkeypatch, tmp_path, runner, deadline)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)

    async def control_start(_self):
        return False

    monkeypatch.setattr("gateway.control_socket.GatewayControlServer.start", control_start)
    monkeypatch.setattr("gateway.lifecycle_ledger.record_startup", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive", lambda: None
    )
    monkeypatch.setattr("gateway.run._ensure_windows_gateway_venv_imports", lambda: None)

    async def discover_and_start(_runner, *, deadline):
        return True, deadline

    monkeypatch.setattr(
        "gateway.run._discover_mcp_and_start_runner", discover_and_start
    )
    monkeypatch.setattr("gateway.shutdown_flush.recover_pending_to_db", lambda: 0)

    def fail_cron_resolution():
        raise RuntimeError("cron resolution failed")

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler", fail_cron_resolution
    )

    with pytest.raises(RuntimeError, match="cron resolution failed"):
        await start_gateway(config=runner.config, verbosity=None)

    assert deadline.stop_calls == 1


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
async def test_startup_deadline_covers_runner_construction(monkeypatch):
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
        def __init__(self, config):
            order.append("runner.init")
            self.config = config

    monkeypatch.setattr("gateway.run.SystemdStartupDeadline", _FakeDeadline)
    monkeypatch.setattr("gateway.run.GatewayRunner", _Runner)

    runner, deadline = await _construct_runner_with_startup_deadline(
        GatewayConfig(systemd_watchdog_seconds=120)
    )

    assert runner.config.systemd_watchdog_seconds == 120
    assert order == ["deadline.init:True", "deadline.start", "runner.init"]
    await deadline.stop()


@pytest.mark.asyncio
async def test_startup_deadline_stops_when_runner_construction_fails(monkeypatch):
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
        def __init__(self, config):
            order.append("runner.init:failed")
            raise RuntimeError("constructor failed")

    monkeypatch.setattr("gateway.run.SystemdStartupDeadline", _FakeDeadline)
    monkeypatch.setattr("gateway.run.GatewayRunner", _Runner)

    with pytest.raises(RuntimeError, match="constructor failed"):
        await _construct_runner_with_startup_deadline(
            GatewayConfig(systemd_watchdog_seconds=120)
        )

    assert order == [
        "deadline.init:True",
        "deadline.start",
        "runner.init:failed",
        "deadline.stop",
    ]


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
async def test_cancelled_ready_handoff_never_signals_ready(monkeypatch):
    calls: list[str] = []
    child_cancelled = asyncio.Event()
    release_child = asyncio.Event()
    monkeypatch.setenv("NOTIFY_SOCKET", "/tmp/hermes-test-notify")
    monkeypatch.setenv("WATCHDOG_USEC", "1000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))

    import gateway.systemd_notify as notify_mod

    monkeypatch.setattr(
        notify_mod, "notify", lambda message: calls.append(message) or True
    )

    class _SlowCancellationDeadline(notify_mod.SystemdStartupDeadline):
        async def _run(self):
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                child_cancelled.set()
                await release_child.wait()

    deadline = _SlowCancellationDeadline(interval_seconds=60)
    assert deadline.start() is True
    child_task = deadline.task
    assert child_task is not None

    class _Runner:
        def _start_systemd_watchdog(self):
            calls.append("READY")
            return True

    handoff_task = asyncio.create_task(_complete_systemd_startup(deadline, _Runner()))
    await asyncio.wait_for(child_cancelled.wait(), timeout=1)
    assert handoff_task.done() is False
    assert handoff_task.cancel() is True

    with pytest.raises(asyncio.CancelledError):
        await handoff_task

    assert "READY" not in calls
    assert deadline.task is None
    release_child.set()
    await child_task
    assert child_task.done() is True
    assert child_task.cancelled() is False


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


