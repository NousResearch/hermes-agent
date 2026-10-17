"""Gateway stop persistence, cleanup, and exit outcomes (#138525, #138530)."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import run as gateway_run, run_bootstrap, run_runtime
from gateway.config import GatewayConfig
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.fixture
def live_shutdown(monkeypatch, tmp_path):
    """Exercise start_gateway, the real shared stop task, and the real shutdown tail."""
    events = []
    stop_entered, persist_finished = asyncio.Event(), asyncio.Event()
    persist_finished.set()
    state = SimpleNamespace(stop_error=None, runtime_error=None, restart=True)
    cron_stop = threading.Event()
    threads = [threading.Thread(target=cron_stop.wait, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()

    class LiveRunner:
        stop = gateway_run.GatewayRunner.stop

        def __init__(self, config):
            self.config = config
            self._stop_task = None
            self._running = False
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self._restart_requested = True
            self._restart_via_service = True

        async def start(self):
            self._running = True
            return True

        def _start_systemd_watchdog(self):
            pass

        async def _stop_impl(self):
            events.append("stop_entered")
            stop_entered.set()
            await persist_finished.wait()
            if state.stop_error is not None:
                raise state.stop_error
            self._running = False
            self._restart_via_service = state.restart
            self.exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE if state.restart else None
            events.append("persisted")

    async def wait_runtime(runner):
        state.runner = runner
        if state.runtime_error is not None:
            raise state.runtime_error

    async def control_stop():
        events.append("control")

    async def drain(runner):
        events.append("drain")

    async def mcp_shutdown(**kwargs):
        events.append("mcp")

    control = SimpleNamespace(stop=AsyncMock(side_effect=control_stop))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(gateway_run, "GatewayRunner", LiveRunner)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_reserve_profiles", lambda homes: True)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_claim_pid_file", lambda **kwargs: True)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_early_environment_probe", lambda: None)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_configure_logging", lambda verbosity: None)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_install_signal_handlers", lambda *args: None)
    monkeypatch.setattr(run_bootstrap, "_start_gateway_start_control_socket", AsyncMock(return_value=control))
    monkeypatch.setattr(run_bootstrap, "_start_gateway_start_cron_and_housekeeping",
                        lambda runner: (cron_stop, object(), *threads))
    monkeypatch.setattr("gateway.run_startup.recover_left_core_at_gateway_start", lambda: None)
    monkeypatch.setattr(gateway_run, "_host_attach_or_none", AsyncMock(return_value=None))
    monkeypatch.setattr(gateway_run, "_enable_multiplex_log_routing", lambda config: None)
    monkeypatch.setattr(gateway_run, "_refresh_host_gateway_record", lambda runner: None)
    monkeypatch.setattr(gateway_run, "_log_standalone_profiles_at_boot", lambda runner: None)
    monkeypatch.setattr(gateway_run, "_ensure_windows_gateway_venv_imports", lambda: None)
    monkeypatch.setattr(gateway_run, "_run_planned_stop_watcher", lambda stop, *args: stop.wait())
    monkeypatch.setattr(gateway_run, "_discover_gateway_mcp_tools", AsyncMock())
    monkeypatch.setattr(gateway_run, "_stop_cron_provider", lambda provider: events.append("cron"))
    monkeypatch.setattr(gateway_run, "_shutdown_mcp_servers_nonblocking", mcp_shutdown)
    monkeypatch.setattr("gateway.lifecycle_ledger.record_startup", lambda: None)
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.stop_nous_auth_keepalive",
                        lambda: events.append("keepalive"))
    monkeypatch.setattr(run_runtime, "initialize_gateway_runtime", AsyncMock())
    monkeypatch.setattr(run_runtime, "start_gateway_runtime_api", AsyncMock())
    monkeypatch.setattr(run_runtime, "publish_gateway_runtime_ready", lambda runner: None)
    monkeypatch.setattr(run_runtime, "wait_gateway_runtime", wait_runtime)
    monkeypatch.setattr(run_runtime, "drain_gateway_runtime", drain)
    monkeypatch.setattr("gateway.run_api.warm_gateway_dashboard", lambda runner: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: events.append("pid"))
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: events.append("lock"))

    real_tail = run_bootstrap._start_gateway_shutdown_tail

    async def tail(*args, **kwargs):
        state.watcher_stop = args[6]
        state.watcher_thread = args[7]
        return await real_tail(*args, **kwargs)

    monkeypatch.setattr(run_bootstrap, "_start_gateway_shutdown_tail", tail)
    state.events, state.stop_entered, state.persist_finished = events, stop_entered, persist_finished
    state.cron_stop, state.threads = cron_stop, threads
    yield state
    persist_finished.set()
    cron_stop.set()
    for thread in threads:
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_shutdown_waits_for_stop_task_before_service_restart_verdict(live_shutdown):
    """Exit 75 and shared-service teardown must wait for marker persistence (#138525)."""
    state = live_shutdown
    state.persist_finished.clear()

    async def run():
        with pytest.raises(SystemExit) as exc:
            await gateway_run.start_gateway(GatewayConfig(), verbosity=None)
        return exc.value.code

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(state.stop_entered.wait(), 10)
        assert not task.done()
        assert state.events == ["stop_entered"]
        assert not state.cron_stop.is_set()
    finally:
        state.persist_finished.set()
        code = await asyncio.wait_for(task, 10)
    assert code == GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert state.events.index("persisted") < state.events.index("control")
    assert state.runner._stop_task.done()


@pytest.mark.asyncio
async def test_failing_stop_cleans_up_and_returns_failure(live_shutdown, caplog):
    state = live_shutdown
    state.stop_error = RuntimeError("stop persistence failed")
    result = await gateway_run.start_gateway(GatewayConfig(), verbosity=None)

    assert result is False
    assert state.events == ["stop_entered", "control", "keepalive", "cron", "mcp",
                            "drain", "control", "pid", "lock"]
    assert state.cron_stop.is_set() and state.watcher_stop.is_set()
    assert not any(thread.is_alive() for thread in state.threads + [state.watcher_thread])
    assert state.runner._stop_task.exception() is state.stop_error
    assert "stop persistence failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True], ids=["success", "service-restart"])
async def test_shutdown_inside_callers_except_block(live_shutdown, restart):
    live_shutdown.restart = restart
    try:
        raise ValueError("unrelated caller failure")
    except ValueError:
        if restart:
            with pytest.raises(SystemExit) as exc:
                await gateway_run.start_gateway(GatewayConfig(), verbosity=None)
            assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
        else:
            assert await gateway_run.start_gateway(GatewayConfig(), verbosity=None) is True
    assert "persisted" in live_shutdown.events
    assert live_shutdown.events[-2:] == ["pid", "lock"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True], ids=["runtime-error", "cancellation"])
@pytest.mark.parametrize("stop_fails", [False, True], ids=["stop-success", "stop-failure"])
async def test_runtime_exception_survives_shutdown(live_shutdown, cancelled, stop_fails):
    state = live_shutdown
    state.runtime_error = asyncio.CancelledError() if cancelled else ValueError("runtime failed")
    state.stop_error = RuntimeError("stop failed too") if stop_fails else None
    with pytest.raises(type(state.runtime_error)) as exc:
        await gateway_run.start_gateway(GatewayConfig(), verbosity=None)
    assert exc.value is state.runtime_error
    assert "mcp" in state.events and "drain" in state.events
    assert state.events[-2:] == ["pid", "lock"]


@pytest.mark.asyncio
async def test_cancelled_stop_preserves_cancellation_and_cleanup(live_shutdown):
    state = live_shutdown
    state.stop_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await gateway_run.start_gateway(GatewayConfig(), verbosity=None)
    assert state.runner._stop_task.cancelled()
    assert state.events == ["stop_entered", "control", "keepalive", "cron", "mcp",
                            "drain", "control", "pid", "lock"]
