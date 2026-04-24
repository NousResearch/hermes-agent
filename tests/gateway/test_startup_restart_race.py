import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
)


class StartupRaceAdapter(BasePlatformAdapter):
    def __init__(
        self,
        platform: Platform,
        *,
        on_connect=None,
        wait_for_disconnect: asyncio.Event | None = None,
    ):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.on_connect = on_connect
        self.wait_for_disconnect = wait_for_disconnect
        self.connected = False
        self.disconnected = False
        self.background_cancelled = False

    async def connect(self):
        if self.on_connect:
            self.on_connect()
        if self.wait_for_disconnect is not None:
            await self.wait_for_disconnect.wait()
        self.connected = True
        return True

    async def disconnect(self):
        self.disconnected = True

    async def cancel_background_tasks(self):
        self.background_cancelled = True
        await super().cancel_background_tasks()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def make_startup_runner(tmp_path):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.SLACK: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner.adapters = {}
    runner._running = False
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._exit_cleanly = False
    runner._exit_with_failure = False
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._stop_task = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._failed_platforms = {}
    runner._voice_mode = {}

    runner.hooks = MagicMock()
    runner.hooks.loaded_hooks = []
    runner.hooks.discover_and_load = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.suspend_recently_active.return_value = 0
    runner.delivery_router = MagicMock()
    runner.delivery_router.adapters = {}

    runner._update_runtime_status = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    runner._sync_voice_mode_state_to_adapter = MagicMock()
    runner._suspend_stuck_loop_sessions = MagicMock(return_value=0)
    runner._notify_active_sessions_of_shutdown = AsyncMock()
    runner._drain_active_agents = AsyncMock(return_value=({}, False))
    runner._finalize_shutdown_agents = MagicMock()
    runner._send_update_notification = AsyncMock(return_value=False)
    runner._schedule_update_notification_watch = MagicMock()
    runner._send_restart_notification = AsyncMock()
    runner.wait_for_shutdown = gateway_run.GatewayRunner.wait_for_shutdown.__get__(
        runner, gateway_run.GatewayRunner
    )

    async def no_op_watcher(*args, **kwargs):
        await asyncio.Event().wait()

    runner._session_expiry_watcher = no_op_watcher
    runner._platform_reconnect_watcher = no_op_watcher
    runner._run_process_watcher = no_op_watcher
    runner._safe_adapter_disconnect = gateway_run.GatewayRunner._safe_adapter_disconnect.__get__(
        runner, gateway_run.GatewayRunner
    )
    runner.request_restart = gateway_run.GatewayRunner.request_restart.__get__(
        runner, gateway_run.GatewayRunner
    )
    runner.stop = gateway_run.GatewayRunner.stop.__get__(runner, gateway_run.GatewayRunner)
    return runner


def patch_startup_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("agent.shell_hooks.register_from_config", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.process_registry.process_registry.recover_from_checkpoint", lambda: 0)


@pytest.mark.asyncio
async def test_startup_aborts_when_restart_requested_before_start(tmp_path, monkeypatch):
    patch_startup_side_effects(monkeypatch, tmp_path)
    runner = make_startup_runner(tmp_path)
    runner.request_restart(detached=False, via_service=True)
    runner._create_adapter = MagicMock()

    result = await asyncio.wait_for(runner.start(), timeout=2)

    assert result is True
    runner._create_adapter.assert_not_called()
    assert runner.delivery_router.adapters == {}
    assert runner._running is False
    assert not any(
        call.args[:1] == ("running",)
        for call in runner._update_runtime_status.call_args_list
    )


@pytest.mark.asyncio
async def test_startup_aborts_when_restart_begins_during_platform_connect(tmp_path, monkeypatch):
    patch_startup_side_effects(monkeypatch, tmp_path)

    runner = make_startup_runner(tmp_path)
    first_disconnected = asyncio.Event()
    telegram = StartupRaceAdapter(
        Platform.TELEGRAM,
        on_connect=lambda: runner.request_restart(detached=False, via_service=True),
    )
    slack = StartupRaceAdapter(Platform.SLACK, wait_for_disconnect=first_disconnected)

    async def disconnect_and_release():
        telegram.disconnected = True
        first_disconnected.set()

    telegram.disconnect = disconnect_and_release
    runner._create_adapter = MagicMock(side_effect=[telegram, slack])

    result = await asyncio.wait_for(runner.start(), timeout=2)

    assert result is True
    assert telegram.disconnected is True
    assert telegram.background_cancelled is True
    assert slack.connected is False
    assert runner._running is False
    assert runner.adapters == {}
    assert runner._update_runtime_status.call_args_list[-1].args[0] == "stopped"
    assert not any(
        call.args[:1] == ("running",)
        for call in runner._update_runtime_status.call_args_list
    )
    assert not any(
        call.args[:2] == (Platform.SLACK.value, "connected")
        for call in runner._update_platform_runtime_status.call_args_list
    )


@pytest.mark.asyncio
async def test_startup_abort_waits_for_existing_stop_task(tmp_path):
    runner = make_startup_runner(tmp_path)
    runner._restart_requested = True
    runner.stop = AsyncMock(side_effect=AssertionError("stop should not be called"))
    stop_completed = asyncio.Event()

    async def existing_stop():
        await asyncio.sleep(0.01)
        stop_completed.set()

    runner._stop_task = asyncio.create_task(existing_stop())
    adapter = StartupRaceAdapter(Platform.TELEGRAM)

    result = await asyncio.wait_for(
        runner._abort_startup_if_shutdown_requested(adapter, Platform.TELEGRAM),
        timeout=2,
    )

    assert result is True
    assert stop_completed.is_set()
    assert runner._stop_task.done()
    runner.stop.assert_not_called()
    assert adapter.background_cancelled is True
    assert adapter.disconnected is True


@pytest.mark.asyncio
async def test_startup_aborts_after_registered_adapter_restart(tmp_path, monkeypatch):
    patch_startup_side_effects(monkeypatch, tmp_path)
    runner = make_startup_runner(tmp_path)
    telegram = StartupRaceAdapter(Platform.TELEGRAM)
    slack = StartupRaceAdapter(Platform.SLACK)
    runner._create_adapter = MagicMock(side_effect=[telegram, slack])

    def update_platform_runtime_status(platform, platform_state, **kwargs):
        if (platform, platform_state) == (Platform.TELEGRAM.value, "connected"):
            runner.request_restart(detached=False, via_service=True)

    runner._update_platform_runtime_status = MagicMock(side_effect=update_platform_runtime_status)

    result = await asyncio.wait_for(runner.start(), timeout=2)

    assert result is True
    assert telegram.connected is True
    assert telegram.disconnected is True
    assert slack.connected is False
    assert runner._running is False
    assert runner.adapters == {}
    assert runner._update_runtime_status.call_args_list[-1].args[0] == "stopped"
    assert not any(
        call.args[:1] == ("running",)
        for call in runner._update_runtime_status.call_args_list
    )
    assert not any(
        call.args[:2] == (Platform.SLACK.value, "connected")
        for call in runner._update_platform_runtime_status.call_args_list
    )


@pytest.mark.asyncio
async def test_start_gateway_does_not_start_cron_after_aborted_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cron_started = False

    class AbortedStartupRunner:
        def __init__(self, config):
            self.config = config
            self.adapters = {}
            self._running = False
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE

        async def start(self):
            return True

        async def wait_for_shutdown(self):
            return None

    def fail_if_cron_starts(*args, **kwargs):
        nonlocal cron_started
        cron_started = True

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", AbortedStartupRunner)
    monkeypatch.setattr("gateway.run._start_cron_ticker", fail_if_cron_starts)
    monkeypatch.setattr("tools.mcp_tool.shutdown_mcp_servers", lambda: None)

    with pytest.raises(SystemExit) as exc:
        await gateway_run.start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert cron_started is False
