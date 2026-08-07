import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class RestartTestAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list[str] = []
        self.sent_calls: list[tuple[str, str, object]] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.sent_calls.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def make_restart_source(
    chat_id: str = "123456",
    chat_type: str = "dm",
    thread_id: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id="u1",
        thread_id=thread_id,
    )


def make_restart_runner(
    adapter: BasePlatformAdapter | None = None,
) -> tuple[GatewayRunner, BasePlatformAdapter]:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._signal_initiated_shutdown = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._detached_restart_helper_started = False
    runner._restart_command_source = None
    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._restart_after_turn_timeout = DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_sources = OrderedDict()
    runner._session_sources_max = 512
    runner._shutdown_all_gateway_honcho = lambda: None
    runner._update_runtime_status = MagicMock()
    runner._queue_or_replace_pending_event = GatewayRunner._queue_or_replace_pending_event.__get__(
        runner, GatewayRunner
    )
    runner._session_key_for_source = GatewayRunner._session_key_for_source.__get__(
        runner, GatewayRunner
    )
    runner._handle_active_session_busy_message = (
        GatewayRunner._handle_active_session_busy_message.__get__(runner, GatewayRunner)
    )
    runner._handle_restart_command = GatewayRunner._handle_restart_command.__get__(
        runner, GatewayRunner
    )
    runner._handle_set_home_command = GatewayRunner._handle_set_home_command.__get__(
        runner, GatewayRunner
    )
    runner._send_restart_notification = GatewayRunner._send_restart_notification.__get__(
        runner, GatewayRunner
    )
    runner._send_home_channel_startup_notifications = (
        GatewayRunner._send_home_channel_startup_notifications.__get__(runner, GatewayRunner)
    )
    runner._status_action_label = GatewayRunner._status_action_label.__get__(
        runner, GatewayRunner
    )
    runner._status_action_gerund = GatewayRunner._status_action_gerund.__get__(
        runner, GatewayRunner
    )
    runner._queue_during_drain_enabled = GatewayRunner._queue_during_drain_enabled.__get__(
        runner, GatewayRunner
    )
    runner._running_agent_count = GatewayRunner._running_agent_count.__get__(
        runner, GatewayRunner
    )
    runner._active_cron_job_count = GatewayRunner._active_cron_job_count.__get__(
        runner, GatewayRunner
    )
    runner._active_api_run_count = GatewayRunner._active_api_run_count.__get__(
        runner, GatewayRunner
    )
    runner._active_work_count = GatewayRunner._active_work_count.__get__(
        runner, GatewayRunner
    )
    runner._persist_active_agents = GatewayRunner._persist_active_agents.__get__(
        runner, GatewayRunner
    )
    runner._snapshot_running_agents = GatewayRunner._snapshot_running_agents.__get__(
        runner, GatewayRunner
    )
    runner._notify_active_sessions_of_shutdown = (
        GatewayRunner._notify_active_sessions_of_shutdown.__get__(runner, GatewayRunner)
    )
    runner._cache_session_source = GatewayRunner._cache_session_source.__get__(
        runner, GatewayRunner
    )
    runner._get_cached_session_source = GatewayRunner._get_cached_session_source.__get__(
        runner, GatewayRunner
    )
    runner._handle_message = GatewayRunner._handle_message.__get__(
        runner, GatewayRunner
    )
    runner._await_active_work_before_restart = (
        GatewayRunner._await_active_work_before_restart.__get__(runner, GatewayRunner)
    )
    # Fail-closed default: tests must opt into the real launcher via
    # attach_real_launcher() if they need to assert Popen arguments.
    runner._launch_detached_restart_command = AsyncMock(return_value=True)
    real_req = GatewayRunner.request_restart.__get__(runner, GatewayRunner)
    runner.request_restart = MagicMock(side_effect=real_req)
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner.session_store._entries = {}
    runner.delivery_router = MagicMock()

    platform_adapter = adapter or RestartTestAdapter()
    platform_adapter.set_message_handler(AsyncMock(return_value=None))
    platform_adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: platform_adapter}

    # Fail-closed safety: stop() must always be AsyncMock. Tests that
    # need the real stop behavior must explicitly rebind runner.stop.
    runner.stop = AsyncMock()

    # Safety guard: tests must NOT use this fixture to launch real
    # subprocesses. If a test calls runner._launch_detached_restart_command
    # and the value is still the default AsyncMock(return_value=True), no
    # real Popen ever runs.
    assert isinstance(runner._launch_detached_restart_command, AsyncMock), (
        "make_restart_runner: _launch_detached_restart_command must remain "
        "an AsyncMock to prevent real subprocess.Popen"
    )
    assert isinstance(runner.stop, AsyncMock), (
        "make_restart_runner: stop() must remain an AsyncMock"
    )

    return runner, platform_adapter


def attach_real_launcher_under_mocked_popen(runner: GatewayRunner) -> None:
    """DANGEROUS: re-bind ``runner._launch_detached_restart_command`` to the
    real production launcher method so a test can inspect the arguments
    passed to ``subprocess.Popen``.

    Preconditions (verified by the caller — see warning below):

      1. ``subprocess.Popen`` MUST be replaced with a controlled fake via
         ``monkeypatch.setattr(subprocess, "Popen", fake_popen)`` BEFORE this
         helper is called. Without that, the production launcher will spawn
         a real detached watcher subprocess and leak a watcher (see PID 5896
         regression).
      2. The test MUST NOT call ``dispatch_gateway_restart`` (which has its
         own dispatch / abort / cancel state machine). Use this helper only
         for direct unit-tests of the launcher method itself.

    WARNING: this helper binds the real launcher method to the runner.
    It does NOT mock ``subprocess.Popen``. The caller is responsible for
    doing that BEFORE invoking the launcher-under-test. If the caller
    forgets, ``subprocess.Popen`` will run for real and may spawn a
    detached watcher subprocess that escapes the test process tree.

    Use ONLY in dedicated launcher-argument tests (currently the 4
    ``tests/gateway/test_restart_drain.py`` Popen-inspection tests).
    """
    runner._launch_detached_restart_command = (
        GatewayRunner._launch_detached_restart_command.__get__(runner, GatewayRunner)
    )


def attach_real_stop(runner: GatewayRunner) -> None:
    """Re-bind ``runner.stop`` to the real production stop() method.

    ``make_restart_runner()`` installs ``runner.stop`` as an AsyncMock by
    default (fail-closed: restart tests must not accidentally invoke the real
    shutdown path). Tests that assert the real shutdown behavior — e.g.
    ``test_gateway_shutdown.py``, ``test_cron_active_work_drain.py`` and
    ``test_restart_resume_pending.py`` — must explicitly opt in by calling
    this helper after ``make_restart_runner()``.

    This mirrors ``attach_real_launcher_under_mocked_popen``: the mock is the
    safe default, real behavior is an explicit per-test choice.
    """
    runner.stop = GatewayRunner.stop.__get__(runner, GatewayRunner)  # type: ignore[method-assign]
