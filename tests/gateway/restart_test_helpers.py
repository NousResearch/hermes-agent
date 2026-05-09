import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class RestartTestAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list[str] = []
        self.sent_calls: list[tuple[str, str, object]] = []

    async def connect(self):
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
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
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
    runner._launch_detached_restart_command = GatewayRunner._launch_detached_restart_command.__get__(
        runner, GatewayRunner
    )
    runner.request_restart = GatewayRunner.request_restart.__get__(runner, GatewayRunner)
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
    return runner, platform_adapter
