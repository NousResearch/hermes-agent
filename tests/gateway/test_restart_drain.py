import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class RecordingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list[str] = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id="123456"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
    )


def _make_runner() -> tuple[GatewayRunner, RecordingAdapter]:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {}
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 60.0
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._update_runtime_status = MagicMock()
    runner._queue_or_replace_pending_event = GatewayRunner._queue_or_replace_pending_event.__get__(runner, GatewayRunner)
    runner._session_key_for_source = GatewayRunner._session_key_for_source.__get__(runner, GatewayRunner)
    runner._handle_active_session_busy_message = GatewayRunner._handle_active_session_busy_message.__get__(runner, GatewayRunner)
    runner._handle_restart_command = GatewayRunner._handle_restart_command.__get__(runner, GatewayRunner)
    runner._status_action_label = GatewayRunner._status_action_label.__get__(runner, GatewayRunner)
    runner._status_action_gerund = GatewayRunner._status_action_gerund.__get__(runner, GatewayRunner)
    runner._queue_during_drain_enabled = GatewayRunner._queue_during_drain_enabled.__get__(runner, GatewayRunner)
    runner._running_agent_count = GatewayRunner._running_agent_count.__get__(runner, GatewayRunner)
    runner.request_restart = MagicMock(return_value=True)
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner.delivery_router = MagicMock()

    adapter = RecordingAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, adapter


@pytest.mark.asyncio
async def test_restart_command_while_busy_requests_drain_without_interrupt():
    runner, _adapter = _make_runner()
    event = MessageEvent(text="/restart", message_type=MessageType.TEXT, source=_source(), message_id="m1")
    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    result = await runner._handle_message(event)

    assert result == "⏳ Draining 1 active agent(s) before restart..."
    running_agent.interrupt.assert_not_called()
    runner.request_restart.assert_called_once_with(detached=True, via_service=False)


@pytest.mark.asyncio
async def test_drain_queue_mode_queues_follow_up_without_interrupt():
    runner, adapter = _make_runner()
    runner._draining = True
    runner._restart_requested = True
    runner._busy_input_mode = "queue"

    event = MessageEvent(text="follow up", message_type=MessageType.TEXT, source=_source(), message_id="m2")
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert session_key in adapter._pending_messages
    assert adapter._pending_messages[session_key].text == "follow up"
    assert not adapter._active_sessions[session_key].is_set()
    assert any("queued for the next turn" in message for message in adapter.sent)


@pytest.mark.asyncio
async def test_draining_rejects_new_session_messages():
    runner, _adapter = _make_runner()
    runner._draining = True
    runner._restart_requested = True

    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=_source("fresh"), message_id="m3")

    result = await runner._handle_message(event)

    assert result == "⏳ Gateway is restarting and is not accepting new work right now."
