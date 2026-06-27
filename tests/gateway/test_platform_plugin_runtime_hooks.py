from __future__ import annotations

import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import Platform, PlatformConfig  # noqa: E402
from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner  # noqa: E402


class _StubAdapter(BasePlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="user1",
    )


def _event(text: str = "hello", *, delivery_intent: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="msg1",
        delivery_intent=delivery_intent,
    )


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = "interrupt"
    runner._queued_events = {}
    runner.adapters = {}
    runner._is_user_authorized = lambda _source: True
    return runner


def _running_agent() -> MagicMock:
    agent = MagicMock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    return agent


@pytest.mark.asyncio
async def test_delivery_intent_queue_overrides_interrupt_mode(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner()
    runner._busy_input_mode = "interrupt"
    adapter = _StubAdapter()
    adapter._on_runtime_queue_changed = AsyncMock()
    adapter.on_gateway_message_accepted = AsyncMock()
    event = _event(delivery_intent="queue")
    session_key = build_session_key(event.source)
    agent = _running_agent()
    runner.adapters[event.source.platform] = adapter
    runner._running_agents[session_key] = agent

    handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    assert adapter._pending_messages[session_key] is event
    agent.interrupt.assert_not_called()
    adapter.on_gateway_message_accepted.assert_awaited_once_with(
        event,
        session_key,
        phase="busy",
    )
    adapter._on_runtime_queue_changed.assert_awaited_once_with(session_key)


@pytest.mark.asyncio
async def test_delivery_intent_interrupt_overrides_queue_mode(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner()
    runner._busy_input_mode = "queue"
    adapter = _StubAdapter()
    adapter._on_runtime_queue_changed = AsyncMock()
    event = _event("now", delivery_intent="interrupt")
    session_key = build_session_key(event.source)
    old_event = _event("old")
    adapter._pending_messages[session_key] = old_event
    agent = _running_agent()
    runner.adapters[event.source.platform] = adapter
    runner._running_agents[session_key] = agent

    handled = await runner._handle_active_session_busy_message(event, session_key)

    assert handled is True
    assert adapter._pending_messages[session_key] is event
    agent.interrupt.assert_called_once_with("now")
    adapter._on_runtime_queue_changed.assert_awaited_once_with(session_key)


@pytest.mark.asyncio
async def test_runtime_control_stop_and_drop_clears_pending_work() -> None:
    runner = _runner()
    adapter = _StubAdapter()
    event = _event()
    session_key = build_session_key(event.source)
    adapter._pending_messages[session_key] = event
    runner._queued_events[session_key] = [_event("later")]
    runner.adapters[event.source.platform] = adapter
    runner._interrupt_and_clear_session = AsyncMock()

    handled = await runner._handle_runtime_control_signal(event, session_key, "stop_and_drop")

    assert handled is True
    assert session_key not in adapter._pending_messages
    assert session_key not in runner._queued_events
    runner._interrupt_and_clear_session.assert_awaited_once()
    assert runner._interrupt_and_clear_session.await_args.kwargs["discard_pending"] is True


@pytest.mark.asyncio
async def test_accepted_message_hook_is_best_effort() -> None:
    runner = _runner()
    adapter = _StubAdapter()
    adapter.on_gateway_message_accepted = AsyncMock()
    event = _event()
    session_key = build_session_key(event.source)
    runner.adapters[event.source.platform] = adapter

    await runner._notify_adapter_message_accepted(event, session_key, phase="queued")

    adapter.on_gateway_message_accepted.assert_awaited_once_with(
        event,
        session_key,
        phase="queued",
    )


def test_streaming_preview_hook_can_override_consumer_options() -> None:
    class Adapter:
        def prepare_streaming_preview(self, *, metadata, cursor, buffer_only):
            return {
                "metadata": {"thread_id": "preview"},
                "cursor": "",
                "buffer_only": True,
            }

    metadata, cursor, buffer_only = GatewayRunner._apply_streaming_preview_hook(
        Adapter(),
        metadata={"thread_id": "original"},
        cursor="▌",
        buffer_only=False,
    )

    assert metadata == {"thread_id": "preview"}
    assert cursor == ""
    assert buffer_only is True


@pytest.mark.asyncio
async def test_base_adapter_runtime_lifecycle_hooks() -> None:
    adapter = _StubAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter._on_runtime_turn_start = AsyncMock()
    adapter._on_runtime_turn_complete = AsyncMock()
    adapter._on_runtime_queue_changed = AsyncMock()
    event = _event()
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    adapter._on_runtime_turn_start.assert_awaited_once_with(event, session_key)
    adapter._on_runtime_turn_complete.assert_awaited_once_with(
        event,
        session_key,
        ProcessingOutcome.SUCCESS,
    )
    adapter._on_runtime_queue_changed.assert_awaited_once_with(session_key)
