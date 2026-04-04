import pytest
from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _dequeue_pending_text
from gateway.session import SessionSource, build_session_key


class _QueueAdapter:
    def __init__(self):
        self._pending_messages = {}

    def enqueue_pending_message(self, session_key, event, merge_photo_bursts=False):
        queue = self._pending_messages.setdefault(session_key, [])
        queue.append(event)

    def peek_pending_message(self, session_key):
        queue = self._pending_messages.get(session_key) or []
        return queue[0] if queue else None

    def get_pending_message(self, session_key):
        queue = self._pending_messages.get(session_key) or []
        if not queue:
            self._pending_messages.pop(session_key, None)
            return None
        event = queue.pop(0)
        if not queue:
            self._pending_messages.pop(session_key, None)
        return event


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: _QueueAdapter()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_running_agent_interrupt_path_queues_multiple_messages_fifo():
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    session_key = build_session_key(source)

    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    e1 = MessageEvent(text="first follow-up", message_type=MessageType.TEXT, source=source, message_id="m1")
    e2 = MessageEvent(text="second follow-up", message_type=MessageType.TEXT, source=source, message_id="m2")

    result1 = await runner._handle_message(e1)
    result2 = await runner._handle_message(e2)

    assert result1 is None
    assert result2 is None
    assert running_agent.interrupt.call_count == 2

    adapter = runner.adapters[Platform.TELEGRAM]
    first = adapter.get_pending_message(session_key)
    second = adapter.get_pending_message(session_key)
    assert first and first.text == "first follow-up"
    assert second and second.text == "second follow-up"


def test_dequeue_pending_text_returns_messages_in_fifo_order():
    adapter = _QueueAdapter()
    session_key = "agent:main:telegram:dm:12345"

    adapter.enqueue_pending_message(
        session_key,
        MessageEvent(text="first", message_type=MessageType.TEXT),
    )
    adapter.enqueue_pending_message(
        session_key,
        MessageEvent(text="second", message_type=MessageType.TEXT),
    )

    assert _dequeue_pending_text(adapter, session_key) == "first"
    assert _dequeue_pending_text(adapter, session_key) == "second"
    assert _dequeue_pending_text(adapter, session_key) is None
