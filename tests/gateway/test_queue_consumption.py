"""Tests for /queue message consumption after normal agent completion.

Verifies that messages queued via /queue (which store in
adapter._pending_messages WITHOUT triggering an interrupt) are consumed
after the agent finishes its current task — not silently dropped.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    PlatformConfig,
    Platform,
)
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Minimal adapter for testing pending message storage
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueueMessageStorage:
    """Verify /queue stores messages correctly in adapter._pending_messages."""

    def test_queue_stores_message_in_pending(self):
        adapter = _StubAdapter()
        session_key = "telegram:user:123"
        event = MessageEvent(
            text="do this next",
            message_type=MessageType.TEXT,
            source=MagicMock(chat_id="123", platform=Platform.TELEGRAM),
            message_id="q1",
        )
        adapter._pending_messages[session_key] = event

        assert session_key in adapter._pending_messages
        assert adapter._pending_messages[session_key].text == "do this next"

    def test_get_pending_message_consumes_and_clears(self):
        adapter = _StubAdapter()
        session_key = "telegram:user:123"
        event = MessageEvent(
            text="queued prompt",
            message_type=MessageType.TEXT,
            source=MagicMock(chat_id="123", platform=Platform.TELEGRAM),
            message_id="q2",
        )
        adapter._pending_messages[session_key] = event

        retrieved = adapter.get_pending_message(session_key)
        assert retrieved is not None
        assert retrieved.text == "queued prompt"
        # Should be consumed (cleared)
        assert adapter.get_pending_message(session_key) is None

    def test_queue_does_not_set_interrupt_event(self):
        """The whole point of /queue — no interrupt signal."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        # Simulate an active session (agent running)
        adapter._active_sessions[session_key] = asyncio.Event()

        # Store a queued message (what /queue does)
        event = MessageEvent(
            text="queued",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="q3",
        )
        adapter._pending_messages[session_key] = event

        # The interrupt event should NOT be set
        assert not adapter._active_sessions[session_key].is_set()
        assert not adapter.has_pending_interrupt(session_key)

    def test_regular_message_sets_interrupt_event(self):
        """Contrast: regular messages DO trigger interrupt."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        adapter._active_sessions[session_key] = asyncio.Event()

        # Simulate regular message arrival (what handle_message does)
        event = MessageEvent(
            text="new message",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="m1",
        )
        adapter._pending_messages[session_key] = event
        adapter._active_sessions[session_key].set()  # this is what handle_message does

        assert adapter.has_pending_interrupt(session_key)


class TestQueueConsumptionAfterCompletion:
    """Verify that pending messages are consumed after normal completion."""

    def test_pending_message_available_after_normal_completion(self):
        """After agent finishes without interrupt, pending message should
        still be retrievable from adapter._pending_messages."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        # Simulate: agent starts, /queue stores a message, agent finishes
        adapter._active_sessions[session_key] = asyncio.Event()
        event = MessageEvent(
            text="process this after",
            message_type=MessageType.TEXT,
            source=MagicMock(),
            message_id="q4",
        )
        adapter._pending_messages[session_key] = event

        # Agent finishes (no interrupt)
        del adapter._active_sessions[session_key]

        # The queued message should still be retrievable
        retrieved = adapter.get_pending_message(session_key)
        assert retrieved is not None
        assert retrieved.text == "process this after"

    def test_multiple_queues_last_one_wins(self):
        """If user /queue's multiple times, last message overwrites."""
        adapter = _StubAdapter()
        session_key = "telegram:user:123"

        for text in ["first", "second", "third"]:
            event = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=MagicMock(),
                message_id=f"q-{text}",
            )
            adapter._pending_messages[session_key] = event

        retrieved = adapter.get_pending_message(session_key)
        assert retrieved.text == "third"


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(adapter: BasePlatformAdapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._draining = False
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    runner._session_key_for_source = GatewayRunner._session_key_for_source.__get__(
        runner, GatewayRunner
    )
    runner._handle_message_with_agent = AsyncMock(return_value="agent path")
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    return runner


class TestQueueCommandRouting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("command_text", ["/queue after this", "/q after this"])
    async def test_active_session_commands_bypass_adapter_interrupt_path(self, command_text):
        adapter = _StubAdapter()
        adapter._send_with_retry = AsyncMock()
        adapter.set_message_handler(AsyncMock(return_value="Queued for the next turn."))

        source = _make_source()
        session_key = build_session_key(source)
        interrupt_event = asyncio.Event()
        adapter._active_sessions[session_key] = interrupt_event

        event = MessageEvent(
            text=command_text,
            source=source,
            message_id="m-active",
            message_type=MessageType.COMMAND,
        )

        await adapter.handle_message(event)

        adapter._message_handler.assert_awaited_once_with(event)
        adapter._send_with_retry.assert_awaited_once()
        assert not interrupt_event.is_set()
        assert session_key not in adapter._pending_messages

    @pytest.mark.asyncio
    @pytest.mark.parametrize("command_text", ["/queue after this", "/q after this"])
    async def test_idle_queue_dispatches_to_queue_handler(self, command_text):
        adapter = _StubAdapter()
        runner = _make_runner(adapter)
        source = _make_source()
        event = MessageEvent(
            text=command_text,
            source=source,
            message_id="m-idle",
            message_type=MessageType.COMMAND,
        )

        result = await runner._handle_message(event)

        session_key = runner._session_key_for_source(source)
        queued_event = adapter._pending_messages[session_key]
        assert result == "Queued for the next turn."
        assert queued_event.text == "after this"
        assert queued_event.message_type == MessageType.TEXT
        runner._handle_message_with_agent.assert_not_awaited()
