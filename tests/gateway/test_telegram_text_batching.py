"""Tests for Telegram text message aggregation.

When a user sends a long message, Telegram clients split it into multiple
updates.  The TelegramAdapter should buffer rapid successive text messages
from the same session and aggregate them before dispatching.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_adapter():
    """Create a minimal TelegramAdapter for testing text batching."""
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(
    text: str,
    chat_id: str = "12345",
    *,
    message_id: str | None = None,
    reply_to_message_id: str | None = None,
    reply_to_text: str | None = None,
    metadata: dict | None = None,
    thread_id: str | None = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="dm",
            thread_id=thread_id,
        ),
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
        reply_to_text=reply_to_text,
        metadata=metadata,
    )


class TestTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_adapter()
        event = _make_event("hello world")

        adapter._enqueue_text_event(event)

        # Not dispatched yet
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        """Two rapid messages from the same chat should be merged."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("This is part one of a long"))
        await asyncio.sleep(0.02)  # small gap, within batch window
        adapter._enqueue_text_event(_make_event("message that was split by Telegram."))

        # Not dispatched yet (timer restarted)
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "part one" in dispatched.text
        assert "split by Telegram" in dispatched.text

    @pytest.mark.asyncio
    async def test_three_way_split_aggregated(self):
        """Three rapid messages should all merge."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("chunk 1"))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("chunk 2"))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("chunk 3"))

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "chunk 1" in text
        assert "chunk 2" in text
        assert "chunk 3" in text

    @pytest.mark.asyncio
    async def test_different_chats_not_merged(self):
        """Messages from different chats should be separate batches."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("from user A", chat_id="111"))
        adapter._enqueue_text_event(_make_event("from user B", chat_id="222"))

        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_cleans_up_after_flush(self):
        """After flushing, internal state should be clean."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("test"))
        await asyncio.sleep(0.2)

        assert len(adapter._pending_text_batches) == 0
        assert len(adapter._pending_text_batch_tasks) == 0

    @pytest.mark.asyncio
    async def test_incompatible_reply_context_flushes_current_batch_immediately(self):
        adapter = _make_adapter()

        adapter._enqueue_text_event(
            _make_event(
                "reply chunk 1",
                message_id="m1",
                reply_to_message_id="bot-msg-1",
                reply_to_text="上一条",
            )
        )
        adapter._enqueue_text_event(
            _make_event(
                "reply chunk 2",
                message_id="m2",
                reply_to_message_id="bot-msg-2",
                reply_to_text="另一条",
            )
        )

        await asyncio.sleep(0.05)
        assert adapter.handle_message.call_count == 1
        first = adapter.handle_message.call_args_list[0].args[0]
        assert first.text == "reply chunk 1"
        assert first.reply_to_message_id == "bot-msg-1"

        await asyncio.sleep(0.2)
        assert adapter.handle_message.call_count == 2
        second = adapter.handle_message.call_args_list[1].args[0]
        assert second.text == "reply chunk 2"
        assert second.reply_to_message_id == "bot-msg-2"

    @pytest.mark.asyncio
    async def test_compatible_batch_keeps_latest_event_context(self):
        adapter = _make_adapter()

        adapter._enqueue_text_event(
            _make_event(
                "chunk 1",
                message_id="m1",
                metadata={"explicit_addressed": False},
                reply_to_message_id="bot-msg-1",
                reply_to_text="上一条",
            )
        )
        adapter._enqueue_text_event(
            _make_event(
                "chunk 2",
                message_id="m2",
                metadata={"explicit_addressed": True, "address_reason": "reply_to_bot"},
                reply_to_message_id="bot-msg-1",
                reply_to_text="上一条",
            )
        )

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args.args[0]
        assert dispatched.text == "chunk 1\nchunk 2"
        assert dispatched.message_id == "m2"
        assert dispatched.metadata["explicit_addressed"] is True
        assert dispatched.metadata["address_reason"] == "reply_to_bot"
