"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# =====================================================================
# Helpers
# =====================================================================

def _make_event(
    text: str,
    platform: Platform,
    chat_id: str = "12345",
    msg_type: MessageType = MessageType.TEXT,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm"),
    )


# =====================================================================
# Discord text batching
# =====================================================================

def _make_discord_adapter():
    """Create a minimal DiscordAdapter for testing text batching."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._text_batch_split_delay_seconds = 0.3  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestDiscordTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_discord_adapter()
        event = _make_event("hello world", Platform.DISCORD)

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
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("Part one of a long", Platform.DISCORD))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("message that was split.", Platform.DISCORD))

        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "Part one" in text
        assert "split" in text


# =====================================================================
# Matrix text batching
# =====================================================================

def _make_matrix_adapter():
    """Create a minimal MatrixAdapter for testing text batching."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(MatrixAdapter)
    adapter._platform = Platform.MATRIX
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestMatrixTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_matrix_adapter()
        event = _make_event("hello world", Platform.MATRIX)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_matrix_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.MATRIX))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.MATRIX))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text

    @pytest.mark.asyncio
    async def test_different_senders_in_shared_thread_not_merged(self):
        """Two participants in one shared thread must not batch together.

        In a shared thread (``thread_sessions_per_user=False``, the default)
        ``build_session_key`` omits ``user_id``, so both senders resolve to the
        same session key. Batching must still key off the sender: otherwise both
        messages merge into one event attributed to the first sender — dropping
        the second sender's authorship (and letting their text ride the first
        sender's identity through downstream authorization).
        """
        adapter = _make_matrix_adapter()

        def _thread_event(text: str, user_id: str) -> MessageEvent:
            return MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=SessionSource(
                    platform=Platform.MATRIX,
                    chat_id="!room:matrix.org",
                    chat_type="group",
                    thread_id="$thread:matrix.org",
                    user_id=user_id,
                ),
            )

        adapter._enqueue_text_event(_thread_event("hi from alice", "@alice:matrix.org"))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_thread_event("hi from bob", "@bob:matrix.org"))

        await asyncio.sleep(0.2)

        # Each sender is dispatched as its own event — never fused into one.
        assert adapter.handle_message.call_count == 2
        dispatched = {
            call.args[0].source.user_id: call.args[0].text
            for call in adapter.handle_message.call_args_list
        }
        assert dispatched == {
            "@alice:matrix.org": "hi from alice",
            "@bob:matrix.org": "hi from bob",
        }
        # No single dispatched event carries both senders' text.
        for text in dispatched.values():
            assert not ("alice" in text and "bob" in text)

    @pytest.mark.asyncio
    async def test_same_sender_in_shared_thread_still_batches(self):
        """One participant's split chunks in a shared thread still aggregate."""
        adapter = _make_matrix_adapter()

        def _thread_event(text: str) -> MessageEvent:
            return MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=SessionSource(
                    platform=Platform.MATRIX,
                    chat_id="!room:matrix.org",
                    chat_type="group",
                    thread_id="$thread:matrix.org",
                    user_id="@alice:matrix.org",
                ),
            )

        adapter._enqueue_text_event(_thread_event("first part"))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_thread_event("second part"))

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


# =====================================================================
# WeCom text batching
# =====================================================================

def _make_wecom_adapter():
    """Create a minimal WeComAdapter for testing text batching."""
    from plugins.platforms.wecom.adapter import WeComAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(WeComAdapter)
    adapter._platform = Platform.WECOM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestWeComTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_wecom_adapter()
        event = _make_event("hello world", Platform.WECOM)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_wecom_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.WECOM))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("second part", Platform.WECOM))

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "first part" in text
        assert "second part" in text


# =====================================================================
# Telegram adaptive delay (PR #6891)
# =====================================================================

def _make_telegram_adapter():
    """Create a minimal TelegramAdapter for testing adaptive delay."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


class TestTelegramAdaptiveDelay:
    @pytest.mark.asyncio
    async def test_short_chunk_uses_normal_delay(self):
        adapter = _make_telegram_adapter()
        adapter._enqueue_text_event(_make_event("short msg", Platform.TELEGRAM))

        # Should flush after the normal 0.1s delay
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_called_once()


# =====================================================================
# Feishu adaptive delay
# =====================================================================

def _make_feishu_adapter():
    """Create a minimal FeishuAdapter for testing adaptive delay."""
    from plugins.platforms.feishu.adapter import FeishuAdapter, FeishuBatchState

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(FeishuAdapter)
    adapter._platform = Platform.FEISHU
    adapter.config = config
    batch_state = FeishuBatchState()
    adapter._pending_text_batches = batch_state.events
    adapter._pending_text_batch_tasks = batch_state.tasks
    adapter._pending_text_batch_counts = batch_state.counts
    adapter._text_batch_delay_seconds = 0.1
    adapter._text_batch_split_delay_seconds = 0.3
    adapter._text_batch_max_messages = 20
    adapter._text_batch_max_chars = 50000
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter._handle_message_with_guards = AsyncMock()
    return adapter


class TestFeishuAdaptiveDelay:
    @pytest.mark.asyncio
    async def test_short_chunk_uses_normal_delay(self):
        adapter = _make_feishu_adapter()
        event = _make_event("short msg", Platform.FEISHU)
        await adapter._enqueue_text_event(event)

        await asyncio.sleep(0.15)
        adapter._handle_message_with_guards.assert_called_once()


