"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

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
    from gateway.platforms.discord import DiscordAdapter

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

    @pytest.mark.asyncio
    async def test_three_way_split_aggregated(self):
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("chunk 1", Platform.DISCORD))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("chunk 2", Platform.DISCORD))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("chunk 3", Platform.DISCORD))

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "chunk 1" in text
        assert "chunk 2" in text
        assert "chunk 3" in text

    @pytest.mark.asyncio
    async def test_different_chats_not_merged(self):
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("from A", Platform.DISCORD, chat_id="111"))
        adapter._enqueue_text_event(_make_event("from B", Platform.DISCORD, chat_id="222"))

        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_cleans_up_after_flush(self):
        adapter = _make_discord_adapter()

        adapter._enqueue_text_event(_make_event("test", Platform.DISCORD))
        await asyncio.sleep(0.2)

        assert len(adapter._pending_text_batches) == 0

    @pytest.mark.asyncio
    async def test_adaptive_delay_for_near_limit_chunk(self):
        """Chunks near the 2000-char limit should trigger longer delay."""
        adapter = _make_discord_adapter()
        # Simulate a chunk near Discord's 2000-char split point
        long_text = "x" * 1950
        adapter._enqueue_text_event(_make_event(long_text, Platform.DISCORD))

        # After the short delay (0.1s), should NOT have flushed yet (split delay is 0.3s)
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_not_called()

        # After the split delay, should be flushed
        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_shield_protects_handle_message_from_cancel(self):
        """Regression guard: a follow-up chunk arriving while
        handle_message is mid-flight must NOT cancel the running
        dispatch.  _enqueue_text_event fires prior_task.cancel() on
        every new chunk; without asyncio.shield around handle_message
        the cancel propagates into the agent's streaming request and
        aborts the response.
        """
        adapter = _make_discord_adapter()

        handle_started = asyncio.Event()
        release_handle = asyncio.Event()
        first_handle_cancelled = asyncio.Event()
        first_handle_completed = asyncio.Event()
        call_count = [0]

        async def slow_handle(event):
            call_count[0] += 1
            # Only the first call (batch 1) is the one we're protecting.
            if call_count[0] == 1:
                handle_started.set()
                try:
                    await release_handle.wait()
                    first_handle_completed.set()
                except asyncio.CancelledError:
                    first_handle_cancelled.set()
                    raise
            # Second call (batch 2) returns immediately — not the subject
            # of this test.

        adapter.handle_message = slow_handle

        # Prime batch 1 and wait for it to land inside handle_message.
        adapter._enqueue_text_event(_make_event("batch 1", Platform.DISCORD))
        await asyncio.wait_for(handle_started.wait(), timeout=1.0)

        # A new chunk arrives — _enqueue_text_event fires
        # prior_task.cancel() on batch 1's flush task, which is
        # currently awaiting inside handle_message.
        adapter._enqueue_text_event(_make_event("batch 2 follow-up", Platform.DISCORD))

        # Let the cancel propagate.
        await asyncio.sleep(0.05)

        # CRITICAL ASSERTION: batch 1's handle_message must NOT have
        # been cancelled.  Without asyncio.shield this assertion fails
        # because CancelledError propagates from the flush task's
        # `await self.handle_message(event)` into slow_handle.
        assert not first_handle_cancelled.is_set(), (
            "handle_message for batch 1 was cancelled by a follow-up "
            "chunk — asyncio.shield is missing or broken"
        )

        # Release batch 1's handle_message and let it complete.
        release_handle.set()
        await asyncio.wait_for(first_handle_completed.wait(), timeout=1.0)
        assert first_handle_completed.is_set()

        # Cleanup
        for task in list(adapter._pending_text_batch_tasks.values()):
            task.cancel()
        await asyncio.sleep(0.01)


# =====================================================================
# Matrix text batching
# =====================================================================

def _make_matrix_adapter():
    """Create a minimal MatrixAdapter for testing text batching."""
    from gateway.platforms.matrix import MatrixAdapter

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
    async def test_different_rooms_not_merged(self):
        adapter = _make_matrix_adapter()

        adapter._enqueue_text_event(_make_event("room A", Platform.MATRIX, chat_id="!aaa:matrix.org"))
        adapter._enqueue_text_event(_make_event("room B", Platform.MATRIX, chat_id="!bbb:matrix.org"))

        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_adaptive_delay_for_near_limit_chunk(self):
        """Chunks near the 4000-char limit should trigger longer delay."""
        adapter = _make_matrix_adapter()
        long_text = "x" * 3950
        adapter._enqueue_text_event(_make_event(long_text, Platform.MATRIX))

        await asyncio.sleep(0.15)
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_cleans_up_after_flush(self):
        adapter = _make_matrix_adapter()
        adapter._enqueue_text_event(_make_event("test", Platform.MATRIX))
        await asyncio.sleep(0.2)
        assert len(adapter._pending_text_batches) == 0


# =====================================================================
# WeCom text batching
# =====================================================================

def _make_wecom_adapter():
    """Create a minimal WeComAdapter for testing text batching."""
    from gateway.platforms.wecom import WeComAdapter

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

    @pytest.mark.asyncio
    async def test_different_chats_not_merged(self):
        adapter = _make_wecom_adapter()

        adapter._enqueue_text_event(_make_event("chat A", Platform.WECOM, chat_id="chat_a"))
        adapter._enqueue_text_event(_make_event("chat B", Platform.WECOM, chat_id="chat_b"))

        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_adaptive_delay_for_near_limit_chunk(self):
        """Chunks near the 4000-char limit should trigger longer delay."""
        adapter = _make_wecom_adapter()
        long_text = "x" * 3950
        adapter._enqueue_text_event(_make_event(long_text, Platform.WECOM))

        await asyncio.sleep(0.15)
        adapter.handle_message.assert_not_called()

        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_cleans_up_after_flush(self):
        adapter = _make_wecom_adapter()
        adapter._enqueue_text_event(_make_event("test", Platform.WECOM))
        await asyncio.sleep(0.2)
        assert len(adapter._pending_text_batches) == 0


# =====================================================================
# Telegram adaptive delay (PR #6891)
# =====================================================================

def _make_telegram_adapter():
    """Create a minimal TelegramAdapter for testing adaptive delay."""
    from gateway.platforms.telegram import TelegramAdapter

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

    @pytest.mark.asyncio
    async def test_near_limit_chunk_uses_split_delay(self):
        """A chunk near the 4096-char limit should trigger longer delay."""
        adapter = _make_telegram_adapter()
        long_text = "x" * 4050  # near the 4096 limit
        adapter._enqueue_text_event(_make_event(long_text, Platform.TELEGRAM))

        # After the short delay, should NOT have flushed yet
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_not_called()

        # After the split delay, should be flushed
        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_split_continuation_merged(self):
        """Two near-limit chunks should both be merged."""
        adapter = _make_telegram_adapter()

        adapter._enqueue_text_event(_make_event("x" * 4050, Platform.TELEGRAM))
        await asyncio.sleep(0.05)
        adapter._enqueue_text_event(_make_event("continuation text", Platform.TELEGRAM))

        # Short chunk arrived → should use normal delay now
        await asyncio.sleep(0.15)
        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "continuation text" in text


# =====================================================================
# Feishu adaptive delay
# =====================================================================

def _make_feishu_adapter():
    """Create a minimal FeishuAdapter for testing adaptive delay."""
    from gateway.platforms.feishu import FeishuAdapter, FeishuBatchState

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

    @pytest.mark.asyncio
    async def test_near_limit_chunk_uses_split_delay(self):
        """A chunk near the 4096-char limit should trigger longer delay."""
        adapter = _make_feishu_adapter()
        long_text = "x" * 4050
        event = _make_event(long_text, Platform.FEISHU)
        await adapter._enqueue_text_event(event)

        await asyncio.sleep(0.15)
        adapter._handle_message_with_guards.assert_not_called()

        await asyncio.sleep(0.25)
        adapter._handle_message_with_guards.assert_called_once()

    @pytest.mark.asyncio
    async def test_split_continuation_merged(self):
        adapter = _make_feishu_adapter()

        await adapter._enqueue_text_event(_make_event("x" * 4050, Platform.FEISHU))
        await asyncio.sleep(0.05)
        await adapter._enqueue_text_event(_make_event("continuation text", Platform.FEISHU))

        await asyncio.sleep(0.15)
        adapter._handle_message_with_guards.assert_called_once()
        text = adapter._handle_message_with_guards.call_args[0][0].text
        assert "continuation text" in text
