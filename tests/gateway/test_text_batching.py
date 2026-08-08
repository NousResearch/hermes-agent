"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    merge_pending_message_event,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_event(
    text: str,
    platform: Platform,
    chat_id: str = "12345",
    msg_type: MessageType = MessageType.TEXT,
    msg_id: Optional[str] = None,
    reply_to_id: Optional[str] = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=SessionSource(platform=platform, chat_id=chat_id, chat_type="dm"),
        message_id=msg_id,
        reply_to_message_id=reply_to_id,
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


# =====================================================================
# Busy-session text merge keeps the latest message id as the reply anchor
# =====================================================================

class TestMergePendingMessageEventLatestIdentity:
    def test_text_merge_carries_latest_message_id_and_reply_anchor(self):
        """When two text events are merged, the resulting event's message_id
        and reply_to_message_id must reflect the LATEST chunk so the bot's
        reply quotes the newest user message, not the first one (#59582)."""
        from gateway.platforms.base import build_session_key

        first = _make_event("first part", Platform.WHATSAPP, msg_id="wamid.A")
        second = _make_event("second part", Platform.WHATSAPP, msg_id="wamid.B")
        pending: dict[str, MessageEvent] = {}
        sk = build_session_key(first.source)

        merge_pending_message_event(pending, sk, first, merge_text=True)
        merge_pending_message_event(pending, sk, second, merge_text=True)

        merged = pending[sk]
        assert merged.text == "first part\nsecond part"
        assert merged.message_id == "wamid.B"
        assert merged.reply_to_message_id == "wamid.B"

    def test_text_merge_falls_back_to_latest_reply_anchor(self):
        """A later chunk with no message_id of its own but a reply_to anchor
        still upgrades the merged event's reply anchor."""
        from gateway.platforms.base import build_session_key

        first = _make_event("first part", Platform.WHATSAPP, msg_id="wamid.A")
        second = _make_event(
            "second part",
            Platform.WHATSAPP,
            msg_id=None,
            reply_to_id="wamid.PRIOR",
        )
        pending: dict[str, MessageEvent] = {}
        sk = build_session_key(first.source)

        merge_pending_message_event(pending, sk, first, merge_text=True)
        merge_pending_message_event(pending, sk, second, merge_text=True)

        merged = pending[sk]
        assert merged.message_id == "wamid.A"
        assert merged.reply_to_message_id == "wamid.PRIOR"


# =====================================================================
# WhatsApp (Baileys bridge) text batching
# =====================================================================

def _make_whatsapp_adapter():
    """Create a minimal WhatsAppAdapter for testing text batching."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(WhatsAppAdapter)
    adapter._platform = Platform.WHATSAPP
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


class TestWhatsAppTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_whatsapp_adapter()
        event = _make_event("hello world", Platform.WHATSAPP)

        adapter._enqueue_text_event(event)

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated_with_latest_message_id(self):
        """Two rapid WhatsApp messages should be merged, and the merged event
        must carry the latest message_id as the reply anchor (#59582)."""
        adapter = _make_whatsapp_adapter()

        adapter._enqueue_text_event(
            _make_event("first part", Platform.WHATSAPP, msg_id="wamid.A")
        )
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(
            _make_event("second part", Platform.WHATSAPP, msg_id="wamid.B")
        )

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "first part" in dispatched.text
        assert "second part" in dispatched.text
        assert dispatched.message_id == "wamid.B"
        assert dispatched.reply_to_message_id == "wamid.B"


