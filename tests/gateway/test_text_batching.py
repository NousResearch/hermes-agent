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
from gateway.platforms.base import SessionSource, _reply_anchor_for_event, merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType


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
    adapter._platform = adapter.platform = Platform.DISCORD
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
    adapter._platform = adapter.platform = Platform.MATRIX
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
    adapter._platform = adapter.platform = Platform.WECOM
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
        assert _reply_anchor_for_event(merged) == "wamid.B"

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

    def test_anchorless_followup_keeps_existing_identity(self):
        first = _make_event("first", Platform.WHATSAPP, msg_id="wamid.A", reply_to_id="wamid.A")
        pending = {"session": first}
        merge_pending_message_event(pending, "session", _make_event("second", Platform.WHATSAPP), merge_text=True)

        assert pending["session"].text == "first\nsecond"
        assert pending["session"].message_id == "wamid.A"
        assert pending["session"].reply_to_message_id == "wamid.A"
        assert _reply_anchor_for_event(pending["session"]) == "wamid.A"


# =====================================================================
# WhatsApp (Baileys bridge) text batching
# =====================================================================

def _make_whatsapp_adapter():
    """Create a minimal WhatsAppAdapter for testing text batching."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
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
        await asyncio.wait_for(
            asyncio.gather(*adapter._pending_text_batch_tasks.values()), timeout=2,
        )

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
        adapter._enqueue_text_event(
            _make_event("second part", Platform.WHATSAPP, msg_id="wamid.B")
        )

        adapter.handle_message.assert_not_called()
        await asyncio.wait_for(
            asyncio.gather(*adapter._pending_text_batch_tasks.values()), timeout=2,
        )

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "first part\nsecond part"
        assert dispatched.message_id == "wamid.B"
        assert dispatched.reply_to_message_id == "wamid.B"
        assert _reply_anchor_for_event(dispatched) == "wamid.B"

    @pytest.mark.asyncio
    async def test_rejected_route_keeps_pending_message_and_timer(self):
        adapter = _make_whatsapp_adapter()
        first = _make_event("first", Platform.WHATSAPP, msg_id="wamid.A")
        adapter._enqueue_text_event(first)
        key = adapter._text_batch_key(first)
        original_task = adapter._pending_text_batch_tasks[key]
        rejected = _make_event("rejected", Platform.WHATSAPP, msg_id="wamid.B")
        rejected.source.profile_route_rejected = True

        adapter._enqueue_text_event(rejected)

        assert adapter._pending_text_batches[key] is first
        assert first.text == "first" and first.message_id == "wamid.A"
        assert adapter._pending_text_batch_tasks[key] is original_task
        await asyncio.wait_for(original_task, timeout=2)
        adapter.handle_message.assert_awaited_once_with(first)
        assert _reply_anchor_for_event(first) == "wamid.A"
