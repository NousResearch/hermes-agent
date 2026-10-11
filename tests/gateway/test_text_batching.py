"""Tests for text message batching across all gateway adapters.

When a user sends a long message, the messaging client splits it at the
platform's character limit.  Each adapter should buffer rapid successive
text messages from the same session and aggregate them before dispatching.

Covers: Discord, Matrix, WeCom, and the adaptive delay logic for
Telegram and Feishu.
"""

import asyncio
from dataclasses import replace
from importlib import import_module
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


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
    adapter._client = None
    adapter._text_batch_intakes = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.0
    adapter._text_batch_split_delay_seconds = 0.0
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
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        adapter = _make_matrix_adapter()

        adapter._enqueue_text_event(_make_event("first part", Platform.MATRIX))
        adapter._enqueue_text_event(_make_event("second part", Platform.MATRIX))

        adapter.handle_message.assert_not_called()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values())

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


_ADAPTER_TYPES = {
    Platform.DISCORD: ("plugins.platforms.discord.adapter", "DiscordAdapter"),
    Platform.MATRIX: ("plugins.platforms.matrix.adapter", "MatrixAdapter"),
    Platform.WHATSAPP: ("plugins.platforms.whatsapp.adapter", "WhatsAppAdapter"),
    Platform("simplex"): ("plugins.platforms.simplex.adapter", "SimplexAdapter"),
    Platform.WECOM: ("plugins.platforms.wecom.adapter", "WeComAdapter"),
}


def _make_reply_batch_adapter(platform: Platform):
    module, adapter_type = _ADAPTER_TYPES[platform]
    adapter = object.__new__(getattr(import_module(module), adapter_type))
    if platform == Platform.MATRIX:
        adapter._client = None
        adapter._text_batch_intakes = {}
        adapter._buffered_intakes = {}
    adapter._platform = adapter.platform = platform
    adapter.config = PlatformConfig(enabled=True, token="test-token")
    adapter._background_tasks = set()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0
    adapter._text_batch_split_delay_seconds = 0
    adapter._attachment_text_merge_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", tuple(_ADAPTER_TYPES))
@pytest.mark.parametrize(
    "boundary",
    (
        "same", "plain-first", "plain-last", "reply_to_message_id",
        "reply_to_text", "reply_to_author_id", "reply_to_author_name",
        "reply_to_is_own_message", "attachment-caption",
    ),
)
async def test_shared_batches_preserve_reply_context_and_attachment_positions(
    platform, boundary
):
    adapter = _make_reply_batch_adapter(platform)
    first = _make_event("first", platform)
    first.reply_to_message_id = "reply-a"
    first.reply_to_text = "quoted text"
    first.reply_to_author_id = "author-a"
    first.reply_to_author_name = "Author A"
    first.reply_to_is_own_message = False
    first.media_urls = ["/tmp/first.png"]
    first.media_types = ["image/png"]
    first.media_text_inlined = []
    second = replace(
        first, text="second", media_urls=["/tmp/second.txt"],
        media_types=["text/plain"], media_text_inlined=[False],
    )
    if boundary.startswith("reply_to_"):
        setattr(second, boundary, True if boundary == "reply_to_is_own_message" else "other")
    if boundary in {"plain-first", "plain-last"}:
        plain = first if boundary == "plain-first" else second
        plain.reply_to_message_id = plain.reply_to_text = None
        plain.reply_to_author_id = plain.reply_to_author_name = None
        plain.reply_to_is_own_message = False
    if boundary == "attachment-caption":
        first.text = ""
        first.message_type = MessageType.PHOTO
        second.message_type = MessageType.TEXT
        second.reply_to_message_id = "reply-b"
    separated = boundary.startswith("reply_to_") or boundary == "attachment-caption"
    if separated:
        expected = [replace(first), replace(second)]
    else:
        context = second if boundary == "plain-first" else first
        expected = [replace(
            first, text="first\nsecond",
            reply_to_message_id=context.reply_to_message_id,
            reply_to_text=context.reply_to_text,
            reply_to_author_id=context.reply_to_author_id,
            reply_to_author_name=context.reply_to_author_name,
            reply_to_is_own_message=context.reply_to_is_own_message,
            media_urls=first.media_urls + second.media_urls,
            media_types=first.media_types + second.media_types,
            media_text_inlined=[None, False],
        )]

    adapter._enqueue_text_event(first)
    adapter._enqueue_text_event(second)
    await asyncio.wait_for(asyncio.gather(
        *adapter._background_tasks, *adapter._pending_text_batch_tasks.values()
    ), timeout=5)
    actual = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", tuple(_ADAPTER_TYPES))
async def test_shared_reply_boundary_preserves_an_in_flight_dispatch(platform):
    adapter = _make_reply_batch_adapter(platform)
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    first = _make_event("first", platform)
    first.reply_to_message_id = "reply-a"
    second = _make_event("second", platform)
    second.reply_to_message_id = "reply-b"
    expected = [replace(first), replace(second)]

    async def dispatch(event):
        if event is first:
            started.set()
            await release.wait()
            finished.set()

    adapter.handle_message.side_effect = dispatch
    try:
        adapter._enqueue_text_event(first)
        adapter._enqueue_text_event(second)
        boundary_tasks = tuple(adapter._background_tasks)
        await asyncio.wait_for(started.wait(), timeout=5)
        for task in boundary_tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*boundary_tasks), timeout=5)
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        await asyncio.wait_for(asyncio.gather(
            *adapter._pending_text_batch_tasks.values()
        ), timeout=5)
    finally:
        release.set()
        tasks = (*adapter._background_tasks, *adapter._pending_text_batch_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
    actual = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert actual == expected
