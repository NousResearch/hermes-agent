"""Regression tests for long slash-command pastes split by Telegram clients.

Telegram splits messages above 4096 chars into multiple updates.  A long
``/queue <huge prompt>`` paste arrives as a COMMAND chunk near the limit plus
plain TEXT continuation chunk(s).  Before the fix, `_handle_command`
dispatched the command chunk immediately, orphaning the continuation — which
then landed as a separate plain message and interrupted the running agent.

Near-limit command chunks must route through the text-batching pipeline so
continuations merge in before dispatch; short commands stay immediate.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import utf16_len


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.05
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter.handle_message = AsyncMock()
    return adapter


def _make_message(text, *, chat_id=12345):
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=chat_id, type="private", title=None, is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


def _make_update(text):
    return SimpleNamespace(update_id=1, message=_make_message(text), effective_message=None)


@pytest.mark.asyncio
async def test_short_command_dispatched_immediately():
    adapter = _make_adapter()

    await adapter._handle_command(_make_update("/stop"), SimpleNamespace())

    adapter.handle_message.assert_awaited_once()
    assert not adapter._pending_text_batches


@pytest.mark.asyncio
async def test_near_limit_command_is_batched_not_dispatched():
    adapter = _make_adapter()
    long_cmd = "/queue " + "x" * 4090  # near Telegram's 4096 split point

    await adapter._handle_command(_make_update(long_cmd), SimpleNamespace())

    # Must NOT dispatch immediately — a continuation chunk is almost certain.
    adapter.handle_message.assert_not_awaited()
    assert len(adapter._pending_text_batches) == 1


@pytest.mark.asyncio
async def test_split_queue_command_merges_with_continuation():
    adapter = _make_adapter()
    first = "/queue " + "x" * 4089  # 4096-char first chunk
    continuation = "y" * 500

    await adapter._handle_command(_make_update(first), SimpleNamespace())
    adapter.handle_message.assert_not_awaited()

    # Continuation arrives as a plain text update moments later.
    await adapter._handle_text_message(_make_update(continuation), SimpleNamespace())

    # Wait past the split-delay flush window.
    await asyncio.sleep(0.3)

    adapter.handle_message.assert_awaited_once()
    dispatched = adapter.handle_message.call_args[0][0]
    assert dispatched.text.startswith("/queue ")
    assert "y" * 500 in dispatched.text
    # One merged event — the continuation never dispatched on its own.
    assert not adapter._pending_text_batches


def _emoji_near_utf16_split(prefix: str = "/queue ") -> str:
    """Chunk with Python len() < 4000 but UTF-16 >= 4000 (Telegram's split unit)."""
    emoji = "\U0001F600"  # 1 Python char, 2 UTF-16 units
    text = prefix + emoji * 2000
    assert len(text) < 4000
    assert utf16_len(text) >= 4000
    return text


@pytest.mark.asyncio
async def test_emoji_near_utf16_limit_command_is_batched_not_dispatched():
    adapter = _make_adapter()

    await adapter._handle_command(_make_update(_emoji_near_utf16_split()), SimpleNamespace())

    adapter.handle_message.assert_not_awaited()
    assert len(adapter._pending_text_batches) == 1


@pytest.mark.asyncio
async def test_emoji_chunk_records_utf16_last_chunk_len():
    adapter = _make_adapter()
    text = "\U0001F600" * 2000
    assert len(text) == 2000
    assert utf16_len(text) == 4000

    await adapter._handle_text_message(_make_update(text), SimpleNamespace())

    pending = next(iter(adapter._pending_text_batches.values()))
    assert pending._last_chunk_len == utf16_len(text)
