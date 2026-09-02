"""Tests for TelegramAdapter.send_or_update_status (issue #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys independent (no cross-talk).
"""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _install_fake_telegram(monkeypatch):
    """Stub the python-telegram-bot package so TelegramAdapter can be imported."""
    fake_telegram = types.ModuleType("telegram")
    fake_telegram.Update = SimpleNamespace(ALL_TYPES=())
    fake_telegram.Bot = object
    fake_telegram.Message = object
    fake_telegram.InlineKeyboardButton = object
    fake_telegram.InlineKeyboardMarkup = object

    fake_error = types.ModuleType("telegram.error")
    fake_error.NetworkError = type("NetworkError", (Exception,), {})
    fake_error.BadRequest = type("BadRequest", (Exception,), {})
    fake_error.TimedOut = type("TimedOut", (Exception,), {})
    fake_telegram.error = fake_error

    fake_constants = types.ModuleType("telegram.constants")
    fake_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    fake_constants.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup",
        CHANNEL="channel", PRIVATE="private",
    )
    fake_telegram.constants = fake_constants

    fake_ext = types.ModuleType("telegram.ext")
    fake_ext.Application = object
    fake_ext.CommandHandler = object
    fake_ext.CallbackQueryHandler = object
    fake_ext.InlineQueryHandler = object
    fake_ext.MessageHandler = object
    fake_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    fake_ext.filters = object

    fake_request = types.ModuleType("telegram.request")
    fake_request.HTTPXRequest = object

    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", fake_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", fake_request)


@pytest.fixture
def adapter(monkeypatch):
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    # Patch send / edit_message so tests can drive them directly.
    a.send = AsyncMock()
    a.edit_message = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_first_call_sends_and_caches_message_id(adapter):
    """First call for a (chat, key) pair must send and remember the id."""
    adapter.send.return_value = SendResult(success=True, message_id="100")

    result = await adapter.send_or_update_status("chat-1", "lifecycle", "starting")

    assert result.success is True
    assert result.message_id == "100"
    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "100"


@pytest.mark.asyncio
async def test_distinct_status_keys_do_not_collide(adapter):
    """A different status_key gets its own message; the original isn't touched."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]

    await adapter.send_or_update_status("chat-1", "lifecycle", "ctx pressure")
    await adapter.send_or_update_status("chat-1", "model-switch", "switched to opus")

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "100"
    assert adapter._status_message_ids[("chat-1", "model-switch")] == "200"


@pytest.mark.asyncio
async def test_status_message_cache_stays_bounded_and_keeps_newest_entry(adapter):
    """Distinct chats cannot grow the cache past its configured bound."""
    adapter._STATUS_MESSAGE_IDS_MAX = 4
    adapter.send.side_effect = [
        SendResult(success=True, message_id=str(message_id))
        for message_id in range(10)
    ]

    for chat_id in range(10):
        await adapter.send_or_update_status(str(chat_id), "lifecycle", "starting")

    assert len(adapter._status_message_ids) <= adapter._STATUS_MESSAGE_IDS_MAX
    assert adapter._status_message_ids[("9", "lifecycle")] == "9"


@pytest.mark.asyncio
async def test_concurrent_cached_edits_do_not_reinsert_evicted_status_keys(adapter):
    adapter._STATUS_MESSAGE_IDS_MAX = 4
    old_keys = [(str(chat_id), "lifecycle") for chat_id in range(4)]
    adapter._status_message_ids.update(
        {key: str(message_id) for message_id, key in enumerate(old_keys)}
    )
    edit_started = {"0": asyncio.Event(), "1": asyncio.Event()}
    release_edits = asyncio.Event()

    async def blocked_successful_edit(chat_id, message_id, content, **kwargs):
        edit_started[message_id].set()
        await release_edits.wait()
        return SendResult(success=True, message_id=message_id)

    adapter.edit_message.side_effect = blocked_successful_edit
    adapter.send.return_value = SendResult(success=True, message_id="4")

    pending_edits = [
        asyncio.create_task(
            adapter.send_or_update_status(str(chat_id), "lifecycle", "updating")
        )
        for chat_id in range(2)
    ]
    await asyncio.gather(*(event.wait() for event in edit_started.values()))

    fresh_key = ("4", "lifecycle")
    await adapter.send_or_update_status(*fresh_key, "starting")
    release_edits.set()
    await asyncio.gather(*pending_edits)

    assert len(adapter._status_message_ids) <= adapter._STATUS_MESSAGE_IDS_MAX
    assert adapter._status_message_ids[fresh_key] == "4"
    assert old_keys[0] not in adapter._status_message_ids
    assert old_keys[1] not in adapter._status_message_ids


