"""Tests for TelegramAdapter.send_or_update_status (issue #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys independent (no cross-talk).
"""

from __future__ import annotations

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
async def test_delete_message_drops_cached_status_id(adapter):
    """cleanup_progress deletes the bubble; the status cache must not keep that id."""
    adapter._status_message_ids[("chat-1", "lifecycle")] = "3202"
    adapter._bot.delete_message = AsyncMock(return_value=True)

    ok = await adapter.delete_message("chat-1", "3202")

    assert ok is True
    assert ("chat-1", "lifecycle") not in adapter._status_message_ids


@pytest.mark.asyncio
async def test_status_after_delete_sends_fresh_instead_of_editing_tombstone(adapter):
    """Next status emit after cleanup must send, not edit the deleted id."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="3202"),
        SendResult(success=True, message_id="3203"),
    ]
    adapter._bot.delete_message = AsyncMock(return_value=True)

    first = await adapter.send_or_update_status("chat-1", "lifecycle", "starting")
    assert first.message_id == "3202"
    await adapter.delete_message("chat-1", "3202")

    second = await adapter.send_or_update_status("chat-1", "lifecycle", "still working")

    assert second.message_id == "3203"
    adapter.edit_message.assert_not_awaited()
    assert adapter.send.await_count == 2
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "3203"


@pytest.mark.asyncio
async def test_finalize_missing_message_does_not_retry_plain_text(monkeypatch):
    """A gone message is not a MarkdownV2 format failure (no second plain-text edit)."""
    _install_fake_telegram(monkeypatch)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    gone = Exception("Bad Request: message to edit not found")
    adapter._bot.edit_message_text = AsyncMock(side_effect=gone)
    adapter._status_message_ids[("123", "lifecycle")] = "456"

    result = await adapter.edit_message("123", "456", "still working", finalize=True)

    assert result.success is False
    assert "not found" in (result.error or "").lower()
    assert adapter._bot.edit_message_text.await_count == 1
    assert ("123", "lifecycle") not in adapter._status_message_ids

