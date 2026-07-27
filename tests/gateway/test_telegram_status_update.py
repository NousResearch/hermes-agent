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
async def test_second_call_edits_in_place(adapter):
    """Same (chat, key) on the second call must edit, not send."""
    adapter.send.return_value = SendResult(success=True, message_id="100")
    adapter.edit_message.return_value = SendResult(success=True, message_id="100")

    await adapter.send_or_update_status("chat-1", "lifecycle", "step 1")
    await adapter.send_or_update_status("chat-1", "lifecycle", "step 2")

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    # Edit was directed at the cached message id.
    args, kwargs = adapter.edit_message.call_args
    assert args[0] == "chat-1"
    assert args[1] == "100"
    assert args[2] == "step 2"


@pytest.mark.asyncio
async def test_edit_failure_falls_back_to_fresh_send(adapter):
    """When edit_message fails the cache is cleared and a new send happens."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]
    adapter.edit_message.return_value = SendResult(
        success=False, error="Bad Request: message to edit not found",
    )

    await adapter.send_or_update_status("chat-1", "lifecycle", "step 1")
    result = await adapter.send_or_update_status("chat-1", "lifecycle", "step 2")

    assert result.success is True
    assert result.message_id == "200"
    assert adapter.send.await_count == 2
    assert adapter.edit_message.await_count == 1
    # Cache now points at the fresh message id.
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "200"


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
async def test_distinct_chat_ids_do_not_collide(adapter):
    """Same status_key in different chats must not edit each other's messages."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]

    await adapter.send_or_update_status("chat-1", "lifecycle", "first")
    await adapter.send_or_update_status("chat-2", "lifecycle", "second")

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[("chat-1", "lifecycle")] == "100"
    assert adapter._status_message_ids[("chat-2", "lifecycle")] == "200"
