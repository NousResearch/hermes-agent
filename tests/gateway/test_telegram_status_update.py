"""Tests for TelegramAdapter.send_or_update_status (issue #30045).

The status-update path must:
  1. Send a fresh message on the first call for a (chat_id, status_key) pair.
  2. Edit that same message on subsequent calls with the same key.
  3. Fall back to sending fresh when the cached message edit fails.
  4. Keep distinct keys independent (no cross-talk).
  5. Keep distinct Telegram thread/topic ids independent (no cross-talk).
  6. Edit in place within the same thread/topic.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _status_key(chat_id: str, status_key: str, thread_id=None, dm_topic_id=None):
    return (chat_id, status_key, thread_id, dm_topic_id)


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
    assert adapter._status_message_ids[_status_key("chat-1", "lifecycle")] == "100"


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
    assert adapter._status_message_ids[_status_key("chat-1", "lifecycle")] == "200"


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
    assert adapter._status_message_ids[_status_key("chat-1", "lifecycle")] == "100"
    assert adapter._status_message_ids[_status_key("chat-1", "model-switch")] == "200"


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
    assert adapter._status_message_ids[_status_key("chat-1", "lifecycle")] == "100"
    assert adapter._status_message_ids[_status_key("chat-2", "lifecycle")] == "200"


@pytest.mark.asyncio
async def test_distinct_thread_ids_do_not_collide(adapter):
    """Same chat/status_key in different Telegram topics must not reuse a status bubble."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]

    await adapter.send_or_update_status(
        "chat-1",
        "provider-error",
        "rate limited",
        metadata={"thread_id": "11"},
    )
    await adapter.send_or_update_status(
        "chat-1",
        "provider-error",
        "rate limited",
        metadata={"thread_id": "22"},
    )

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[_status_key("chat-1", "provider-error", "11")] == "100"
    assert adapter._status_message_ids[_status_key("chat-1", "provider-error", "22")] == "200"


@pytest.mark.asyncio
async def test_same_thread_id_edits_in_place(adapter):
    """The routing key still edits in place within the same Telegram topic."""
    adapter.send.return_value = SendResult(success=True, message_id="100")
    adapter.edit_message.return_value = SendResult(success=True, message_id="100")

    await adapter.send_or_update_status(
        "chat-1",
        "provider-error",
        "rate limited",
        metadata={"thread_id": "11"},
    )
    await adapter.send_or_update_status(
        "chat-1",
        "provider-error",
        "still rate limited",
        metadata={"thread_id": "11"},
    )

    adapter.send.assert_awaited_once()
    adapter.edit_message.assert_awaited_once()
    args, kwargs = adapter.edit_message.call_args
    assert args[0] == "chat-1"
    assert args[1] == "100"
    assert args[2] == "still rate limited"
    assert kwargs["metadata"] == {"thread_id": "11"}


@pytest.mark.asyncio
async def test_distinct_dm_topic_ids_do_not_collide(adapter):
    """Same chat/status_key in different DM topics must not reuse a bubble."""
    adapter.send.side_effect = [
        SendResult(success=True, message_id="100"),
        SendResult(success=True, message_id="200"),
    ]

    await adapter.send_or_update_status(
        "chat-1",
        "lifecycle",
        "starting",
        metadata={"direct_messages_topic_id": "7"},
    )
    await adapter.send_or_update_status(
        "chat-1",
        "lifecycle",
        "step 2",
        metadata={"direct_messages_topic_id": "9"},
    )

    assert adapter.send.await_count == 2
    adapter.edit_message.assert_not_awaited()
    assert adapter._status_message_ids[_status_key("chat-1", "lifecycle", None, "7")] == "100"
    assert adapter._status_message_ids[_status_key("chat-1", "lifecycle", None, "9")] == "200"


@pytest.mark.asyncio
async def test_concurrent_same_key_is_serialized(adapter):
    """Two concurrent send_or_update_status calls with the same key must be
    serialized — the second must wait for the first to complete (send + cache),
    then edit the cached message instead of sending a duplicate."""
    import asyncio as _asyncio

    send_started = _asyncio.Event()
    release_send = _asyncio.Event()

    async def slow_send(*args, **kwargs):
        send_started.set()
        await release_send.wait()
        return SendResult(success=True, message_id="100")

    adapter.send = AsyncMock(side_effect=slow_send)
    adapter.edit_message.return_value = SendResult(success=True, message_id="100")

    # Start the first call — it will block inside send().
    first = _asyncio.create_task(
        adapter.send_or_update_status("chat-1", "lifecycle", "call A")
    )
    await send_started.wait()  # ensure first call is inside send()

    # Start the second call while the first is still blocked.
    second = _asyncio.create_task(
        adapter.send_or_update_status("chat-1", "lifecycle", "call B")
    )
    await _asyncio.sleep(0.01)  # let it settle

    # While the first send is in-flight, the second must NOT have sent.
    assert not second.done(), "second call should be blocked waiting for the lock"
    assert adapter.send.await_count == 1, "exactly one send should be in-flight"

    # Release the first send — both should complete.
    release_send.set()
    await _asyncio.gather(first, second)

    # One send (first call) + one edit (second call, found cached_id).
    assert adapter.send.await_count == 1
    assert adapter.edit_message.await_count == 1
