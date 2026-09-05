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

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


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
async def test_gateway_statuses_edit_only_within_their_turn(adapter, monkeypatch):
    pending = []
    monkeypatch.setattr(
        "gateway.run.safe_schedule_threadsafe",
        lambda coro, *args, **kwargs: pending.append(coro),
    )
    # Exercise gateway routing and the real adapter send/edit path; only the
    # Telegram network transport is replaced.
    del adapter.send
    del adapter.edit_message
    adapter._bot.send_message = AsyncMock(side_effect=[SimpleNamespace(message_id=i) for i in range(3)])
    adapter._bot.edit_message_text = AsyncMock()
    for session, generation in [("topic-a", 1), ("topic-a", 2), ("topic-b", 1)]:
        ctx = TurnContext(
            source=SimpleNamespace(platform=Platform.TELEGRAM),
            session_key=session, run_generation=generation,
            _status_adapter=adapter, _status_chat_id="123",
            _run_still_current=lambda: True,
        )
        turn = TurnRunner(MagicMock(), ctx)
        turn._status_callback_sync("lifecycle", "Starting")
        turn._status_callback_sync("lifecycle", "Continuing")
        for coro in pending:
            await coro
        pending.clear()

    assert adapter._bot.send_message.await_count == 3
    assert [call.kwargs["message_id"] for call in adapter._bot.edit_message_text.await_args_list] == [0, 1, 2]


@pytest.mark.asyncio
async def test_overlapping_statuses_send_once_and_old_turn_cache_is_bounded(adapter):
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_send(*args, **kwargs):
        started.set()
        await release.wait()
        return SendResult(success=True, message_id="first")

    adapter.send.side_effect = delayed_send
    adapter.edit_message.return_value = SendResult(success=True)
    first = asyncio.create_task(adapter.send_or_update_status("chat-1", "turn-1", "Starting"))
    await asyncio.wait_for(started.wait(), timeout=5)
    second = asyncio.create_task(adapter.send_or_update_status("chat-1", "turn-1", "Continuing"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert adapter.send.await_count == 1
    adapter.edit_message.assert_awaited_once()

    adapter._STATUS_MESSAGE_IDS_MAX = 4
    adapter.send.side_effect = None
    adapter.send.return_value = SendResult(success=True, message_id="next")
    for i in range(10):
        await adapter.send_or_update_status("chat-1", f"later-{i}", "Starting")
    assert len(adapter._status_message_ids) <= adapter._STATUS_MESSAGE_IDS_MAX
    assert adapter._status_message_ids[("chat-1", "later-9")] == "next"
