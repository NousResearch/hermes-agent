"""Tests for Telegram status send-or-edit storage."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len


class FakeNetworkError(Exception):
    pass


class FakeBadRequest(FakeNetworkError):
    pass


class FakeTimedOut(FakeNetworkError):
    pass


def _install_fake_telegram(monkeypatch):
    fake_telegram = types.ModuleType("telegram")
    fake_telegram.Update = SimpleNamespace(ALL_TYPES=())
    fake_telegram.Bot = object
    fake_telegram.Message = object
    fake_telegram.InlineKeyboardButton = object
    fake_telegram.InlineKeyboardMarkup = object

    fake_error = types.ModuleType("telegram.error")
    fake_error.NetworkError = FakeNetworkError
    fake_error.BadRequest = FakeBadRequest
    fake_error.TimedOut = FakeTimedOut
    fake_telegram.error = fake_error

    fake_constants = types.ModuleType("telegram.constants")
    fake_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    fake_constants.ChatType = SimpleNamespace(
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
        PRIVATE="private",
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
    from gateway.platforms.telegram import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_chat_action = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_first_send_stores_mapping_and_returns_message_id(adapter):
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))

    result = await adapter.send_or_update_status("123", "77", "cron", "Working")

    assert result.success is True
    assert result.message_id == "101"
    assert adapter._status_message_ids[("123", "77", "cron")] == "101"
    adapter._bot.send_message.assert_awaited_once()
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["message_thread_id"] == 77


@pytest.mark.asyncio
async def test_second_call_edits_prior_message_without_new_send(adapter):
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))
    adapter._bot.edit_message_text = AsyncMock()

    await adapter.send_or_update_status("123", "77", "cron", "Starting")
    adapter._bot.send_message.reset_mock()
    result = await adapter.send_or_update_status("123", "77", "cron", "Still working")

    assert result.success is True
    assert result.message_id == "101"
    adapter._bot.send_message.assert_not_awaited()
    adapter._bot.edit_message_text.assert_awaited_once()
    kwargs = adapter._bot.edit_message_text.await_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["message_id"] == 101
    assert kwargs["text"] == "Still working"


@pytest.mark.asyncio
async def test_concurrent_same_key_status_updates_do_not_duplicate_send(adapter):
    send_started = asyncio.Event()
    allow_send = asyncio.Event()

    async def fake_send(**kwargs):
        send_started.set()
        await allow_send.wait()
        return SimpleNamespace(message_id=101)

    adapter._bot.send_message = AsyncMock(side_effect=fake_send)
    adapter._bot.edit_message_text = AsyncMock()

    first = asyncio.create_task(
        adapter.send_or_update_status("123", "77", "cron", "Starting")
    )
    await asyncio.wait_for(send_started.wait(), timeout=1)

    second = asyncio.create_task(
        adapter.send_or_update_status("123", "77", "cron", "Still working")
    )
    await asyncio.sleep(0)

    assert adapter._bot.send_message.await_count == 1

    allow_send.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.success is True
    assert second_result.success is True
    assert adapter._bot.send_message.await_count == 1
    adapter._bot.edit_message_text.assert_awaited_once()
    kwargs = adapter._bot.edit_message_text.await_args.kwargs
    assert kwargs["message_id"] == 101
    assert kwargs["text"] == "Still working"
    assert adapter._status_message_ids[("123", "77", "cron")] == "101"


@pytest.mark.asyncio
async def test_permanent_edit_failure_sends_new_and_refreshes_mapping(adapter):
    adapter._bot.send_message = AsyncMock(
        side_effect=[
            SimpleNamespace(message_id=101),
            SimpleNamespace(message_id=202),
        ]
    )
    adapter._bot.edit_message_text = AsyncMock(
        side_effect=FakeBadRequest("Bad Request: message to edit not found")
    )

    await adapter.send_or_update_status("123", "77", "cron", "Starting")
    result = await adapter.send_or_update_status("123", "77", "cron", "Replacement")

    assert result.success is True
    assert result.message_id == "202"
    assert adapter._status_message_ids[("123", "77", "cron")] == "202"
    assert adapter._bot.send_message.await_count == 2


@pytest.mark.asyncio
async def test_transient_edit_failure_does_not_duplicate_or_refresh(adapter):
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))
    adapter._bot.edit_message_text = AsyncMock(
        side_effect=FakeTimedOut("Timed out waiting for Telegram response")
    )

    await adapter.send_or_update_status("123", "77", "cron", "Starting")
    adapter._bot.send_message.reset_mock()
    result = await adapter.send_or_update_status("123", "77", "cron", "Still working")

    assert result.success is False
    assert result.retryable is True
    assert "Timed out" in result.error
    assert adapter._status_message_ids[("123", "77", "cron")] == "101"
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_and_status_key_scoping_create_separate_messages(adapter):
    next_id = 100

    async def fake_send(**kwargs):
        nonlocal next_id
        next_id += 1
        return SimpleNamespace(message_id=next_id)

    adapter._bot.send_message = AsyncMock(side_effect=fake_send)

    await adapter.send_or_update_status("123", "1", "cron", "A")
    await adapter.send_or_update_status("123", "2", "cron", "B")
    await adapter.send_or_update_status("123", "1", "other", "C")

    assert adapter._bot.send_message.await_count == 3
    assert adapter._status_message_ids[("123", "1", "cron")] == "101"
    assert adapter._status_message_ids[("123", "2", "cron")] == "102"
    assert adapter._status_message_ids[("123", "1", "other")] == "103"


@pytest.mark.asyncio
async def test_status_message_id_cache_is_bounded_and_fallback_refreshes_recency(adapter):
    adapter.STATUS_MESSAGE_ID_CACHE_LIMIT = 2
    adapter._bot.send_message = AsyncMock(
        side_effect=[
            SimpleNamespace(message_id=101),
            SimpleNamespace(message_id=102),
            SimpleNamespace(message_id=111),
            SimpleNamespace(message_id=103),
        ]
    )
    adapter._bot.edit_message_text = AsyncMock(
        side_effect=FakeBadRequest("Bad Request: message to edit not found")
    )

    await adapter.send_or_update_status("123", "1", "first", "A")
    await adapter.send_or_update_status("123", "2", "second", "B")
    await adapter.send_or_update_status("123", "1", "first", "A replacement")
    await adapter.send_or_update_status("123", "3", "third", "C")

    assert ("123", "2", "second") not in adapter._status_message_ids
    assert adapter._status_message_ids[("123", "1", "first")] == "111"
    assert adapter._status_message_ids[("123", "3", "third")] == "103"
    assert len(adapter._status_message_ids) == 2


@pytest.mark.asyncio
async def test_long_status_text_is_bounded_to_single_safe_message(adapter):
    sent_texts = []

    async def fake_send(**kwargs):
        sent_texts.append(kwargs["text"])
        return SimpleNamespace(message_id=101)

    adapter._bot.send_message = AsyncMock(side_effect=fake_send)
    long_text = "x" * 6000

    result = await adapter.send_or_update_status("123", None, "cron", long_text)

    assert result.success is True
    assert adapter._bot.send_message.await_count == 1
    assert len(sent_texts) == 1
    assert utf16_len(sent_texts[0]) <= adapter.STATUS_MESSAGE_LENGTH
    assert sent_texts[0].endswith(r"\.\.\.")


@pytest.mark.asyncio
async def test_normal_send_behavior_remains_unaffected(adapter):
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=303))

    result = await adapter.send("123", "Normal send", metadata={"thread_id": "77"})

    assert result.success is True
    assert result.message_id == "303"
    assert adapter._status_message_ids == {}
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["message_thread_id"] == 77
