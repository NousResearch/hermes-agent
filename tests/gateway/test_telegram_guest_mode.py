"""Tests for Telegram Bot API 10.0 guest mode support.

Covers:
- _handle_guest_message_update: payload parsing, state setup, routing
- send() / send_or_update_status() / send_draft(): suppression for guest chats
- on_processing_complete(): answerGuestQuery flush and state cleanup
- Denial path: both _pending_guest_queries and _guest_only_chats cleared on reject
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult
from gateway.session import SessionSource


# ── fake telegram package ─────────────────────────────────────────────────────


def _install_fake_telegram(monkeypatch):
    """Stub python-telegram-bot so TelegramAdapter can be imported without it."""
    fake_telegram = types.ModuleType("telegram")
    fake_update = MagicMock()
    fake_update.ALL_TYPES = ()
    fake_telegram.Update = fake_update
    fake_telegram.Bot = object
    fake_telegram.Message = MagicMock()
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
    fake_ext.TypeHandler = object
    fake_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    fake_ext.filters = object

    fake_request = types.ModuleType("telegram.request")
    fake_request.HTTPXRequest = object

    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", fake_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", fake_request)


# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def adapter(monkeypatch):
    _install_fake_telegram(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GUEST_MODE", "true")
    from gateway.platforms.telegram import TelegramAdapter

    a = object.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a.config = PlatformConfig(enabled=True, token="fake-token")
    a._bot = MagicMock()
    a._bot.do_api_request = AsyncMock()
    a._pending_guest_queries: dict = {}
    a._guest_only_chats: set = set()
    a._guest_reply_buffer: dict = {}
    a._status_message_ids: dict = {}
    return a


def _make_event(chat_id: str = "999") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="group",
            user_id="42",
            user_name="Tester",
        ),
        message_id="1",
    )


# ── _handle_guest_message_update ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_guest_message_stores_query_id(monkeypatch, adapter):
    """guest_query_id must be stored so on_processing_complete can flush."""
    _install_fake_telegram(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GUEST_MODE", "true")

    raw_gm = {
        "message_id": 1,
        "date": 0,
        "chat": {"id": -100123, "type": "supergroup"},
        "from": {"id": 42, "is_bot": False, "first_name": "Tester"},
        "text": "@bot hello",
        "guest_query_id": "gqid-abc",
    }
    fake_update = SimpleNamespace(
        update_id=1,
        api_kwargs={"guest_message": raw_gm},
    )

    fake_msg = SimpleNamespace(
        text="@bot hello",
        caption=None,
        chat=SimpleNamespace(id=-100123),
    )

    with (
        patch.object(type(adapter), "_telegram_guest_mode", return_value=True),
        patch("gateway.platforms.telegram.Message") as MockMsg,
        patch.object(adapter, "_should_process_message", return_value=True),
        patch.object(adapter, "_build_message_event", return_value=_make_event("-100123")),
        patch.object(adapter, "_clean_bot_trigger_text", side_effect=lambda t: t),
        patch.object(adapter, "_apply_telegram_group_observe_attribution", side_effect=lambda e: e),
        patch.object(adapter, "_enqueue_text_event"),
    ):
        MockMsg.de_json.return_value = fake_msg
        await adapter._handle_guest_message_update(fake_update, None)

    assert adapter._pending_guest_queries.get("-100123") == "gqid-abc"
    assert "-100123" in adapter._guest_only_chats


@pytest.mark.asyncio
async def test_handle_guest_message_skips_when_guest_mode_off(monkeypatch, adapter):
    """Nothing should happen when guest_mode is disabled."""
    monkeypatch.setenv("TELEGRAM_GUEST_MODE", "false")

    fake_update = SimpleNamespace(update_id=1, api_kwargs={"guest_message": {"guest_query_id": "x"}})

    with patch.object(type(adapter), "_telegram_guest_mode", return_value=False):
        await adapter._handle_guest_message_update(fake_update, None)

    assert adapter._pending_guest_queries == {}
    assert adapter._guest_only_chats == set()


@pytest.mark.asyncio
async def test_handle_guest_message_denial_clears_both_state_dicts(monkeypatch, adapter):
    """When _should_process_message returns False, both state dicts must be cleared.

    Without discarding from _guest_only_chats, future send() calls to the same
    chat would be silently suppressed even after the bot joins the group.
    """
    raw_gm = {
        "message_id": 2,
        "date": 0,
        "chat": {"id": -100456, "type": "supergroup"},
        "text": "@bot hi",
        "guest_query_id": "gqid-denied",
    }
    fake_update = SimpleNamespace(update_id=2, api_kwargs={"guest_message": raw_gm})
    fake_msg = SimpleNamespace(text="@bot hi", caption=None, chat=SimpleNamespace(id=-100456))

    with (
        patch.object(type(adapter), "_telegram_guest_mode", return_value=True),
        patch("gateway.platforms.telegram.Message") as MockMsg,
        patch.object(adapter, "_should_process_message", return_value=False),
        patch.object(adapter, "_enqueue_text_event") as mock_enqueue,
    ):
        MockMsg.de_json.return_value = fake_msg
        await adapter._handle_guest_message_update(fake_update, None)

    assert "-100456" not in adapter._pending_guest_queries
    assert "-100456" not in adapter._guest_only_chats
    mock_enqueue.assert_not_called()


# ── send() buffering ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_buffers_content_for_guest_chat(adapter):
    """send() must buffer content rather than calling sendMessage for guest chats."""
    adapter._pending_guest_queries["chat-1"] = "gqid-xyz"

    result = await adapter.send("chat-1", "The answer is 42.")

    assert result.success is True
    assert result.message_id is None
    assert adapter._guest_reply_buffer["chat-1"] == "The answer is 42."
    adapter._bot.send_message = MagicMock()  # verify it was never called
    adapter._bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_last_write_wins_for_guest_chat(adapter):
    """Each send() overwrites the buffer so only the final answer is kept."""
    adapter._pending_guest_queries["chat-1"] = "gqid-xyz"

    await adapter.send("chat-1", "Thinking about it...")
    await adapter.send("chat-1", "Here is your answer.")

    assert adapter._guest_reply_buffer["chat-1"] == "Here is your answer."


@pytest.mark.asyncio
async def test_send_buffers_for_guest_only_chat_without_query_id(adapter):
    """_guest_only_chats membership alone should also trigger buffering."""
    adapter._guest_only_chats.add("chat-2")

    result = await adapter.send("chat-2", "Extra chunk.")

    assert result.success is True
    assert adapter._guest_reply_buffer.get("chat-2") == "Extra chunk."


# ── send_or_update_status() suppression ──────────────────────────────────────


@pytest.mark.asyncio
async def test_send_or_update_status_suppressed_for_guest_chat(adapter):
    """Status messages must be fully suppressed for guest chats."""
    adapter._pending_guest_queries["chat-1"] = "gqid-xyz"

    with patch.object(adapter, "send", new_callable=AsyncMock) as mock_send:
        result = await adapter.send_or_update_status("chat-1", "thinking", "Searching...")

    assert result.success is True
    mock_send.assert_not_awaited()
    assert adapter._guest_reply_buffer == {}


@pytest.mark.asyncio
async def test_send_or_update_status_suppressed_for_guest_only_chat(adapter):
    """_guest_only_chats membership also suppresses send_or_update_status."""
    adapter._guest_only_chats.add("chat-3")

    with patch.object(adapter, "send", new_callable=AsyncMock) as mock_send:
        result = await adapter.send_or_update_status("chat-3", "lifecycle", "Done.")

    assert result.success is True
    mock_send.assert_not_awaited()


# ── on_processing_complete() flush ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_flushes_via_answer_guest_query(monkeypatch, adapter):
    """on_processing_complete must call answerGuestQuery with the buffered reply."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)

    adapter._pending_guest_queries["-100999"] = "gqid-flush"
    adapter._guest_only_chats.add("-100999")
    adapter._guest_reply_buffer["-100999"] = "The final answer."

    with patch.object(type(adapter), "_reactions_enabled", return_value=False):
        await adapter.on_processing_complete(_make_event("-100999"), ProcessingOutcome.SUCCESS)

    adapter._bot.do_api_request.assert_awaited_once()
    call_kwargs = adapter._bot.do_api_request.call_args
    assert call_kwargs[0][0] == "answerGuestQuery"
    api_kw = call_kwargs[1]["api_kwargs"]
    assert api_kw["guest_query_id"] == "gqid-flush"
    result = api_kw["result"]
    assert result["type"] == "article"
    assert "The final answer" in result["input_message_content"]["message_text"]


@pytest.mark.asyncio
async def test_on_processing_complete_cleans_up_all_state(monkeypatch, adapter):
    """All three guest state dicts must be cleared after flushing."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)

    adapter._pending_guest_queries["-100999"] = "gqid-flush"
    adapter._guest_only_chats.add("-100999")
    adapter._guest_reply_buffer["-100999"] = "Answer."

    with patch.object(type(adapter), "_reactions_enabled", return_value=False):
        await adapter.on_processing_complete(_make_event("-100999"), ProcessingOutcome.SUCCESS)

    assert "-100999" not in adapter._pending_guest_queries
    assert "-100999" not in adapter._guest_only_chats
    assert "-100999" not in adapter._guest_reply_buffer


@pytest.mark.asyncio
async def test_on_processing_complete_no_op_for_non_guest_chat(monkeypatch, adapter):
    """answerGuestQuery must NOT be called for regular (non-guest) chats."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)

    with patch.object(type(adapter), "_reactions_enabled", return_value=False):
        await adapter.on_processing_complete(_make_event("777"), ProcessingOutcome.SUCCESS)

    adapter._bot.do_api_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_processing_complete_flush_error_does_not_raise(monkeypatch, adapter):
    """A failed answerGuestQuery call must be logged but not propagate."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)

    adapter._pending_guest_queries["-100999"] = "gqid-bad"
    adapter._guest_reply_buffer["-100999"] = "Answer."
    adapter._bot.do_api_request.side_effect = RuntimeError("QUERY_ID_INVALID")

    with patch.object(type(adapter), "_reactions_enabled", return_value=False):
        await adapter.on_processing_complete(_make_event("-100999"), ProcessingOutcome.SUCCESS)

    # State is still cleaned up even when the API call fails.
    assert "-100999" not in adapter._pending_guest_queries
    assert "-100999" not in adapter._guest_only_chats
