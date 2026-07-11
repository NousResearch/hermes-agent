"""Regression coverage for Telegram Bot API Guest Mode queries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageType, _thread_metadata_for_source
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._bot = MagicMock()
    adapter._send_path_degraded = False
    return adapter


def test_guest_source_metadata_routes_to_answer_guest_query():
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        thread_id="guest:5323706597129951744",
        message_id="17",
    )

    assert _thread_metadata_for_source(source, "17") == {
        "thread_id": "guest:5323706597129951744",
        "telegram_guest_query_id": "5323706597129951744",
    }


@pytest.mark.asyncio
async def test_guest_message_enters_normal_agent_queue_once():
    adapter = _adapter()
    event = SimpleNamespace(
        source=SimpleNamespace(thread_id=None, chat_topic=None),
        text="@darrenslavebot are you there?",
    )
    adapter._build_message_event = MagicMock(return_value=event)
    adapter._clean_bot_trigger_text = MagicMock(return_value="are you there?")
    adapter._enqueue_text_event = MagicMock()
    message = SimpleNamespace(
        text="@darrenslavebot are you there?",
        guest_query_id="5323706597129951744",
    )
    update = SimpleNamespace(update_id=91, guest_message=message)

    await adapter._handle_guest_message(update, None)

    adapter._build_message_event.assert_called_once_with(
        message, MessageType.TEXT, update_id=91
    )
    assert event.source.thread_id == "guest:5323706597129951744"
    assert event.source.chat_topic == "Guest Bot Mention"
    assert event.text == "are you there?"
    adapter._enqueue_text_event.assert_called_once_with(event)


@pytest.mark.asyncio
async def test_unauthorized_guest_message_is_rejected():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = MagicMock(return_value=False)
    adapter._build_message_event = MagicMock()
    adapter._enqueue_text_event = MagicMock()
    message = SimpleNamespace(text="summon", guest_query_id="guest-query-1")
    update = SimpleNamespace(update_id=92, guest_message=message)

    await adapter._handle_guest_message(update, None)

    adapter._is_user_authorized_from_message.assert_called_once_with(message)
    adapter._build_message_event.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_generic_text_handler_ignores_guest_message():
    adapter = _adapter()
    adapter._effective_update_message = MagicMock()
    adapter._enqueue_text_event = MagicMock()
    update = SimpleNamespace(
        guest_message=SimpleNamespace(text="guest summon"),
        effective_message=SimpleNamespace(text="guest summon"),
    )

    await adapter._handle_text_message(update, None)

    adapter._effective_update_message.assert_not_called()
    adapter._enqueue_text_event.assert_not_called()


@pytest.mark.asyncio
async def test_generic_command_handler_ignores_guest_message():
    adapter = _adapter()
    adapter._effective_update_message = MagicMock()
    adapter._should_process_message = MagicMock()
    update = SimpleNamespace(
        guest_message=SimpleNamespace(text="/status", guest_query_id="guest-query-2"),
        effective_message=SimpleNamespace(text="/status"),
    )

    await adapter._handle_command(update, None)

    adapter._effective_update_message.assert_not_called()
    adapter._should_process_message.assert_not_called()


@pytest.mark.asyncio
async def test_intermediate_guest_send_is_suppressed():
    adapter = _adapter()
    adapter._bot.answer_guest_query = AsyncMock()

    result = await adapter.send(
        "1482299073",
        "Working…",
        metadata={"telegram_guest_query_id": "5323706597129951744"},
    )

    assert result.success is True
    assert result.message_id is None
    adapter._bot.answer_guest_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_guest_send_uses_answer_guest_query_once(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InputTextMessageContent",
        lambda text: SimpleNamespace(message_text=text),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineQueryResultArticle",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    adapter._bot.answer_guest_query = AsyncMock(
        return_value=SimpleNamespace(message_id=777)
    )

    result = await adapter.send(
        "1482299073",
        "Yes — I’m here.",
        metadata={
            "telegram_guest_query_id": "5323706597129951744",
            "notify": True,
        },
    )

    assert result.success is True
    assert result.message_id == "777"
    adapter._bot.answer_guest_query.assert_awaited_once()
    query_id, article = adapter._bot.answer_guest_query.await_args.args
    assert query_id == "5323706597129951744"
    assert article.input_message_content.message_text == "Yes — I’m here."


@pytest.mark.asyncio
async def test_guest_answer_failure_is_non_retryable(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InputTextMessageContent",
        lambda text: SimpleNamespace(message_text=text),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineQueryResultArticle",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    adapter._bot.answer_guest_query = AsyncMock(side_effect=RuntimeError("expired"))

    result = await adapter.send(
        "1482299073",
        "Too late",
        metadata={
            "telegram_guest_query_id": "5323706597129951744",
            "notify": True,
        },
    )

    assert result.success is False
    assert result.retryable is False
    assert result.error == "expired"
