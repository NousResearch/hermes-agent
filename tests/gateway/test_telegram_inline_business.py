from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.platforms.telegram as telegram_platform
from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.telegram import TelegramAdapter


@pytest.mark.asyncio
async def test_inline_query_invokes_agent_and_answers_result(monkeypatch):
    monkeypatch.setattr(
        telegram_platform,
        "InlineQueryResultArticle",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        telegram_platform,
        "InputTextMessageContent",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._message_handler = AsyncMock(return_value="Inline answer")
    query = SimpleNamespace(
        id="inline-1",
        query="make reply polite",
        from_user=SimpleNamespace(id=42, full_name="Alice"),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(update_id=100, inline_query=query)

    await adapter._handle_inline_query(update, MagicMock())

    adapter._message_handler.assert_awaited_once()
    event = adapter._message_handler.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.text == "make reply polite"
    assert event.message_type == MessageType.TEXT
    assert event.source.chat_id == "inline:42"
    assert event.source.chat_type == "inline"
    query.answer.assert_awaited_once()
    results = query.answer.await_args.kwargs["results"]
    assert len(results) == 1
    assert results[0].input_message_content.message_text == "Inline answer"


@pytest.mark.asyncio
async def test_business_message_uses_business_connection_for_reply():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._message_handler = AsyncMock(return_value="Business reply")
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=777)))

    chat = SimpleNamespace(id=123, type="private", title=None, full_name="Bob")
    user = SimpleNamespace(id=42, full_name="Alice")
    message = SimpleNamespace(
        text="hello business",
        chat=chat,
        from_user=user,
        message_id=55,
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=None,
        date=None,
    )
    update = SimpleNamespace(
        update_id=101,
        business_message=message,
        business_connection_id="bc-1",
    )

    await adapter._handle_business_message(update, MagicMock())

    adapter._message_handler.assert_awaited_once()
    event = adapter._message_handler.await_args.args[0]
    assert event.text == "hello business"
    assert getattr(event, "business_connection_id") == "bc-1"
    adapter._bot.send_message.assert_awaited_once()
    assert adapter._bot.send_message.await_args.kwargs["business_connection_id"] == "bc-1"
    assert adapter._bot.send_message.await_args.kwargs["chat_id"] == 123
