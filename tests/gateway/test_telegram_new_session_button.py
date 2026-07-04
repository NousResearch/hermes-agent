from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_send_new_session_button_renders_inline_new_button(monkeypatch):
    from plugins.platforms.telegram import adapter as adapter_mod

    class Button:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data

    class Markup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(adapter_mod, "InlineKeyboardButton", Button)
    monkeypatch.setattr(adapter_mod, "InlineKeyboardMarkup", Markup)

    adapter = _adapter()
    sent = SimpleNamespace(message_id=55)
    adapter._send_message_with_thread_fallback = AsyncMock(return_value=sent)

    result = await adapter.send_new_session_button(
        "-1001",
        metadata={"thread_id": "2"},
    )

    assert result.success is True
    kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
    assert "Новая тема" in kwargs["text"]
    assert kwargs["reply_markup"].inline_keyboard[0][0].text == "New"
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "ns:new"
    assert kwargs["message_thread_id"] == 2


@pytest.mark.asyncio
async def test_new_session_button_callback_runs_reset_without_typing(monkeypatch):
    adapter = _adapter()
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True

    reset_events: list[MessageEvent] = []

    class Runner:
        async def _handle_reset_command(self, event: MessageEvent):
            reset_events.append(event)
            return "✨ New session started!"

    runner = Runner()
    adapter.set_message_handler(runner._handle_reset_command)
    adapter._send_message_with_thread_fallback = AsyncMock(return_value=SimpleNamespace(message_id=77))

    query = SimpleNamespace(
        data="ns:new",
        from_user=SimpleNamespace(id=42, first_name="Sasha", username="sasha"),
        message=SimpleNamespace(
            chat_id=-1001,
            message_id=55,
            message_thread_id=2,
            text="Готово. Новая тема?",
            chat=SimpleNamespace(id=-1001, type="supergroup", is_forum=True),
        ),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())

    assert reset_events
    event = reset_events[0]
    assert event.text == "/new"
    assert event.message_type == MessageType.COMMAND
    assert event.source == SessionSource(
        platform=adapter.platform,
        chat_id="-1001",
        chat_type="forum",
        user_id="42",
        user_name="sasha",
        thread_id="2",
    )
    query.answer.assert_awaited()
    query.edit_message_text.assert_awaited()
