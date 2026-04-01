from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.telegram import TelegramAdapter
from gateway.session import SessionSource


def _make_event(text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            user_id="orig-user",
            user_name="orig-name",
        ),
        message_id="55",
    )


@pytest.mark.asyncio
async def test_callback_query_maps_session_approval_to_command():
    adapter = object.__new__(TelegramAdapter)
    event = _make_event()
    adapter._build_message_event = MagicMock(return_value=event)
    adapter.handle_message = AsyncMock()

    query = SimpleNamespace(
        data="approval:session",
        from_user=SimpleNamespace(id=999, username="tyler", full_name="Tyler Martin"),
        message=SimpleNamespace(reply_markup=object()),
        answer=AsyncMock(),
        edit_message_reply_markup=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await TelegramAdapter._handle_callback_query(adapter, update, None)

    assert event.text == "/approve session"
    assert event.source.user_id == "999"
    assert event.source.user_name == "tyler"
    adapter.handle_message.assert_awaited_once_with(event)
    query.answer.assert_awaited()
    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)


@pytest.mark.asyncio
async def test_send_approval_prompt_uses_inline_keyboard():
    adapter = object.__new__(TelegramAdapter)
    adapter._bot = SimpleNamespace(
        send_message=AsyncMock(return_value=SimpleNamespace(message_id=42))
    )
    adapter.format_message = MagicMock(side_effect=lambda content: content)
    adapter.platform = Platform.TELEGRAM

    result = await TelegramAdapter.send_approval_prompt(
        adapter,
        chat_id="123",
        content="Dangerous command requires approval.",
        reply_to="55",
        metadata={"thread_id": "77"},
        allow_permanent=False,
    )

    assert result.success is True
    adapter._bot.send_message.assert_awaited_once()
    kwargs = adapter._bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == 123
    assert kwargs["reply_to_message_id"] == 55
    assert kwargs["message_thread_id"] == 77
    markup = kwargs["reply_markup"]
    labels = [button.text for row in markup.inline_keyboard for button in row]
    assert labels == ["Approve once", "Approve session", "Deny"]
