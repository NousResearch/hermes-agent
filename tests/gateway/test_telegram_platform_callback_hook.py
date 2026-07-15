"""Behavior contract for plugin-owned Telegram inline actions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._message_handler = None
    return adapter


def _make_update(data: str = "zbr:relevant"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = 12345
    query.message.chat = SimpleNamespace(type="channel")
    query.message.message_id = 77
    query.message.message_thread_id = None
    query.message.text = "[zoon-brain:output:daily-radar:2026-07-16:01]"
    query.from_user = SimpleNamespace(id=999, first_name="Peter")
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    return SimpleNamespace(callback_query=query), query


@pytest.mark.asyncio
async def test_authorized_unknown_callback_is_handled_by_plugin_hook(monkeypatch):
    """A plugin can acknowledge an action and replace its inline keyboard."""
    from plugins.platforms.telegram import adapter as telegram_adapter

    class _Button(SimpleNamespace):
        def __init__(self, *, text, callback_data):
            super().__init__(text=text, callback_data=callback_data)

    class _Markup(SimpleNamespace):
        def __init__(self, rows):
            super().__init__(inline_keyboard=rows)

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "999")
    adapter = _make_adapter()
    update, query = _make_update()
    hook_result = {
        "handled": True,
        "answer": "Gespeichert",
        "buttons": [[
            {"text": "✓ Relevant · 1", "callback_data": "zbr:relevant"},
            {"text": "Zu technisch · 0", "callback_data": "zbr:technical"},
        ]],
    }

    with patch("hermes_cli.plugins.invoke_hook", return_value=[hook_result]) as invoke:
        await adapter._handle_callback_query(update, MagicMock())

    invoke.assert_called_once_with(
        "platform_callback",
        platform="telegram",
        callback_data="zbr:relevant",
        user_id="999",
        user_name="Peter",
        chat_id="12345",
        chat_type="channel",
        thread_id=None,
        message_id="77",
        message_text="[zoon-brain:output:daily-radar:2026-07-16:01]",
    )
    query.answer.assert_awaited_once_with(text="Gespeichert", show_alert=False)
    markup = query.edit_message_reply_markup.await_args.kwargs["reply_markup"]
    assert [[button.text for button in row] for row in markup.inline_keyboard] == [[
        "✓ Relevant · 1",
        "Zu technisch · 0",
    ]]


@pytest.mark.asyncio
async def test_send_attaches_normalized_keyboard_to_final_chunk(monkeypatch):
    """Callers can add actions without taking over Telegram delivery."""
    from plugins.platforms.telegram import adapter as telegram_adapter
    from plugins.platforms.telegram.adapter import TelegramAdapter

    class _Button(SimpleNamespace):
        def __init__(self, *, text, callback_data):
            super().__init__(text=text, callback_data=callback_data)

    class _Markup(SimpleNamespace):
        def __init__(self, rows):
            super().__init__(inline_keyboard=rows)

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", _Markup)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(
        side_effect=[SimpleNamespace(message_id=1), SimpleNamespace(message_id=2)]
    )
    adapter.truncate_message = lambda *_args, **_kwargs: ["part one", "part two"]
    buttons = [[
        {"text": "Relevant", "callback_data": "zbr:relevant"},
        {"text": "Zu technisch", "callback_data": "zbr:technical"},
    ]]

    result = await adapter.send(
        "12345",
        "Daily Radar",
        metadata={"inline_keyboard": buttons, "notify": True},
    )

    assert result.success is True
    calls = adapter._bot.send_message.await_args_list
    assert "reply_markup" not in calls[0].kwargs
    markup = calls[1].kwargs["reply_markup"]
    assert [[button.callback_data for button in row] for row in markup.inline_keyboard] == [[
        "zbr:relevant",
        "zbr:technical",
    ]]
