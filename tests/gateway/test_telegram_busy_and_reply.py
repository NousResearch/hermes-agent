from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


class _FakeButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data
        self.kwargs = kwargs


class _FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _adapter(monkeypatch):
    import plugins.platforms.telegram.adapter as telegram_mod
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(telegram_mod, "InlineKeyboardButton", _FakeButton)
    monkeypatch.setattr(telegram_mod, "InlineKeyboardMarkup", _FakeMarkup)
    adapter = object.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._config = adapter.config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._bot = SimpleNamespace(id=999, username="HermesBot")
    adapter._reply_to_mode = "first"
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._send_message_with_thread_fallback = AsyncMock(return_value=SimpleNamespace(message_id=123))
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    return adapter


@pytest.mark.asyncio
async def test_telegram_busy_prompt_renders_all_action_buttons(monkeypatch):
    adapter = _adapter(monkeypatch)

    result = await adapter.send_busy_prompt(
        chat_id="12345",
        prompt="Hermes is still working. What should I do?",
        prompt_id="42",
        can_steer=True,
        reply_to="777",
    )

    assert result.success is True
    kwargs = adapter._send_message_with_thread_fallback.await_args.kwargs
    assert kwargs["chat_id"] == 12345
    assert kwargs["reply_to_message_id"] == 777
    keyboard = kwargs["reply_markup"].inline_keyboard
    callback_data = [button.callback_data for row in keyboard for button in row]
    assert callback_data == [
        "bp:queue:42",
        "bp:interrupt:42",
        "bp:steer:42",
        "bp:ignore:42",
    ]


@pytest.mark.asyncio
async def test_telegram_busy_callback_calls_runner_and_removes_buttons(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = []

    class Runner:
        def marker(self):
            return None

        async def resolve_busy_prompt_choice(self, prompt_id, choice):
            calls.append((prompt_id, choice))
            return True

    runner = Runner()
    adapter._message_handler = runner.marker
    query = SimpleNamespace(
        data="bp:interrupt:42",
        message=SimpleNamespace(
            chat_id=12345,
            chat=SimpleNamespace(type="private"),
            message_thread_id=None,
        ),
        from_user=SimpleNamespace(id=111, first_name="Michael"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )

    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)

    assert calls == [("42", "interrupt")]
    query.answer.assert_awaited_once_with(text="⚡ Interrupting")
    edit_kwargs = query.edit_message_text.await_args.kwargs
    assert edit_kwargs["reply_markup"] is None
    assert "Interrupting" in edit_kwargs["text"]


def test_telegram_reply_to_own_bot_message_marks_own_context(monkeypatch):
    adapter = _adapter(monkeypatch)
    replied = SimpleNamespace(
        message_id=555,
        text="Previous Hermes answer",
        caption=None,
        from_user=SimpleNamespace(id=999, username="HermesBot", full_name="Hermes"),
    )
    message = SimpleNamespace(
        text="continue from this",
        caption=None,
        chat=SimpleNamespace(id=12345, type="private", is_forum=False, title=None, full_name="Michael"),
        from_user=SimpleNamespace(id=111, full_name="Michael"),
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=replied,
        message_id=556,
        date=None,
    )

    event = adapter._build_message_event(message, msg_type=MessageType.TEXT)

    assert event.reply_to_message_id == "555"
    assert event.reply_to_text == "Previous Hermes answer"
    assert event.reply_to_author_id == "999"
    assert event.reply_to_author_name == "Hermes"
    assert event.reply_to_is_own_message is True


def test_telegram_reply_to_human_message_is_not_own_context(monkeypatch):
    adapter = _adapter(monkeypatch)
    replied = SimpleNamespace(
        message_id=555,
        text="Another human message",
        caption=None,
        from_user=SimpleNamespace(id=222, username="alice", full_name="Alice"),
    )
    message = SimpleNamespace(
        text="replying",
        caption=None,
        chat=SimpleNamespace(id=12345, type="private", is_forum=False, title=None, full_name="Michael"),
        from_user=SimpleNamespace(id=111, full_name="Michael"),
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=replied,
        message_id=556,
        date=None,
    )

    event = adapter._build_message_event(message, msg_type=MessageType.TEXT)

    assert event.reply_to_text == "Another human message"
    assert event.reply_to_author_id == "222"
    assert event.reply_to_is_own_message is False
