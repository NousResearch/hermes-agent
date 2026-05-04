"""Tests for Telegram question-session topic spawning."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from hermes_cli.commands import resolve_command


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter(extra=None):
    if extra is None:
        extra = {"question_sessions": {"enabled": True}}
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))


def _make_message(text="/qopen Test title\n\nDo the thing", thread_id=2609):
    return SimpleNamespace(
        text=text,
        message_id=42,
        message_thread_id=thread_id,
        date=None,
        reply_to_message=None,
        chat=SimpleNamespace(
            id=-100123,
            type="supergroup",
            title="Hermes Agent - Mina",
            full_name=None,
            is_forum=True,
        ),
        from_user=SimpleNamespace(id=777, full_name="Joohyun Kim"),
    )


def test_question_session_payload_parses_title_and_body():
    adapter = _make_adapter()

    parsed = adapter._parse_question_session_payload(
        "/qopen 텔레그램 세션형 질문 만들기\n\n고정 토픽을 오염시키고 싶지 않아"
    )

    assert parsed == (
        "텔레그램 세션형 질문 만들기",
        "고정 토픽을 오염시키고 싶지 않아",
    )


def test_question_session_topic_name_is_prefixed_and_truncated():
    adapter = _make_adapter()

    topic_name = adapter._question_session_topic_name("  " + "a" * 200 + "  ")

    assert topic_name.startswith("Q · ")
    assert len(topic_name) <= 128
    assert topic_name == "Q · " + ("a" * 124)


def test_qopen_is_registered_as_gateway_command():
    command = resolve_command("qopen")

    assert command is not None
    assert command.name == "qopen"
    assert command.gateway_only is True


@pytest.mark.asyncio
async def test_handle_question_session_command_is_disabled_by_default():
    adapter = _make_adapter(extra={})
    adapter._bot = AsyncMock()
    adapter.handle_message = AsyncMock()
    update = SimpleNamespace(update_id=123, message=_make_message())

    handled = await adapter._handle_question_session_command(update)

    assert handled is False
    adapter._bot.create_forum_topic.assert_not_called()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_question_session_command_creates_topic_and_dispatches_new_event():
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.create_forum_topic.return_value = SimpleNamespace(message_thread_id=9876)
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=99)
    adapter.handle_message = AsyncMock()

    update = SimpleNamespace(update_id=123, message=_make_message())

    handled = await adapter._handle_question_session_command(update)

    assert handled is True
    adapter._bot.create_forum_topic.assert_awaited_once_with(
        chat_id=-100123,
        name="Q · Test title",
    )
    adapter._bot.send_message.assert_awaited_once()
    sent_kwargs = adapter._bot.send_message.await_args.kwargs
    assert sent_kwargs["chat_id"] == -100123
    assert sent_kwargs["message_thread_id"] == 9876
    assert "Test title" in sent_kwargs["text"]

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "Do the thing"
    assert event.message_type == MessageType.TEXT
    assert event.source.chat_id == "-100123"
    assert event.source.thread_id == "9876"
    assert event.source.chat_topic == "Q · Test title"
    assert event.message_id == "99"
    assert event.platform_update_id == 123


@pytest.mark.asyncio
async def test_handle_command_does_not_bypass_group_processing_gate():
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._should_process_message = MagicMock(return_value=False)
    update = SimpleNamespace(update_id=123, message=_make_message())

    await adapter._handle_command(update, SimpleNamespace())

    adapter._should_process_message.assert_called_once()
    adapter._bot.create_forum_topic.assert_not_called()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_question_session_command_sends_usage_when_title_missing():
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter.handle_message = AsyncMock()
    update = SimpleNamespace(update_id=123, message=_make_message(text="/qopen"))

    handled = await adapter._handle_question_session_command(update)

    assert handled is True
    adapter._bot.create_forum_topic.assert_not_called()
    adapter.handle_message.assert_not_called()
    adapter._bot.send_message.assert_awaited_once()
    assert "Usage" in adapter._bot.send_message.await_args.kwargs["text"]
