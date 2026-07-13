from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.task_status import CallbackResolution


class FakeButton:
    def __init__(self, text, callback_data):
        self.text = text
        self.callback_data = callback_data


class FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _install_fake_telegram(monkeypatch):
    fake_telegram = types.ModuleType("telegram")
    fake_telegram.Update = SimpleNamespace(ALL_TYPES=())
    fake_telegram.Bot = object
    fake_telegram.Message = object
    fake_telegram.InlineKeyboardButton = FakeButton
    fake_telegram.InlineKeyboardMarkup = FakeMarkup

    fake_error = types.ModuleType("telegram.error")
    fake_error.NetworkError = type("NetworkError", (Exception,), {})
    fake_error.BadRequest = type("BadRequest", (Exception,), {})
    fake_error.TimedOut = type("TimedOut", (Exception,), {})
    fake_telegram.error = fake_error

    fake_constants = types.ModuleType("telegram.constants")
    fake_constants.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    fake_constants.ChatType = SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private"
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
    from plugins.platforms.telegram.adapter import TelegramAdapter

    instance = TelegramAdapter(PlatformConfig(enabled=True, token="x"))
    instance._bot = MagicMock()
    instance._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=123))
    instance._bot.edit_message_text = AsyncMock()
    return instance


@pytest.mark.asyncio
async def test_task_status_create_and_edit_preserve_exact_topic_and_keyboard(adapter):
    buttons = [
        {"label": "Approve", "callback_data": "ts:approve:t_12345678:2"},
        {"label": "Deny", "callback_data": "ts:deny:t_12345678:2"},
    ]

    sent = await adapter.send_task_status(
        "-1001",
        "Task card",
        buttons=buttons,
        metadata={"thread_id": "77", "notify": False},
    )
    edited = await adapter.edit_task_status(
        "-1001",
        "123",
        "Task card updated",
        buttons=buttons,
        metadata={"thread_id": "77", "notify": False},
    )

    assert sent.success is True and sent.message_id == "123"
    assert edited.success is True and edited.message_id == "123"
    send_kwargs = adapter._bot.send_message.call_args.kwargs
    assert send_kwargs["chat_id"] == -1001
    assert send_kwargs["message_thread_id"] == 77
    assert send_kwargs["disable_notification"] is True
    assert send_kwargs["reply_markup"].inline_keyboard[0][0].callback_data == buttons[0]["callback_data"]
    edit_kwargs = adapter._bot.edit_message_text.call_args.kwargs
    assert edit_kwargs["chat_id"] == -1001
    assert edit_kwargs["message_id"] == 123
    assert edit_kwargs["reply_markup"].inline_keyboard[0][1].callback_data == buttons[1]["callback_data"]


@pytest.mark.asyncio
async def test_task_status_push_is_not_silent_and_never_drops_exact_thread(adapter):
    result = await adapter.send_task_status_push(
        "-1001",
        "Decision required",
        buttons=[],
        metadata={"thread_id": "900", "notify": True},
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args.kwargs
    assert kwargs["message_thread_id"] == 900
    assert "disable_notification" not in kwargs


def test_telegram_important_notifications_use_display_platform_path(adapter):
    from plugins.platforms.telegram.adapter import _resolve_notifications_mode

    config = {
        "display": {
            "platforms": {"telegram": {"notifications": "important"}}
        }
    }
    with patch.dict(
        os.environ, {"HERMES_TELEGRAM_NOTIFICATIONS": ""}, clear=False
    ), patch("gateway.config.load_gateway_config", return_value=config):
        assert _resolve_notifications_mode() == "important"


@pytest.mark.asyncio
async def test_task_status_transport_rejects_missing_thread_metadata(adapter):
    result = await adapter.send_task_status(
        "-1001", "Task card", buttons=[], metadata={"notify": False}
    )

    assert result.success is False
    assert "thread_id" in result.error
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_negative_resolution_never_invokes_action_handler(adapter):
    handler = AsyncMock()
    adapter.set_task_status_action_handler(
        handler, allowed_actions=("approve", "deny"), board="project"
    )
    query = MagicMock()
    query.data = "approve"
    query.from_user = SimpleNamespace(id=7, first_name="Aja")
    query.message = SimpleNamespace(
        chat_id=-1001,
        message_id=123,
        message_thread_id=77,
        chat=SimpleNamespace(type="supergroup"),
    )
    query.answer = AsyncMock()

    await adapter._handle_callback_query(
        SimpleNamespace(callback_query=query), MagicMock()
    )

    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_correlated_callback_invokes_handler_with_exact_task_and_version(adapter):
    handler = AsyncMock()
    adapter.set_task_status_action_handler(
        handler, allowed_actions=("approve", "deny"), board="project"
    )
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    query = MagicMock()
    query.data = "ts:approve:t_12345678:4"
    query.from_user = SimpleNamespace(id=7, first_name="Aja")
    query.message = SimpleNamespace(
        chat_id=-1001,
        message_id=123,
        message_thread_id=77,
        chat=SimpleNamespace(type="supergroup"),
    )
    query.answer = AsyncMock()

    with patch(
        "gateway.task_status.resolve_task_status_callback_across_boards",
        return_value=CallbackResolution(
            ok=True,
            kanban_task_id="t_12345678",
            state_version=4,
            action="approve",
        ),
    ):
        await adapter._handle_callback_query(
            SimpleNamespace(callback_query=query), MagicMock()
        )

    handler.assert_awaited_once_with(
        kanban_task_id="t_12345678", state_version=4, action="approve"
    )
    query.answer.assert_awaited_once()
