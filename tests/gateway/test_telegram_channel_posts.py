"""Regression tests for Telegram channel_post updates.

Telegram channel broadcasts are delivered as ``Update.channel_post`` rather than
``Update.message``.  The adapter should use ``effective_message`` so channel
posts are converted into Hermes gateway events instead of being silently
ignored.
"""

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


def _build_telegram_stubs():
    telegram_mod = types.ModuleType("telegram")
    telegram_mod.Update = object
    telegram_mod.Bot = object
    telegram_mod.Message = object
    telegram_mod.InlineKeyboardButton = object
    telegram_mod.InlineKeyboardMarkup = object
    telegram_mod.LinkPreviewOptions = object

    telegram_ext_mod = types.ModuleType("telegram.ext")
    telegram_ext_mod.Application = object
    telegram_ext_mod.CommandHandler = object
    telegram_ext_mod.CallbackQueryHandler = object
    telegram_ext_mod.MessageHandler = object
    telegram_ext_mod.ContextTypes = SimpleNamespace(DEFAULT_TYPE=type(None))
    telegram_ext_mod.filters = SimpleNamespace()

    telegram_constants_mod = types.ModuleType("telegram.constants")
    telegram_constants_mod.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    telegram_constants_mod.ChatType = SimpleNamespace(
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
        PRIVATE="private",
    )

    telegram_request_mod = types.ModuleType("telegram.request")
    telegram_request_mod.HTTPXRequest = object

    telegram_mod.ext = telegram_ext_mod
    telegram_mod.constants = telegram_constants_mod
    telegram_mod.request = telegram_request_mod

    return {
        "telegram": telegram_mod,
        "telegram.ext": telegram_ext_mod,
        "telegram.constants": telegram_constants_mod,
        "telegram.request": telegram_request_mod,
    }


@pytest.fixture
def telegram_adapter_cls(monkeypatch):
    """Import TelegramAdapter without leaking temporary telegram stubs."""
    module_name = "gateway.platforms.telegram"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        yield existing_module.TelegramAdapter
        return

    telegram_pkg = sys.modules.get("telegram")
    installed = isinstance(getattr(telegram_pkg, "__file__", None), str)
    if telegram_pkg is None:
        try:
            installed = importlib.util.find_spec("telegram") is not None
        except ValueError:
            installed = False

    if not installed:
        for name, module in _build_telegram_stubs().items():
            monkeypatch.setitem(sys.modules, name, module)

    module = importlib.import_module(module_name)
    try:
        yield module.TelegramAdapter
    finally:
        if not installed:
            sys.modules.pop(module_name, None)


def _make_adapter(telegram_adapter_cls):
    return telegram_adapter_cls(PlatformConfig(enabled=True, token="***", extra={}))


def _make_channel_message(text="channel id test @hermes_bot"):
    chat = SimpleNamespace(
        id=-1003950368353,
        type="channel",
        title="wzrd",
        full_name=None,
        is_forum=False,
    )
    return SimpleNamespace(
        chat=chat,
        from_user=None,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=11,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
    )


def _make_channel_update(msg):
    return SimpleNamespace(
        update_id=12345,
        message=None,
        channel_post=msg,
        effective_message=msg,
    )


def test_build_message_event_uses_channel_identity_for_channel_posts(telegram_adapter_cls):
    adapter = _make_adapter(telegram_adapter_cls)
    msg = _make_channel_message()

    event = adapter._build_message_event(msg, MessageType.TEXT, update_id=12345)

    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "-1003950368353"
    # Channel posts often have no from_user.  Preserve an identity so the
    # gateway authorization layer can allowlist the channel by numeric ID.
    assert event.source.user_id == "-1003950368353"
    assert event.source.user_name == "wzrd"
    assert event.platform_update_id == 12345


@pytest.mark.asyncio
async def test_text_handler_uses_effective_message_for_channel_post(telegram_adapter_cls):
    adapter = _make_adapter(telegram_adapter_cls)
    msg = _make_channel_message()
    update = _make_channel_update(msg)
    adapter._enqueue_text_event = MagicMock()

    await adapter._handle_text_message(update, MagicMock())

    adapter._enqueue_text_event.assert_called_once()
    event = adapter._enqueue_text_event.call_args.args[0]
    assert event.text == "channel id test @hermes_bot"
    assert event.message_type == MessageType.TEXT
    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "-1003950368353"


@pytest.mark.asyncio
async def test_command_handler_uses_effective_message_for_channel_post(telegram_adapter_cls):
    adapter = _make_adapter(telegram_adapter_cls)
    msg = _make_channel_message(text="/status")
    update = _make_channel_update(msg)
    adapter.handle_message = AsyncMock()

    await adapter._handle_command(update, MagicMock())

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "/status"
    assert event.message_type == MessageType.COMMAND
    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "-1003950368353"
