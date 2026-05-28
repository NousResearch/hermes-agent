"""Regression tests for Telegram post-send typing refresh behavior.

Telegram clears the typing bubble when a message is delivered, so Hermes
re-triggers ``send_chat_action('typing')`` after intermediate progress/status
messages. That refresh must NOT run for the terminal reply, or the client keeps
showing the bot as typing for a few extra seconds after the answer already
landed.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _install_fake_telegram(monkeypatch):
    fake_telegram = types.ModuleType("telegram")
    setattr(fake_telegram, "Update", SimpleNamespace(ALL_TYPES=()))
    setattr(fake_telegram, "Bot", object)
    setattr(fake_telegram, "Message", object)
    setattr(fake_telegram, "InlineKeyboardButton", object)
    setattr(fake_telegram, "InlineKeyboardMarkup", object)

    fake_error = types.ModuleType("telegram.error")
    setattr(fake_error, "NetworkError", type("NetworkError", (Exception,), {}))
    setattr(fake_error, "BadRequest", type("BadRequest", (Exception,), {}))
    setattr(fake_error, "TimedOut", type("TimedOut", (Exception,), {}))
    setattr(fake_telegram, "error", fake_error)

    fake_constants = types.ModuleType("telegram.constants")
    setattr(fake_constants, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2"))
    setattr(fake_constants, "ChatType", SimpleNamespace(
        GROUP="group", SUPERGROUP="supergroup", CHANNEL="channel", PRIVATE="private"
    ))
    setattr(fake_telegram, "constants", fake_constants)

    fake_ext = types.ModuleType("telegram.ext")
    setattr(fake_ext, "Application", object)
    setattr(fake_ext, "CommandHandler", object)
    setattr(fake_ext, "CallbackQueryHandler", object)
    setattr(fake_ext, "MessageHandler", object)
    setattr(fake_ext, "ContextTypes", SimpleNamespace(DEFAULT_TYPE=object))
    setattr(fake_ext, "filters", object)

    fake_request = types.ModuleType("telegram.request")
    setattr(fake_request, "HTTPXRequest", object)

    monkeypatch.setitem(sys.modules, "telegram", fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", fake_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", fake_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", fake_request)


@pytest.fixture
def adapter(monkeypatch):
    _install_fake_telegram(monkeypatch)
    sys.modules.pop("plugins.platforms.telegram.adapter", None)
    from plugins.platforms.telegram.adapter import TelegramAdapter

    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    assert a._bot is not None
    a._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    a.send_typing = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_send_retriggers_typing_by_default(adapter):
    result = await adapter.send("12345", "intermediate progress")

    assert result.success is True
    adapter.send_typing.assert_awaited_once_with("12345", metadata=None)


@pytest.mark.asyncio
async def test_send_notify_without_suppression_still_retriggers_typing(adapter):
    result = await adapter.send(
        "12345",
        "notify-only message",
        metadata={"notify": True},
    )

    assert result.success is True
    adapter.send_typing.assert_awaited_once_with("12345", metadata={"notify": True})


@pytest.mark.asyncio
async def test_send_skips_post_send_typing_when_suppressed(adapter):
    result = await adapter.send(
        "12345",
        "final answer",
        metadata={"notify": True, "suppress_post_send_typing": True},
    )

    assert result.success is True
    adapter.send_typing.assert_not_awaited()
