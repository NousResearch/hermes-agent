"""Tests for Telegram skill command menu refresh."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.BotCommand = FakeBotCommand
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


class FakeBotCommand:
    def __init__(self, command, description):
        self.command = command
        self.description = description


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = SimpleNamespace(set_my_commands=AsyncMock())
    return adapter


@pytest.mark.asyncio
async def test_refresh_skill_group_registers_current_telegram_menu(monkeypatch):
    import telegram

    monkeypatch.setattr(telegram, "BotCommand", FakeBotCommand, raising=False)

    with patch(
        "hermes_cli.commands.telegram_menu_commands",
        return_value=([("write_trip_report", "Generate a trip report")], 0),
    ):
        result = await _make_adapter().refresh_skill_group()

    assert result == (1, 0)


@pytest.mark.asyncio
async def test_refresh_skill_group_no_bot_is_noop():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    assert await adapter.refresh_skill_group() == (0, 0)
