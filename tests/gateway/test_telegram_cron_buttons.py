"""Tests for Telegram cron-delivery accept/dismiss buttons (button mode)."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
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


_ensure_telegram_mock()

from gateway.platforms import telegram as tg
from gateway.platforms.telegram import TelegramAdapter
from gateway.platforms.base import BasePlatformAdapter
from gateway.config import PlatformConfig


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramCronNotice:
    @pytest.mark.asyncio
    async def test_sends_accept_dismiss_buttons(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 77
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)
        tg.InlineKeyboardButton.reset_mock()

        result = await adapter.send_cron_notice(chat_id="12345", notice_id="ab12cd34")

        assert result.success is True
        assert result.message_id == "77"
        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert kwargs["reply_markup"] is not None

        callback_data = [
            c.kwargs.get("callback_data") for c in tg.InlineKeyboardButton.call_args_list
        ]
        assert "cron:accept:ab12cd34" in callback_data
        assert "cron:dismiss:ab12cd34" in callback_data

    @pytest.mark.asyncio
    async def test_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._bot = None
        result = await adapter.send_cron_notice(chat_id="12345", notice_id="x")
        assert result.success is False


class TestCronButtonCapability:
    def test_base_adapter_does_not_support_cron_buttons(self):
        assert BasePlatformAdapter.SUPPORTS_CRON_BUTTONS is False

    def test_telegram_supports_cron_buttons(self):
        assert TelegramAdapter.SUPPORTS_CRON_BUTTONS is True
