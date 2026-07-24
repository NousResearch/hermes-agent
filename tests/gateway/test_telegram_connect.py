"""Tests for Telegram connect() non-retryable fatal error on missing credentials.

When Telegram has no bot token or no python-telegram-bot installed, connect()
must set a non-retryable fatal error so the gateway does not queue it for
background reconnection (#31049).
"""

import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


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

    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    telegram_mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

import plugins.platforms.telegram.adapter as telegram_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class TestTelegramUnconfiguredNonRetryable:
    """Verify that missing dependency/token sets a non-retryable fatal error."""

    @pytest.mark.asyncio
    async def test_no_telegram_lib_sets_non_retryable_fatal(self, monkeypatch):
        """connect() with python-telegram-bot unavailable → non-retryable fatal error."""
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
        monkeypatch.setattr(telegram_mod, "TELEGRAM_AVAILABLE", False)
        result = await adapter.connect()
        assert result is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "missing_dependency"

    @pytest.mark.asyncio
    async def test_no_bot_token_sets_non_retryable_fatal(self, monkeypatch):
        """connect() with empty token → non-retryable fatal error."""
        monkeypatch.setattr(telegram_mod, "TELEGRAM_AVAILABLE", True)
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token=""))
        result = await adapter.connect()
        assert result is False
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_retryable is False
        assert adapter.fatal_error_code == "missing_credentials"
