"""Tests for the Telegram bot status indicator.

Telegram bots have no real online/offline presence dot (that's a user-account
feature). The closest Bot API surface is the bot's *short description* — the
line shown under the bot's name in its profile. When `extra.status_indicator`
is enabled, the adapter sets it to "Online" on connect and "Offline" on clean
disconnect so users can tell whether the gateway is up.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

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

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter(extra):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._bot = MagicMock()
    adapter._bot.set_my_short_description = AsyncMock()
    return adapter


def test_enabled_via_extra():
    adapter = _make_adapter(extra={"status_indicator": True})
    assert adapter._status_indicator_enabled is True


@pytest.mark.asyncio
async def test_disabled_is_noop():
    adapter = _make_adapter(extra={"status_indicator": False})
    await adapter._set_status_indicator(online=True)
    adapter._bot.set_my_short_description.assert_not_called()


@pytest.mark.asyncio
async def test_online_sets_default_text():
    adapter = _make_adapter(extra={"status_indicator": True})
    await adapter._set_status_indicator(online=True)
    adapter._bot.set_my_short_description.assert_awaited_once_with(
        short_description="Online"
    )


