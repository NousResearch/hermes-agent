"""Tests for the Telegram bot status indicator.

Telegram bots have no real online/offline presence dot (that's a user-account
feature). The closest Bot API surface is the bot's *short description* — the
line shown under the bot's name in its profile. When `extra.status_indicator`
is enabled, the adapter sets it to "Online" on connect and restores the
operator's original description on clean disconnect so the profile text
survives reconnect cycles (#78784).
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
    adapter._bot.get_my_short_description = AsyncMock(
        return_value=MagicMock(short_description="Original profile text")
    )
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


class TestStatusDescriptionStoreRestore:
    """The short description must survive connect/disconnect cycles (#78784).

    Before the fix, every connect overwrote the operator's profile text with
    "Online" and every disconnect overwrote it with "Offline" — no capture,
    no restore, so the original was permanently lost.
    """

    @pytest.mark.asyncio
    async def test_online_captures_original_before_overwriting(self):
        adapter = _make_adapter(extra={"status_indicator": True})
        adapter._bot.get_my_short_description = AsyncMock(
            return_value=MagicMock(short_description="Set via BotFather")
        )
        await adapter._set_status_indicator(online=True)
        # Must fetch the existing description before clobbering it.
        adapter._bot.get_my_short_description.assert_awaited_once()
        # And still set "Online".
        adapter._bot.set_my_short_description.assert_awaited_once_with(
            short_description="Online"
        )

    @pytest.mark.asyncio
    async def test_offline_restores_original_not_offline_text(self):
        adapter = _make_adapter(extra={"status_indicator": True})
        adapter._bot.get_my_short_description = AsyncMock(
            return_value=MagicMock(short_description="My real description")
        )
        await adapter._set_status_indicator(online=True)   # capture + Online
        await adapter._set_status_indicator(online=False)  # restore original
        # Last call must restore the original, NOT "Offline".
        adapter._bot.set_my_short_description.assert_awaited_with(
            short_description="My real description"
        )

    @pytest.mark.asyncio
    async def test_description_survives_reconnect_cycle(self):
        adapter = _make_adapter(extra={"status_indicator": True})
        adapter._bot.get_my_short_description = AsyncMock(
            return_value=MagicMock(short_description="Persistent text")
        )
        await adapter._set_status_indicator(online=True)   # capture
        await adapter._set_status_indicator(online=False)  # restore
        await adapter._set_status_indicator(online=True)   # reconnect
        # After a full cycle the saved original must be unchanged.
        assert adapter._status_saved_description == "Persistent text"

    @pytest.mark.asyncio
    async def test_reconnect_does_not_recapture_status_text(self):
        """Second online must not re-capture "Online" as the original."""
        adapter = _make_adapter(extra={"status_indicator": True})
        adapter._bot.get_my_short_description = AsyncMock(
            return_value=MagicMock(short_description="Real desc")
        )
        await adapter._set_status_indicator(online=True)
        first_call_count = adapter._bot.get_my_short_description.await_count
        await adapter._set_status_indicator(online=True)  # second online
        # get_my_short_description must NOT be called again.
        assert adapter._bot.get_my_short_description.await_count == first_call_count

    @pytest.mark.asyncio
    async def test_capture_failure_falls_back_gracefully(self):
        """If get_my_short_description raises, Online is still set and Offline
        is used on disconnect (no crash, no garbage restore)."""
        adapter = _make_adapter(extra={"status_indicator": True})
        adapter._bot.get_my_short_description = AsyncMock(
            side_effect=RuntimeError("network error")
        )
        await adapter._set_status_indicator(online=True)
        adapter._bot.set_my_short_description.assert_awaited_once_with(
            short_description="Online"
        )
        await adapter._set_status_indicator(online=False)
        adapter._bot.set_my_short_description.assert_awaited_with(
            short_description="Offline"
        )


