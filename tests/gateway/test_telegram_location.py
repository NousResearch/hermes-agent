"""Tests for Telegram location/venue handling in gateway/platforms/telegram.py.

Covers: LOCATION message injection (lat/lon), venue vs pin handling, and
ensures the gateway produces a MessageEvent with MessageType.LOCATION.

Note: python-telegram-bot may not be installed in the test environment.
We mock the telegram module at import time to avoid collection errors.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed
# ---------------------------------------------------------------------------

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

    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build mock Telegram objects
# ---------------------------------------------------------------------------

def _make_message(*, location=None, venue=None):
    msg = MagicMock()
    msg.message_id = 42
    msg.text = ""
    msg.date = None
    msg.message_thread_id = None

    msg.location = location
    msg.venue = venue

    msg.chat = MagicMock()
    msg.chat.id = 100
    msg.chat.type = "private"
    msg.chat.title = None
    msg.chat.full_name = "Test Chat"

    msg.from_user = MagicMock()
    msg.from_user.id = 1
    msg.from_user.full_name = "Test User"

    return msg


def _make_update(msg):
    update = MagicMock()
    update.message = msg
    return update


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(config)
    a.handle_message = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTelegramLocationHandling:
    @pytest.mark.asyncio
    async def test_location_pin_injected(self, adapter):
        loc = MagicMock()
        loc.latitude = 12.3456
        loc.longitude = 56.789
        loc.horizontal_accuracy = 25

        msg = _make_message(location=loc)
        update = _make_update(msg)

        await adapter._handle_location_message(update, MagicMock())

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.LOCATION
        assert "[The user shared a location pin via Telegram.]" in event.text
        assert "latitude: 12.3456" in event.text
        assert "longitude: 56.789" in event.text
        assert "horizontal_accuracy_m: 25" in event.text

    @pytest.mark.asyncio
    async def test_venue_injected(self, adapter):
        vloc = MagicMock()
        vloc.latitude = 40.7128
        vloc.longitude = -74.006

        venue = MagicMock()
        venue.location = vloc
        venue.title = "Test Venue"
        venue.address = "123 Example St"
        venue.foursquare_id = "abc123"

        msg = _make_message(venue=venue)
        update = _make_update(msg)

        await adapter._handle_location_message(update, MagicMock())

        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.LOCATION
        assert "[The user shared a venue via Telegram.]" in event.text
        assert "title: Test Venue" in event.text
        assert "address: 123 Example St" in event.text
        assert "latitude: 40.7128" in event.text
        assert "longitude: -74.006" in event.text
        assert "foursquare_id: abc123" in event.text

    @pytest.mark.asyncio
    async def test_ignores_update_without_message(self, adapter):
        update = MagicMock()
        update.message = None
        await adapter._handle_location_message(update, MagicMock())
        adapter.handle_message.assert_not_called()
