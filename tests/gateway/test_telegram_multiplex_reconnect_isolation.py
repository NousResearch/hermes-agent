"""Replacement adapters remain isolated to their owning Telegram profile."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = object()
    return adapter


def test_secondary_reconnect_resolves_its_own_live_telegram_adapter():
    retired = _adapter()
    retired._bot = None
    retired.set_owner_profile("research")
    primary = _adapter()
    replacement = _adapter()
    retired.gateway_runner = SimpleNamespace(
        adapters={retired.platform: primary},
        _profile_adapters={"research": {retired.platform: replacement}},
    )

    assert retired._replacement_telegram_adapter() is replacement


def test_primary_reconnect_resolves_the_primary_telegram_adapter():
    retired = _adapter()
    retired._bot = None
    primary = _adapter()
    secondary = _adapter()
    retired.gateway_runner = SimpleNamespace(
        adapters={retired.platform: primary},
        _profile_adapters={"research": {retired.platform: secondary}},
    )

    assert retired._replacement_telegram_adapter() is primary


def test_secondary_reconnect_without_a_live_replacement_fails_closed():
    retired = _adapter()
    retired._bot = None
    retired.set_owner_profile("research")
    primary = _adapter()
    retired.gateway_runner = SimpleNamespace(
        adapters={retired.platform: primary},
        _profile_adapters={"research": {}},
    )

    assert retired._replacement_telegram_adapter() is None
