from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.account_usage_presence import (
    AccountUsagePresenceCapabilities,
    AccountUsagePresencePayload,
    AccountUsagePresenceRestoreResult,
)
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, discord


def _payload(percent=75, *, cached=False):
    return AccountUsagePresencePayload(
        label="Five hour",
        remaining_percent=percent,
        cached=cached,
    )


def _adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.change_presence = AsyncMock()
    return adapter


def _install_activity_double(monkeypatch):
    factory = MagicMock(
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(discord, "Activity", factory)
    monkeypatch.setattr(
        discord,
        "ActivityType",
        SimpleNamespace(watching="watching"),
    )
    return factory


def test_discord_advertises_activity_capacity():
    adapter = _adapter()

    assert adapter.account_usage_presence_capabilities == AccountUsagePresenceCapabilities(
        activity=True
    )
    assert adapter.account_usage_presence_state_key() == "discord"


@pytest.mark.asyncio
async def test_discord_sets_watching_activity(monkeypatch):
    factory = _install_activity_double(monkeypatch)
    adapter = _adapter()

    changed = await adapter.apply_account_usage_presence(_payload(75), None)

    assert changed is True
    factory.assert_called_once_with(
        type="watching",
        name="Five hour 75% remaining",
    )
    activity = adapter._client.change_presence.await_args.kwargs["activity"]
    assert activity.type == "watching"
    assert activity.name == "Five hour 75% remaining"


@pytest.mark.asyncio
async def test_discord_unknown_usage_is_truthful(monkeypatch):
    _install_activity_double(monkeypatch)
    adapter = _adapter()

    await adapter.apply_account_usage_presence(
        AccountUsagePresencePayload.unknown(), None
    )

    activity = adapter._client.change_presence.await_args.kwargs["activity"]
    assert activity.name == "Usage unavailable"


@pytest.mark.asyncio
async def test_discord_cached_usage_is_visible(monkeypatch):
    _install_activity_double(monkeypatch)
    adapter = _adapter()

    await adapter.apply_account_usage_presence(_payload(75, cached=True), None)

    activity = adapter._client.change_presence.await_args.kwargs["activity"]
    assert activity.name == "Five hour 75% remaining (cached)"


@pytest.mark.asyncio
async def test_discord_does_not_clear_activity_on_restore():
    adapter = _adapter()

    restored = await adapter.restore_account_usage_presence({}, {})

    assert restored is AccountUsagePresenceRestoreResult.RETRY
    adapter._client.change_presence.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_capacity_is_noop_before_client_initializes():
    adapter = _adapter()
    adapter._client = None

    assert await adapter.apply_account_usage_presence(_payload(), None) is False
