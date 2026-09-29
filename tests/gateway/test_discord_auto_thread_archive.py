"""Invariant: a thread Hermes opens inherits the parent channel's auto-archive window.

Regression for the hardcoded ``auto_archive_duration=1440`` in ``_auto_create_thread``.
The adapter contradicted the channel's own default: on a channel set to 10080 (7 days),
a thread opened by the agent dropped out of the sidebar 24 h after its last message while
a thread opened by a human in the SAME channel survived a week. Measured live on
2026-09-15: every thread the gateway had opened carried ``dur=1440`` while the channel
default was 10080.

The contract asserted here is a relationship between two pieces of data (the channel's
default and the duration passed to ``create_thread``), not a snapshot — no assertion on
any specific channel, and the fallback case is pinned because an unknown/absent channel
default must not silently produce an invalid duration.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


# The tests/gateway/conftest.py installs the discord mock at collection time; import after.
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    VALID_THREAD_AUTO_ARCHIVE_MINUTES,
)

FALLBACK = 1440


class _Channel:
    def __init__(self, default_auto_archive_duration=None):
        self.id = 100
        self.name = "organization"
        self.topic = None
        self.default_auto_archive_duration = default_auto_archive_duration
        self.send = AsyncMock()


def _message(channel):
    """A message that can open a thread; ``create_thread`` records its kwargs."""
    saved = {}

    async def create_thread(**kwargs):
        saved.update(kwargs)
        return SimpleNamespace(id=55555, name=kwargs.get("name"), _saved=saved)

    async def channel_send(*a, **k):
        raise AssertionError("fallback path not expected here")

    channel.send = channel_send
    return SimpleNamespace(
        id=42,
        content="please open a thread",
        mentions=[],
        attachments=[],
        created_at=datetime.now(timezone.utc),
        channel=channel,
        author=SimpleNamespace(id=7, display_name="Alice", name="Alice", bot=False),
        create_thread=create_thread,
    )


@pytest.fixture
def adapter():
    a = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    a._client = SimpleNamespace(user=SimpleNamespace(id=999, bot=True))
    return a


class TestAutoThreadArchiveWindow:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("channel_default", sorted(VALID_THREAD_AUTO_ARCHIVE_MINUTES))
    async def test_inherits_channel_default(self, adapter, channel_default):
        """Whatever valid default the channel carries is the duration Hermes uses."""
        channel = _Channel(default_auto_archive_duration=channel_default)
        msg = _message(channel)
        thread = await adapter._auto_create_thread(msg)
        assert thread is not None
        assert thread._saved["auto_archive_duration"] == channel_default

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [None, 0, 999, "seven", 60 * 24 * 30])
    async def test_falls_back_when_channel_default_is_absent_or_invalid(self, adapter, bad):
        """An unusable channel default must still yield a duration Discord accepts."""
        channel = _Channel(default_auto_archive_duration=bad)
        msg = _message(channel)
        thread = await adapter._auto_create_thread(msg)
        assert thread is not None
        assert thread._saved["auto_archive_duration"] == FALLBACK
        assert FALLBACK in VALID_THREAD_AUTO_ARCHIVE_MINUTES

    @pytest.mark.asyncio
    async def test_seed_message_fallback_inherits_too(self, adapter):
        """The seed-message retry path opens its thread with the same window, not a stale 1440."""
        channel = _Channel(default_auto_archive_duration=10080)
        msg = _message(channel)

        async def boom(**kwargs):
            raise RuntimeError("primary thread creation failed")

        msg.create_thread = boom
        seeded = {}

        async def seed_send(*a, **k):
            async def create_thread(**kwargs):
                seeded.update(kwargs)
                return SimpleNamespace(id=55666, name=kwargs.get("name"))
            return SimpleNamespace(create_thread=create_thread)

        channel.send = seed_send
        thread = await adapter._auto_create_thread(msg)
        assert thread is not None
        assert seeded["auto_archive_duration"] == 10080


class TestSharedArchiveHelperIsChannelBased:
    """The window is derived from the CHANNEL, so every thread-opening path shares one rule."""

    def test_valid_defaults_pass_through(self, adapter):
        for default in sorted(VALID_THREAD_AUTO_ARCHIVE_MINUTES):
            assert adapter._auto_thread_archive_minutes(_Channel(default)) == default

    def test_unusable_defaults_fall_back(self, adapter):
        for bad in (None, 0, 999, "seven"):
            assert adapter._auto_thread_archive_minutes(_Channel(bad)) == FALLBACK
        assert adapter._auto_thread_archive_minutes(None) == FALLBACK


class TestHandoffThreadArchiveWindow:
    """The sibling path (session handoff) must not keep its own stale 1440."""

    @pytest.mark.asyncio
    async def test_handoff_thread_inherits_channel_default(self, adapter):
        parent = _Channel(default_auto_archive_duration=10080)
        recorded = {}

        async def create_thread(**kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(id=77777)

        parent.create_thread = create_thread
        adapter._client = SimpleNamespace(get_channel=lambda cid: parent)

        thread_id = await adapter.create_handoff_thread(str(parent.id), "handoff")
        assert thread_id == "77777"
        assert recorded["auto_archive_duration"] == 10080