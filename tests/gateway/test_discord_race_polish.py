"""Discord adapter race polish: concurrent join_voice_channel must not
double-invoke channel.connect() on the same guild."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t", extra=extra or {})
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._allowed_role_ids = set()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._client = MagicMock()
    return adapter


@pytest.mark.asyncio
async def test_concurrent_joins_do_not_double_connect():
    """Two concurrent join_voice_channel calls on the same guild must
    serialize through the per-guild lock — only ONE channel.connect()
    actually fires; the second sees the _voice_clients entry the first
    just installed."""
    adapter = _make_adapter()

    connect_count = [0]
    release = asyncio.Event()

    class FakeVC:
        def __init__(self, channel):
            self.channel = channel

        def is_connected(self):
            return True

        async def move_to(self, _channel):
            return None

    async def slow_connect(self):
        connect_count[0] += 1
        await release.wait()
        return FakeVC(self)

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42
    channel.connect = lambda: slow_connect(channel)

    from plugins.platforms.discord import adapter as discord_mod
    with patch.object(discord_mod, "VoiceReceiver",
                      MagicMock(return_value=MagicMock(start=lambda: None))):
        with patch.object(discord_mod.asyncio, "ensure_future",
                          lambda _c: asyncio.create_task(asyncio.sleep(0))):
            t1 = asyncio.create_task(adapter.join_voice_channel(channel))
            t2 = asyncio.create_task(adapter.join_voice_channel(channel))
            await asyncio.sleep(0.05)
            release.set()
            r1, r2 = await asyncio.gather(t1, t2)

    assert connect_count[0] == 1, (
        f"expected 1 channel.connect() call, got {connect_count[0]} — "
        "per-guild lock is not serializing join_voice_channel"
    )
    assert r1 is True and r2 is True
    assert 42 in adapter._voice_clients


@pytest.mark.asyncio
async def test_auto_voice_join_configured_user_joining_configured_channel_triggers_callback():
    adapter = _make_adapter({
        "voice_auto_join": {
            "enabled": True,
            "guild_id": "42",
            "voice_channel_id": "111",
            "text_channel_id": "222",
            "user_ids": '["999"]',
        }
    })
    adapter._voice_auto_join_callback = AsyncMock(return_value=True)

    channel = SimpleNamespace(id=111, name="hermes-voice")
    guild = SimpleNamespace(id=42, get_channel=lambda cid: SimpleNamespace(id=cid, name="hermes"))
    member = SimpleNamespace(id=999, guild=guild)
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=channel)

    result = await adapter._maybe_auto_join_from_voice_state(member, before, after)

    assert result is True
    adapter._voice_auto_join_callback.assert_awaited_once_with(
        guild_id=42,
        voice_channel=channel,
        text_channel_id=222,
        text_channel_name="hermes",
    )


@pytest.mark.asyncio
async def test_auto_voice_join_ignores_unconfigured_user():
    adapter = _make_adapter({
        "voice_auto_join": {
            "enabled": True,
            "guild_id": "42",
            "voice_channel_id": "111",
            "text_channel_id": "222",
            "user_ids": ["999"],
        }
    })
    adapter._voice_auto_join_callback = AsyncMock(return_value=True)

    channel = SimpleNamespace(id=111, name="hermes-voice")
    member = SimpleNamespace(id=123, guild=SimpleNamespace(id=42))
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=channel)

    result = await adapter._maybe_auto_join_from_voice_state(member, before, after)

    assert result is False
    adapter._voice_auto_join_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_voice_join_leaves_when_configured_user_leaves_channel():
    adapter = _make_adapter({
        "voice_auto_join": {
            "enabled": True,
            "guild_id": "42",
            "voice_channel_id": "111",
            "text_channel_id": "222",
            "user_ids": ["999"],
        }
    })
    adapter._voice_auto_join_callback = AsyncMock(return_value=True)
    adapter._voice_auto_leave_callback = AsyncMock(return_value=True)

    channel = SimpleNamespace(id=111, name="hermes-voice")
    member = SimpleNamespace(id=999, guild=SimpleNamespace(id=42))
    before = SimpleNamespace(channel=channel)
    after = SimpleNamespace(channel=None)

    result = await adapter._maybe_auto_join_from_voice_state(member, before, after)

    assert result is True
    adapter._voice_auto_join_callback.assert_not_awaited()
    adapter._voice_auto_leave_callback.assert_awaited_once_with(guild_id=42, text_channel_id=222)


@pytest.mark.asyncio
async def test_auto_voice_join_leaves_when_configured_user_moves_out_of_channel():
    adapter = _make_adapter({
        "voice_auto_join": {
            "enabled": True,
            "guild_id": "42",
            "voice_channel_id": "111",
            "text_channel_id": "222",
            "user_ids": ["999"],
        }
    })
    adapter._voice_auto_join_callback = AsyncMock(return_value=True)
    adapter._voice_auto_leave_callback = AsyncMock(return_value=True)

    configured_channel = SimpleNamespace(id=111, name="hermes-voice")
    other_channel = SimpleNamespace(id=333, name="other")
    member = SimpleNamespace(id=999, guild=SimpleNamespace(id=42))
    before = SimpleNamespace(channel=configured_channel)
    after = SimpleNamespace(channel=other_channel)

    result = await adapter._maybe_auto_join_from_voice_state(member, before, after)

    assert result is True
    adapter._voice_auto_join_callback.assert_not_awaited()
    adapter._voice_auto_leave_callback.assert_awaited_once_with(guild_id=42, text_channel_id=222)
