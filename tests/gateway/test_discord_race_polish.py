"""Discord adapter race polish: concurrent join_voice_channel must not
double-invoke channel.connect() on the same guild."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter._platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="t")
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

    def schedule_and_close(coro):
        coro.close()
        return asyncio.create_task(asyncio.sleep(0))

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42
    channel.connect = lambda: slow_connect(channel)

    from plugins.platforms.discord import adapter as discord_mod
    with patch.object(discord_mod, "VoiceReceiver",
                      MagicMock(return_value=MagicMock(start=lambda: None))):
        with patch.object(discord_mod.asyncio, "ensure_future",
                          schedule_and_close):
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
async def test_forced_voice_move_refreshes_packet_receiver():
    """A Discord transport reconnect must not leave the receiver on its old UDP reader."""
    adapter = _make_adapter()
    guild_id = 42
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    old_receiver = MagicMock()
    old_listen_task = MagicMock()
    replacement_receiver = MagicMock()
    replacement_listen_task = MagicMock()
    adapter._voice_clients[guild_id] = voice_client
    adapter._voice_receivers[guild_id] = old_receiver
    adapter._voice_listen_tasks[guild_id] = old_listen_task

    def schedule_and_close(coro):
        coro.close()
        return replacement_listen_task

    from plugins.platforms.discord import adapter as discord_mod
    with patch.object(discord_mod, "VoiceReceiver", return_value=replacement_receiver) as receiver_cls, \
            patch.object(discord_mod.asyncio, "ensure_future", side_effect=schedule_and_close), \
            patch.object(discord_mod.asyncio, "sleep", return_value=None):
        await adapter._refresh_voice_receiver_after_forced_move(guild_id)

    old_receiver.stop.assert_called_once_with()
    old_listen_task.cancel.assert_called_once_with()
    receiver_cls.assert_called_once_with(voice_client, allowed_user_ids=set())
    replacement_receiver.start.assert_called_once_with()
    assert adapter._voice_receivers[guild_id] is replacement_receiver
    assert adapter._voice_listen_tasks[guild_id] is replacement_listen_task


@pytest.mark.asyncio
async def test_auto_join_requires_configured_user_and_voice_channel():
    """Only the configured user entering the configured room can auto-join Hermes."""
    adapter = _make_adapter()
    adapter._voice_auto_join_channel_id = 111
    adapter._voice_auto_join_text_channel_id = 222
    adapter._voice_auto_join_user_ids = {"158441542070173697"}
    adapter.join_voice_channel = AsyncMock(return_value=True)

    channel = MagicMock()
    channel.id = 111
    member = MagicMock()
    member.id = 158441542070173697
    member.guild.id = 42
    before = MagicMock(channel=None)
    after = MagicMock(channel=channel)

    await adapter._auto_join_voice_for_member(member, before, after)

    adapter.join_voice_channel.assert_awaited_once_with(
        channel,
        text_channel_id=222,
        source={
            "platform": "discord",
            "chat_id": "222",
            "user_id": "158441542070173697",
            "user_name": "158441542070173697",
            "chat_type": "channel",
        },
    )


@pytest.mark.asyncio
async def test_auto_join_on_ready_joins_when_configured_user_is_present():
    """A gateway restart rejoins when the configured user is already in the room."""
    adapter = _make_adapter()
    adapter._voice_auto_join_channel_id = 111
    adapter._voice_auto_join_text_channel_id = 222
    adapter._voice_auto_join_user_ids = {"158441542070173697"}
    adapter.join_voice_channel = AsyncMock(return_value=True)

    channel = MagicMock()
    channel.id = 111
    member = MagicMock()
    member.id = 158441542070173697
    channel.members = [member]
    adapter._client.get_channel.return_value = channel

    await adapter._auto_join_configured_voice_channel_if_user_present()

    adapter.join_voice_channel.assert_awaited_once_with(
        channel,
        text_channel_id=222,
        source={
            "platform": "discord",
            "chat_id": "222",
            "user_id": "158441542070173697",
            "user_name": "158441542070173697",
            "chat_type": "channel",
        },
    )


@pytest.mark.asyncio
async def test_auto_join_leaves_when_configured_user_leaves_room():
    """The dedicated auto-join session ends when its configured user leaves."""
    adapter = _make_adapter()
    adapter._voice_auto_join_channel_id = 111
    adapter._voice_auto_join_user_ids = {"158441542070173697"}
    adapter.leave_voice_channel = AsyncMock()

    configured_channel = MagicMock()
    configured_channel.id = 111
    other_channel = MagicMock()
    other_channel.id = 222
    member = MagicMock()
    member.id = 158441542070173697
    member.guild.id = 42
    before = MagicMock(channel=configured_channel)
    after = MagicMock(channel=other_channel)

    await adapter._auto_join_voice_for_member(member, before, after)

    adapter.leave_voice_channel.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_auto_join_stays_when_another_configured_user_remains():
    adapter = _make_adapter()
    adapter._voice_auto_join_channel_id = 111
    adapter._voice_auto_join_user_ids = {"1001", "1002"}
    adapter.leave_voice_channel = AsyncMock()

    configured_channel = MagicMock()
    configured_channel.id = 111
    departing = MagicMock()
    departing.id = 1001
    departing.guild.id = 42
    remaining = MagicMock()
    remaining.id = 1002
    configured_channel.members = [remaining]
    other_channel = MagicMock()
    other_channel.id = 222

    await adapter._auto_join_voice_for_member(
        departing,
        MagicMock(channel=configured_channel),
        MagicMock(channel=other_channel),
    )

    adapter.leave_voice_channel.assert_not_awaited()
