"""Discord adapter race polish: concurrent join_voice_channel must not
double-invoke channel.connect() on the same guild."""

import asyncio
from unittest.mock import MagicMock, patch

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
    entered = asyncio.Event()
    listener_stop = asyncio.Event()
    adapter._voice_timeout_seconds = 0

    async def listen(guild_id):
        await listener_stop.wait()

    adapter._voice_listen_loop = listen

    class FakeVC:
        def __init__(self, channel):
            self.channel = channel

        def is_connected(self):
            return True

        async def move_to(self, _channel):
            return None

    async def slow_connect(self):
        connect_count[0] += 1
        entered.set()
        await release.wait()
        return FakeVC(self)

    channel = MagicMock()
    channel.id = 111
    channel.guild.id = 42
    channel.connect = lambda: slow_connect(channel)

    from plugins.platforms.discord import adapter as discord_mod
    tasks = []
    try:
        with patch.object(discord_mod, "VoiceReceiver",
                          MagicMock(return_value=MagicMock(start=lambda: None))):
            tasks.append(asyncio.create_task(adapter.join_voice_channel(channel)))
            await asyncio.wait_for(entered.wait(), timeout=10)
            tasks.append(asyncio.create_task(adapter.join_voice_channel(channel)))
            release.set()
            r1, r2 = await asyncio.gather(*tasks)
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        listener_stop.set()
        await asyncio.gather(*adapter._voice_listen_tasks.values())

    assert connect_count[0] == 1, (
        f"expected 1 channel.connect() call, got {connect_count[0]} — "
        "per-guild lock is not serializing join_voice_channel"
    )
    assert r1 is True and r2 is True
    assert 42 in adapter._voice_clients
