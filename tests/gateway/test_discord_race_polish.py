"""Regression tests for the Discord adapter race-polish fix.

Two races are addressed:
1. on_message allowlist check racing on_ready's _resolve_allowed_usernames
   resolution window.  Username-based entries in DISCORD_ALLOWED_USERS
   appear in the set as raw strings for several seconds after
   connect/reconnect; author.id is always numeric, so legitimate users
   are silently rejected until resolution finishes.
2. join_voice_channel check-and-connect: concurrent /voice channel
   invocations both see _voice_clients.get(guild_id) is None, both call
   channel.connect(), second raises ClientException ('Already connected').
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    """Bare DiscordAdapter for testing — object.__new__ pattern per AGENTS.md."""
    from gateway.platforms.discord import DiscordAdapter

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


class TestJoinVoiceSerialization:
    @pytest.mark.asyncio
    async def test_concurrent_joins_do_not_double_connect(self):
        """Two concurrent join_voice_channel calls on the same guild
        must serialize through the per-guild lock — only ONE
        channel.connect() actually fires; the second sees the
        _voice_clients entry the first just installed."""
        adapter = _make_adapter()

        connect_count = [0]
        connect_event = asyncio.Event()

        class FakeVC:
            def __init__(self, channel):
                self.channel = channel

            def is_connected(self):
                return True

            async def move_to(self, _channel):
                return None

            async def disconnect(self):
                return None

        async def slow_connect(self):
            connect_count[0] += 1
            # Widen the race window
            await connect_event.wait()
            return FakeVC(self)

        channel = MagicMock()
        channel.id = 111
        channel.guild.id = 42
        channel.connect = lambda: slow_connect(channel)

        # Swap out VoiceReceiver so it doesn't try to set up real audio
        from gateway.platforms import discord as discord_mod
        with patch.object(discord_mod, "VoiceReceiver", MagicMock(return_value=MagicMock(start=lambda: None))):
            with patch.object(discord_mod.asyncio, "ensure_future", lambda _c: asyncio.create_task(asyncio.sleep(0))):
                # Fire two joins concurrently
                t1 = asyncio.create_task(adapter.join_voice_channel(channel))
                t2 = asyncio.create_task(adapter.join_voice_channel(channel))
                # Let them run until they're blocked on our event
                await asyncio.sleep(0.05)
                # Release connect so both can finish
                connect_event.set()
                r1, r2 = await asyncio.gather(t1, t2)

        assert connect_count[0] == 1, (
            f"Expected exactly 1 channel.connect() call, got {connect_count[0]} — "
            "per-guild voice lock is not serializing join_voice_channel"
        )
        assert r1 is True and r2 is True
        assert 42 in adapter._voice_clients


class TestOnMessageWaitsForReadyEvent:
    @pytest.mark.asyncio
    async def test_on_message_blocks_until_ready_event_set(self):
        """A message arriving before on_ready finishes
        _resolve_allowed_usernames must wait, not proceed with a
        half-resolved allowlist."""
        # This is an integration-style check — we pull out the
        # on_message handler by asserting the source contains the
        # expected wait pattern.  A full end-to-end test would require
        # setting up the discord.py client machinery, which is not
        # practical here.
        import inspect
        from gateway.platforms import discord as discord_mod

        src = inspect.getsource(discord_mod.DiscordAdapter.connect)
        assert "_ready_event.is_set()" in src, (
            "on_message must gate on _ready_event so username-based "
            "allowlist entries are resolved before the allowlist check"
        )
        assert "await asyncio.wait_for(" in src and "_ready_event.wait()" in src, (
            "Expected asyncio.wait_for(_ready_event.wait(), timeout=...) "
            "pattern in on_message"
        )
