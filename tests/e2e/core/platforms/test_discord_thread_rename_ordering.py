"""Real SDK dispatch must preserve moderator names before cached callbacks run.

Regression for #131614. The messaging E2E lane owns this SDK contract; gateway
unit tests exercise the same state machine without the optional Discord extra.
"""

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import plugins.platforms.discord.adapter as discord_platform
from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_batched_thread_updates_retire_the_alias_through_connected_bot(monkeypatch):
    # The parent E2E conftest installs SDK mocks. Require the real messaging
    # extra here: an absent dependency must fail the owning CI lane, never skip.
    with patch.dict(sys.modules):
        for name in tuple(sys.modules):
            if name == "discord" or name.startswith("discord."):
                del sys.modules[name]
        discord = importlib.import_module("discord")
        commands = importlib.import_module("discord.ext.commands")
        state_type = importlib.import_module("discord.state").ConnectionState
        monkeypatch.setattr(discord_platform, "discord", discord)
        monkeypatch.setattr(discord_platform, "commands", commands)
        monkeypatch.setattr(discord_platform, "Intents", discord.Intents)
        monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", True)
        monkeypatch.setattr(discord_platform, "_load_opus_codec", lambda: None)
        adapter = discord_platform.DiscordAdapter(PlatformConfig(
            enabled=True, token="fake", extra={"slash_commands": False}))
        monkeypatch.setattr(adapter, "_start_liveness_probe", lambda: None)

        async def start(client, token, **kwargs):
            await client._async_setup_hook()
            adapter._ready_event.set()
            await asyncio.Future()  # transport remains open until disconnect

        monkeypatch.setattr(commands.Bot, "start", start)
        assert await adapter.connect()
        try:
            assert not adapter._platform_events_subscribed()
            parent = SimpleNamespace(id=700, name="ops", guild=SimpleNamespace(name="Hermes Server"))

            class CachedThread:
                id, parent_id = 800, 700
                name = "what broke?"
                guild = parent.guild
                archived = False

                def _update(self, data):
                    self.name = data["name"]

                async def edit(self, *, name, reason=None):
                    return SimpleNamespace(name=name)

            thread = CachedThread()
            thread.parent = parent
            monkeypatch.setattr(adapter._client, "get_channel", lambda _id: thread)
            first = adapter._format_thread_chat_name(thread)
            assert await adapter.rename_thread("800", "Database outage", only_if_current_name=thread.name)
            # Do not call the formatter on Hermes's title before the moderator
            # roundtrip: its cache-lag fallback would otherwise hide this seam.
            guild = SimpleNamespace(get_thread=lambda _id: thread)
            state = SimpleNamespace(_get_guild=lambda _id: guild, dispatch=adapter._client.dispatch)
            scheduled_before = asyncio.all_tasks()
            state_type.parse_thread_update(state, {
                "id": "800", "guild_id": "1", "parent_id": "700", "type": 11, "name": "Database outage",
            })
            await asyncio.gather(*(asyncio.all_tasks() - scheduled_before))
            scheduled_before = asyncio.all_tasks()
            for name in ("what broke?", "Database outage"):
                state_type.parse_thread_update(state, {
                    "id": "800", "guild_id": "1", "parent_id": "700", "type": 11, "name": name,
                })
            # Both cached callbacks now share the final name. Only raw payloads
            # retain the intermediate moderator edit, with no hook subscriber.
            assert thread.name == "Database outage"
            await asyncio.gather(*(asyncio.all_tasks() - scheduled_before))
            rendered = adapter._format_thread_chat_name(thread)
            assert rendered.endswith(" / Database outage")
            assert rendered != first
        finally:
            await adapter.disconnect()
