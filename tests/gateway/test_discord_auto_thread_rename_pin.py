"""Hermes's own semantic rename of an auto-created thread keeps the thread's pinned prompt.

``chat_name`` is part of the pinned session-context prompt's key. The title lane renames a new
auto-thread once the LLM title arrives, normally before the user's second message, so turn 2 read
the new name and re-rendered the already-sent prompt: a prompt-cache miss on turn 2 of every
auto-threaded conversation on the default config. A rename by anyone else is a real metadata
change and still re-renders.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import plugins.platforms.discord.adapter as discord_platform
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import build_session_context
from plugins.platforms.discord.adapter import DiscordAdapter


class _Text:
    def __init__(self, channel_id, name="ops"):
        self.id, self.name, self.topic = channel_id, name, None
        self.guild = SimpleNamespace(id=1, name="Hermes Server")


class _Thread:
    def __init__(self, channel_id, parent, name):
        self.id, self.name, self.parent, self.parent_id = channel_id, name, parent, parent.id
        self.guild = parent.guild

    async def edit(self, *, name, reason=None):
        self.name = name


_USER = SimpleNamespace(id=42, display_name="Alice", name="alice")


def _message(channel, message_id):
    return SimpleNamespace(
        id=message_id, content="what broke?", mentions=[], attachments=[], reference=None,
        created_at=datetime.now(timezone.utc), channel=channel, author=_USER)


@pytest.mark.asyncio
async def test_hermes_title_rename_keeps_the_pin_and_a_human_rename_does_not(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "Thread", _Thread, raising=False)
    monkeypatch.setattr(discord_platform, "DISCORD_AVAILABLE", True)
    monkeypatch.setenv("DISCORD_REQUIRE_MENTION", "false")
    monkeypatch.setenv("DISCORD_AUTO_THREAD", "true")
    parent = _Text(700)
    thread = _Thread(800, parent, name="what broke?")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999), get_channel=lambda _id: thread)
    adapter._text_batch_delay_seconds = 0
    adapter._discord_history_backfill = lambda: False
    adapter._auto_create_thread = AsyncMock(return_value=thread)
    adapter.handle_message = AsyncMock()

    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})

    async def turn(channel, message_id):
        await adapter._handle_message(_message(channel, message_id))
        source = adapter.handle_message.await_args.args[0].source
        assert source.chat_id == "800"
        return runner._pinned_session_context_prompt(build_session_context(source, config), False, "k")

    first = await turn(parent, 100)
    # The title lane's call, as gateway/run_topics.py makes it for a native auto-thread.
    assert await adapter.rename_thread("800", "Database outage", only_if_current_name="what broke?")
    assert thread.name == "Database outage"
    assert await turn(thread, 101) == first

    thread.name = "Renamed by a moderator"
    after_human_rename = await turn(thread, 102)
    assert after_human_rename != first
    assert "Renamed by a moderator" in after_human_rename
