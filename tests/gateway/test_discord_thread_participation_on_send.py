"""A conversational send into a Discord thread records participation. Regression for #160499:
a thread the bot had only posted into (cron delivery) kept the full mention gate, so a voice
note there, which cannot carry a mention, was dropped without a log line."""
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter



import discord  # the real module, or the mock installed above


def _adapter(monkeypatch, tmp_path, channel) -> DiscordAdapter:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(get_channel=MagicMock(return_value=channel), fetch_channel=AsyncMock())
    return adapter


def _channel(spec, channel_id: int):
    channel = MagicMock(spec=spec)
    channel.id = channel_id
    channel.send = AsyncMock(return_value=SimpleNamespace(id=9001))
    return channel


@pytest.mark.asyncio
async def test_send_into_a_thread_makes_follow_ups_skip_the_mention_gate(monkeypatch, tmp_path):
    thread = _channel(discord.Thread, 777)
    adapter = _adapter(monkeypatch, tmp_path, thread)
    inbound = SimpleNamespace(channel=thread)
    assert adapter._in_bot_thread(inbound) is False

    result = await adapter.send("555", "scheduled report", metadata={"thread_id": "777"})

    assert result.success is True
    assert adapter._in_bot_thread(inbound) is True


@pytest.mark.asyncio
async def test_status_only_send_into_a_thread_is_not_participation(monkeypatch, tmp_path):
    thread = _channel(discord.Thread, 778)
    adapter = _adapter(monkeypatch, tmp_path, thread)

    result = await adapter.send("555", "working", metadata={"thread_id": "778", "non_conversational": True})

    assert result.success is True
    assert "778" not in adapter._threads


@pytest.mark.asyncio
async def test_send_into_a_plain_channel_records_no_thread(monkeypatch, tmp_path):
    channel = _channel(type("TextChannel", (), {}), 555)
    adapter = _adapter(monkeypatch, tmp_path, channel)

    result = await adapter.send("555", "hello")

    assert result.success is True
    assert "555" not in adapter._threads
