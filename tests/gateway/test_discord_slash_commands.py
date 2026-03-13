"""Tests for native Discord slash command fast-paths."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(tree=FakeTree(), get_channel=lambda _id: None, fetch_channel=AsyncMock())
    return adapter


@pytest.mark.asyncio
async def test_registers_native_thread_slash_command(adapter):
    adapter._handle_thread_create_slash = AsyncMock()
    adapter._register_slash_commands()

    command = adapter._client.tree.commands["thread"]
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock()),
    )

    await command(interaction, name="Planning", message="", auto_archive_duration=1440)

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter._handle_thread_create_slash.assert_awaited_once_with(interaction, "Planning", "", 1440)


@pytest.mark.asyncio
async def test_registers_native_channel_slash_command(adapter):
    adapter._handle_channel_create_slash = AsyncMock()
    adapter._register_slash_commands()

    command = adapter._client.tree.commands["channel"]
    interaction = SimpleNamespace(
        response=SimpleNamespace(defer=AsyncMock()),
    )

    await command(interaction, name="planning-room", topic="Roadmap", nsfw=True)

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    adapter._handle_channel_create_slash.assert_awaited_once_with(interaction, "planning-room", "Roadmap", True)


@pytest.mark.asyncio
async def test_handle_thread_create_slash_reports_success(adapter):
    created_thread = SimpleNamespace(id=555, name="Planning", send=AsyncMock())
    parent_channel = SimpleNamespace(create_thread=AsyncMock(return_value=created_thread), send=AsyncMock())
    interaction_channel = SimpleNamespace(parent=parent_channel)
    interaction = SimpleNamespace(
        channel=interaction_channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    parent_channel.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    created_thread.send.assert_awaited_once_with("Kickoff")
    interaction.followup.send.assert_awaited_once()
    args, kwargs = interaction.followup.send.await_args
    assert "<#555>" in args[0]
    assert kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_handle_thread_create_slash_falls_back_to_seed_message(adapter):
    created_thread = SimpleNamespace(id=555, name="Planning")
    seed_message = SimpleNamespace(id=777, create_thread=AsyncMock(return_value=created_thread))
    channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(return_value=seed_message),
    )
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "Kickoff", 1440)

    channel.send.assert_awaited_once_with("Kickoff")
    seed_message.create_thread.assert_awaited_once_with(
        name="Planning",
        auto_archive_duration=1440,
        reason="Requested by Jezza via /thread",
    )
    interaction.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_thread_create_slash_reports_failure(adapter):
    channel = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct failed")),
        send=AsyncMock(side_effect=RuntimeError("nope")),
    )
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_thread_create_slash(interaction, "Planning", "", 1440)

    interaction.followup.send.assert_awaited_once()
    args, kwargs = interaction.followup.send.await_args
    assert "Failed to create thread:" in args[0]
    assert "nope" in args[0]
    assert kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_handle_channel_create_slash_reports_success(adapter):
    created_channel = SimpleNamespace(id=777, name="planning-room")
    guild = SimpleNamespace(create_text_channel=AsyncMock(return_value=created_channel))
    category = object()
    channel = SimpleNamespace(guild=guild, category=category)
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_channel_create_slash(interaction, "planning-room", "Roadmap", False)

    guild.create_text_channel.assert_awaited_once_with(
        name="planning-room",
        nsfw=False,
        reason="Requested by Jezza via /channel",
        topic="Roadmap",
        category=category,
    )
    interaction.followup.send.assert_awaited_once()
    args, kwargs = interaction.followup.send.await_args
    assert "<#777>" in args[0]
    assert kwargs["ephemeral"] is True


@pytest.mark.asyncio
async def test_handle_channel_create_slash_from_thread_uses_parent_channel_category(adapter):
    created_channel = SimpleNamespace(id=777, name="planning-room")
    guild = SimpleNamespace(create_text_channel=AsyncMock(return_value=created_channel))
    category = object()
    parent_channel = SimpleNamespace(guild=guild, category=category)
    thread_channel = SimpleNamespace(parent=parent_channel, guild=guild)
    interaction = SimpleNamespace(
        channel=thread_channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_channel_create_slash(interaction, "planning-room", "", True)

    guild.create_text_channel.assert_awaited_once_with(
        name="planning-room",
        nsfw=True,
        reason="Requested by Jezza via /channel",
        category=category,
    )


@pytest.mark.asyncio
async def test_handle_channel_create_slash_reports_failure(adapter):
    guild = SimpleNamespace(create_text_channel=AsyncMock(side_effect=RuntimeError("nope")))
    channel = SimpleNamespace(guild=guild, category=None)
    interaction = SimpleNamespace(
        channel=channel,
        channel_id=123,
        user=SimpleNamespace(display_name="Jezza"),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await adapter._handle_channel_create_slash(interaction, "planning-room", "", False)

    interaction.followup.send.assert_awaited_once_with("Failed to create channel: nope", ephemeral=True)
