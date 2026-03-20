from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import importlib
import sys

import pytest

def _install_discord_mock():
    class FakeIntents:
        @staticmethod
        def default():
            return SimpleNamespace()

    discord_mod = ModuleType("discord")
    discord_mod.__file__ = "mock-discord.py"
    discord_mod.Message = type("Message", (), {})
    discord_mod.Intents = FakeIntents
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1,
        primary=2,
        danger=3,
        green=1,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(
        is_loaded=lambda: True,
        load_opus=lambda *_args, **_kwargs: None,
        Decoder=MagicMock,
    )
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    sys.modules["discord"] = discord_mod
    ext_mod = ModuleType("discord.ext")
    commands_mod = ModuleType("discord.ext.commands")
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules["discord.ext"] = ext_mod
    sys.modules["discord.ext.commands"] = commands_mod


def _load_permissions_module():
    _install_discord_mock()
    importlib.reload(importlib.import_module("gateway.platforms.discord_impl.delivery"))
    return importlib.reload(importlib.import_module("gateway.platforms.discord_impl.permissions"))


permissions = _load_permissions_module()


def _make_client(channel=None, guilds=None):
    return SimpleNamespace(
        get_channel=MagicMock(return_value=channel),
        fetch_channel=AsyncMock(return_value=None if channel is None else channel),
        guilds=list(guilds or []),
        user=SimpleNamespace(id=999),
    )


def _make_guild_channel(channel_id, name, *, can_read=True, can_read_history=True):
    guild_member = SimpleNamespace(id=999)
    guild = SimpleNamespace(id=777, me=guild_member, name="Hermes")
    permission_bits = SimpleNamespace(
        view_channel=can_read,
        read_messages=can_read,
        send_messages=True,
        read_message_history=can_read_history,
        attach_files=True,
        embed_links=True,
        add_reactions=True,
        manage_threads=False,
        create_public_threads=True,
        create_private_threads=False,
    )
    return SimpleNamespace(
        id=channel_id,
        name=name,
        guild=guild,
        permissions_for=MagicMock(return_value=permission_bits),
    )


@pytest.mark.asyncio
async def test_check_channel_permissions_for_guild_text_channel():
    channel = _make_guild_channel(123, "general")
    client = _make_client(channel=channel)

    result = await permissions.check_channel_permissions(client, "123")

    assert result == permissions.ChannelPermissions(
        channel_id="123",
        channel_name="general",
        can_read=True,
        can_send=True,
        can_read_history=True,
        can_attach_files=True,
        can_embed_links=True,
        can_add_reactions=True,
        can_manage_threads=False,
        can_create_threads=True,
    )
    channel.permissions_for.assert_called_once_with(channel.guild.me)


@pytest.mark.asyncio
async def test_check_channel_permissions_for_dm_channel_disables_thread_flags():
    dm_channel = permissions.discord.DMChannel()
    dm_channel.id = 321
    dm_channel.name = None
    dm_channel.guild = None
    dm_channel.recipient = SimpleNamespace(name="Alan")
    client = _make_client(channel=dm_channel)

    result = await permissions.check_channel_permissions(client, "321")

    assert result == permissions.ChannelPermissions(
        channel_id="321",
        channel_name="Alan",
        can_read=True,
        can_send=True,
        can_read_history=True,
        can_attach_files=True,
        can_embed_links=True,
        can_add_reactions=True,
        can_manage_threads=False,
        can_create_threads=False,
    )


@pytest.mark.asyncio
async def test_check_channel_permissions_returns_none_for_missing_channel():
    client = _make_client(channel=None)

    result = await permissions.check_channel_permissions(client, "123")

    assert result is None


@pytest.mark.asyncio
async def test_list_accessible_channels_filters_to_readable_channels_only():
    readable = _make_guild_channel(1, "general", can_read=True)
    hidden = _make_guild_channel(2, "private", can_read=False)
    guild = SimpleNamespace(id=777, name="Hermes", text_channels=[readable, hidden])
    readable.guild = guild
    hidden.guild = guild
    client = _make_client(guilds=[guild])

    result = await permissions.list_accessible_channels(client)

    assert result == [
        permissions.AccessibleChannel(
            channel_id="1",
            channel_name="general",
            guild_id="777",
            guild_name="Hermes",
            channel_kind="channel",
            qualified_name="Hermes/general",
            mention="<#1>",
            can_read=True,
            can_send=True,
            can_read_history=True,
            can_attach_files=True,
            can_embed_links=True,
            can_add_reactions=True,
            can_manage_threads=False,
            can_create_threads=True,
        )
    ]


@pytest.mark.asyncio
async def test_list_accessible_channels_respects_guild_id_filter():
    guild_one = SimpleNamespace(id=111, name="One", text_channels=[_make_guild_channel(1, "one")])
    guild_two = SimpleNamespace(id=222, name="Two", text_channels=[_make_guild_channel(2, "two")])
    for guild in (guild_one, guild_two):
        for channel in guild.text_channels:
            channel.guild = guild
    client = _make_client(guilds=[guild_one, guild_two])

    result = await permissions.list_accessible_channels(client, guild_id="222")

    assert [channel.channel_id for channel in result] == ["2"]
    assert [channel.channel_name for channel in result] == ["two"]


@pytest.mark.asyncio
async def test_list_accessible_channels_includes_normalized_target_metadata():
    guild = SimpleNamespace(id=777, name="Hermes", text_channels=[_make_guild_channel(1, "general")])
    for channel in guild.text_channels:
        channel.guild = guild
    client = _make_client(guilds=[guild])

    result = await permissions.list_accessible_channels(client)

    assert result == [
        permissions.AccessibleChannel(
            channel_id="1",
            channel_name="general",
            guild_id="777",
            guild_name="Hermes",
            channel_kind="channel",
            qualified_name="Hermes/general",
            mention="<#1>",
            can_read=True,
            can_send=True,
            can_read_history=True,
            can_attach_files=True,
            can_embed_links=True,
            can_add_reactions=True,
            can_manage_threads=False,
            can_create_threads=True,
        )
    ]


@pytest.mark.asyncio
async def test_can_read_channel_returns_true_and_false_correctly():
    readable = _make_guild_channel(123, "general", can_read=True, can_read_history=True)
    unreadable = _make_guild_channel(456, "private", can_read=False, can_read_history=False)

    readable_client = _make_client(channel=readable)
    unreadable_client = _make_client(channel=unreadable)

    assert await permissions.can_read_channel(readable_client, "123") is True
    assert await permissions.can_read_channel(unreadable_client, "456") is False
