"""Build Discord channels for adapter boundary tests with the installed SDK."""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import AsyncIterator, TYPE_CHECKING
from unittest.mock import MagicMock

import discord

if TYPE_CHECKING:
    from discord.state import ConnectionState
    from discord.types.channel import DMChannel as DMPayload, ForumChannel as ForumPayload, TextChannel as TextPayload
    from discord.types.guild import Guild as GuildPayload
    from discord.types.threads import Thread as ThreadPayload


class _ThreadChannel(discord.Thread):
    """Allow tests to replace the history boundary on a slot-based SDK thread."""

    topic: None = None

    async def history(self, *args, **kwargs) -> AsyncIterator[discord.Message]:
        return
        yield  # pragma: no cover


class _ForumChannel(discord.ForumChannel):
    """Allow tests to replace the network method on a slot-based SDK channel."""


def _guild(guild_id: int, name: str) -> tuple[discord.Guild, ConnectionState]:
    state = discord.Client(intents=discord.Intents.none())._connection
    data: GuildPayload = {
        "id": str(guild_id), "name": name, "owner_id": "333", "icon": None,
        "splash": None, "discovery_splash": None, "emojis": [], "stickers": [],
        "features": [], "description": None, "incidents_data": None,
        "region": "", "afk_channel_id": None, "afk_timeout": 60,
        "verification_level": 0, "default_message_notifications": 0,
        "explicit_content_filter": 0, "roles": [], "mfa_level": 0,
        "nsfw_level": 0, "application_id": None, "system_channel_id": None,
        "system_channel_flags": 0, "rules_channel_id": None,
        "vanity_url_code": None, "banner": None, "premium_tier": 0,
        "preferred_locale": "en-US", "public_updates_channel_id": None,
        "stage_instances": [], "guild_scheduled_events": [],
    }
    return discord.Guild(state=state, data=data), state


def make_thread(*, channel_id: int, parent_id: int, name: str = "test-thread",
                owner_id: int = 777, guild_id: int = 999,
                guild_name: str = "TestGuild") -> _ThreadChannel:
    if not isinstance(discord, ModuleType):
        channel = MagicMock(spec=_ThreadChannel)
        channel.history = MagicMock(side_effect=lambda *args, **kwargs: _ThreadChannel.history(channel))
        channel.id, channel.name = channel_id, name
        channel.owner_id, channel.parent_id = owner_id, parent_id
        channel.guild = SimpleNamespace(id=guild_id, name=guild_name)
        channel.parent = SimpleNamespace(id=parent_id, name="general", guild=channel.guild)
        return channel

    guild, state = _guild(guild_id, guild_name)
    parent_data: TextPayload = {
        "id": str(parent_id), "type": 0, "name": "general", "position": 0,
        "permission_overwrites": [], "rate_limit_per_user": 0,
        "nsfw": False, "parent_id": None, "last_message_id": None, "guild_id": str(guild_id),
    }
    parent = discord.TextChannel(state=state, guild=guild, data=parent_data)
    guild._channels[parent_id] = parent
    data: ThreadPayload = {
        "id": str(channel_id), "guild_id": str(guild_id), "type": 11, "name": name,
        "parent_id": str(parent_id), "owner_id": str(owner_id or 0),
        "message_count": 0, "member_count": 1, "total_message_sent": 0,
        "rate_limit_per_user": 0,
        "thread_metadata": {"archived": False, "auto_archive_duration": 60,
                            "archive_timestamp": "2026-10-02T00:00:00+00:00", "locked": False},
    }
    return _ThreadChannel(guild=guild, state=state, data=data)


def make_forum(*, channel_id: int, name: str = "ideas") -> _ForumChannel:
    if not isinstance(discord, ModuleType):
        channel = MagicMock(spec=discord.ForumChannel)
        channel.id, channel.name = channel_id, name
        return channel

    guild, state = _guild(999, "TestGuild")
    data: ForumPayload = {
        "id": str(channel_id), "type": 15, "name": name, "position": 0,
        "permission_overwrites": [], "parent_id": None, "guild_id": "999",
        "rate_limit_per_user": 0, "nsfw": False, "available_tags": [],
        "default_reaction_emoji": None, "default_sort_order": None,
        "default_forum_layout": 0, "default_thread_rate_limit_per_user": 0,
        "default_auto_archive_duration": 1440, "flags": 0,
    }
    return _ForumChannel(state=state, guild=guild, data=data)


def make_dm(*, channel_id: int) -> discord.DMChannel:
    if not isinstance(discord, ModuleType):
        channel = MagicMock(spec=discord.DMChannel)
        channel.id = channel_id
        return channel

    state = discord.Client(intents=discord.Intents.none())._connection
    data: DMPayload = {"id": str(channel_id), "name": "dm", "type": 1,
                       "recipients": [], "last_message_id": None}
    return discord.DMChannel(me=MagicMock(spec=discord.ClientUser), state=state, data=data)
