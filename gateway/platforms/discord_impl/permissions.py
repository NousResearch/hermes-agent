"""Discord permission introspection.

Channel/thread access checks, bot permission queries, and visibility helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from gateway.platforms.discord_impl.delivery import resolve_channel

try:
    import discord
except ImportError:  # pragma: no cover - import guard
    discord = None


@dataclass
class ChannelPermissions:
    """Bot's effective permissions in a Discord channel."""

    channel_id: str
    channel_name: str
    can_read: bool
    can_send: bool
    can_read_history: bool
    can_attach_files: bool
    can_embed_links: bool
    can_add_reactions: bool
    can_manage_threads: bool
    can_create_threads: bool


@dataclass
class AccessibleChannel:
    """Readable Discord channel entry with normalized target metadata."""

    channel_id: str
    channel_name: str
    guild_id: Optional[str]
    guild_name: Optional[str]
    channel_kind: str
    qualified_name: str
    mention: str
    can_read: bool
    can_send: bool
    can_read_history: bool
    can_attach_files: bool
    can_embed_links: bool
    can_add_reactions: bool
    can_manage_threads: bool
    can_create_threads: bool


def _channel_name(channel: Any) -> str:
    recipient = getattr(channel, "recipient", None)
    recipient_name = getattr(recipient, "name", None)
    return (
        getattr(channel, "name", None)
        or recipient_name
        or str(getattr(channel, "id", "unknown"))
    )


def _permission_flag(perms: Any, *names: str) -> bool:
    for name in names:
        value = getattr(perms, name, None)
        if value is not None:
            return bool(value)
    return False


def _is_dm_channel(channel: Any) -> bool:
    dm_cls = getattr(discord, "DMChannel", None) if discord else None
    if dm_cls and isinstance(channel, dm_cls):
        return True

    channel_type = getattr(channel, "type", None)
    type_value = getattr(channel_type, "value", channel_type)
    if type_value in {"dm", "private"}:
        return True

    return getattr(channel, "guild", None) is None and getattr(channel, "recipient", None) is not None


def _build_channel_permissions(client: Any, channel: Any) -> ChannelPermissions:
    if _is_dm_channel(channel):
        return ChannelPermissions(
            channel_id=str(getattr(channel, "id", "")),
            channel_name=_channel_name(channel),
            can_read=True,
            can_send=True,
            can_read_history=True,
            can_attach_files=True,
            can_embed_links=True,
            can_add_reactions=True,
            can_manage_threads=False,
            can_create_threads=False,
        )

    guild = getattr(channel, "guild", None)
    member = getattr(guild, "me", None) or getattr(client, "user", None)
    permissions_for = getattr(channel, "permissions_for", None)

    if guild is None or member is None or not callable(permissions_for):
        return ChannelPermissions(
            channel_id=str(getattr(channel, "id", "")),
            channel_name=_channel_name(channel),
            can_read=False,
            can_send=False,
            can_read_history=False,
            can_attach_files=False,
            can_embed_links=False,
            can_add_reactions=False,
            can_manage_threads=False,
            can_create_threads=False,
        )

    perms = permissions_for(member)
    return ChannelPermissions(
        channel_id=str(getattr(channel, "id", "")),
        channel_name=_channel_name(channel),
        can_read=_permission_flag(perms, "view_channel", "read_messages"),
        can_send=_permission_flag(perms, "send_messages"),
        can_read_history=_permission_flag(perms, "read_message_history"),
        can_attach_files=_permission_flag(perms, "attach_files"),
        can_embed_links=_permission_flag(perms, "embed_links"),
        can_add_reactions=_permission_flag(perms, "add_reactions"),
        can_manage_threads=_permission_flag(perms, "manage_threads"),
        can_create_threads=_permission_flag(
            perms,
            "create_public_threads",
            "create_private_threads",
        ),
    )


def _build_accessible_channel(client: Any, channel: Any) -> AccessibleChannel:
    perms = _build_channel_permissions(client, channel)
    guild = getattr(channel, "guild", None)
    guild_id = str(getattr(guild, "id", "")) if getattr(guild, "id", None) is not None else None
    guild_name = getattr(guild, "name", None)
    channel_name = _channel_name(channel)
    qualified_name = f"{guild_name}/{channel_name}" if guild_name else channel_name
    return AccessibleChannel(
        channel_id=perms.channel_id,
        channel_name=channel_name,
        guild_id=guild_id,
        guild_name=guild_name,
        channel_kind="channel",
        qualified_name=qualified_name,
        mention=f"<#{perms.channel_id}>",
        can_read=perms.can_read,
        can_send=perms.can_send,
        can_read_history=perms.can_read_history,
        can_attach_files=perms.can_attach_files,
        can_embed_links=perms.can_embed_links,
        can_add_reactions=perms.can_add_reactions,
        can_manage_threads=perms.can_manage_threads,
        can_create_threads=perms.can_create_threads,
    )


async def check_channel_permissions(
    client: Any,
    channel_id: str,
) -> Optional[ChannelPermissions]:
    """Check bot's effective permissions in a channel."""
    channel = await resolve_channel(client, channel_id)
    if channel is None:
        return None
    return _build_channel_permissions(client, channel)


async def list_accessible_channels(
    client: Any,
    guild_id: Optional[str] = None,
) -> list[AccessibleChannel]:
    """List channels the bot can access, optionally filtered by guild."""
    if not client:
        return []

    accessible_channels: list[AccessibleChannel] = []
    for guild in getattr(client, "guilds", []) or []:
        if guild_id is not None and str(getattr(guild, "id", "")) != str(guild_id):
            continue

        for channel in getattr(guild, "text_channels", []) or []:
            accessible_channel = _build_accessible_channel(client, channel)
            if accessible_channel.can_read:
                accessible_channels.append(accessible_channel)

    return accessible_channels


async def can_read_channel(client: Any, channel_id: str) -> bool:
    """Quick check: can the bot read messages in this channel?"""
    channel_permissions = await check_channel_permissions(client, channel_id)
    if channel_permissions is None:
        return False
    return channel_permissions.can_read and channel_permissions.can_read_history
