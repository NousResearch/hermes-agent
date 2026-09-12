"""Discord channel metadata and conversation-context helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .. import adapter as _adapter

logger = _adapter.logger
discord = _adapter.discord


def _multiplex_active() -> bool:
    return _adapter._multiplex_active()


class ChannelContextMixin:
    """Resolve channel metadata and normalize Discord channel context."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a Discord channel."""
        if not self._client:
            return {"name": "Unknown", "type": "dm"}
        try:
            channel = await self._resolve_channel(chat_id)
            if not channel:
                return {"name": str(chat_id), "type": "dm"}
            if isinstance(channel, discord.DMChannel):
                chat_type = "dm"
                name = channel.recipient.name if channel.recipient else str(chat_id)
            elif isinstance(channel, discord.Thread):
                chat_type = "thread"
                name = channel.name
            elif isinstance(channel, discord.TextChannel):
                chat_type = "channel"
                name = f"#{channel.name}"
                if channel.guild:
                    name = f"{channel.guild.name} / {name}"
            else:
                chat_type = "channel"
                name = getattr(channel, "name", str(chat_id))
            return {
                "name": name, "type": chat_type,
                "guild_id": str(channel.guild.id) if hasattr(channel, "guild") and channel.guild else None,
                "guild_name": channel.guild.name if hasattr(channel, "guild") and channel.guild else None,
            }
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to get chat info for %s: %s", self.name, chat_id, e, exc_info=True)
            return {"name": str(chat_id), "type": "dm", "error": str(e)}

    async def _resolve_allowed_usernames(self) -> None:
        """Resolve username/display-name entries in DISCORD_ALLOWED_USERS to numeric IDs."""
        if not self._allowed_user_ids or not self._client:
            return
        numeric_ids = set()
        to_resolve = set()
        for entry in self._allowed_user_ids:
            if entry.isdigit():
                numeric_ids.add(entry)
            elif entry == "*":
                # Keep the "*" wildcard verbatim; it can't resolve and would be silently dropped.
                numeric_ids.add(entry)
            else:
                to_resolve.add(entry.lower())
        if not to_resolve:
            return
        print(f"[{self.name}] Resolving {len(to_resolve)} username(s): {', '.join(to_resolve)}")
        resolved_count = 0
        for guild in self._client.guilds:
            # Fetch full member list (requires members intent)
            try:
                members = guild.members
                if len(members) < guild.member_count:
                    members = [m async for m in guild.fetch_members(limit=None)]
            except Exception as e:
                logger.warning("Failed to fetch members for guild %s: %s", guild.name, e)
                continue
            for member in members:
                name_lower = member.name.lower()
                display_lower = member.display_name.lower()
                global_lower = (member.global_name or "").lower()
                matched = name_lower in to_resolve or display_lower in to_resolve or global_lower in to_resolve
                if matched:
                    uid = str(member.id)
                    numeric_ids.add(uid)
                    resolved_count += 1
                    matched_name = name_lower if name_lower in to_resolve else (
                        display_lower if display_lower in to_resolve else global_lower
                    )
                    to_resolve.discard(matched_name)
                    print(f"[{self.name}] Resolved '{matched_name}' -> {uid} ({member.name}#{member.discriminator})")
            if not to_resolve:
                break
        if to_resolve:
            print(f"[{self.name}] Could not resolve usernames: {', '.join(to_resolve)}")
        # Adapter-local: under multiplex_profiles os.environ writes would clobber other profiles.
        # Update the internal set. Keep the resolved IDs adapter-local first: under multiplex_profiles,
        # writing os.environ here would clobber every OTHER profile's DISCORD_ALLOWED_USERS after this
        # adapter's on_ready — an unguarded runtime mutation of process-global state (issue #72348). Refresh
        # this adapter's own snapshot instead.
        self._allowed_user_ids = numeric_ids
        snap = getattr(self, "_gate_env_snapshot", None)
        if snap is not None:
            snap["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if not _multiplex_active():
            # Single-profile: legacy env rewrite so gateway env-based auth sees numeric IDs.
            os.environ["DISCORD_ALLOWED_USERS"] = ",".join(sorted(numeric_ids))
        if resolved_count:
            print(f"[{self.name}] Updated DISCORD_ALLOWED_USERS with {resolved_count} resolved ID(s)")


    def _get_parent_channel_id(self, channel: Any) -> Optional[str]:
        """Return the parent channel ID for a Discord thread-like channel, if present."""
        parent = getattr(channel, "parent", None)
        if parent is not None and getattr(parent, "id", None) is not None:
            return str(parent.id)
        parent_id = getattr(channel, "parent_id", None)
        if parent_id is not None:
            return str(parent_id)
        return None

    def _is_forum_parent(self, channel: Any) -> bool:
        """Best-effort check for whether a Discord channel is a forum channel."""
        if channel is None:
            return False
        forum_cls = getattr(discord, "ForumChannel", None)
        if forum_cls and isinstance(channel, forum_cls):
            return True
        channel_type = getattr(channel, "type", None)
        if channel_type is not None:
            type_value = getattr(channel_type, "value", channel_type)
            if type_value == 15:
                return True
        return False

    def _get_effective_topic(self, channel: Any, is_thread: bool = False) -> Optional[str]:
        """Return the channel topic, falling back to the parent forum's topic for forum threads."""
        topic = getattr(channel, "topic", None)
        if not topic and is_thread:
            parent = getattr(channel, "parent", None)
            if parent and self._is_forum_parent(parent):
                topic = getattr(parent, "topic", None)
        return topic

    def _format_thread_chat_name(self, thread: Any) -> str:
        """Build a readable chat name for thread-like Discord channels, including forum context when available."""
        thread_name = getattr(thread, "name", None) or str(getattr(thread, "id", "thread"))
        parent = getattr(thread, "parent", None)
        guild = getattr(thread, "guild", None) or getattr(parent, "guild", None)
        guild_name = getattr(guild, "name", None)
        parent_name = getattr(parent, "name", None)
        if self._is_forum_parent(parent) and guild_name and parent_name:
            return f"{guild_name} / {parent_name} / {thread_name}"
        if parent_name and guild_name:
            return f"{guild_name} / #{parent_name} / {thread_name}"
        if parent_name:
            return f"{parent_name} / {thread_name}"
        return thread_name

    # ------------------------------------------------------------------
