"""Small Discord interaction context helpers mixed into the adapter facade."""

from __future__ import annotations

from typing import Any


class DiscordSlashInteractionContextMixin:
    def format_message(self, content: str) -> str:
        """Format for Discord: GFM tables become bullet lists."""
        from plugins.platforms.discord.adapter import convert_table_to_bullets

        if not content:
            return content
        return convert_table_to_bullets(content)

    def _interaction_guild_id(self, interaction: Any) -> str | None:
        """Resolve the guild ID of a slash interaction (mirrors the message path)."""
        guild_id = getattr(interaction, "guild_id", None)
        if guild_id is None:
            guild_id = getattr(getattr(getattr(interaction, "channel", None), "guild", None), "id", None)
        return str(guild_id) if guild_id else None
