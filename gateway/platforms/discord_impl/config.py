"""Discord configuration and policy helpers."""

from __future__ import annotations

import os


def clean_discord_id(entry: str) -> str:
    """Normalize a Discord user ID or username entry."""
    entry = entry.strip()
    if entry.startswith("<@") and entry.endswith(">"):
        entry = entry.lstrip("<@!").rstrip(">")
    if entry.lower().startswith("user:"):
        entry = entry[5:]
    return entry.strip()


def parse_allowed_users(env_val: str) -> set[str]:
    """Parse ``DISCORD_ALLOWED_USERS`` into cleaned entries."""
    return {
        clean_discord_id(entry)
        for entry in env_val.split(",")
        if entry.strip()
    }


def get_bot_filter_policy() -> str:
    """Return the ``DISCORD_ALLOW_BOTS`` policy."""
    return os.getenv("DISCORD_ALLOW_BOTS", "none").lower().strip()


def get_free_response_channels() -> set[str]:
    """Return the free-response channel ID set."""
    return {
        channel_id.strip()
        for channel_id in os.getenv("DISCORD_FREE_RESPONSE_CHANNELS", "").split(",")
        if channel_id.strip()
    }


def is_mention_required() -> bool:
    """Return whether Discord server messages require an explicit mention."""
    return os.getenv("DISCORD_REQUIRE_MENTION", "true").lower() not in ("false", "0", "no")


def is_auto_thread_enabled() -> bool:
    """Return whether Discord auto-threading is enabled."""
    return os.getenv("DISCORD_AUTO_THREAD", "true").lower() in ("true", "1", "yes")
