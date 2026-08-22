"""Validated Discord REST v10 reaction request builders for the platform plugin."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import quote


MAX_REACTION_PAGE = 100
_CUSTOM_EMOJI_RE = re.compile(r"^[A-Za-z0-9_]{2,32}:[1-9][0-9]{14,19}$")
_KEYCAP_EMOJI_RE = re.compile(r"^[0-9#*]\ufe0f?\u20e3$")
_FORBIDDEN = re.compile(r"[\s/@#]|@everyone|@here")
_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]*$")


class ReactionError(ValueError):
    """Raised when a Discord reaction request is invalid."""


def validate_emoji(emoji: str) -> str:
    """Return a validated Unicode or ``name:id`` Discord reaction emoji."""
    if not isinstance(emoji, str) or not emoji:
        raise ReactionError("emoji must be a non-empty string")
    if emoji.strip() != emoji:
        raise ReactionError("emoji must not have surrounding whitespace")
    if _CUSTOM_EMOJI_RE.fullmatch(emoji) or _KEYCAP_EMOJI_RE.fullmatch(emoji):
        return emoji
    if _FORBIDDEN.search(emoji):
        raise ReactionError(f"emoji {emoji!r} contains forbidden characters")

    codepoints = list(emoji)
    if any(char.isascii() and char.isalnum() for char in codepoints):
        raise ReactionError(
            f"{emoji!r} is not a recognized emoji; use a Unicode emoji or custom 'name:id'"
        )
    if any(unicodedata.category(char).startswith("S") for char in codepoints):
        return emoji
    raise ReactionError(f"{emoji!r} is not a recognized emoji")


def encode_emoji_path(emoji: str) -> str:
    """Percent-encode a validated emoji for a Discord REST URL path."""
    return quote(validate_emoji(emoji), safe="")


def _base_path(channel_id: str, message_id: str) -> str:
    if not _SNOWFLAKE_RE.fullmatch(str(channel_id)):
        raise ReactionError(f"channel_id must be a snowflake, got {channel_id!r}")
    if not _SNOWFLAKE_RE.fullmatch(str(message_id)):
        raise ReactionError(f"message_id must be a snowflake, got {message_id!r}")
    return f"/channels/{channel_id}/messages/{message_id}"


def _request(method: str, path: str, *, query: dict[str, str] | None = None) -> dict[str, Any]:
    return {"method": method, "path": path, "payload": None, "query": query or {}}


def add_reaction_request(channel_id: str, message_id: str, emoji: str) -> dict[str, Any]:
    """Build the request to add the bot's own reaction."""
    return _request("PUT", f"{_base_path(channel_id, message_id)}/reactions/{encode_emoji_path(emoji)}/@me")


def remove_own_reaction_request(channel_id: str, message_id: str, emoji: str) -> dict[str, Any]:
    """Build the request to remove the bot's own reaction."""
    return _request("DELETE", f"{_base_path(channel_id, message_id)}/reactions/{encode_emoji_path(emoji)}/@me")


def remove_user_reaction_request(
    channel_id: str, message_id: str, emoji: str, user_id: str
) -> dict[str, Any]:
    """Build the request to remove another user's reaction."""
    if not _SNOWFLAKE_RE.fullmatch(str(user_id)):
        raise ReactionError(f"user_id must be a snowflake, got {user_id!r}")
    return _request(
        "DELETE",
        f"{_base_path(channel_id, message_id)}/reactions/{encode_emoji_path(emoji)}/{user_id}",
    )


def remove_all_reactions_request(channel_id: str, message_id: str) -> dict[str, Any]:
    """Build the request to remove all reactions from a message."""
    return _request("DELETE", f"{_base_path(channel_id, message_id)}/reactions")


def list_reactions_request(
    channel_id: str, message_id: str, emoji: str, *, limit: int = 25
) -> dict[str, Any]:
    """Build the request to list users reacting with an emoji."""
    if isinstance(limit, bool):
        raise ReactionError("limit must be an integer")
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ReactionError("limit must be an integer") from exc
    return _request(
        "GET",
        f"{_base_path(channel_id, message_id)}/reactions/{encode_emoji_path(emoji)}",
        query={"limit": str(max(1, min(limit, MAX_REACTION_PAGE)))},
    )
