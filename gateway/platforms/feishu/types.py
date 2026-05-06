"""Shared types and helpers for the Feishu adapter package.

Lives in its own module to break the runtime import cycle between
``adapter.py`` and ``events_mapping.py`` (the former imports from the
latter at module load; the latter previously lazy-imported back from
adapter at call time to reach a dataclass and two helpers).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FeishuMentionRef:
    """Hermes-side representation of a single @mention in a Feishu message."""

    name: str = ""
    open_id: str = ""
    is_all: bool = False
    is_self: bool = False


def map_chat_type(raw_chat_type: str) -> str:
    """Project a Feishu/Lark chat-type string to Hermes' canonical taxonomy.

    Returns one of: ``"dm"``, ``"group"``, ``"forum"``.
    """
    normalized = (raw_chat_type or "").strip().lower()
    if normalized == "p2p":
        return "dm"
    if "topic" in normalized or "thread" in normalized or "forum" in normalized:
        return "forum"
    if normalized == "group":
        return "group"
    return "dm"


def resolve_source_chat_type(
    *, chat_info: Dict[str, Any], event_chat_type: str
) -> str:
    """Compute the SessionSource ``chat_type`` from a chat_info dict + the
    raw chat_type carried on the event.

    chat_info["type"] takes precedence when it's already in {group, forum};
    otherwise falls back to the event's chat_type ("p2p" -> "dm", else "group").
    """
    resolved = str(chat_info.get("type") or "").strip().lower()
    if resolved in {"group", "forum"}:
        return resolved
    if event_chat_type == "p2p":
        return "dm"
    return "group"
