"""Discord history fetch and search.

Bounded read-only channel/thread history retrieval with permission checks.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

from gateway.platforms.discord_impl.delivery import resolve_channel
from gateway.platforms.discord_impl import permissions as discord_permissions

SEARCH_SCAN_LIMIT = 500
SEARCH_PAGE_SIZE = 100


@dataclass
class HistoryMessage:
    """Lightweight representation of a Discord message from history."""

    id: str
    author_id: str
    author_name: str
    content: str
    timestamp: str
    is_bot: bool
    attachments: list[str]
    reply_to: Optional[str] = None


def _clamp_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 100
    return max(1, min(500, value))


def _history_anchor(message_id: Optional[str]) -> Optional[Any]:
    if message_id is None:
        return None
    try:
        anchor_id: Any = int(message_id)
    except (TypeError, ValueError):
        anchor_id = str(message_id)
    return SimpleNamespace(id=anchor_id)


async def _collect_history(history_iter: Any) -> list[Any]:
    if history_iter is None:
        return []

    if inspect.isawaitable(history_iter):
        history_iter = await history_iter

    if hasattr(history_iter, "__aiter__"):
        return [message async for message in history_iter]

    return list(history_iter)


def _timestamp_to_iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _author_name(message: Any) -> str:
    author = getattr(message, "author", None)
    if author is None:
        return "unknown"
    return (
        getattr(author, "display_name", None)
        or getattr(author, "name", None)
        or str(getattr(author, "id", "unknown"))
    )


def _attachments(message: Any) -> list[str]:
    urls: list[str] = []
    for attachment in getattr(message, "attachments", []) or []:
        url = getattr(attachment, "url", None)
        if url:
            urls.append(str(url))
    return urls


def _reply_to(message: Any) -> Optional[str]:
    reference = getattr(message, "reference", None)
    if reference is None:
        return None

    reference_id = getattr(reference, "message_id", None)
    if reference_id is None:
        resolved = getattr(reference, "resolved", None)
        reference_id = getattr(resolved, "id", None)
    if reference_id is None:
        return None
    return str(reference_id)


def _to_history_message(message: Any) -> HistoryMessage:
    author = getattr(message, "author", None)
    return HistoryMessage(
        id=str(getattr(message, "id", "")),
        author_id=str(getattr(author, "id", "")),
        author_name=_author_name(message),
        content=str(getattr(message, "content", "") or ""),
        timestamp=_timestamp_to_iso(getattr(message, "created_at", None)),
        is_bot=bool(getattr(author, "bot", False)),
        attachments=_attachments(message),
        reply_to=_reply_to(message),
    )


async def fetch_history(
    client: Any,
    channel_id: str,
    limit: int = 100,
    before: Optional[str] = None,
    after: Optional[str] = None,
) -> list[HistoryMessage]:
    """Fetch bounded message history from a Discord channel."""
    channel = await resolve_channel(client, channel_id)
    if channel is None:
        return []

    channel_permissions = discord_permissions._build_channel_permissions(client, channel)
    if not (channel_permissions.can_read and channel_permissions.can_read_history):
        return []

    history_iter = channel.history(
        limit=_clamp_limit(limit),
        before=_history_anchor(before),
        after=_history_anchor(after),
    )
    return [_to_history_message(message) for message in await _collect_history(history_iter)]


async def search_history(
    client: Any,
    channel_id: str,
    query: str,
    limit: int = 50,
    author_id: Optional[str] = None,
) -> list[HistoryMessage]:
    """Search recent channel history with bounded text matching."""
    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return []

    clamped_limit = _clamp_limit(limit)
    filtered_messages: list[HistoryMessage] = []
    author_filter = str(author_id) if author_id is not None else None
    remaining_scan = SEARCH_SCAN_LIMIT
    before: Optional[str] = None

    while remaining_scan > 0 and len(filtered_messages) < clamped_limit:
        page_size = min(SEARCH_PAGE_SIZE, remaining_scan)
        messages = await fetch_history(
            client,
            channel_id,
            limit=page_size,
            before=before,
        )
        if not messages:
            break

        remaining_scan -= len(messages)

        for message in messages:
            if author_filter is not None and message.author_id != author_filter:
                continue
            if normalized_query not in message.content.lower():
                continue
            filtered_messages.append(message)
            if len(filtered_messages) >= clamped_limit:
                break

        if len(messages) < page_size:
            break
        before = messages[-1].id

    return filtered_messages
