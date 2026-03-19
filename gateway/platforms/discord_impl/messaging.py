"""Discord message-operation helpers."""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Callable, Optional

from gateway.platforms.base import SendResult
from gateway.platforms.discord_impl.delivery import resolve_channel


def normalize_edit_content(
    content: str,
    *,
    format_message: Optional[Callable[[str], str]] = None,
    max_message_length: int = 2000,
) -> str:
    """Format and truncate edited message content to Discord limits."""
    formatter = format_message or (lambda value: value)
    formatted = formatter(content)
    if len(formatted) > max_message_length:
        return formatted[: max_message_length - 3] + "..."
    return formatted


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


def _serialize_message(message: Any) -> dict[str, Any]:
    author = getattr(message, "author", None)
    attachments = [
        str(getattr(attachment, "url"))
        for attachment in (getattr(message, "attachments", []) or [])
        if getattr(attachment, "url", None)
    ]
    reference = getattr(message, "reference", None)
    reply_to = getattr(reference, "message_id", None) if reference is not None else None
    return {
        "id": str(getattr(message, "id", "")),
        "author_id": str(getattr(author, "id", "")),
        "author_name": _author_name(message),
        "content": str(getattr(message, "content", "") or ""),
        "timestamp": _timestamp_to_iso(getattr(message, "created_at", None)),
        "is_bot": bool(getattr(author, "bot", False)),
        "attachments": attachments,
        "reply_to": str(reply_to) if reply_to is not None else None,
    }


async def _collect_items(iterable: Any) -> list[Any]:
    if iterable is None:
        return []
    if inspect.isawaitable(iterable):
        iterable = await iterable
    if hasattr(iterable, "__aiter__"):
        return [item async for item in iterable]
    return list(iterable)


def _clamp_limit(limit: int, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _thread_dict(thread: Any) -> dict[str, Any]:
    parent = getattr(thread, "parent", None)
    guild = getattr(thread, "guild", None) or getattr(parent, "guild", None)
    return {
        "id": str(getattr(thread, "id", "")),
        "name": getattr(thread, "name", "") or "",
        "parent_id": str(getattr(parent, "id", "")) if getattr(parent, "id", None) is not None else None,
        "parent_name": getattr(parent, "name", None),
        "guild_id": str(getattr(guild, "id", "")) if getattr(guild, "id", None) is not None else None,
        "guild_name": getattr(guild, "name", None),
        "archived": bool(getattr(thread, "archived", False)),
        "locked": bool(getattr(thread, "locked", False)),
        "message_count": getattr(thread, "message_count", None),
        "member_count": getattr(thread, "member_count", None),
    }


async def fetch_channel_message(
    client: Any,
    chat_id: str,
    message_id: str,
) -> tuple[Any | None, Any | None, SendResult | None]:
    """Resolve a Discord channel and fetch a specific message from it."""
    channel = await resolve_channel(client, chat_id)
    if not channel:
        return None, None, SendResult(success=False, error=f"Channel {chat_id} not found")

    try:
        message = await channel.fetch_message(int(message_id))
    except Exception as exc:
        return channel, None, SendResult(success=False, error=str(exc))

    return channel, message, None


async def edit_message(
    client: Any,
    chat_id: str,
    message_id: str,
    content: str,
    *,
    format_message: Optional[Callable[[str], str]] = None,
    max_message_length: int = 2000,
) -> SendResult:
    """Edit a previously sent Discord message."""
    if not client:
        return SendResult(success=False, error="Not connected")

    _channel, message, error = await fetch_channel_message(client, chat_id, message_id)
    if error is not None:
        return error

    formatted = normalize_edit_content(
        content,
        format_message=format_message,
        max_message_length=max_message_length,
    )

    try:
        await message.edit(content=formatted)
    except Exception as exc:
        return SendResult(success=False, error=str(exc))

    return SendResult(success=True, message_id=message_id)


async def list_threads(
    client: Any,
    channel_id: str,
    *,
    include_archived: bool = False,
    limit: int = 100,
    before: Any = None,
    private: bool = False,
    joined: bool = False,
) -> list[dict[str, Any]]:
    """List active and optionally archived threads for a channel."""
    channel = await resolve_channel(client, channel_id)
    if channel is None:
        return []

    active_threads = [_thread_dict(thread) for thread in (getattr(channel, "threads", []) or [])]
    if not include_archived:
        return active_threads

    archived_threads_fn = getattr(channel, "archived_threads", None)
    if not callable(archived_threads_fn):
        return active_threads

    archived_threads = await _collect_items(
        archived_threads_fn(
            private=private,
            joined=joined,
            limit=_clamp_limit(limit, default=100, minimum=1, maximum=100),
            before=before,
        )
    )
    return active_threads + [_thread_dict(thread) for thread in archived_threads]


async def reply_in_thread(
    client: Any,
    thread_id: str,
    content: str,
    *,
    reply_to: Optional[str] = None,
    format_message: Optional[Callable[[str], str]] = None,
    truncate_message: Optional[Callable[[str, int], list[str]]] = None,
    max_message_length: int = 2000,
    send_text_message: Optional[Callable[..., Any]] = None,
) -> SendResult:
    """Send a message to a Discord thread after validating the target."""
    if not client:
        return SendResult(success=False, error="Not connected")

    channel = await resolve_channel(client, thread_id)
    if channel is None:
        return SendResult(success=False, error=f"Channel {thread_id} not found")

    if getattr(channel, "parent", None) is None:
        return SendResult(success=False, error=f"Channel {thread_id} is not a thread")

    formatter = format_message or (lambda value: value)
    truncater = truncate_message or (lambda value, _max_len: [value])
    sender = send_text_message
    if sender is None:
        from gateway.platforms.discord_impl.delivery import send_text_message as default_send_text_message
        sender = default_send_text_message

    formatted = formatter(content)
    chunks = truncater(formatted, max_message_length)
    message_ids: list[str] = []
    reference = None

    if reply_to:
        try:
            reference = await channel.fetch_message(int(reply_to))
        except Exception:
            reference = None

    try:
        for index, chunk in enumerate(chunks):
            message = await sender(
                channel,
                chunk,
                reference=reference if index == 0 else None,
            )
            message_ids.append(str(message.id))
    except Exception as exc:
        return SendResult(success=False, error=str(exc))

    return SendResult(
        success=True,
        message_id=message_ids[0] if message_ids else None,
        raw_response={"message_ids": message_ids},
    )


async def list_pins(
    client: Any,
    channel_id: str,
    *,
    limit: int = 50,
    before: Any = None,
    oldest_first: bool = False,
) -> list[dict[str, Any]]:
    """List pinned messages for a channel or thread."""
    channel = await resolve_channel(client, channel_id)
    if channel is None:
        return []

    pins_fn = getattr(channel, "pins", None)
    if not callable(pins_fn):
        return []

    pins = await _collect_items(
        pins_fn(
            limit=_clamp_limit(limit, default=50, minimum=1, maximum=50),
            before=before,
            oldest_first=oldest_first,
        )
    )
    return [_serialize_message(message) for message in pins]


async def pin_message(
    client: Any,
    chat_id: str,
    message_id: str,
    *,
    reason: Optional[str] = None,
) -> SendResult:
    """Pin a Discord message."""
    if not client:
        return SendResult(success=False, error="Not connected")

    _channel, message, error = await fetch_channel_message(client, chat_id, message_id)
    if error is not None:
        return error

    try:
        await message.pin(reason=reason)
    except Exception as exc:
        return SendResult(success=False, error=str(exc))

    return SendResult(success=True, message_id=message_id)


async def unpin_message(
    client: Any,
    chat_id: str,
    message_id: str,
    *,
    reason: Optional[str] = None,
) -> SendResult:
    """Unpin a Discord message."""
    if not client:
        return SendResult(success=False, error="Not connected")

    _channel, message, error = await fetch_channel_message(client, chat_id, message_id)
    if error is not None:
        return error

    try:
        await message.unpin(reason=reason)
    except Exception as exc:
        return SendResult(success=False, error=str(exc))

    return SendResult(success=True, message_id=message_id)
