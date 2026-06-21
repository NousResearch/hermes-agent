"""Discord startup-context helpers for gateway sessions."""

from __future__ import annotations

import os
from typing import Any, Protocol


class StartupContextModule(Protocol):
    def external_context_row(self, **kwargs: Any) -> dict[str, Any]: ...
    def is_noise(self, row: dict[str, Any]) -> bool: ...


def discord_context_channel_id(event: Any, source: Any) -> str:
    raw_channel = getattr(getattr(event, "raw_message", None), "channel", None)
    raw_channel_id = str(getattr(raw_channel, "id", "") or "")
    source_chat_id = str(getattr(source, "chat_id", "") or "")
    parent_chat_id = str(getattr(source, "parent_chat_id", "") or "")
    if (
        getattr(source, "thread_id", None)
        and parent_chat_id
        and raw_channel_id == parent_chat_id
        and raw_channel_id != source_chat_id
    ):
        return parent_chat_id
    return source_chat_id or parent_chat_id


def discord_api_timeout() -> float:
    try:
        return max(0.25, min(float(os.getenv("HERMES_DISCORD_CONTEXT_API_TIMEOUT", "2.0")), 8.0))
    except (TypeError, ValueError):
        return 2.0


def discord_api_context_row(module: StartupContextModule, message: Any) -> dict[str, Any] | None:
    content = (getattr(message, "content", None) or "").strip()
    if not content:
        attachments = getattr(message, "attachments", None) or []
        if attachments:
            first = attachments[0]
            content = f"(attachment: {getattr(first, 'filename', 'file')})"
    author_obj = getattr(message, "author", None)
    author = (
        getattr(author_obj, "display_name", None)
        or getattr(author_obj, "name", None)
        or "someone"
    )
    reference = getattr(message, "reference", None)
    reply_to_id = getattr(reference, "message_id", None) if reference else None
    timestamp = getattr(message, "created_at", None)
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()
    row = module.external_context_row(
        message_id=getattr(message, "id", ""),
        channel_id=getattr(getattr(message, "channel", None), "id", ""),
        author_id=getattr(author_obj, "id", ""),
        author=author,
        content=content,
        timestamp=str(timestamp or ""),
        reply_to_id=reply_to_id,
        is_bot=bool(getattr(author_obj, "bot", False)),
        username=getattr(author_obj, "name", None),
    )
    if module.is_noise(row):
        return None
    return row


def render_reply_chain_payload(payload: dict[str, Any]) -> str:
    chain = payload.get("reply_chain") or []
    if not chain:
        return ""
    lines = [
        "## Discord Reply Chain",
        "The user is replying to this chain. Use it to disambiguate the current request.",
    ]
    for msg in chain:
        lines.append(
            f"- [{msg.get('timestamp')}] {msg.get('author')} ({msg.get('id')}): {msg.get('content')}"
        )
    if payload.get("reply_chain_truncated"):
        lines.append("- [older reply-chain messages omitted]")
    return "\n".join(lines)
