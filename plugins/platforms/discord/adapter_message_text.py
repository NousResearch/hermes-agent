"""Readable text of a Discord message for agent context (inbound, reply quote, history backfill).

Discord leaves ``content`` empty in two common shapes: a forward keeps its text and files in
``message_snapshots``, and bot alerts / error relays carry their whole payload in embeds.
Every reader of another message's text goes through :func:`message_text` so those shapes
read the same on the trigger message, the replied-to message and the backfilled history.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Callable, Optional

FORWARD_PREFIX = "[Forwarded] "


def embed_text(obj: Any) -> str:
    """Flatten a message's (or snapshot's) embeds: title, description, fields, footer."""
    parts: list[str] = []
    for embed in getattr(obj, "embeds", None) or []:
        parts.extend(str(v) for v in (getattr(embed, "title", None), getattr(embed, "description", None)) if v)
        for field in getattr(embed, "fields", None) or []:
            line = f"{getattr(field, 'name', None) or ''}: {getattr(field, 'value', None) or ''}".strip(": ").strip()
            if line:
                parts.append(line)
        footer = getattr(getattr(embed, "footer", None), "text", None)
        if footer:
            parts.append(str(footer))
    return "\n".join(parts).strip()


def snapshot_text(msg: Any) -> str:
    """Text of a forward's snapshots (content, else embeds), "" for a message that is not a forward."""
    parts: list[str] = []
    for snap in getattr(msg, "message_snapshots", None) or []:
        text = (getattr(snap, "content", None) or "").strip() or embed_text(snap)
        if text:
            parts.append(text)
    return "\n".join(parts)


def snapshot_attachments(msg: Any) -> list:
    """Files a forward carries inside its snapshots (the wrapper's own ``attachments`` is empty)."""
    return [a for snap in getattr(msg, "message_snapshots", None) or [] for a in getattr(snap, "attachments", None) or []]


def message_text(msg: Any, *, clean: bool = True) -> str:
    """The message's own text, else its forwarded text (marked), else its embed text."""
    own = (getattr(msg, "clean_content", None) if clean else None) or getattr(msg, "content", None) or ""
    if own.strip():
        return own
    forwarded = snapshot_text(msg)
    return f"{FORWARD_PREFIX}{forwarded}" if forwarded else embed_text(msg)


def is_forward_reference(reference: Any) -> bool:
    """A forward also fills ``message.reference`` (type forward) with the ORIGINAL message,
    usually in another channel; it is not a reply and must not anchor reply context."""
    return getattr(getattr(reference, "type", None), "name", "") == "forward"


async def resolve_reference(message: Any) -> Optional[Any]:
    """The replied-to message: the gateway-resolved copy, else one REST fetch (same channel for
    a reply), else None. Forwards and deleted targets give None."""
    reference = getattr(message, "reference", None)
    if reference is None or is_forward_reference(reference) or not getattr(reference, "message_id", None):
        return None
    resolved = getattr(reference, "resolved", None)
    if resolved is not None and getattr(resolved, "id", None) is not None:
        return resolved
    fetch = getattr(getattr(message, "channel", None), "fetch_message", None)
    if fetch is None:
        return None
    with suppress(Exception):  # 404 (deleted), 403 (no history permission), transport: no quote
        return await fetch(reference.message_id)
    return None


async def thread_starter(thread: Any) -> Optional[Any]:
    """The channel message a thread was opened from, else None (standalone thread, no access).

    That starter lives in the PARENT channel under the thread's id; the thread's own history
    holds only a system pointer, so without it a thread opened from an alert backfills empty.
    """
    starter = getattr(thread, "starter_message", None)
    if starter is not None:
        return starter
    fetch = getattr(getattr(thread, "parent", None), "fetch_message", None)
    if fetch is None:
        return None
    with suppress(Exception):  # 404 for a standalone thread, 403 without history access
        return await fetch(thread.id)
    return None


async def thread_starter_rows(thread: Any, keep: Callable[[Any], Optional[str]], *, is_thread: bool,
                              warm: bool) -> list[tuple[str, str]]:
    """Backfill row for the thread starter on a COLD thread scan, formatted by the caller's ``keep``
    (same bot/trust filters as any history line); none once the bot already spoke in the thread."""
    if not is_thread or warm:
        return []
    starter = await thread_starter(thread)
    line = keep(starter) if starter is not None else None
    return [("", f"(thread starter) {line}")] if line else []
