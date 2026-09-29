"""Read earlier Matrix thread messages for a new gateway session."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection
from urllib.parse import quote

from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
    _effective_content,
    _label_body,
    _own_text,
)


logger = logging.getLogger(__name__)

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


class UndecryptableEvent(Exception):
    """An encrypted history event could not be decrypted. ``str()`` gives the reason."""


@dataclass(frozen=True)
class HistoryMessage:
    msgtype: str
    text: str
    content: dict


async def decrypt_history_event(client: Any, raw: dict) -> Any:
    if raw.get("type") != "m.room.encrypted":
        return raw
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        raise UndecryptableEvent("missing decryption keys")
    try:
        from mautrix.types import Event

        return await asyncio.wait_for(crypto.decrypt_megolm_event(Event.deserialize(raw)), timeout=10.0)
    except Exception as exc:
        logger.debug("Matrix: could not decrypt history event %s: %s", raw.get("event_id"), exc)
        reason = "missing decryption keys" if type(exc).__name__ == "SessionNotFound" else "decryption failed"
        raise UndecryptableEvent(reason) from exc


def history_message(event: Any) -> HistoryMessage | None:
    content, edited = _effective_content(event)
    body = content.get("body")
    if not isinstance(body, str):
        return None
    body = body.strip()
    if edited and body.startswith("* "):
        body = body[2:].strip()
    msgtype = str(content.get("msgtype") or "")
    text = _label_body(msgtype, _own_text(body, content))
    return HistoryMessage(msgtype, text, content) if text else None


async def fetch_thread_entries(
    client: Any,
    cache: MatrixEventContextCache,
    room_id: str,
    thread_id: str,
    *,
    limit: int,
    exclude_event_ids: Collection[str] = (),
) -> list[MatrixEventContext]:
    if client is None or limit <= 0 or not thread_id:
        return []

    path = (
        f"/_matrix/client/v1/rooms/{quote(room_id, safe='')}"
        f"/relations/{quote(thread_id, safe='')}/m.thread"
    )
    try:
        response = await asyncio.wait_for(
            client.api.request(Method.GET, path, query_params={"dir": "b", "limit": str(limit)}),
            timeout=10.0,
        )
    except Exception as exc:
        logger.debug("Matrix: could not fetch thread %s in %s: %s", thread_id, room_id, exc)
        return []

    entries: list[MatrixEventContext] = []
    if thread_id not in exclude_event_ids:
        root = await cache.resolve(client, room_id, thread_id)
        if root is not None:
            entries.append(root)

    chunk = response.get("chunk") if isinstance(response, dict) else None
    if not isinstance(chunk, list):
        return entries

    for raw in reversed(chunk[:limit]):
        if not isinstance(raw, dict):
            continue
        event_id = raw.get("event_id")
        if not isinstance(event_id, str) or event_id in exclude_event_ids:
            continue

        try:
            message = history_message(await decrypt_history_event(client, raw))
        except UndecryptableEvent:
            continue
        if message is None:
            continue
        sender = str(raw.get("sender") or "")
        entry = MatrixEventContext(sender, message.text, is_image=message.msgtype == "m.image")
        stored = cache.store(room_id, event_id, entry)
        if stored is not None:
            entries.append(stored)

    return entries
