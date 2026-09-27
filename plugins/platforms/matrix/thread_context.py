"""Read earlier Matrix thread messages for a new gateway session."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
    _effective_content,
    _label_body,
    _own_text,
)
from plugins.platforms.matrix.relations import MatrixRelation


logger = logging.getLogger(__name__)

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


async def _decrypt_thread_event(client: Any, raw: dict) -> Any | None:
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        return None
    try:
        from mautrix.types import Event

        event = Event.deserialize(raw)
        return await asyncio.wait_for(crypto.decrypt_megolm_event(event), timeout=10.0)
    except Exception as exc:
        logger.debug("Matrix: could not decrypt thread event %s: %s", raw.get("event_id"), exc)
        return None


async def history_entry(client: Any, raw: dict) -> tuple[MatrixEventContext, dict] | None:
    if raw.get("type", "m.room.message") not in {"m.room.message", "m.room.encrypted"}:
        return None
    if raw.get("type") == "m.room.encrypted":
        event = await _decrypt_thread_event(client, raw)
        if event is None:
            return None
    else:
        event = raw

    content, edited = _effective_content(event)
    body = content.get("body")
    if not isinstance(body, str):
        return None
    body = body.strip()
    if edited and body.startswith("* "):
        body = body[2:].strip()
    text = _label_body(str(content.get("msgtype") or ""), _own_text(body))
    if not text:
        return None
    sender = str(raw.get("sender") or "")
    return MatrixEventContext(sender, text, is_image=content.get("msgtype") == "m.image"), content


async def fetch_thread_entries(
    client: Any,
    cache: MatrixEventContextCache,
    room_id: str,
    thread_id: str,
    *,
    limit: int,
    exclude_event_id: str | None = None,
    before_ts: float | None = None,
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
        if event_id == exclude_event_id or not isinstance(event_id, str):
            continue
        if before_ts is not None:
            timestamp = raw.get("origin_server_ts")
            if not isinstance(timestamp, (int, float)) or timestamp >= before_ts * 1000:
                continue
        parsed = await history_entry(client, raw)
        if parsed is None:
            continue
        entry, content = parsed
        if MatrixRelation.from_content(content.get("m.relates_to")).thread_root != thread_id:
            continue
        stored = cache.store(room_id, event_id, entry)
        if stored is not None:
            entries.append(stored)

    return entries
