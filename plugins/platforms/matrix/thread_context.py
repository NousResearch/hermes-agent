"""Read earlier Matrix thread messages for a new gateway session."""

from __future__ import annotations

import asyncio
import logging
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


async def _decrypt_thread_event(client: Any, raw: dict) -> Any | None:
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        return None
    try:
        from mautrix.types import EncryptedEvent, JSON

        event = EncryptedEvent.deserialize(JSON(raw))
        return await asyncio.wait_for(crypto.decrypt_megolm_event(event), timeout=10.0)
    except Exception as exc:
        logger.debug("Matrix: could not decrypt thread event %s: %s", raw.get("event_id"), exc)
        return None


async def fetch_thread_entries(
    client: Any,
    cache: MatrixEventContextCache,
    room_id: str,
    thread_id: str,
    *,
    limit: int,
    exclude_event_ids: Collection[str | None] = (),
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

        event: Any = raw
        if raw.get("type") == "m.room.encrypted":
            event = await _decrypt_thread_event(client, raw)
            if event is None:
                continue

        content, edited = _effective_content(event)
        body = content.get("body")
        if not isinstance(body, str):
            continue
        body = body.strip()
        if edited and body.startswith("* "):
            body = body[2:].strip()
        msgtype = str(content.get("msgtype") or "")
        text = _label_body(msgtype, _own_text(body, content))
        if not text:
            continue
        sender = str(raw.get("sender") or "")
        entry = MatrixEventContext(sender, text, is_image=msgtype == "m.image")
        stored = cache.store(room_id, event_id, entry)
        if stored is not None:
            entries.append(stored)

    return entries
