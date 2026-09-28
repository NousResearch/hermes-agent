"""Read earlier Matrix thread messages at an event boundary."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from enum import Enum
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.effective_event import effective_event, event_content
from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
    _label_body,
    _own_text,
)
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reaction_context import fetch_reactions_for_events


logger = logging.getLogger(__name__)

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


async def history_entry(
    client: Any, raw: dict, cache: MatrixEventContextCache, room_id: str,
    *, before: MatrixEventContext | None,
) -> tuple[MatrixEventContext, dict] | None:
    if raw.get("room_id", room_id) != room_id:
        return None
    event_id = raw.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return None
    unsigned = raw.get("unsigned")
    if isinstance(raw.get("event_id"), str) and isinstance(unsigned, dict) and unsigned.get("redacted_because"):
        cache.store(room_id, raw["event_id"], MatrixEventContext(
            str(raw.get("sender") or ""), "[redacted]", redacted=True,
        ))
    if raw.get("type", "m.room.message") not in {"m.room.message", "m.room.encrypted"}:
        return None
    if MatrixRelation.from_content(event_content(raw).get("m.relates_to")).is_edit:
        return None
    retained = cache.retain(room_id, event_id)
    if before is None and not retained.text and not retained.redacted and not retained.state_error:
        before = retained
    state = await effective_event(client, raw, cache=cache, room_id=room_id)
    content = state.content
    if content is None:
        entry = MatrixEventContext(
            str(raw.get("sender") or ""), "[encrypted message could not be decrypted]",
            state_error=state.error["error"] if state.error else None,
        )
        stored = cache.store_resolved(room_id, event_id, entry, before)
        return (stored, state.original_content) if stored is not None else None
    if state.redacted:
        entry = MatrixEventContext(
            str(raw.get("sender") or ""), "[redacted]", redacted=True,
        )
        stored = cache.store_resolved(room_id, event_id, entry, before)
        return (stored, state.original_content) if stored is not None else None
    body = content.get("body")
    if not isinstance(body, str):
        return None
    body = body.strip()
    text = _label_body(str(content.get("msgtype") or ""), _own_text(body))
    if not text:
        return None
    sender = str(raw.get("sender") or "")
    entry = MatrixEventContext(
        sender, text, is_image=content.get("msgtype") == "m.image",
        media_content=MatrixEventContext.image_content(content),
        state_error=state.error["error"] if state.error else None,
        replacement_id=state.replacement_id,
    )
    stored = cache.store_resolved(room_id, event_id, entry, before)
    return (stored, state.original_content) if stored is not None else None


async def fetch_thread_entries(
    client: Any,
    cache: MatrixEventContextCache,
    room_id: str,
    thread_id: str,
    *,
    limit: int,
    before_event_id: str | None = None,
) -> list[MatrixEventContext]:
    if client is None or limit <= 0 or not thread_id or not before_event_id:
        return []

    cached = cache.snapshot(room_id)
    cached.setdefault(thread_id, cache.retain(room_id, thread_id))
    retained: dict[str, MatrixEventContext] = {}

    context_path = (
        f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
        f"/context/{quote(before_event_id, safe='')}"
    )
    path = (
        f"/_matrix/client/v1/rooms/{quote(room_id, safe='')}"
        f"/relations/{quote(thread_id, safe='')}/m.thread"
    )
    messages_path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/messages"
    response: dict | None = None
    event_key = "events_before"
    try:
        boundary = await asyncio.wait_for(
            client.api.request(Method.GET, context_path, query_params={"limit": "0"}), timeout=10.0,
        )
        token = boundary.get("start") if isinstance(boundary, dict) else None
        if isinstance(token, str) and token:
            room_page = await asyncio.wait_for(
                client.api.request(
                    Method.GET, messages_path,
                    query_params={"from": token, "dir": "b", "limit": str(limit)},
                ),
                timeout=10.0,
            )
            response = room_page if isinstance(room_page, dict) else None
            event_key = "chunk"
            if response is not None and isinstance(response.get("chunk"), list):
                retained.update(cache.retain_events(room_id, [raw for raw in response["chunk"][:limit] if isinstance(raw, dict)]))
            relation_token = room_page.get("start") if isinstance(room_page, dict) else None
            try:
                if isinstance(relation_token, str) and relation_token:
                    relations_page = await asyncio.wait_for(
                        client.api.request(
                            Method.GET, path,
                            query_params={"dir": "b", "limit": str(limit), "from": relation_token},
                        ),
                        timeout=10.0,
                    )
                    if isinstance(relations_page, dict) and isinstance(relations_page.get("chunk"), list):
                        response = relations_page
            except Exception as exc:
                logger.debug("Matrix: thread cursor rejected for %s in %s: %s", thread_id, room_id, exc)
        if response is None:
            response = await asyncio.wait_for(
                client.api.request(Method.GET, context_path, query_params={"limit": str(limit * 2)}),
                timeout=10.0,
            )
            event_key = "events_before"
    except Exception as exc:
        logger.debug("Matrix: could not fetch thread %s in %s: %s", thread_id, room_id, exc)
        return []

    entries: list[MatrixEventContext] = []
    entry_ids: list[str] = []
    reaction_ids: list[str] = []
    chunk = response.get(event_key) if isinstance(response, dict) else None
    if not isinstance(chunk, list):
        chunk = []
    retained.update(cache.retain_events(room_id, [raw for raw in chunk[:limit] if isinstance(raw, dict)]))
    root = cache.history_entry(room_id, thread_id)
    root_path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(thread_id, safe='')}"
    try:
        raw_root = await asyncio.wait_for(client.api.request(Method.GET, root_path), timeout=10.0)
        if (isinstance(raw_root, dict) and raw_root.get("event_id") == thread_id
                and raw_root.get("room_id", room_id) == room_id):
            before = cached.get(thread_id)
            parsed_root = await history_entry(client, raw_root, cache, room_id, before=before)
            if parsed_root is not None:
                root = parsed_root[0]
    except Exception as exc:
        logger.debug("Matrix: could not fetch thread root %s in %s: %s", thread_id, room_id, exc)
        root = cache.history_entry(room_id, thread_id)
    if root is not None and (root.redacted or root.text or root.state_error):
        entries.append(root)
        entry_ids.append(thread_id)
        if not root.redacted:
            reaction_ids.append(thread_id)

    for raw in reversed(chunk[:limit]):
        if not isinstance(raw, dict):
            continue
        event_id = raw.get("event_id")
        if event_id == before_event_id or not isinstance(event_id, str):
            continue
        if raw.get("room_id", room_id) != room_id:
            continue
        before = cached.get(event_id)
        parsed = await history_entry(client, raw, cache, room_id, before=before)
        if parsed is None:
            continue
        entry, content = parsed
        # Redaction removes a child's relation, so a room page cannot identify
        # its thread from the redacted event alone.
        if MatrixRelation.from_content(content.get("m.relates_to")).thread_root != thread_id:
            continue
        entries.append(entry)
        entry_ids.append(event_id)
        if not entry.redacted:
            reaction_ids.append(event_id)

    snapshots = await fetch_reactions_for_events(client, room_id, reaction_ids, cache=cache)
    by_id = dict(zip(reaction_ids, snapshots))
    entries = [
        cache.recheck(room_id, cache.history_entry(room_id, event_id) or entry)
        for event_id, entry in zip(entry_ids, entries)
    ]
    return [
        cache.recheck(room_id, replace(entry, reactions=by_id[event_id].reactions, reactions_truncated=by_id[event_id].truncated,
                reaction_keys_missing=bool(by_id[event_id].missing_keys),
                reactions_unavailable=bool(by_id[event_id].error)))
        if event_id in by_id and not entry.redacted else entry
        for event_id, entry in zip(entry_ids, entries)
    ]
