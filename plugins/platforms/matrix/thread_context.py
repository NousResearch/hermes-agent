"""Read earlier Matrix thread messages at an event boundary."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from collections.abc import Callable
from enum import Enum
from typing import Any, Collection
from urllib.parse import quote

from plugins.platforms.matrix.reply_context import (
    MatrixEventContext,
    MatrixEventContextCache,
    _content_dict,
    _effective_content,
    _event_sender,
    _label_body,
    _own_text,
)
from plugins.platforms.matrix.relations import MatrixRelation


logger = logging.getLogger(__name__)

# Receives an earlier event's sender and original content. It returns True when the event
# belongs to a turn that the transcript already contains, and catch-up stops at that event.
PreviousTurnCheck = Callable[[str, dict], bool]

NON_CONVERSATIONAL_KEY = "com.nousresearch.hermes.non_conversational"

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
    text = _label_body(msgtype, _own_text(body))
    return HistoryMessage(msgtype, text, content) if text else None


async def history_entry(client: Any, raw: dict) -> tuple[MatrixEventContext, dict] | None:
    if raw.get("type", "m.room.message") not in {"m.room.message", "m.room.encrypted"}:
        return None
    try:
        event = await decrypt_history_event(client, raw)
    except UndecryptableEvent:
        return None

    original_content = _content_dict(event)
    if original_content.get(NON_CONVERSATIONAL_KEY) is True:
        return None
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
    return MatrixEventContext(sender, text, is_image=content.get("msgtype") == "m.image"), original_content


async def _thread_root(
    client: Any, cache: MatrixEventContextCache, room_id: str, thread_id: str,
    is_previous_turn: PreviousTurnCheck | None,
) -> MatrixEventContext | None:
    if is_previous_turn is None:
        return await cache.resolve(client, room_id, thread_id)
    event = await cache.fetch_event(client, room_id, thread_id)
    if event is None or is_previous_turn(_event_sender(event), _content_dict(event)):
        return None
    return await cache.store_event(room_id, thread_id, event)


async def fetch_thread_entries(
    client: Any,
    cache: MatrixEventContextCache,
    room_id: str,
    thread_id: str,
    *,
    limit: int,
    before_event_id: str | None = None,
    exclude_event_ids: Collection[str] = (),
    is_previous_turn: PreviousTurnCheck | None = None,
) -> list[MatrixEventContext]:
    if client is None or limit <= 0 or not thread_id or not before_event_id:
        return []

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

    chunk = response.get(event_key) if isinstance(response, dict) else None
    newest_first: list[MatrixEventContext] = []
    reached_previous_turn = False
    for raw in chunk[:limit] if isinstance(chunk, list) else []:
        if not isinstance(raw, dict):
            continue
        event_id = raw.get("event_id")
        if not isinstance(event_id, str) or event_id == before_event_id or event_id in exclude_event_ids:
            continue
        if raw.get("room_id", room_id) != room_id:
            continue
        parsed = await history_entry(client, raw)
        if parsed is None:
            continue
        entry, content = parsed
        if MatrixRelation.from_content(content.get("m.relates_to")).thread_root != thread_id:
            continue
        if is_previous_turn is not None and is_previous_turn(entry.sender, content):
            reached_previous_turn = True
            break
        stored = cache.store(room_id, event_id, entry)
        if stored is not None:
            newest_first.append(stored)

    entries: list[MatrixEventContext] = []
    if not reached_previous_turn:
        root = await _thread_root(client, cache, room_id, thread_id, is_previous_turn)
        if root is not None:
            entries.append(root)
    entries.extend(reversed(newest_first))
    return entries
