"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.thread_context import (
    Method,
    UndecryptableEvent,
    decrypt_history_event,
    history_message,
)


def _raw_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    serialize = getattr(event, "serialize", None)
    return serialize() if callable(serialize) else {}


async def _visible_event(adapter: Any, raw: dict[str, Any], room_id: str, chat_type: str) -> dict | None:
    message = history_message(await decrypt_history_event(adapter._client, raw))
    if message is None:
        return None
    relation = MatrixRelation.from_content(message.content.get("m.relates_to"))
    sender = str(raw.get("sender") or "")
    authorized = sender == adapter._user_id or adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=room_id
    ) is True
    return {
        "event_id": raw.get("event_id"),
        "sender": sender,
        "body": message.text[:1200],
        "msgtype": message.msgtype,
        "thread_id": relation.thread_root,
        "timestamp": raw.get("origin_server_ts"),
        "sender_authorized": authorized,
    }


async def _thread_root(client: Any, room_id: str, event_id: str) -> dict[str, Any] | None:
    try:
        root = _raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))
    except Exception:
        return None
    return root if root.get("event_id") == event_id else None


async def read_matrix_context(
    adapter: Any, kind: str, room_id: str, event_id: str | None, limit: int,
    *, requester: str,
) -> dict[str, Any]:
    if room_id not in adapter._joined_rooms or not await adapter._is_allowed_matrix_room_event(room_id):
        return {"error": "Matrix room is not allowed or joined"}
    chat_type = "dm" if await adapter._is_dm_room(room_id) else "group"
    if room_id not in adapter._joined_rooms or not adapter._is_allowed_matrix_room(
        room_id, chat_type
    ):
        return {"error": "Matrix room is not allowed or joined"}
    if adapter._is_sender_authorized(requester, chat_type=chat_type, chat_id=room_id) is not True:
        return {"error": "Matrix requester is not authorized for this room"}
    client = adapter._client
    if client is None:
        return {"error": "Matrix client is disconnected"}

    root = await _thread_root(client, room_id, event_id) if kind == "thread" else None
    remaining = limit - (root is not None)
    try:
        if kind == "event":
            chunk = [_raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))]
        elif remaining == 0:
            chunk = []
        else:
            room = quote(room_id, safe="")
            if kind == "thread":
                path = f"/_matrix/client/v1/rooms/{room}/relations/{quote(event_id or '', safe='')}/m.thread"
                query = {"dir": "b", "limit": str(remaining)}
            else:
                token = await asyncio.wait_for(client.sync_store.get_next_batch(), timeout=10.0)
                if not token:
                    return {"error": "Matrix history is unavailable until the first sync completes"}
                path = f"/_matrix/client/v3/rooms/{room}/messages"
                query = {"from": token, "dir": "b", "limit": str(remaining)}
            response = await asyncio.wait_for(client.api.request(Method.GET, path, query_params=query), timeout=10.0)
            newest_first = response.get("chunk") if isinstance(response, dict) else None
            chunk = list(reversed(newest_first[:remaining])) if isinstance(newest_first, list) else []
    except Exception as exc:
        return {"error": f"Matrix read failed: {type(exc).__name__}"}

    events: list[dict] = []
    errors: list[dict] = []
    for raw in ([root] if root is not None else []) + chunk:
        if not isinstance(raw, dict):
            continue
        try:
            visible = await _visible_event(adapter, raw, room_id, chat_type)
        except UndecryptableEvent as exc:
            errors.append({"event_id": raw.get("event_id"), "error": str(exc)})
            continue
        if visible is None or (kind == "thread" and event_id not in (visible["event_id"], visible["thread_id"])):
            continue
        events.append(visible)

    return {"events": events, "errors": errors}
