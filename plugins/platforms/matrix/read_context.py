"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.effective_event import effective_event, event_content
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reaction_context import fetch_reactions_for_events
from plugins.platforms.matrix.reply_context import _label_body, _own_text

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


def _raw_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    serialize = getattr(event, "serialize", None)
    return serialize() if callable(serialize) else {}


async def _visible_event(
    adapter: Any, raw: dict[str, Any], room_id: str, chat_type: str,
) -> tuple[dict | None, dict | None, str | None]:
    event_id = raw.get("event_id")
    if raw.get("room_id", room_id) != room_id:
        return None, None, None
    if MatrixRelation.from_content(event_content(raw).get("m.relates_to")).is_edit:
        return None, None, None
    state = await effective_event(
        adapter._client, raw,
        is_redacted=lambda target: adapter._event_context_cache.is_redacted(room_id, target),
    )
    content = state.content
    if content is None:
        return None, state.error, None
    if not state.redacted and not content.get("msgtype"):
        return None, None, None
    body = content.get("body")
    if not isinstance(body, str):
        body = ""
    body = body.strip()
    body = "[redacted]" if state.redacted else _label_body(str(content.get("msgtype")), _own_text(body))[:1200]
    relation = MatrixRelation.from_content(state.original_content.get("m.relates_to"))
    sender = str(raw.get("sender") or "")
    authorized = sender == adapter._user_id or adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=room_id
    ) is True
    visible = {
        "event_id": event_id,
        "sender": sender,
        "body": body,
        "msgtype": None if state.redacted else str(content.get("msgtype")),
        "thread_id": relation.thread_root,
        "timestamp": raw.get("origin_server_ts"),
        "sender_authorized": authorized,
    }
    if state.edited:
        visible["edited"] = True
    if state.redacted:
        visible["redacted"] = True
    return visible, state.error, state.replacement_id


async def read_matrix_context(
    adapter: Any, kind: str, room_id: str, event_id: str | None, limit: int,
    *, requester: str,
) -> dict[str, Any]:
    if room_id not in adapter._joined_rooms or not await adapter._is_allowed_matrix_room_event(room_id):
        return {"error": "Matrix room is not allowed or joined"}
    chat_type = "dm" if await adapter._is_dm_room(room_id) else "group"
    if adapter._is_sender_authorized(requester, chat_type=chat_type, chat_id=room_id) is not True:
        return {"error": "Matrix requester is not authorized for this room"}
    client = adapter._client
    if client is None:
        return {"error": "Matrix client is disconnected"}

    root: dict[str, Any] | None = None
    if kind == "thread":
        try:
            path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id or '', safe='')}"
            root = _raw_event(await asyncio.wait_for(client.api.request(Method.GET, path), timeout=10.0))
            if root.get("event_id") != event_id:
                root = None
        except Exception:
            root = None

    try:
        if kind == "event":
            path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id or '', safe='')}"
            raw = _raw_event(await asyncio.wait_for(client.api.request(Method.GET, path), timeout=10.0))
            if raw.get("event_id") != event_id:
                return {"error": "Matrix event not found in this room"}
            chunk = [raw]
        else:
            room = quote(room_id, safe="")
            if kind == "thread":
                path = f"/_matrix/client/v1/rooms/{room}/relations/{quote(event_id or '', safe='')}/m.thread"
                query = {"dir": "b", "limit": str(limit - 1 if root is not None else limit)}
            else:
                token = await asyncio.wait_for(client.sync_store.get_next_batch(), timeout=10.0)
                if not token:
                    return {"error": "Matrix history is unavailable until the first sync completes"}
                path = f"/_matrix/client/v3/rooms/{room}/messages"
                query = {"from": token, "dir": "b", "limit": str(limit)}
            if kind == "thread" and root is not None and limit == 1:
                chunk = []
            else:
                response = await asyncio.wait_for(client.api.request(Method.GET, path, query_params=query), timeout=10.0)
                chunk = response.get("chunk", []) if isinstance(response, dict) else []
    except Exception as exc:
        return {"error": f"Matrix read failed: {type(exc).__name__}"}

    events: list[dict] = []
    errors: list[dict] = []
    resolved: list[tuple[dict, dict, str | None]] = []
    for raw in ([root] if root is not None else []) + chunk[:limit - bool(root)]:
        if not isinstance(raw, dict):
            continue
        visible, error, replacement_id = await _visible_event(adapter, raw, room_id, chat_type)
        if error is not None:
            errors.append(error)
        if visible is None:
            continue
        if kind == "thread" and visible["event_id"] != event_id and visible["thread_id"] != event_id:
            continue
        events.append(visible)
        resolved.append((raw, visible, replacement_id))

    targets = [event for event in events if isinstance(event["event_id"], str) and not event.get("redacted")]
    snapshots = await fetch_reactions_for_events(
        client, room_id, [event["event_id"] for event in targets],
        limit=50 if kind == "event" else 8,
    )
    by_id = {event["event_id"]: snapshot for event, snapshot in zip(targets, snapshots)}
    events = []
    dependencies: dict[str, str | None] = {}
    for raw, visible, replacement_id in resolved:
        if replacement_id and adapter._event_context_cache.is_redacted(room_id, replacement_id):
            visible, error, replacement_id = await _visible_event(adapter, raw, room_id, chat_type)
            if error is not None:
                errors.append(error)
            if visible is None:
                continue
        events.append(visible)
        dependencies[visible["event_id"]] = replacement_id

    for event in events:
        snapshot = by_id.get(event["event_id"])
        if snapshot is None:
            continue
        if snapshot.reactions:
            event["reactions"] = [
                reaction.to_dict(sender_authorized=(
                    reaction.sender == adapter._user_id or adapter._is_sender_authorized(
                        reaction.sender, chat_type=chat_type, chat_id=room_id,
                    ) is True
                ))
                for reaction in snapshot.reactions
            ]
        if snapshot.truncated:
            event["reactions_truncated"] = True
        for reaction_id in snapshot.missing_keys:
            errors.append({"event_id": reaction_id, "error": "missing decryption keys"})
        if snapshot.error:
            errors.append({"event_id": event["event_id"], "error": snapshot.error})

    for event in events:
        redacted = adapter._event_context_cache.is_redacted(room_id, event["event_id"])
        replacement_id = dependencies[event["event_id"]]
        replacement_redacted = replacement_id and adapter._event_context_cache.is_redacted(room_id, replacement_id)
        if not redacted and not replacement_redacted:
            continue
        event.update(body="[redacted]" if redacted else "[event content unavailable]", msgtype=None)
        if redacted:
            event["redacted"] = True
        if replacement_redacted and not redacted:
            errors.append({"event_id": event["event_id"], "error": "replacement was redacted"})
        event.pop("edited", None)
        event.pop("reactions", None)
        event.pop("reactions_truncated", None)

    return {"events": events, "errors": errors}
