"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.client_events import Method, decrypt_raw_event, raw_event
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reaction_context import fetch_reactions_for_events
from plugins.platforms.matrix.reply_context import _effective_content, _label_body, _own_text


async def _visible_event(adapter: Any, raw: dict[str, Any], room_id: str, chat_type: str) -> tuple[dict | None, dict | None]:
    event_id = raw.get("event_id")
    event: Any = raw
    if raw.get("type") == "m.room.encrypted":
        event, error = await decrypt_raw_event(adapter._client, raw)
        if error is not None:
            return None, {"event_id": event_id, "error": error}

    content, edited = _effective_content(event)
    if not content.get("msgtype"):
        return None, None
    body = content.get("body")
    if not isinstance(body, str):
        body = ""
    body = body.strip()
    if edited and body.startswith("* "):
        body = body[2:].strip()
    body = _label_body(str(content.get("msgtype")), _own_text(body))[:1200]
    relation = MatrixRelation.from_content(content.get("m.relates_to"))
    sender = str(raw.get("sender") or "")
    authorized = sender == adapter._user_id or adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=room_id
    ) is True
    return {
        "event_id": event_id,
        "sender": sender,
        "body": body,
        "msgtype": str(content.get("msgtype")),
        "thread_id": relation.thread_root,
        "timestamp": raw.get("origin_server_ts"),
        "sender_authorized": authorized,
    }, None


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
            root = raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))
            if root.get("event_id") != event_id:
                root = None
        except Exception:
            root = None

    try:
        if kind == "event":
            raw = raw_event(await asyncio.wait_for(client.get_event(room_id, event_id), timeout=10.0))
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
    for raw in ([root] if root is not None else []) + chunk[:limit - bool(root)]:
        if not isinstance(raw, dict):
            continue
        visible, error = await _visible_event(adapter, raw, room_id, chat_type)
        if error is not None:
            errors.append(error)
        if visible is None:
            continue
        if kind == "thread" and visible["event_id"] != event_id and visible["thread_id"] != event_id:
            continue
        events.append(visible)

    targets = [event for event in events if isinstance(event["event_id"], str)]
    snapshots = await fetch_reactions_for_events(
        client, room_id, [event["event_id"] for event in targets],
        limit=50 if kind == "event" else 8,
    )
    for event, snapshot in zip(targets, snapshots):
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
        for reaction in snapshot.undecryptable:
            errors.append({
                "event_id": event["event_id"], "reaction_event_id": reaction.event_id,
                "error": f"reaction {reaction.error}",
            })
        if snapshot.error:
            errors.append({"event_id": event["event_id"], "error": snapshot.error})

    return {"events": events, "errors": errors}
