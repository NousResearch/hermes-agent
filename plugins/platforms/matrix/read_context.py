"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.effective_event import effective_event, event_content
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reaction_context import fetch_reactions_for_events
from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache, _label_body, _own_text

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
    *, before: MatrixEventContext | None,
) -> tuple[dict | None, dict | None, str | None]:
    event_id = raw.get("event_id")
    if raw.get("room_id", room_id) != room_id:
        return None, None, None
    cache = adapter._event_context_cache
    unsigned = raw.get("unsigned")
    if isinstance(event_id, str) and isinstance(unsigned, dict) and unsigned.get("redacted_because"):
        cache.redact(room_id, event_id)
    if MatrixRelation.from_content(event_content(raw).get("m.relates_to")).is_edit:
        return None, None, None
    state = await effective_event(
        adapter._client, raw,
        is_redacted=lambda target: adapter._event_context_cache.is_redacted(room_id, target),
    )
    content = state.content
    if state.redacted and isinstance(event_id, str):
        cache.redact(room_id, event_id)
    if content is None:
        return None, state.error, None
    if not state.redacted and not content.get("msgtype") and not state.error:
        return None, None, None
    body = content.get("body")
    if not isinstance(body, str):
        body = ""
    body = body.strip()
    text = _label_body(str(content.get("msgtype") or ""), _own_text(body))
    body = "[redacted]" if state.redacted else text[:1200]
    relation = MatrixRelation.from_content(state.original_content.get("m.relates_to"))
    sender = str(raw.get("sender") or "")
    authorized = sender == adapter._user_id or adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=room_id
    ) is True
    visible = {
        "event_id": event_id,
        "sender": sender,
        "body": body,
        "msgtype": None if state.redacted or not content.get("msgtype") else str(content.get("msgtype")),
        "thread_id": relation.thread_root,
        "timestamp": raw.get("origin_server_ts"),
        "sender_authorized": authorized,
    }
    if state.edited:
        visible["edited"] = True
    if state.redacted:
        visible["redacted"] = True
    if state.error is not None and before is not None and before.state_error:
        visible.update(body="[event content unavailable]", msgtype=None)
        visible.pop("edited", None)
    if not state.redacted and cache.history_entry(room_id, event_id) is not before:
        visible.update(body="[event content unavailable]", msgtype=None)
        visible.pop("edited", None)
        return visible, {"event_id": event_id, "error": "event content changed"}, state.replacement_id
    if isinstance(event_id, str) and not state.redacted and state.error is None:
        if before is None or before.state_error or before.text != text or before.replacement_id != state.replacement_id:
            cache.store(room_id, event_id, MatrixEventContext(
                sender, text, is_image=content.get("msgtype") == "m.image", replacement_id=state.replacement_id,
            ))
    return visible, state.error, state.replacement_id


@dataclass
class MatrixReadEvent:
    raw: dict[str, Any]
    visible: dict[str, Any]
    error: dict[str, str] | None
    replacement_id: str | None
    cache_entry: MatrixEventContext | None

    async def refresh(self, adapter: Any, room_id: str, chat_type: str) -> bool:
        cache = adapter._event_context_cache
        event_id = self.visible["event_id"]
        current = cache.history_entry(room_id, event_id)
        if (current is self.cache_entry and not cache.is_redacted(room_id, self.replacement_id)
                and not (self.error and self.error["error"] == "event content changed")):
            return True
        self.cache_entry = current
        raw = self.raw
        if current is None or not current.redacted:
            try:
                path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
                fresh = _raw_event(await asyncio.wait_for(adapter._client.api.request(Method.GET, path), timeout=10.0))
            except Exception:
                fresh = {}
            if fresh.get("event_id") == event_id and fresh.get("room_id", room_id) == room_id:
                raw = fresh
            else:
                self.invalidate(
                    "replacement was redacted" if cache.is_redacted(room_id, self.replacement_id) else
                    current.state_error if current and current.state_error else "event content changed"
                )
                return True
        visible, error, replacement_id = await _visible_event(adapter, raw, room_id, chat_type, before=current)
        if visible is None:
            self.error = error
            return False
        self.raw, self.visible, self.error, self.replacement_id = raw, visible, error, replacement_id
        if error is not None:
            self.visible.update(body="[event content unavailable]", msgtype=None)
            self.visible.pop("edited", None)
        self.cache_entry = cache.history_entry(room_id, event_id)
        return True

    def invalidate(self, error: str) -> None:
        self.visible.update(body="[event content unavailable]", msgtype=None)
        self.visible.pop("edited", None)
        self.error = {"event_id": self.visible["event_id"], "error": error}

    def recheck(self, cache: MatrixEventContextCache, room_id: str) -> None:
        event_id = self.visible["event_id"]
        current = cache.history_entry(room_id, event_id)
        if current is not None and current.redacted:
            self.visible.update(body="[redacted]", msgtype=None, redacted=True)
            self.error = None
        elif cache.is_redacted(room_id, self.replacement_id):
            self.invalidate("replacement was redacted")
        elif current is not self.cache_entry:
            self.invalidate(current.state_error if current and current.state_error else "event content changed")
        else:
            return
        self.visible.pop("edited", None)
        self.visible.pop("reactions", None)
        self.visible.pop("reactions_truncated", None)


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

    cached = adapter._event_context_cache.snapshot(room_id)

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
    resolved: list[MatrixReadEvent] = []
    for raw in ([root] if root is not None else []) + chunk[:limit - bool(root)]:
        if not isinstance(raw, dict):
            continue
        visible, error, replacement_id = await _visible_event(
            adapter, raw, room_id, chat_type, before=cached.get(raw.get("event_id")),
        )
        if visible is None:
            if error is not None:
                errors.append(error)
            continue
        if kind == "thread" and visible["event_id"] != event_id and visible["thread_id"] != event_id:
            continue
        events.append(visible)
        resolved.append(MatrixReadEvent(
            raw, visible, error, replacement_id,
            adapter._event_context_cache.history_entry(room_id, raw.get("event_id")),
        ))

    targets = [event for event in events if isinstance(event["event_id"], str) and not event.get("redacted")]
    snapshots = await fetch_reactions_for_events(
        client, room_id, [event["event_id"] for event in targets],
        limit=50 if kind == "event" else 8,
    )
    by_id = {event["event_id"]: snapshot for event, snapshot in zip(targets, snapshots)}
    events = []
    refreshed: list[MatrixReadEvent] = []
    for snapshot in resolved:
        if not await snapshot.refresh(adapter, room_id, chat_type):
            if snapshot.error is not None:
                errors.append(snapshot.error)
            continue
        events.append(snapshot.visible)
        refreshed.append(snapshot)

    for event in events:
        snapshot = by_id.get(event["event_id"])
        if snapshot is None:
            continue
        reactions = [
            reaction for reaction in snapshot.reactions
            if not adapter._event_context_cache.is_redacted(room_id, reaction.event_id)
        ]
        if reactions:
            event["reactions"] = [
                reaction.to_dict(sender_authorized=(
                    reaction.sender == adapter._user_id or adapter._is_sender_authorized(
                        reaction.sender, chat_type=chat_type, chat_id=room_id,
                    ) is True
                ))
                for reaction in reactions
            ]
        if snapshot.truncated:
            event["reactions_truncated"] = True
        for reaction_id in snapshot.missing_keys:
            errors.append({"event_id": reaction_id, "error": "missing decryption keys"})
        if snapshot.error:
            errors.append({"event_id": event["event_id"], "error": snapshot.error})

    for snapshot in refreshed:
        snapshot.recheck(adapter._event_context_cache, room_id)
        if snapshot.error is not None:
            errors.append(snapshot.error)

    return {"events": events, "errors": errors}
