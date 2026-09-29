"""Bounded Matrix history reads for a live Matrix session."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.client_events import Method, raw_event
from plugins.platforms.matrix.effective_event import effective_event, event_content
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.polls import poll_context
from plugins.platforms.matrix.reaction_context import fetch_reactions_for_events
from plugins.platforms.matrix.reply_context import MatrixEventContext, _label_body, _own_text

_MESSAGE_FILTER = json.dumps({"types": ["m.room.message", "m.room.encrypted", "m.sticker"]})


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
    retained = cache.retain(room_id, event_id) if isinstance(event_id, str) else None
    if before is None and retained is not None and not retained.text and not retained.redacted and not retained.state_error:
        before = retained
    state = await effective_event(adapter._client, raw, cache=cache, room_id=room_id)
    content = state.content
    if state.redacted and isinstance(event_id, str):
        cache.redact(room_id, event_id)
    if content is None:
        if isinstance(event_id, str) and state.error is not None:
            cache.store_resolved(room_id, event_id, MatrixEventContext(
                str(raw.get("sender") or ""), "", state_error=state.error["error"],
            ), before)
        return None, state.error, None
    poll_text = poll_context(content, state.event_type)
    if not state.redacted and not content.get("msgtype") and not state.error and poll_text is None:
        return None, None, None
    body = poll_text or content.get("body")
    if not isinstance(body, str):
        body = ""
    body = body.strip()
    text = _label_body(str(content.get("msgtype") or ""), _own_text(body, content))
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
    if isinstance(event_id, str) and not state.redacted:
        if (before is None or before.state_error or state.error or before.text != text
                or before.replacement_id != state.replacement_id
                or before.media_content != MatrixEventContext.image_content(content)):
            cache.store_resolved(room_id, event_id, MatrixEventContext(
                sender, text, is_image=content.get("msgtype") == "m.image", replacement_id=state.replacement_id,
                media_content=MatrixEventContext.image_content(content),
                state_error=state.error["error"] if state.error else None,
            ), before)
    return visible, state.error, state.replacement_id


@dataclass
class MatrixReadEvent:
    raw: dict[str, Any]
    visible: dict[str, Any] | None
    error: dict[str, str] | None
    replacement_id: str | None
    cache_entry: MatrixEventContext | None

    async def refresh(self, adapter: Any, room_id: str, chat_type: str) -> None:
        cache = adapter._event_context_cache
        event_id = self.raw["event_id"]
        current = cache.history_entry(room_id, event_id)
        if (current is self.cache_entry and not cache.is_redacted(room_id, self.replacement_id)
                and not (self.error and self.error["error"] == "event content changed")):
            return
        self.cache_entry = current
        raw = self.raw
        if current is None or not current.redacted:
            try:
                path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
                fresh = raw_event(await asyncio.wait_for(adapter._client.api.request(Method.GET, path), timeout=10.0))
            except Exception:
                fresh = {}
            if fresh.get("event_id") == event_id and fresh.get("room_id", room_id) == room_id:
                raw = fresh
            else:
                self.invalidate(
                    "replacement was redacted" if cache.is_redacted(room_id, self.replacement_id) else
                    current.state_error if current and current.state_error else "event content changed"
                )
                return
        visible, error, replacement_id = await _visible_event(adapter, raw, room_id, chat_type, before=current)
        self.raw, self.visible, self.error, self.replacement_id = raw, visible, error, replacement_id
        if error is not None and self.visible is not None:
            self.visible.update(body="[event content unavailable]", msgtype=None)
            self.visible.pop("edited", None)
        self.cache_entry = cache.history_entry(room_id, event_id)

    def invalidate(self, error: str) -> None:
        if self.visible is not None:
            self.visible.update(body="[event content unavailable]", msgtype=None)
            self.visible.pop("edited", None)
        self.error = {"event_id": self.raw["event_id"], "error": error}

    def recheck(self, adapter: Any, room_id: str, chat_type: str) -> None:
        cache = adapter._event_context_cache
        event_id = self.raw["event_id"]
        current = cache.history_entry(room_id, event_id)
        if current is not None and current.redacted:
            if self.visible is None:
                sender = str(self.raw.get("sender") or "")
                self.visible = {
                    "event_id": event_id, "sender": sender,
                    "thread_id": MatrixRelation.from_content(event_content(self.raw).get("m.relates_to")).thread_root,
                    "timestamp": self.raw.get("origin_server_ts"),
                    "sender_authorized": sender == adapter._user_id or adapter._is_sender_authorized(
                        sender, chat_type=chat_type, chat_id=room_id,
                    ) is True,
                }
            self.visible.update(body="[redacted]", msgtype=None, redacted=True)
            self.error = None
        elif cache.is_redacted(room_id, self.replacement_id):
            self.invalidate("replacement was redacted")
        elif current is not self.cache_entry:
            self.invalidate(current.state_error if current and current.state_error else "event content changed")
        else:
            return
        if self.visible is not None:
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
    if room_id not in adapter._joined_rooms or not adapter._is_allowed_matrix_room(
        room_id, chat_type
    ):
        return {"error": "Matrix room is not allowed or joined"}
    if adapter._is_sender_authorized(requester, chat_type=chat_type, chat_id=room_id) is not True:
        return {"error": "Matrix requester is not authorized for this room"}
    client = adapter._client
    if client is None:
        return {"error": "Matrix client is disconnected"}

    if kind != "room" and event_id is None:
        return {"error": "event_id is required for thread and event reads"}

    cache = adapter._event_context_cache
    cached = cache.snapshot(room_id)
    if event_id and kind in {"event", "thread"}:
        cached.setdefault(event_id, cache.retain(room_id, event_id))

    root: dict[str, Any] | None = None
    if kind == "thread":
        try:
            path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id or '', safe='')}"
            root = raw_event(await asyncio.wait_for(client.api.request(Method.GET, path), timeout=10.0))
            if root.get("event_id") != event_id:
                root = None
            if root is not None:
                for target, dependency in cache.retain_events(room_id, [root]).items():
                    cached.setdefault(target, dependency)
        except Exception:
            root = None

    remaining = limit - (root is not None)
    try:
        if kind == "event":
            path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id or '', safe='')}"
            raw = raw_event(await asyncio.wait_for(client.api.request(Method.GET, path), timeout=10.0))
            if raw.get("event_id") != event_id:
                return {"error": "Matrix event not found in this room"}
            chunk = [raw]
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
                query = {"from": token, "dir": "b", "limit": str(remaining), "filter": _MESSAGE_FILTER}
            response = await asyncio.wait_for(client.api.request(Method.GET, path, query_params=query), timeout=10.0)
            newest_first = response.get("chunk") if isinstance(response, dict) else None
            chunk = list(reversed(newest_first[:remaining])) if isinstance(newest_first, list) else []
    except Exception as exc:
        return {"error": f"Matrix read failed: {type(exc).__name__}"}

    events: list[dict] = []
    errors: list[dict] = []
    resolved: list[MatrixReadEvent] = []
    candidates = ([root] if root is not None else []) + chunk
    returned = [raw for raw in candidates if isinstance(raw, dict)]
    skipped = len(candidates) - len(returned)
    _retained = cache.retain_events(room_id, returned)
    for raw in returned:
        visible, error, replacement_id = await _visible_event(
            adapter, raw, room_id, chat_type, before=cached.get(raw.get("event_id")),
        )
        if visible is None and error is None:
            skipped += 1
            continue
        relation = MatrixRelation.from_content(event_content(raw).get("m.relates_to"))
        if visible is not None and kind == "thread" and raw.get("event_id") != event_id and relation.thread_root != event_id:
            skipped += 1
            continue
        if visible is not None:
            events.append(visible)
        resolved.append(MatrixReadEvent(
            raw, visible, error, replacement_id,
            adapter._event_context_cache.history_entry(room_id, raw.get("event_id")),
        ))

    targets = [event for event in events if isinstance(event["event_id"], str) and not event.get("redacted")]
    snapshots = await fetch_reactions_for_events(
        client, room_id, [event["event_id"] for event in targets],
        limit=50 if kind == "event" else 8, cache=adapter._event_context_cache,
    )
    by_id = {event["event_id"]: snapshot for event, snapshot in zip(targets, snapshots)}
    events = []
    for snapshot in resolved:
        await snapshot.refresh(adapter, room_id, chat_type)
    for snapshot in resolved:
        snapshot.recheck(adapter, room_id, chat_type)
        if snapshot.visible is not None:
            events.append(snapshot.visible)
        if snapshot.error is not None:
            errors.append(snapshot.error)

    for event in events:
        if event.get("redacted") or event["msgtype"] is None:
            continue
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
        for reaction in snapshot.undecryptable:
            errors.append({
                "event_id": event["event_id"], "reaction_event_id": reaction.event_id,
                "error": f"reaction {reaction.error}",
            })
        if snapshot.error:
            errors.append({"event_id": event["event_id"], "error": snapshot.error})

    if kind == "event" and not events and not errors:
        return {"error": "Matrix event has no message content"}
    return {"events": events, "errors": errors, "skipped": skipped}
