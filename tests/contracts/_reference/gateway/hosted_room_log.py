"""The room and event-log verbs of the in-memory ``HostedRoomRepository``, over the placement's hosted-room tables. ``append_event`` runs the attachment step ``retain_message_attachments`` (``hosted_room_attachment_steps``) inside its write for ``message.user`` and ``message.member``.

Reference limit: a probe reads memory and cannot fail, so ``RoomProbeUnavailableError`` is never raised here.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    EventConflictError,
    EventCursorConflictError,
    HostedPlacement,
    HostedRoomError,
    RoomNotFoundError,
)
from session_rules.hosted_rooms import (
    _CONTEXT_ACTOR,
    _EMPTY_CONTEXT,
    CONTEXT_EVENT_KIND,
    context_from_payload,
    retry_fields,
    validate_context,
)

from . import hosted_room_rules as rules
from .hosted_room_attachment_steps import retain_message_attachments
from .hosted_room_tables import (
    HostedTables,
    event_from_row,
    find_event,
    insert_event,
    peek_tables,
    prepare_event,
    prune_disbanded_rooms_locked,
    read_tables,
    require_authority,
    retire,
    room_events,
    room_from_row,
    room_row,
    write_tables,
)

__all__ = ["HostedRoomLogMixin"]


def _disband_replay(tables: HostedTables, room_id: str, room: dict[str, Any] | None) -> dict[str, Any] | None:
    """The tombstone of a retired or disbanded room; None while the room is live."""
    if room is None:
        retired_at = tables["hosted_room_retired_ids"].get(room_id)
        if retired_at is None:
            raise RoomNotFoundError("hosted room not found")
        return {"room_id": room_id, "disbanded_at": float(retired_at), "idempotent": True, "history_expired": True}
    if room["disbanded_at"] is None:
        return None
    retire(tables, room_id, room["disbanded_at"])
    event = find_event(tables, room_id, "system:room-disbanded")
    return {
        "room_id": room_id, "disbanded_at": float(room["disbanded_at"]), "idempotent": True,
        **({"event": event_from_row(event, idempotent=True)} if event is not None else {})}


def _log_page(events: list[dict[str, Any]], *, since_seq: int, latest_seq: int, authority: dict[str, Any]) -> dict[str, Any]:
    """The largest prefix whose serialized page fits the page budget."""
    def build_page(page_events: list[dict[str, Any]]) -> dict[str, Any]:
        cursor = page_events[-1]["seq"] if page_events else since_seq
        return {"events": page_events, "cursor": cursor, "latest_seq": latest_seq, "has_more": cursor < latest_seq, "authority": authority}

    def fits(page_events: list[dict[str, Any]]) -> bool:
        return rules.utf8_len(json.dumps(build_page(page_events), ensure_ascii=False, separators=(",", ":"))) <= rules.MAX_LOG_PAGE_BYTES

    if events and not fits(events):
        low, high = 1, len(events)
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if fits(events[:middle]) else (low, middle - 1)
        events = events[:low]
        if not fits(events):
            raise HostedRoomError("hosted room event exceeds replay page limit")
    return build_page(events)


def _append_locked(tables: HostedTables, *, room_id: str, event_id: str, kind: str, actor: dict[str, Any], actor_json: str, payload_json: str, authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None, control_reserve: bool, timestamp: float) -> dict[str, Any]:
    """Append one event inside the caller's write: a stored ``event_id`` replays or conflicts, then the room, authority, cursor, attachment and capacity checks and the insert."""
    existing = find_event(tables, room_id, event_id)
    if existing is not None:
        if (existing["kind"], existing["actor_json"], existing["authority_epoch"], existing["payload_json"]) != (kind, actor_json, authority_epoch, payload_json):
            raise EventConflictError("event_id already exists with different content")
        return event_from_row(existing, idempotent=True)
    room = room_row(tables, room_id, include_disbanded=False)
    require_authority(room, authority_gateway_id, authority_epoch, "stale hosted room authority")
    seq = int(room["next_seq"])
    if expected_latest_seq is not None and seq - 1 != expected_latest_seq:
        raise EventCursorConflictError("room changed before event publication")
    if kind in {"message.user", "message.member"}:
        retain_message_attachments(tables, room_id=room_id, event_id=event_id, manifest=json.loads(payload_json).get("attachments", []), now=timestamp)
    event_bytes = prepare_event(tables, room, event_id, kind, actor_json, payload_json, allow_control=kind in rules.CONTROL_EVENT_KINDS or (control_reserve and actor.get("kind") == "gateway"))
    insert_event(tables, room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, timestamp)
    room.update(next_seq=seq + 1, event_bytes=int(room["event_bytes"]) + event_bytes, updated_at=timestamp)
    return {**event_from_row(tables["hosted_room_events"][room_id][seq]), "actor": dict(actor)}


class HostedRoomLogMixin:
    """Room and event verbs; the repository provides ``_store``."""

    _store: Any

    def rename_room(self, *, placement: HostedPlacement, room_id: str, event_id: str, name: str, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        actor_json = rules.system_actor_json("room-control")
        payload_json = rules.payload_json({"name": name})
        with write_tables(self._store, placement) as tables:
            room = room_row(tables, room_id, include_disbanded=True)
            if room["disbanded_at"] is not None:
                raise RoomNotFoundError("hosted room not found")
            existing = find_event(tables, room_id, event_id)
            if existing is not None:
                if existing["kind"] != "room.renamed" or existing["payload_json"] != payload_json:
                    raise EventConflictError("event_id already exists with different immutable content")
                return {**room_from_row(room, idempotent=True), "event": event_from_row(existing, idempotent=True)}
            seq = int(room["next_seq"])
            event_bytes = prepare_event(tables, room, event_id, "room.renamed", actor_json, payload_json)
            room.update(name=name, next_seq=seq + 1, event_bytes=int(room["event_bytes"]) + event_bytes, revision=int(room["revision"]) + 1, updated_at=timestamp)
            insert_event(tables, room_id, seq, event_id, "room.renamed", actor_json, int(room["authority_epoch"]), payload_json, timestamp)
            return {**room_from_row(room), "event": event_from_row(tables["hosted_room_events"][room_id][seq])}

    def append_event(self, *, placement: HostedPlacement, room_id: str, event_id: str, kind: str, actor: Mapping[str, str], payload: Mapping[str, JsonValue], authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None = None, control_reserve: bool = False, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        actor_json = rules.actor_json(dict(actor))
        payload_json = rules.payload_json(dict(payload))
        with write_tables(self._store, placement) as tables:
            return _append_locked(tables, room_id=room_id, event_id=event_id, kind=kind, actor=dict(actor), actor_json=actor_json, payload_json=payload_json, authority_gateway_id=authority_gateway_id, authority_epoch=authority_epoch, expected_latest_seq=expected_latest_seq, control_reserve=control_reserve, timestamp=timestamp)

    def save_room_settings(self, *, placement: HostedPlacement, room_id: str, event_id: str, purpose: str, notes: str, routing: str | None, lead_member_id: str | None, seated_member_ids: Collection[str], authority_gateway_id: str, authority_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        actor = dict(_CONTEXT_ACTOR)
        actor_json = rules.actor_json(actor)
        with write_tables(self._store, placement) as tables:
            first = find_event(tables, room_id, event_id)
            original = json.loads(first["payload_json"]) if first is not None and first["kind"] == CONTEXT_EVENT_KIND else None
            settings = [row for row in room_events(tables, room_id) if row["kind"] == CONTEXT_EVENT_KIND]
            current = {**_EMPTY_CONTEXT, **original} if original is not None else context_from_payload(json.loads(settings[-1]["payload_json"]) if settings else None)
            context = validate_context(purpose, notes, current["routing"] if routing is None else routing, current["lead_member_id"] if lead_member_id is None else lead_member_id)
            if first is None and context["routing"] == "lead" and context["lead_member_id"] not in set(seated_member_ids):
                raise HostedRoomError("the lead must be a member of this Group Chat")
            payload = context if original is None else retry_fields(context, original)
            event = _append_locked(tables, room_id=room_id, event_id=event_id, kind=CONTEXT_EVENT_KIND, actor=actor, actor_json=actor_json, payload_json=rules.payload_json(payload), authority_gateway_id=authority_gateway_id, authority_epoch=authority_epoch, expected_latest_seq=None, control_reserve=False, timestamp=timestamp)
        return {"context": context, "event": event}

    def disband_room(self, *, placement: HostedPlacement, room_id: str, expected_gateway_id: str, expected_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            room = tables["hosted_rooms"].get(room_id)
            replay = _disband_replay(tables, room_id, room)
            if replay is not None:
                return replay
            assert room is not None
            require_authority(room, expected_gateway_id, expected_epoch, "stale hosted room authority")
            actor_json, payload_json = rules.system_actor_json("room-control"), rules.payload_json({"room_id": room_id})
            seq = int(room["next_seq"])
            disband_bytes = prepare_event(tables, room, "system:room-disbanded", "room.disbanded", actor_json, payload_json, allow_control=True)
            insert_event(tables, room_id, seq, "system:room-disbanded", "room.disbanded", actor_json, int(room["authority_epoch"]), payload_json, timestamp)
            room.update(disbanded_at=timestamp, updated_at=timestamp, revision=int(room["revision"]) + 1, next_seq=seq + 1, event_bytes=int(room["event_bytes"]) + disband_bytes)
            retire(tables, room_id, timestamp)
            event = event_from_row(tables["hosted_room_events"][room_id][seq])
            prune_disbanded_rooms_locked(tables, now=timestamp, max_gateway_event_bytes=rules.MAX_GATEWAY_EVENT_BYTES)
        return {"room_id": room_id, "disbanded_at": timestamp, "idempotent": False, "event": event}

    def prune_disbanded_rooms(self, *, placement: HostedPlacement, now: float | None = None) -> int:
        with write_tables(self._store, placement) as tables:
            return prune_disbanded_rooms_locked(tables, now=rules.clock(now))

    def list_rooms(self, *, placement: HostedPlacement, include_disbanded: bool = False, limit: int = 500, offset: int = 0) -> list[Mapping[str, JsonValue]]:
        tables = read_tables(self._store, placement)
        rows = [row for row in tables["hosted_rooms"].values() if include_disbanded or row["disbanded_at"] is None]
        rows.sort(key=lambda row: (-float(row["updated_at"]), row["room_id"]))
        return [room_from_row(row) for row in rows[offset:offset + limit]]

    def room_state(self, *, placement: HostedPlacement, room_id: str, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        tables = read_tables(self._store, placement)
        room = room_row(tables, room_id, include_disbanded=include_disbanded)
        epoch = int(room["authority_epoch"])
        claims = [row for row in room_events(tables, room_id) if row["kind"] == "authority.claimed" and row["authority_epoch"] == epoch]
        return {**room_from_row(room), **({"authority_claim": event_from_row(claims[-1])} if claims else {})}

    def read_events(self, *, placement: HostedPlacement, room_id: str, since_seq: int = 0, limit: int = 100, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        tables = read_tables(self._store, placement)
        room = room_row(tables, room_id, include_disbanded=include_disbanded)
        latest_seq = int(room["next_seq"]) - 1
        authority = {"gateway_id": str(room["authority_gateway_id"]), "epoch": int(room["authority_epoch"])}
        if since_seq > latest_seq:
            raise HostedRoomError("since_seq is ahead of the hosted room log")
        events, total = [], 0
        for row in [row for row in room_events(tables, room_id) if row["seq"] > since_seq][:limit]:
            total += rules.utf8_len(row["event_id"], row["kind"], row["actor_json"], row["payload_json"])
            if total > rules.MAX_LOG_PAGE_BYTES:
                break
            events.append(event_from_row(row))
        return _log_page(events, since_seq=since_seq, latest_seq=latest_seq, authority=authority)

    def latest_event_payload(self, *, placement: HostedPlacement, room_id: str, kind: str, before_seq: int | None = None) -> Mapping[str, JsonValue] | None:
        rows = [row for row in room_events(read_tables(self._store, placement), room_id) if row["kind"] == kind and (before_seq is None or row["seq"] < before_seq)]
        return json.loads(rows[-1]["payload_json"]) if rows else None

    def event_by_id(self, *, placement: HostedPlacement, room_id: str, event_id: str) -> Mapping[str, JsonValue] | None:
        row = find_event(read_tables(self._store, placement), room_id, event_id)
        return None if row is None else event_from_row(row)

    def probe_hosted_room(self, *, placement: HostedPlacement, room_id: str) -> bool:
        tables = peek_tables(self._store, placement)
        if tables is None:
            return False
        row = tables["hosted_rooms"].get(room_id)
        return row is not None and row["disbanded_at"] is None
