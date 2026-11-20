"""In-memory reference ``HostedRoomReplicaRepository`` over the placement's hosted-room tables (``hosted_room_tables``).

Reference limit: ``demote_room``'s authority update freezes stored work records; the reference keeps no work records (``hosted_room_work``), so there is nothing to freeze and no verb could read the effect.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    HostedPlacement,
    ReplicaEpochRegressionError,
    ReplicaError,
    ReplicaGapError,
    ReplicaPage,
    RoomConflictError,
)

from ..backend import register_repository
from . import hosted_room_rules as rules
from .hosted_room_tables import HostedTables, insert_event, read_tables, write_tables

__all__ = ["InMemoryHostedRoomReplicaRepository"]

# Replica caps: rooms and accounted event bytes.
MAX_REPLICA_ROOMS = 256
MAX_REPLICA_EVENT_BYTES = 256 * 1024 * 1024
_SYSTEM_ACTOR = {"kind": "system", "id": "authority-control"}


def _control_event(kind: str, epoch: int, payload: dict[str, Any]) -> tuple[str, str, str, str]:
    """A control event as (event_id, kind, actor_json, payload_json)."""
    return f"system:authority-{kind}:{epoch}", f"authority.{kind}", rules.actor_json(_SYSTEM_ACTOR), rules.payload_json(payload)


def _replica_state(tables: HostedTables, room_id: str) -> tuple[dict[str, Any] | None, int, int, int]:
    """The replica row state: a new room is admitted only under the replica cap."""
    replicas = tables["hosted_room_replicas"]
    row = replicas.get(room_id)
    if row is None:
        if len(replicas) >= MAX_REPLICA_ROOMS:
            raise ReplicaError("replica room capacity exhausted")
        return None, 0, 0, 0
    return row, int(row["authority_epoch"]), int(row["last_seq"]), int(row["event_bytes"])


def _store_events(tables: HostedTables, room_id: str, events: list[Mapping[str, Any]], stored_bytes: int, now: float) -> int:
    """Insert the new page events in order; returns their accounted bytes."""
    added = 0
    rows = tables.rows("hosted_room_replica_events", room_id)
    for event in events:
        size = rules.replica_event_bytes(event)
        if stored_bytes + added + size > MAX_REPLICA_EVENT_BYTES:
            raise ReplicaError("replica event storage exhausted")
        seq = int(event["seq"])
        rows[seq] = {
            "room_id": room_id, "seq": seq, "event_id": event["event_id"], "kind": event["kind"],
            "actor_json": rules.actor_json(event["actor"]), "authority_epoch": event.get("authority_epoch"),
            "payload_json": rules.payload_json(event["payload"]), "created_at": float(event.get("created_at") or now)}
        added += size
    return added


class InMemoryHostedRoomReplicaRepository:
    """See ``persistence.gateway.hosted_rooms.HostedRoomReplicaRepository`` for the contract each method implements."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def ingest_page(self, *, placement: HostedPlacement, room_id: str, room_name: str, members_json: str, page: ReplicaPage, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        authority = {"gateway_id": page.authority_gateway_id, "epoch": page.authority_epoch}
        with write_tables(self._store, placement) as tables:
            row, stored_epoch, last_seq, stored_bytes = _replica_state(tables, room_id)
            if page.authority_epoch < stored_epoch:
                raise ReplicaEpochRegressionError("page authority epoch is older than the stored replica epoch")
            new_events = [event for event in page.events if int(event["seq"]) > last_seq]
            if new_events and int(new_events[0]["seq"]) != last_seq + 1:
                raise ReplicaGapError("page skips sequences the replica has not stored")
            added = _store_events(tables, room_id, new_events, stored_bytes, timestamp)
            new_last = int(new_events[-1]["seq"]) if new_events else last_seq
            latest_seq = new_last if page.latest_seq is None else page.latest_seq
            values = {
                "name": room_name, "members_json": members_json, "authority_gateway_id": page.authority_gateway_id,
                "authority_epoch": page.authority_epoch, "last_seq": new_last, "latest_seq": max(latest_seq, new_last)}
            if row is None:
                tables["hosted_room_replicas"][room_id] = {"room_id": room_id, **values, "event_bytes": added, "created_at": timestamp, "updated_at": timestamp}
            else:
                row.update(values, event_bytes=int(row["event_bytes"]) + added, updated_at=timestamp)
        return {"room_id": room_id, "stored_seq": new_last, "ingested": len(new_events), "authority": authority, "caught_up": new_last >= max(latest_seq, new_last)}

    def replica_state(self, *, placement: HostedPlacement, room_id: str) -> Mapping[str, JsonValue]:
        row = read_tables(self._store, placement)["hosted_room_replicas"].get(room_id)
        if row is None:
            raise ReplicaError("replica not found")
        return {
            "room_id": row["room_id"], "name": row["name"], "members": json.loads(row["members_json"]),
            "authority": {"gateway_id": row["authority_gateway_id"], "epoch": int(row["authority_epoch"])},
            "last_seq": int(row["last_seq"]), "latest_seq": int(row["latest_seq"]), "event_bytes": int(row["event_bytes"]),
            "created_at": float(row["created_at"]), "updated_at": float(row["updated_at"])}

    def promote_replica(self, *, placement: HostedPlacement, room_id: str, local_gateway_id: str, reason: str = 'authority-unreachable', now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            replica = tables["hosted_room_replicas"].get(room_id)
            if replica is None:
                raise ReplicaError("replica not found")
            if replica["authority_gateway_id"] == local_gateway_id:
                raise ReplicaError("this gateway already holds the room authority")
            if room_id in tables["hosted_rooms"]:
                raise RoomConflictError("room_id already exists in the local authoritative store")
            if room_id in tables["hosted_room_retired_ids"]:
                raise RoomConflictError("room_id belongs to a disbanded room")
            previous_gateway, previous_epoch = str(replica["authority_gateway_id"]), int(replica["authority_epoch"])
            target_epoch, claim_seq = previous_epoch + 1, int(replica["last_seq"]) + 1
            event_id, kind, actor_json, payload_json = _control_event("claimed", target_epoch, {
                "previous_gateway_id": previous_gateway, "authority_gateway_id": local_gateway_id,
                "authority_epoch": target_epoch, "promoted_from_replica": True, "reason": reason})
            tables["hosted_rooms"][room_id] = {
                "room_id": room_id, "name": replica["name"], "members_json": replica["members_json"],
                "authority_gateway_id": local_gateway_id, "authority_epoch": target_epoch, "next_seq": claim_seq + 1,
                "event_bytes": int(replica["event_bytes"]) + rules.utf8_len(event_id, kind, actor_json, payload_json),
                "revision": 1, "created_at": timestamp, "updated_at": timestamp, "disbanded_at": None}
            events = tables.rows("hosted_room_events", room_id)
            for seq, row in sorted(tables["hosted_room_replica_events"].get(room_id, {}).items()):
                events[seq] = dict(row)
            insert_event(tables, room_id, claim_seq, event_id, kind, actor_json, target_epoch, payload_json, timestamp)
            tables["hosted_room_replica_events"].pop(room_id, None)
            del tables["hosted_room_replicas"][room_id]
        return {
            "room_id": room_id, "authority_gateway_id": local_gateway_id, "authority_epoch": target_epoch,
            "previous_gateway_id": previous_gateway, "previous_epoch": previous_epoch, "claim_seq": claim_seq,
            "latest_seq": claim_seq}

    def demote_room(self, *, placement: HostedPlacement, room_id: str, observed_gateway_id: str, observed_epoch: int, local_gateway_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        timestamp = rules.clock(now)
        with write_tables(self._store, placement) as tables:
            row = tables["hosted_rooms"].get(room_id)
            if row is None or row["disbanded_at"] is not None:
                raise ReplicaError("room not found in the local authoritative store")
            current_gateway, current_epoch = str(row["authority_gateway_id"]), int(row["authority_epoch"])
            if (current_gateway, current_epoch) == (observed_gateway_id, observed_epoch):
                return {"room_id": room_id, "authority_gateway_id": current_gateway, "authority_epoch": current_epoch, "idempotent": True}
            if observed_epoch <= current_epoch:
                raise ReplicaEpochRegressionError("observed epoch does not supersede the stored authority")
            if current_gateway != local_gateway_id:
                raise ReplicaError("room is not locally authoritative; nothing to demote")
            event_id, kind, actor_json, payload_json = _control_event("lost", observed_epoch, {
                "previous_gateway_id": current_gateway, "authority_gateway_id": observed_gateway_id,
                "authority_epoch": observed_epoch})
            insert_event(tables, room_id, int(row["next_seq"]), event_id, kind, actor_json, observed_epoch, payload_json, timestamp)
            row.update(
                authority_gateway_id=observed_gateway_id, authority_epoch=observed_epoch, next_seq=int(row["next_seq"]) + 1,
                revision=int(row["revision"]) + 1, updated_at=timestamp)
        return {"room_id": room_id, "authority_gateway_id": observed_gateway_id, "authority_epoch": observed_epoch, "idempotent": False}


register_repository("hosted_room_replicas", InMemoryHostedRoomReplicaRepository)
