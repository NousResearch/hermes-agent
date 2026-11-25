"""``HostedRoomReplicaRepository`` on SQLite: replicated room logs and the authority takeover primitives in the file ``placement`` names.

The replica verbs open the file with the rooms step, then the replica tables, before the transaction; ``ingest_page`` and ``promote_replica`` run the replica DDL again first inside it. ``demote_room`` opens it with the rooms step only.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import (
    HostedPlacement, ReplicaEpochRegressionError, ReplicaError, ReplicaGapError, ReplicaPage, RoomConflictError,
)
from session_rules.hosted_rooms import (
    MAX_REPLICA_EVENT_BYTES, MAX_REPLICA_ROOMS, _actor_json, _control_event, _event_bytes, _payload_json, clock,
    utf8_len,
)

from .._backend import SQLiteStore, register_repository, translated
from ._hosted_rooms_common import table_exists
from ._hosted_rooms_schema import ROOMS_STEP
from ._hosted_store import HostedFiles, SchemaStep

__all__ = ["SQLiteHostedRoomReplicaRepository"]

_EVENT_COLUMNS = "(room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at)"
_INSERT_ROOM_EVENT = f"INSERT INTO hosted_room_events {_EVENT_COLUMNS} VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
_INSERT_REPLICA_EVENT = f"INSERT INTO hosted_room_replica_events {_EVENT_COLUMNS} VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
_SELECT_REPLICA = "SELECT * FROM hosted_room_replicas WHERE room_id=?"


def _initialize_replica_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_replicas (
            room_id TEXT PRIMARY KEY, name TEXT NOT NULL, members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
            latest_seq INTEGER NOT NULL DEFAULT 0, event_bytes INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL
        )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_replica_events (
            room_id TEXT NOT NULL, seq INTEGER NOT NULL CHECK (seq >= 1), event_id TEXT NOT NULL,
            kind TEXT NOT NULL, actor_json TEXT NOT NULL, authority_epoch INTEGER,
            payload_json TEXT NOT NULL, created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq)
        )""")


def _replica_schema_ready(conn: sqlite3.Connection, path: Path) -> bool:
    return table_exists(conn, "hosted_room_replicas") and table_exists(conn, "hosted_room_replica_events")


# The replica tables, created before each replica verb's transaction.
REPLICA_STEP: SchemaStep = (_replica_schema_ready, lambda conn, path: _initialize_replica_schema(conn))


# Appends an authority control event to the room log at ``seq``.
def _append_control_event(
    conn: sqlite3.Connection, room_id: str, seq: int, epoch: int, event: tuple[str, str, str, str], now: float
) -> None:
    event_id, kind, actor_json, payload_json = event
    conn.execute(_INSERT_ROOM_EVENT, (room_id, seq, event_id, kind, actor_json, epoch, payload_json, now))


def _replica_row_state(conn: sqlite3.Connection, room_id: str) -> tuple[sqlite3.Row | None, int, int, int]:
    """Return (row, stored_epoch, last_seq, stored_bytes); a new room is admitted only under the room cap."""
    row = conn.execute("""SELECT authority_gateway_id, authority_epoch, last_seq, latest_seq, event_bytes
             FROM hosted_room_replicas WHERE room_id=?""", (room_id,)).fetchone()
    if row is None:
        count = conn.execute("SELECT COUNT(*) FROM hosted_room_replicas").fetchone()[0]
        if int(count) >= MAX_REPLICA_ROOMS:
            raise ReplicaError("replica room capacity exhausted")
        return None, 0, 0, 0
    return row, int(row["authority_epoch"]), int(row["last_seq"]), int(row["event_bytes"])


def _store_replica(
    conn: sqlite3.Connection, *, is_new: bool, room_id: str, room_name: str, members_json: str,
    authority: dict[str, Any], new_last: int, latest_seq: int, added_bytes: int, now: float) -> None:
    """INSERT the replica row for a new room, else UPDATE it (event_bytes accumulates)."""
    values = (room_name, members_json, authority["gateway_id"], authority["epoch"], new_last, max(latest_seq, new_last))
    if is_new:
        conn.execute("""INSERT INTO hosted_room_replicas (room_id, name, members_json,
                authority_gateway_id, authority_epoch, last_seq, latest_seq, event_bytes,
                created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (room_id, *values, added_bytes, now, now))
    else:
        conn.execute("""UPDATE hosted_room_replicas SET name=?, members_json=?, authority_gateway_id=?,
                authority_epoch=?, last_seq=?, latest_seq=?, event_bytes=event_bytes+?,
                updated_at=? WHERE room_id=?""", (*values, added_bytes, now, room_id))


class SQLiteHostedRoomReplicaRepository:
    """Replicated room logs and the authority takeover primitives."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._files = HostedFiles(store, (ROOMS_STEP, REPLICA_STEP))
        self._room_files = HostedFiles(store, (ROOMS_STEP,))

    # ── writes ──────────────────────────────────────────────────────────────

    # The caller validates the page first. Stores its unseen events in order, refusing an older epoch or a gap.
    def ingest_page(self, *, placement: HostedPlacement, room_id: str, room_name: str, members_json: str, page: ReplicaPage, now: float | None = None) -> Mapping[str, JsonValue]:
        events = list(page.events)
        authority = {"gateway_id": page.authority_gateway_id, "epoch": page.authority_epoch}
        now = clock(now)

        def write(conn: sqlite3.Connection) -> tuple[int, int, int]:
            _initialize_replica_schema(conn)
            row, stored_epoch, last_seq, stored_bytes = _replica_row_state(conn, room_id)
            if authority["epoch"] < stored_epoch:
                raise ReplicaEpochRegressionError("page authority epoch is older than the stored replica epoch")
            new_events = [e for e in events if int(e["seq"]) > last_seq]
            if new_events and int(new_events[0]["seq"]) != last_seq + 1:
                raise ReplicaGapError("page skips sequences the replica has not stored")
            added_bytes = 0
            for event in new_events:
                size = _event_bytes(event)
                if stored_bytes + added_bytes + size > MAX_REPLICA_EVENT_BYTES:
                    raise ReplicaError("replica event storage exhausted")
                conn.execute(
                    _INSERT_REPLICA_EVENT,
                    (
                        room_id, int(event["seq"]), event["event_id"], event["kind"], _actor_json(event["actor"]),
                        event.get("authority_epoch"), _payload_json(event["payload"]),
                        float(event.get("created_at") or now)))
                added_bytes += size
            new_last = int(new_events[-1]["seq"]) if new_events else last_seq
            latest_seq = page.latest_seq if page.latest_seq is not None else new_last
            _store_replica(
                conn, is_new=row is None, room_id=room_id, room_name=room_name, members_json=members_json,
                authority=authority, new_last=new_last, latest_seq=latest_seq, added_bytes=added_bytes, now=now)
            return new_last, len(new_events), latest_seq
        with translated():
            new_last, ingested, latest_seq = self._files._hosted_write(placement, write, immediate=True)
        return {
            "room_id": room_id, "stored_seq": new_last, "ingested": ingested, "authority": authority,
            "caught_up": new_last >= max(latest_seq, new_last)}

    # The caller validates the arguments first. Makes the replica a local room at the next epoch, logging the claim.
    def promote_replica(self, *, placement: HostedPlacement, room_id: str, local_gateway_id: str, reason: str = 'authority-unreachable', now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        local_gateway = local_gateway_id

        def write(conn: sqlite3.Connection) -> tuple[str, int, int, int]:
            _initialize_replica_schema(conn)
            replica = conn.execute(_SELECT_REPLICA, (room_id,)).fetchone()
            if replica is None:
                raise ReplicaError("replica not found")
            if replica["authority_gateway_id"] == local_gateway:
                raise ReplicaError("this gateway already holds the room authority")
            if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone():
                raise RoomConflictError("room_id already exists in the local authoritative store")
            if conn.execute("SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)).fetchone():
                raise RoomConflictError("room_id belongs to a disbanded room")
            previous_gateway, previous_epoch = str(replica["authority_gateway_id"]), int(replica["authority_epoch"])
            target_epoch, claim_seq = previous_epoch + 1, int(replica["last_seq"]) + 1
            claim = _control_event("claimed", target_epoch, {
                "previous_gateway_id": previous_gateway, "authority_gateway_id": local_gateway,
                "authority_epoch": target_epoch, "promoted_from_replica": True, "reason": reason})
            conn.execute("""INSERT INTO hosted_rooms
                   (room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq, event_bytes,
                    revision, created_at, updated_at, disbanded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)""",
                (
                    room_id, replica["name"], replica["members_json"], local_gateway, target_epoch, claim_seq + 1,
                    int(replica["event_bytes"]) + utf8_len(*claim), now, now))
            conn.execute(
                f"""INSERT INTO hosted_room_events {_EVENT_COLUMNS}
                   SELECT room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
                     FROM hosted_room_replica_events WHERE room_id=?""", (room_id,))
            _append_control_event(conn, room_id, claim_seq, target_epoch, claim, now)
            conn.execute("DELETE FROM hosted_room_replica_events WHERE room_id=?", (room_id,))
            conn.execute("DELETE FROM hosted_room_replicas WHERE room_id=?", (room_id,))
            return previous_gateway, previous_epoch, target_epoch, claim_seq
        with translated():
            previous_gateway, previous_epoch, target_epoch, claim_seq = self._files._hosted_write(
                placement, write, immediate=True)
        return {
            "room_id": room_id, "authority_gateway_id": local_gateway, "authority_epoch": target_epoch,
            "previous_gateway_id": previous_gateway, "previous_epoch": previous_epoch, "claim_seq": claim_seq,
            "latest_seq": claim_seq}

    # The caller validates the arguments first. Hands the room to a newer observed authority with a loss event.
    def demote_room(self, *, placement: HostedPlacement, room_id: str, observed_gateway_id: str, observed_epoch: int, local_gateway_id: str, now: float | None = None) -> Mapping[str, JsonValue]:
        now = clock(now)
        local_gateway = local_gateway_id

        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute("""SELECT authority_gateway_id, authority_epoch, next_seq
                     FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL""", (room_id,)).fetchone()
            if row is None:
                raise ReplicaError("room not found in the local authoritative store")
            current_gateway, current_epoch = str(row["authority_gateway_id"]), int(row["authority_epoch"])
            if current_gateway == observed_gateway_id and current_epoch == observed_epoch:
                return {
                    "room_id": room_id, "authority_gateway_id": current_gateway, "authority_epoch": current_epoch,
                    "idempotent": True}
            if observed_epoch <= current_epoch:
                raise ReplicaEpochRegressionError("observed epoch does not supersede the stored authority")
            if current_gateway != local_gateway:
                raise ReplicaError("room is not locally authoritative; nothing to demote")
            lost = _control_event("lost", observed_epoch, {
                "previous_gateway_id": current_gateway, "authority_gateway_id": observed_gateway_id,
                "authority_epoch": observed_epoch})
            _append_control_event(conn, room_id, int(row["next_seq"]), observed_epoch, lost, now)
            conn.execute("""UPDATE hosted_rooms
                      SET authority_gateway_id=?, authority_epoch=?, next_seq=next_seq+1, revision=revision+1, updated_at=?
                    WHERE room_id=?""",
                (observed_gateway_id, observed_epoch, now, room_id))
            return {
                "room_id": room_id, "authority_gateway_id": observed_gateway_id, "authority_epoch": observed_epoch,
                "idempotent": False}
        with translated():
            return self._room_files._hosted_write(placement, write, immediate=True)

    # ── reads ───────────────────────────────────────────────────────────────

    # The caller validates the arguments first; refuses a room with no replica.
    def replica_state(self, *, placement: HostedPlacement, room_id: str) -> Mapping[str, JsonValue]:
        with translated():
            row = self._files._hosted_read(placement, lambda conn: conn.execute(_SELECT_REPLICA, (room_id,)).fetchone())
        if row is None:
            raise ReplicaError("replica not found")
        return {
            "room_id": row["room_id"], "name": row["name"], "members": json.loads(row["members_json"]),
            "authority": {"gateway_id": row["authority_gateway_id"], "epoch": int(row["authority_epoch"])},
            "last_seq": int(row["last_seq"]), "latest_seq": int(row["latest_seq"]), "event_bytes": int(row["event_bytes"]),
            "created_at": float(row["created_at"]), "updated_at": float(row["updated_at"])}


register_repository("hosted_room_replicas", SQLiteHostedRoomReplicaRepository)
