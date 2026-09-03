"""Replica store and takeover primitives for hosted Group Chat rooms.

The authority gateway owns a room's ordered log in ``gateway/hosted_rooms.py``.
This module gives every OTHER participant gateway a durable local copy of that
log, and the fenced primitives to continue the room when the authority host
dies:

- ``ingest_page()`` persists replay pages (``groups.log`` output, which carries
  the room's authority stamp) idempotently, refusing sequence gaps and
  authority-epoch regressions.
- ``promote_replica()`` instantiates the replicated log as a locally-owned
  hosted room at ``epoch + 1`` with a lineage-proving ``authority.claimed``
  event, so a surviving participant can resume the room.
- ``demote_room()`` fences a returning stale authority: presented with proof of
  a newer epoch, the local room records ``authority.lost`` and stops being
  authoritative.

Storage primitives only: none of these decide *when* takeover is safe. The
caller (an explicit user action today; a lease/quorum driver later) must
establish that the previous owner can no longer commit before promoting.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from gateway.hosted_rooms import (
    MAX_ACTOR_ID_CHARS,
    MAX_EVENT_JSON_BYTES,
    MAX_ROOM_ID_CHARS,
    HostedRoomError,
    RoomConflictError,
    _canonical_json,
    _connect,
    _transaction,
    _validate_identifier,
    _validate_members,
    _validate_room_name,
    local_authority_gateway_id,
)

MAX_REPLICA_ROOMS = 256
MAX_REPLICA_EVENT_BYTES = 256 * 1024 * 1024


class ReplicaError(HostedRoomError):
    """Base class for invalid or conflicting replica operations."""


class ReplicaGapError(ReplicaError):
    """A page does not start at the replica's next expected sequence."""


class ReplicaEpochRegressionError(ReplicaError):
    """A page or demotion carries an older authority epoch than stored."""


def _initialize_replica_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_replicas (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
            latest_seq INTEGER NOT NULL DEFAULT 0,
            event_bytes INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_replica_events (
            room_id TEXT NOT NULL,
            seq INTEGER NOT NULL CHECK (seq >= 1),
            event_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor_json TEXT NOT NULL,
            authority_epoch INTEGER,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq)
        )"""
    )


def _replica_transaction(db_path: Path | str):
    return _transaction(db_path, immediate=True)


def _ensure_schema(db_path: Path | str) -> None:
    conn = _connect(db_path)
    try:
        with conn:
            _initialize_replica_schema(conn)
    finally:
        conn.close()


def _event_bytes(event: dict[str, Any]) -> int:
    return (
        len(str(event["event_id"]).encode("utf-8"))
        + len(str(event["kind"]).encode("utf-8"))
        + len(
            json.dumps(
                event["actor"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
        + len(
            json.dumps(
                event["payload"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
    )


def _validate_page(page: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(page, dict):
        raise ReplicaError("page must be an object")
    events = page.get("events")
    authority = page.get("authority")
    if not isinstance(events, list):
        raise ReplicaError("page.events must be a list")
    if not isinstance(authority, dict):
        raise ReplicaError("page.authority is required for replication")
    gateway_id = _validate_identifier(
        authority.get("gateway_id"),
        label="page.authority.gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    epoch = authority.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ReplicaError("page.authority.epoch must be a positive integer")
    previous_seq: int | None = None
    for event in events:
        if not isinstance(event, dict):
            raise ReplicaError("page events must be objects")
        seq = event.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ReplicaError("event.seq must be a positive integer")
        if previous_seq is not None and seq != previous_seq + 1:
            raise ReplicaGapError("page events must be contiguous")
        previous_seq = seq
        for field in ("event_id", "kind"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise ReplicaError(f"event.{field} must be a non-empty string")
        if not isinstance(event.get("actor"), dict):
            raise ReplicaError("event.actor must be an object")
        if "payload" not in event:
            raise ReplicaError("event.payload is required")
    return events, {"gateway_id": gateway_id, "epoch": epoch}


def ingest_page(
    db_path: Path | str,
    *,
    room_id: Any,
    room_name: Any,
    members: Any,
    page: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist one replay page for ``room_id``; idempotent, gap- and
    epoch-regression-safe.

    ``page`` is the verbatim result of the authority's ``groups.log`` call
    (``read_events()``), whose ``authority`` stamp proves lineage.
    """
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    room_name = _validate_room_name(room_name)
    _, members_json = _validate_members(members)
    events, authority = _validate_page(page)
    now = time.time() if now is None else float(now)
    _ensure_schema(db_path)

    with _replica_transaction(db_path) as conn:
        _initialize_replica_schema(conn)
        row = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, last_seq,
                      latest_seq, event_bytes
                 FROM hosted_room_replicas WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if row is None:
            count = conn.execute(
                "SELECT COUNT(*) FROM hosted_room_replicas"
            ).fetchone()[0]
            if int(count) >= MAX_REPLICA_ROOMS:
                raise ReplicaError("replica room capacity exhausted")
            stored_epoch = 0
            last_seq = 0
            stored_bytes = 0
        else:
            stored_epoch = int(row["authority_epoch"])
            last_seq = int(row["last_seq"])
            stored_bytes = int(row["event_bytes"])

        if authority["epoch"] < stored_epoch:
            raise ReplicaEpochRegressionError(
                "page authority epoch is older than the stored replica epoch"
            )

        new_events = [e for e in events if int(e["seq"]) > last_seq]
        if new_events and int(new_events[0]["seq"]) != last_seq + 1:
            raise ReplicaGapError(
                "page skips sequences the replica has not stored"
            )
        added_bytes = 0
        for event in new_events:
            size = _event_bytes(event)
            if stored_bytes + added_bytes + size > MAX_REPLICA_EVENT_BYTES:
                raise ReplicaError("replica event storage exhausted")
            actor_json = _canonical_json(
                event["actor"], label="actor", max_bytes=4 * 1024
            )
            payload_json = _canonical_json(
                event["payload"], label="payload", max_bytes=MAX_EVENT_JSON_BYTES
            )
            epoch_value = event.get("authority_epoch")
            conn.execute(
                """INSERT INTO hosted_room_replica_events
                   (room_id, seq, event_id, kind, actor_json, authority_epoch,
                    payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    room_id,
                    int(event["seq"]),
                    event["event_id"],
                    event["kind"],
                    actor_json,
                    epoch_value,
                    payload_json,
                    float(event.get("created_at") or now),
                ),
            )
            added_bytes += size
        new_last = int(new_events[-1]["seq"]) if new_events else last_seq
        latest_seq = page.get("latest_seq")
        if isinstance(latest_seq, bool) or not isinstance(latest_seq, int):
            latest_seq = new_last
        if row is None:
            conn.execute(
                """INSERT INTO hosted_room_replicas
                   (room_id, name, members_json, authority_gateway_id,
                    authority_epoch, last_seq, latest_seq, event_bytes,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    room_id,
                    room_name,
                    members_json,
                    authority["gateway_id"],
                    authority["epoch"],
                    new_last,
                    max(latest_seq, new_last),
                    added_bytes,
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """UPDATE hosted_room_replicas
                      SET name=?, members_json=?, authority_gateway_id=?,
                          authority_epoch=?, last_seq=?, latest_seq=?,
                          event_bytes=event_bytes+?, updated_at=?
                    WHERE room_id=?""",
                (
                    room_name,
                    members_json,
                    authority["gateway_id"],
                    authority["epoch"],
                    new_last,
                    max(latest_seq, new_last),
                    added_bytes,
                    now,
                    room_id,
                ),
            )
    return {
        "room_id": room_id,
        "stored_seq": new_last,
        "ingested": len(new_events),
        "authority": authority,
        "caught_up": new_last >= max(latest_seq, new_last),
    }


def replica_state(db_path: Path | str, *, room_id: Any) -> dict[str, Any]:
    """Return the stored replica's coverage and authority lineage."""
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    _ensure_schema(db_path)
    with _replica_transaction(db_path) as conn:
        _initialize_replica_schema(conn)
        row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, last_seq, latest_seq, event_bytes,
                      created_at, updated_at
                 FROM hosted_room_replicas WHERE room_id=?""",
            (room_id,),
        ).fetchone()
    if row is None:
        raise ReplicaError("replica not found")
    return {
        "room_id": row["room_id"],
        "name": row["name"],
        "members": json.loads(row["members_json"]),
        "authority": {
            "gateway_id": row["authority_gateway_id"],
            "epoch": int(row["authority_epoch"]),
        },
        "last_seq": int(row["last_seq"]),
        "latest_seq": int(row["latest_seq"]),
        "event_bytes": int(row["event_bytes"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
    }


def promote_replica(
    db_path: Path | str,
    *,
    room_id: Any,
    reason: Any = "authority-unreachable",
    now: float | None = None,
) -> dict[str, Any]:
    """Continue a replicated room on THIS gateway at ``epoch + 1``.

    Copies the replica's log into the authoritative store, appends a lineage-
    proving ``authority.claimed`` event, and returns the new room state. The
    old authority is fenced everywhere the claim replicates: its epoch is now
    stale and every fenced primitive rejects it.

    The caller decides that takeover is safe (the previous owner can no longer
    commit). This primitive only makes the takeover atomic and provable.
    """
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    if not isinstance(reason, str) or not reason or len(reason) > 200:
        raise ReplicaError("reason must be a non-empty string of at most 200 chars")
    now = time.time() if now is None else float(now)
    local_gateway = local_authority_gateway_id()
    _ensure_schema(db_path)

    with _replica_transaction(db_path) as conn:
        _initialize_replica_schema(conn)
        replica = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, last_seq, event_bytes
                 FROM hosted_room_replicas WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if replica is None:
            raise ReplicaError("replica not found")
        if replica["authority_gateway_id"] == local_gateway:
            raise ReplicaError("this gateway already holds the room authority")
        if conn.execute(
            "SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)
        ).fetchone():
            raise RoomConflictError(
                "room_id already exists in the local authoritative store"
            )
        if conn.execute(
            "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?",
            (room_id,),
        ).fetchone():
            raise RoomConflictError("room_id belongs to a disbanded room")

        previous_gateway = str(replica["authority_gateway_id"])
        previous_epoch = int(replica["authority_epoch"])
        target_epoch = previous_epoch + 1
        last_seq = int(replica["last_seq"])
        claim_seq = last_seq + 1
        claim_event_id = f"system:authority-claimed:{target_epoch}"
        claim_actor_json = _canonical_json(
            {"kind": "system", "id": "authority-control"},
            label="actor",
            max_bytes=4 * 1024,
        )
        claim_payload_json = _canonical_json(
            {
                "previous_gateway_id": previous_gateway,
                "authority_gateway_id": local_gateway,
                "authority_epoch": target_epoch,
                "promoted_from_replica": True,
                "reason": reason,
            },
            label="payload",
            max_bytes=MAX_EVENT_JSON_BYTES,
        )
        claim_bytes = (
            len(claim_event_id.encode("utf-8"))
            + len(b"authority.claimed")
            + len(claim_actor_json.encode("utf-8"))
            + len(claim_payload_json.encode("utf-8"))
        )

        conn.execute(
            """INSERT INTO hosted_rooms
               (room_id, name, members_json, authority_gateway_id,
                authority_epoch, next_seq, event_bytes, revision,
                created_at, updated_at, disbanded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)""",
            (
                room_id,
                replica["name"],
                replica["members_json"],
                local_gateway,
                target_epoch,
                claim_seq + 1,
                int(replica["event_bytes"]) + claim_bytes,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM hosted_room_replica_events WHERE room_id=?""",
            (room_id,),
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES (?, ?, ?, 'authority.claimed', ?, ?, ?, ?)""",
            (
                room_id,
                claim_seq,
                claim_event_id,
                claim_actor_json,
                target_epoch,
                claim_payload_json,
                now,
            ),
        )
        conn.execute(
            "DELETE FROM hosted_room_replica_events WHERE room_id=?", (room_id,)
        )
        conn.execute(
            "DELETE FROM hosted_room_replicas WHERE room_id=?", (room_id,)
        )
    return {
        "room_id": room_id,
        "authority_gateway_id": local_gateway,
        "authority_epoch": target_epoch,
        "previous_gateway_id": previous_gateway,
        "previous_epoch": previous_epoch,
        "claim_seq": claim_seq,
        "latest_seq": claim_seq,
    }


def demote_room(
    db_path: Path | str,
    *,
    room_id: Any,
    observed_gateway_id: Any,
    observed_epoch: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence THIS gateway's stale room authority against a proven newer epoch.

    Called when a returning gateway observes (via a replicated
    ``authority.claimed`` event or a transport rejection) that another gateway
    now owns the room at a higher epoch. Appends ``authority.lost`` and adopts
    the observed lineage so no further local sends can be committed at the
    stale epoch. Idempotent for repeated observations of the same lineage.
    """
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    observed_gateway_id = _validate_identifier(
        observed_gateway_id,
        label="observed_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    if (
        isinstance(observed_epoch, bool)
        or not isinstance(observed_epoch, int)
        or observed_epoch < 1
    ):
        raise ReplicaError("observed_epoch must be a positive integer")
    now = time.time() if now is None else float(now)
    local_gateway = local_authority_gateway_id()

    with _replica_transaction(db_path) as conn:
        row = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, next_seq
                 FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL""",
            (room_id,),
        ).fetchone()
        if row is None:
            raise ReplicaError("room not found in the local authoritative store")
        current_gateway = str(row["authority_gateway_id"])
        current_epoch = int(row["authority_epoch"])
        if (
            current_gateway == observed_gateway_id
            and current_epoch == observed_epoch
        ):
            return {
                "room_id": room_id,
                "authority_gateway_id": current_gateway,
                "authority_epoch": current_epoch,
                "idempotent": True,
            }
        if observed_epoch <= current_epoch:
            raise ReplicaEpochRegressionError(
                "observed epoch does not supersede the stored authority"
            )
        if current_gateway != local_gateway:
            raise ReplicaError(
                "room is not locally authoritative; nothing to demote"
            )
        seq = int(row["next_seq"])
        lost_actor_json = _canonical_json(
            {"kind": "system", "id": "authority-control"},
            label="actor",
            max_bytes=4 * 1024,
        )
        lost_payload_json = _canonical_json(
            {
                "previous_gateway_id": current_gateway,
                "authority_gateway_id": observed_gateway_id,
                "authority_epoch": observed_epoch,
            },
            label="payload",
            max_bytes=MAX_EVENT_JSON_BYTES,
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES (?, ?, ?, 'authority.lost', ?, ?, ?, ?)""",
            (
                room_id,
                seq,
                f"system:authority-lost:{observed_epoch}",
                lost_actor_json,
                observed_epoch,
                lost_payload_json,
                now,
            ),
        )
        conn.execute(
            """UPDATE hosted_rooms
                  SET authority_gateway_id=?, authority_epoch=?,
                      next_seq=next_seq+1, revision=revision+1, updated_at=?
                WHERE room_id=?""",
            (observed_gateway_id, observed_epoch, now, room_id),
        )
    return {
        "room_id": room_id,
        "authority_gateway_id": observed_gateway_id,
        "authority_epoch": observed_epoch,
        "idempotent": False,
    }
