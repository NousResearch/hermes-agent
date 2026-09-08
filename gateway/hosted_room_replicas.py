"""Passive replica store for hosted Group Chat rooms.

The authority gateway owns a room's ordered log in ``gateway/hosted_rooms.py``.
This module gives every OTHER participant gateway a durable local copy of that
log:

``ingest_page()`` persists ``groups.log`` replay pages idempotently while
refusing gaps, conflicting overlap, forged authority changes, and resurrection
after a terminal disband event. A replica deliberately remains passive: safe
takeover needs a globally exclusive lease or quorum, which this storage layer
does not provide.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gateway.hosted_rooms import (
    MAX_ACTOR_ID_CHARS,
    MAX_EVENT_ID_CHARS,
    MAX_EVENT_JSON_BYTES,
    MAX_GATEWAY_EVENT_BYTES,
    MAX_LOG_LIMIT,
    MAX_LOG_PAGE_BYTES,
    MAX_ROOM_ID_CHARS,
    HostedRoomError,
    _canonical_json,
    _connect,
    _prune_disbanded_rooms_locked,
    _transaction,
    _validate_actor,
    _validate_event_kind,
    _validate_identifier,
    _validate_members,
    _validate_room_name,
)

from gateway.hosted_room_safety import _prune_disbanded_replicas_locked
from gateway import hosted_room_passive_lineage as lineage

MAX_REPLICA_ROOMS = 256
MAX_REPLICA_EVENT_BYTES = MAX_GATEWAY_EVENT_BYTES


class ReplicaError(HostedRoomError):
    """Base class for invalid or conflicting replica operations."""


class ReplicaGapError(ReplicaError):
    """A page does not start at the replica's next expected sequence."""


class ReplicaCapacityError(ReplicaError):
    """Copying may resume after space or a replica slot becomes available."""


class ReplicaHistoryExpiredError(ReplicaError):
    """A compacted replica keeps its identity but no longer has replay data."""

    reason = "replica_history_expired"


class ReplicaLineageUnverifiedError(ReplicaError):
    """A replica cannot prove the complete authority lineage it was given."""

    reason = "replica_lineage_unverified"


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
            updated_at REAL NOT NULL,
            disbanded_at REAL,
            quarantined_at REAL,
            quarantine_reason TEXT
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
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(hosted_room_replicas)")
    }
    for name, kind in (("replica_version", "INTEGER"), ("lineage_sha256", "TEXT")):
        if name not in columns:
            conn.execute(f"ALTER TABLE hosted_room_replicas ADD COLUMN {name} {kind}")
    if "disbanded_at" not in columns:
        conn.execute("ALTER TABLE hosted_room_replicas ADD COLUMN disbanded_at REAL")
    if "quarantined_at" not in columns:
        conn.execute("ALTER TABLE hosted_room_replicas ADD COLUMN quarantined_at REAL")
    if "quarantine_reason" not in columns:
        conn.execute("ALTER TABLE hosted_room_replicas ADD COLUMN quarantine_reason TEXT")
    conn.execute(
        """UPDATE hosted_room_replicas
              SET disbanded_at=(
                    SELECT MIN(created_at)
                      FROM hosted_room_replica_events
                     WHERE hosted_room_replica_events.room_id =
                           hosted_room_replicas.room_id
                       AND kind='room.disbanded'
                  )
            WHERE disbanded_at IS NULL
              AND EXISTS (
                    SELECT 1 FROM hosted_room_replica_events
                     WHERE hosted_room_replica_events.room_id =
                           hosted_room_replicas.room_id
                       AND kind='room.disbanded'
                  )"""
    )


def _audit_existing_replicas_locked(conn: sqlite3.Connection) -> None:
    """Quarantine lineage written by the pre-fix replica implementation."""
    for row in conn.execute(
        """SELECT * FROM hosted_room_replicas"""
    ).fetchall():
        room_id = str(row["room_id"])
        events = conn.execute(
            """SELECT seq, event_id, authority_epoch, kind, actor_json,
                      payload_json, created_at
                 FROM hosted_room_replica_events
                WHERE room_id=? ORDER BY seq""",
            (room_id,),
        ).fetchall()
        reasons: list[str] = []
        seqs = [int(event["seq"]) for event in events]
        event_ids = [str(event["event_id"]) for event in events]
        last_seq = int(row["last_seq"])
        latest_seq = int(row["latest_seq"])
        spans = None
        if "replica_version" in row.keys() and row["replica_version"] is not None:
            try:
                spans = lineage.replica_history_locked(conn, row)
            except lineage.PassiveLineageError:
                reasons.append("unverified_authority_lineage")
        elif int(row["authority_epoch"]) != 1:
            reasons.append("unverified_authority_epoch")
        if seqs != list(range(1, last_seq + 1)):
            reasons.append("non_contiguous_history")
        if len(set(event_ids)) != len(event_ids):
            reasons.append("duplicate_event_id")
        if latest_seq < last_seq:
            reasons.append("coverage_regression")
        disband_positions = [
            index for index, event in enumerate(events)
            if event["kind"] == "room.disbanded"
        ]
        if disband_positions and disband_positions != [len(events) - 1]:
            reasons.append("events_after_disband")
        if disband_positions and last_seq != latest_seq:
            reasons.append("incomplete_terminal_history")
        if spans is None and any(
            event["authority_epoch"] != int(row["authority_epoch"])
            for event in events
        ):
            reasons.append("mixed_authority_lineage")
        try:
            _validate_identifier(
                row["authority_gateway_id"],
                label="authority_gateway_id",
                max_chars=MAX_ACTOR_ID_CHARS,
            )
            for event in events:
                kind = _validate_event_kind(event["kind"])
                _validate_identifier(
                    event["event_id"],
                    label="event_id",
                    max_chars=MAX_EVENT_ID_CHARS,
                )
                actor, _ = _validate_actor(
                    json.loads(event["actor_json"]), kind=kind
                )
                if spans is not None:
                    lineage.event_span(spans, event)
                if (spans is None and actor["kind"] == "gateway"
                        and actor["id"] != str(row["authority_gateway_id"])):
                    reasons.append("gateway_actor_authority_mismatch")
                payload = json.loads(event["payload_json"])
                if not isinstance(payload, dict):
                    raise ReplicaError("event payload is not an object")
                if not math.isfinite(float(event["created_at"])):
                    raise ReplicaError("event timestamp is not finite")
        except (HostedRoomError, TypeError, ValueError, json.JSONDecodeError):
            reasons.append("invalid_event_shape")

        recomputed_bytes = sum(
            len(str(event["event_id"]).encode("utf-8"))
            + len(str(event["kind"]).encode("utf-8"))
            + len(str(event["actor_json"]).encode("utf-8"))
            + len(str(event["payload_json"]).encode("utf-8"))
            for event in events
        )
        if recomputed_bytes != int(row["event_bytes"]):
            conn.execute(
                "UPDATE hosted_room_replicas SET event_bytes=? WHERE room_id=?",
                (recomputed_bytes, room_id),
            )
        if reasons and row["quarantine_reason"] is None:
            conn.execute(
                """UPDATE hosted_room_replicas
                      SET quarantined_at=?, quarantine_reason=?
                    WHERE room_id=?""",
                (time.time(), reasons[0], room_id),
            )


@contextmanager
def _replica_transaction(db_path: Path | str):
    with _transaction(db_path, immediate=True) as conn:
        _initialize_replica_schema(conn)
        # A still-running #99047 process can write after migration. Re-audit
        # inside the same write transaction before every read or extension.
        _audit_existing_replicas_locked(conn)
        yield conn


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


def _validate_non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplicaError(f"{label} must be a non-negative integer")
    return value


def _validate_page(
    page: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int, bool]:
    if not isinstance(page, dict):
        raise ReplicaError("page must be an object")
    v2 = "replica_version" in page
    if v2:
        if type(page["replica_version"]) is not int or page["replica_version"] != 2 or set(page) != {
            "events", "authority", "cursor", "latest_seq", "has_more", "replica_version", "lineage_sha256"
        }:
            raise ReplicaError("invalid v2 replica fields")
    elif "lineage_sha256" in page:
        raise ReplicaError("lineage digest requires v2")
    _canonical_json(page, label="page", max_bytes=MAX_LOG_PAGE_BYTES)
    events = page.get("events")
    authority = page.get("authority")
    if not isinstance(events, list):
        raise ReplicaError("page.events must be a list")
    if len(events) > MAX_LOG_LIMIT:
        raise ReplicaError(f"page.events cannot exceed {MAX_LOG_LIMIT} events")
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
    if v2 and (gateway_id != authority.get("gateway_id") or set(authority) != {"gateway_id", "epoch"} or epoch >= 2**63):
        raise ReplicaError("invalid v2 authority")
    cursor = _validate_non_negative_int(page.get("cursor"), label="page.cursor")
    latest_seq = _validate_non_negative_int(
        page.get("latest_seq"), label="page.latest_seq"
    )
    if v2 and (cursor >= 2**63 or latest_seq >= 2**63):
        raise ReplicaError("v2 replay coordinates exceed SQLite bounds")
    has_more = page.get("has_more")
    if not isinstance(has_more, bool):
        raise ReplicaError("page.has_more must be a boolean")
    if cursor > latest_seq:
        raise ReplicaError("page.cursor cannot exceed page.latest_seq")
    if has_more != (cursor < latest_seq):
        raise ReplicaError("page.has_more does not match its replay cursor")

    normalized_events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
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
        event_room_id = _validate_identifier(
            event.get("room_id"),
            label="event.room_id",
            max_chars=MAX_ROOM_ID_CHARS,
        )
        event_id = _validate_identifier(
            event.get("event_id"),
            label="event.event_id",
            max_chars=MAX_EVENT_ID_CHARS,
        )
        if event_id in event_ids:
            raise ReplicaError("page repeats an event_id")
        event_ids.add(event_id)
        kind = _validate_event_kind(event.get("kind"))
        actor, actor_json = _validate_actor(event.get("actor"), kind=kind)
        if v2 and (actor != event.get("actor") or event_id != event.get("event_id")
                or event_room_id != event.get("room_id") or kind != event.get("kind")):
            raise ReplicaError("v2 event identifiers and actors must be canonical")
        if not v2 and actor["kind"] == "gateway" and actor["id"] != gateway_id:
            raise ReplicaError("gateway actor does not match page authority")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ReplicaError("event.payload must be an object")
        payload_json = _canonical_json(
            payload, label="payload", max_bytes=MAX_EVENT_JSON_BYTES
        )
        event_epoch = event.get("authority_epoch")
        if (
            isinstance(event_epoch, bool)
            or not isinstance(event_epoch, int)
            or event_epoch < 1
            or event_epoch > epoch
        ):
            raise ReplicaError("event.authority_epoch is outside the page lineage")
        created_at = event.get("created_at")
        if (
            isinstance(created_at, bool)
            or not isinstance(created_at, (int, float))
            or not math.isfinite(float(created_at))
        ):
            raise ReplicaError("event.created_at must be a finite number")
        normalized_events.append(
            {
                "room_id": event_room_id,
                "seq": seq,
                "event_id": event_id,
                "kind": kind,
                "actor": actor,
                "actor_json": actor_json,
                "authority_epoch": event_epoch,
                "payload": payload,
                "payload_json": payload_json,
                "created_at": float(created_at),
            }
        )
    if normalized_events and normalized_events[-1]["seq"] != cursor:
        raise ReplicaError("page.cursor must equal the last returned sequence")
    if not normalized_events and cursor != latest_seq:
        raise ReplicaError("an incomplete replay page must include events")
    return (
        normalized_events,
        {"gateway_id": gateway_id, "epoch": epoch},
        cursor,
        latest_seq,
        has_more,
    )


def ingest_page(
    db_path: Path | str,
    *,
    room_id: Any,
    room_name: Any,
    members: Any,
    page: Any,
    now: float | None = None,
    _authorize: Callable[[sqlite3.Connection], None] | None = None,
) -> dict[str, Any]:
    """Persist one replay page for ``room_id``; idempotent, gap- and
    epoch-regression-safe.

    ``page`` is the verbatim result of the authority's ``groups.log`` call
    (``read_events()``). Its authority stamp is metadata, not authentication;
    network callers must use ``ingest_granted_page`` to verify provenance.
    """
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    room_name = _validate_room_name(room_name)
    _, members_json = _validate_members(members)
    events, authority, _cursor, latest_seq, _has_more = _validate_page(page)
    v2 = page.get("replica_version") == 2
    for event in events:
        if event["room_id"] != room_id:
            raise ReplicaError("page contains an event for a different room")
    now = time.time() if now is None else float(now)
    with _replica_transaction(db_path) as conn:
        if _authorize is not None:
            _authorize(conn)
        from gateway.hosted_room_replica_retirement import copy_retired_locked, copy_scope_matches_locked
        if copy_retired_locked(conn, room_id):
            raise ReplicaHistoryExpiredError("Group Chat copy has been retired")
        if not copy_scope_matches_locked(
            conn, room_id=room_id, authority_gateway_id=authority["gateway_id"],
            authority_epoch=authority["epoch"], members_json=members_json,
            replica_version=2 if v2 else None, lineage_sha256=page.get("lineage_sha256"),
        ):
            raise ReplicaError("copy scope differs from owner enrollment")
        spans = None
        if v2:
            enrolled = lineage.current_locked(conn, room_id)
            spans = lineage.enrolled_history(enrolled)
            for event in events:
                lineage.event_span(spans, event)
        _prune_disbanded_replicas_locked(conn, now=now)
        if conn.execute(
            "SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)
        ).fetchone():
            raise ReplicaError("room_id is already locally authoritative")
        if conn.execute(
            "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)
        ).fetchone():
            raise ReplicaError("room_id is permanently retired on this gateway")
        row = conn.execute(
            """SELECT * FROM hosted_room_replicas WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if row is None:
            _prune_disbanded_replicas_locked(
                conn,
                now=None,
                max_replica_rooms=max(0, MAX_REPLICA_ROOMS - 1),
            )
            reservation = conn.execute(
                """SELECT owner_kind FROM hosted_room_id_reservations
                    WHERE room_id=?""",
                (room_id,),
            ).fetchone()
            if reservation is not None and reservation["owner_kind"] == "replica":
                raise ReplicaHistoryExpiredError(
                    "replica history expired; room_id remains permanently retired"
                )
            count = conn.execute(
                "SELECT COUNT(*) FROM hosted_room_replicas"
            ).fetchone()[0]
            if int(count) >= MAX_REPLICA_ROOMS:
                raise ReplicaCapacityError("replica room capacity exhausted")
            stored_epoch = 0
            last_seq = 0
            stored_latest = 0
            disbanded_at = None
            if not v2 and authority["epoch"] != 1:
                raise ReplicaLineageUnverifiedError(
                    "replica lineage is incomplete; the first authority epoch is required"
                )
        else:
            stored_epoch = int(row["authority_epoch"])
            last_seq = int(row["last_seq"])
            stored_latest = int(row["latest_seq"])
            disbanded_at = row["disbanded_at"]
            if row["quarantine_reason"] is not None:
                raise ReplicaError(
                    "stored replica is quarantined: " + str(row["quarantine_reason"])
                )
            if (row["name"] != room_name and _authorize is None) or row["members_json"] != members_json:
                raise ReplicaError("replica metadata conflicts with stored state")
            if not v2 and (
                row["authority_gateway_id"] != authority["gateway_id"]
                or stored_epoch != authority["epoch"]
            ):
                raise ReplicaLineageUnverifiedError(
                    "replica authority changed without a verified takeover lineage"
                )
            if latest_seq < stored_latest:
                raise ReplicaError("page.latest_seq regresses stored replica coverage")

        for event in events:
            if not v2 and event["authority_epoch"] != stored_epoch and row is not None:
                raise ReplicaError("event authority conflicts with stored replica lineage")
            existing = conn.execute(
                """SELECT seq, event_id, kind, actor_json, authority_epoch,
                          payload_json, created_at
                     FROM hosted_room_replica_events
                    WHERE room_id=? AND (seq=? OR event_id=?)""",
                (room_id, int(event["seq"]), event["event_id"]),
            ).fetchall()
            for stored in existing:
                if (
                    int(stored["seq"]) != int(event["seq"])
                    or stored["event_id"] != event["event_id"]
                    or stored["kind"] != event["kind"]
                    or stored["actor_json"] != event["actor_json"]
                    or stored["authority_epoch"] != event["authority_epoch"]
                    or stored["payload_json"] != event["payload_json"]
                    or float(stored["created_at"]) != event["created_at"]
                ):
                    raise ReplicaError("replayed event conflicts with stored history")
            if int(event["seq"]) <= last_seq and not existing:
                raise ReplicaError("stored replica history is incomplete")

        new_events = [e for e in events if int(e["seq"]) > last_seq]
        if new_events and int(new_events[0]["seq"]) != last_seq + 1:
            raise ReplicaGapError(
                "page skips sequences the replica has not stored"
            )
        if disbanded_at is not None and new_events:
            raise ReplicaError("a disbanded Group Chat cannot accept later events")
        disband_indexes = [
            index for index, event in enumerate(new_events)
            if event["kind"] == "room.disbanded"
        ]
        if disband_indexes and disband_indexes != [len(new_events) - 1]:
            raise ReplicaError("room.disbanded must be the terminal event")
        if disband_indexes and int(new_events[-1]["seq"]) != latest_seq:
            raise ReplicaError("room.disbanded must complete the source history")

        projected_name = row["name"] if row is not None else room_name
        if _authorize is not None:
            # A sender can observe metadata ahead of this bounded page. Follow
            # committed rename events instead of relabeling an incomplete prefix.
            for event in new_events:
                if event["kind"] == "room.renamed":
                    projected_name = _validate_room_name(json.loads(event["payload_json"]).get("name"))

        event_sizes = [_event_bytes(event) for event in new_events]
        added_bytes = sum(event_sizes)
        gateway_bytes = int(
            conn.execute(
                """SELECT
                       COALESCE((SELECT SUM(event_bytes) FROM hosted_rooms), 0) +
                       COALESCE((SELECT SUM(event_bytes)
                                   FROM hosted_room_replicas), 0)"""
            ).fetchone()[0]
        )
        if gateway_bytes + added_bytes > MAX_REPLICA_EVENT_BYTES:
            replica_bytes = int(
                conn.execute(
                    "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_room_replicas"
                ).fetchone()[0]
            )
            _prune_disbanded_rooms_locked(
                conn,
                now=None,
                max_gateway_event_bytes=max(
                    0, MAX_REPLICA_EVENT_BYTES - added_bytes - replica_bytes
                ),
            )
            hosted_bytes = int(
                conn.execute(
                    "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
                ).fetchone()[0]
            )
            _prune_disbanded_replicas_locked(
                conn,
                now=None,
                max_replica_event_bytes=max(
                    0, MAX_REPLICA_EVENT_BYTES - added_bytes - hosted_bytes
                ),
            )
            gateway_bytes = hosted_bytes + int(
                conn.execute(
                    "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_room_replicas"
                ).fetchone()[0]
            )
        if gateway_bytes + added_bytes > MAX_REPLICA_EVENT_BYTES:
            raise ReplicaCapacityError("replica event storage exhausted")
        for event in new_events:
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
                    event["actor_json"],
                    event["authority_epoch"],
                    event["payload_json"],
                    event["created_at"],
                ),
            )
        new_last = int(new_events[-1]["seq"]) if new_events else last_seq
        terminal_at = (
            new_events[-1]["created_at"] if disband_indexes else disbanded_at
        )
        prefix = lineage.at_sequence(spans, new_last) if v2 else None
        prefix_authority = {"gateway_id": prefix.gateway_id, "epoch": prefix.epoch} if prefix else authority
        if row is None:
            conn.execute(
                """INSERT INTO hosted_room_replicas
                   (room_id, name, members_json, authority_gateway_id,
                    authority_epoch, last_seq, latest_seq, event_bytes,
                    created_at, updated_at, disbanded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    room_id,
                    projected_name,
                    members_json,
                    prefix_authority["gateway_id"],
                    prefix_authority["epoch"],
                    new_last,
                    latest_seq,
                    added_bytes,
                    now,
                    now,
                    terminal_at,
                ),
            )
        else:
            conn.execute(
                """UPDATE hosted_room_replicas
                      SET last_seq=?, latest_seq=?, event_bytes=event_bytes+?,
                          updated_at=?, disbanded_at=?, name=?
                    WHERE room_id=?""",
                (
                    new_last,
                    latest_seq,
                    added_bytes,
                    now,
                    terminal_at,
                    projected_name,
                    room_id,
                ),
            )
        if v2:
            conn.execute("""UPDATE hosted_room_replicas SET authority_gateway_id=?,authority_epoch=?,
                replica_version=2,lineage_sha256=? WHERE room_id=?""",
                (prefix.gateway_id, prefix.epoch, page["lineage_sha256"], room_id))
    return {
        **({"replica_version": 2, "lineage_sha256": page["lineage_sha256"],
            "lineage_status": lineage.status(spans, new_last)} if v2 else {}),
        "room_id": room_id,
        "stored_seq": new_last,
        "ingested": len(new_events),
        "authority": authority,
        "caught_up": new_last >= latest_seq,
    }


def replica_state(db_path: Path | str, *, room_id: Any) -> dict[str, Any]:
    """Return the stored replica's coverage and authority lineage."""
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    with _replica_transaction(db_path) as conn:
        from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
        from gateway.hosted_rooms_common import table_exists
        retired = conn.execute(f"SELECT retired_at FROM {RETIREMENT_TABLE} WHERE room_id=?", (room_id,)).fetchone() if table_exists(conn, RETIREMENT_TABLE) else None
        row = conn.execute(
            """SELECT * FROM hosted_room_replicas WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        lineage_fields = lineage.state_fields_locked(conn, row) if row is not None else {}
        reservation = (
            conn.execute(
                """SELECT owner_kind FROM hosted_room_id_reservations
                    WHERE room_id=?""",
                (room_id,),
            ).fetchone()
            if row is None
            else None
        )
    if row is None:
        if reservation is not None and reservation["owner_kind"] == "replica":
            raise ReplicaHistoryExpiredError(
                "replica history expired; room_id remains permanently retired"
            )
        raise ReplicaError("replica not found")
    return {
        **lineage_fields,
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
        "disbanded_at": (
            float(row["disbanded_at"]) if row["disbanded_at"] is not None else None
        ),
        "safety_status": (
            "quarantined" if row["quarantine_reason"] is not None else "retired" if retired is not None else "passive"
        ),
        **({"copy_retired_at": float(retired[0])} if retired is not None else {}),
        "safety_reason": row["quarantine_reason"],
    }
