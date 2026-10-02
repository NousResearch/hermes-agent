"""Replica store and gated takeover primitives for hosted Group Chat rooms.

The authority gateway owns a room's ordered log in ``gateway/hosted_rooms.py``. Every OTHER participant
gateway can keep a durable local copy: ``ingest_page()`` persists ``groups.log`` replay pages idempotently
and refuses gaps, conflicting overlap, forged authority changes and resurrection after a terminal disband.
``promote_replica()`` continues a copied room locally at ``epoch + 1`` with a lineage-proving
``authority.claimed`` event; ``demote_room()`` records ``authority.lost`` when a returning stale authority
is shown a newer epoch. Storage primitives only: ``require_takeover()`` is the one gate that decides whether
the RPCs may reach them, and it stays closed until Hermes can select one globally exclusive authority.
Until then the room store quarantines every takeover lineage it records (``gateway/hosted_room_safety.py``).
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator

from gateway.hosted_rooms import (
    MAX_ACTOR_ID_CHARS, MAX_EVENT_ID_CHARS, MAX_GATEWAY_EVENT_BYTES, MAX_LOG_LIMIT, MAX_LOG_PAGE_BYTES,
    MAX_MEMBERS_JSON_BYTES, MAX_ROOM_NAME_CHARS,
    MAX_ROOM_ID_CHARS, HostedRoomError, RoomConflictError, _actor_json, _canonical_json, _payload_json,
    _prune_disbanded_rooms_locked, _room_id, _transaction, _validate_actor, _validate_event_kind,
    _validate_identifier, _validate_members, _validate_room_name, local_authority_gateway_id)
from gateway.hosted_room_safety import _prune_disbanded_replicas_locked, _raise_if_quarantined
from gateway.hosted_rooms_common import DbPath, bounded_int, clock, table_columns, utf8_len

MAX_REPLICA_ROOMS = 256
# Replica payload shares the gateway's event budget: both live in the same bounded store.
MAX_REPLICA_EVENT_BYTES = MAX_GATEWAY_EVENT_BYTES
_SYSTEM_ACTOR = {"kind": "system", "id": "authority-control"}
_EVENT_COLUMNS = "(room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at)"
_INSERT_ROOM_EVENT = f"INSERT INTO hosted_room_events {_EVENT_COLUMNS} VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
_INSERT_REPLICA_EVENT = f"INSERT INTO hosted_room_replica_events {_EVENT_COLUMNS} VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
_SELECT_REPLICA = "SELECT * FROM hosted_room_replicas WHERE room_id=?"


class ReplicaError(HostedRoomError): """Base class for invalid or conflicting replica operations."""
class ReplicaGapError(ReplicaError): """A page does not start at the replica's next expected sequence."""
class ReplicaCapacityError(ReplicaError): """Copying may resume once space or a replica slot is free."""
class ReplicaEpochRegressionError(ReplicaError): """A demotion carries an older authority epoch than stored."""


class ReplicaHistoryExpiredError(ReplicaError):
    """A compacted replica keeps its identity but no longer has replay data."""

    reason = "replica_history_expired"


class ReplicaLineageUnverifiedError(ReplicaError):
    """A replica cannot prove the complete authority lineage it was given."""

    reason = "replica_lineage_unverified"


class TakeoverGateClosedError(ReplicaError):
    """``require_takeover`` refused a takeover RPC; ``reason`` names the missing guarantee."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


#: Capability features the takeover RPCs advertise, only while ``takeover_enabled()``.
TAKEOVER_FEATURES = ("log_replication", "authority_takeover")
_TAKEOVER_REFUSALS = {
    "promote": ("authority_takeover_disabled",
                "Group Chat takeover is disabled until Hermes can select one globally exclusive authority."),
    "demote": ("authority_takeover_disabled",
               "Group Chat demotion is disabled until Hermes can select one globally exclusive authority."),
    "replicate": ("replica_provenance_required",
                  "Group Chat replication is disabled until Hermes can verify that a page came from the "
                  "room's authority."),
}


def takeover_enabled() -> bool:
    """Whether ``groups.promote``, ``groups.demote`` and ``groups.replicate`` may run.

    Closed. Takeover is only safe once Hermes can select one globally exclusive authority for a
    room: today two gateways can both promote a copy, a demoted gateway cannot be proven to have
    stopped writing, and a page handed to ``groups.replicate`` is no proof of what the authority
    wrote. Exclusive-authority recovery opens this gate; it is the only switch.

    Opening it is not enough on its own. The triggers in ``gateway/hosted_room_safety.py`` treat
    every ``authority.claimed`` promoted from a copy, and every ``authority.lost``, as an unproven
    takeover and quarantine the room. Recovery must also mark the takeovers it verifies, and teach
    those triggers to accept that mark; until then a takeover through an open gate still leaves the
    room read-only (``test_open_gate_promotion_still_ends_quarantined`` pins this).
    """
    return False


def require_takeover(operation: str) -> None:
    """The one gate in front of the takeover RPCs: refuse with a typed reason while it is closed."""
    if takeover_enabled():
        return
    reason, message = _TAKEOVER_REFUSALS[operation]
    raise TakeoverGateClosedError(message, reason=reason)


def _initialize_replica_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_replicas (
            room_id TEXT PRIMARY KEY, name TEXT NOT NULL, members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            last_seq INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
            latest_seq INTEGER NOT NULL DEFAULT 0, event_bytes INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            disbanded_at REAL, quarantined_at REAL, quarantine_reason TEXT
        )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS hosted_room_replica_events (
            room_id TEXT NOT NULL, seq INTEGER NOT NULL CHECK (seq >= 1), event_id TEXT NOT NULL,
            kind TEXT NOT NULL, actor_json TEXT NOT NULL, authority_epoch INTEGER,
            payload_json TEXT NOT NULL, created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq)
        )""")
    columns = table_columns(conn, "hosted_room_replicas")
    for column, declaration in (("disbanded_at", "REAL"), ("quarantined_at", "REAL"), ("quarantine_reason", "TEXT")):
        if column not in columns:
            conn.execute(f"ALTER TABLE hosted_room_replicas ADD COLUMN {column} {declaration}")
    # Copies stored before terminal state was tracked keep their disband as an event only.
    conn.execute("""UPDATE hosted_room_replicas
              SET disbanded_at=(SELECT MIN(created_at) FROM hosted_room_replica_events
                                 WHERE hosted_room_replica_events.room_id=hosted_room_replicas.room_id
                                   AND kind='room.disbanded')
            WHERE disbanded_at IS NULL AND EXISTS (
                SELECT 1 FROM hosted_room_replica_events
                 WHERE hosted_room_replica_events.room_id=hosted_room_replicas.room_id
                   AND kind='room.disbanded')""")


def _replica_header_bounds_sql() -> str:
    """Fixed-column byte/type preflight; SQL character length is NUL-truncated."""
    text_limits = {
        "room_id": MAX_ROOM_ID_CHARS * 4, "name": MAX_ROOM_NAME_CHARS * 4,
        "authority_gateway_id": MAX_ACTOR_ID_CHARS * 4,
        "members_json": MAX_MEMBERS_JSON_BYTES, "quarantine_reason": 256 * 4,
    }
    checks = []
    for column, bound in text_limits.items():
        check = f"(typeof({column})='text' AND LENGTH(CAST({column} AS BLOB))<={int(bound)})"
        checks.append(f"({column} IS NULL OR {check})" if column == "quarantine_reason" else check)
    checks.extend(f"typeof({column})='integer'" for column in
                  ("authority_epoch", "last_seq", "latest_seq", "event_bytes"))
    checks.extend(f"typeof({column}) IN ('integer','real')" for column in ("created_at", "updated_at"))
    checks.extend(f"typeof({column}) IN ('null','integer','real')" for column in ("disbanded_at", "quarantined_at"))
    return " AND ".join(checks)


def _audit_existing_replicas_locked(conn: sqlite3.Connection) -> None:
    """Quarantine invalid copies with bounded, streaming history validation.

    SQL preflights size/count; Python streams bounded individual event rows
    with the existing validators. Excess bytes are retained, not a pruning permit.
    """
    from gateway import hosted_rooms as limits

    for row in conn.execute(
        f"""SELECT room_id,authority_gateway_id,authority_epoch,last_seq,
                   latest_seq,event_bytes,disbanded_at,
                   CASE WHEN quarantine_reason IS NOT NULL THEN 1 END AS quarantine_reason
              FROM hosted_room_replicas WHERE {_replica_header_bounds_sql()}"""
    ):
        room_id = str(row["room_id"])
        stats = conn.execute(
            """SELECT COUNT(*) AS count,
                COALESCE(SUM(COALESCE(LENGTH(CAST(event_id AS BLOB)),0) +
                    COALESCE(LENGTH(CAST(kind AS BLOB)),0) +
                    COALESCE(LENGTH(CAST(actor_json AS BLOB)),0) +
                    COALESCE(LENGTH(CAST(payload_json AS BLOB)),0)),0) AS bytes,
                COALESCE(MAX(COALESCE(LENGTH(CAST(event_id AS BLOB)),0) +
                    COALESCE(LENGTH(CAST(kind AS BLOB)),0) +
                    COALESCE(LENGTH(CAST(actor_json AS BLOB)),0) +
                    COALESCE(LENGTH(CAST(payload_json AS BLOB)),0)),0) AS largest,
                COALESCE(MAX(CASE WHEN typeof(seq)!='integer' OR typeof(authority_epoch)!='integer'
                    OR typeof(created_at) NOT IN ('integer','real') THEN 1 ELSE 0 END),0) AS bad_scalars
                FROM hosted_room_replica_events WHERE room_id=?""", (room_id,)).fetchone()
        stored_bytes = int(stats["bytes"])
        if stored_bytes != int(row["event_bytes"]):
            conn.execute("UPDATE hosted_room_replicas SET event_bytes=? WHERE room_id=?", (stored_bytes, room_id))
        # Previously classified evidence stays opaque and read-only. It is never
        # necessary to decode a quarantined payload to keep it or report its size.
        if row["quarantine_reason"] is not None:
            continue
        reasons: list[str] = []
        last_seq, latest_seq = int(row["last_seq"]), int(row["latest_seq"])
        if int(row["authority_epoch"]) != 1:
            reasons.append("unverified_authority_epoch")
        if stats["count"] != last_seq:
            reasons.append("non_contiguous_history")
        if latest_seq < last_seq:
            reasons.append("coverage_regression")
        if stats["bad_scalars"]:
            reasons.append("invalid_event_shape")
        if (stats["count"] > limits.MAX_EVENTS_PER_ROOM + limits.CONTROL_EVENT_COUNT_RESERVE
                or stats["largest"] > MAX_LOG_PAGE_BYTES):
            # A current read envelope is not evidence of corruption. Leave
            # oversized history unvalidated and held, without inventing an
            # integrity quarantine that would forbid authorized retirement.
            pass
        elif not stats["bad_scalars"]:
            identities = conn.execute("SELECT COUNT(DISTINCT event_id) FROM hosted_room_replica_events WHERE room_id=?",
                                      (room_id,)).fetchone()[0]
            if identities != stats["count"]:
                reasons.append("duplicate_event_id")
            try:
                _validate_identifier(row["authority_gateway_id"], label="authority_gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
                for index, event in enumerate(conn.execute(
                    """SELECT seq,event_id,authority_epoch,kind,actor_json,payload_json,created_at
                       FROM hosted_room_replica_events WHERE room_id=? ORDER BY seq""", (room_id,)), 1):
                    if int(event["seq"]) != index:
                        reasons.append("non_contiguous_history")
                    if event["kind"] == "room.disbanded":
                        if index != stats["count"]:
                            reasons.append("events_after_disband")
                        if last_seq != latest_seq:
                            reasons.append("incomplete_terminal_history")
                    if event["authority_epoch"] != int(row["authority_epoch"]):
                        reasons.append("mixed_authority_lineage")
                    kind = _validate_event_kind(event["kind"])
                    _validate_identifier(event["event_id"], label="event_id", max_chars=MAX_EVENT_ID_CHARS)
                    actor, _ = _validate_actor(json.loads(event["actor_json"]), kind=kind)
                    if actor["kind"] == "gateway" and actor["id"] != str(row["authority_gateway_id"]):
                        reasons.append("gateway_actor_authority_mismatch")
                    if not isinstance(json.loads(event["payload_json"]), dict):
                        raise ReplicaError("event payload is not an object")
                    if not math.isfinite(float(event["created_at"])):
                        raise ReplicaError("event timestamp is not finite")
            except (HostedRoomError, TypeError, ValueError, json.JSONDecodeError, RecursionError):
                reasons.append("invalid_event_shape")
        if reasons:
            conn.execute("UPDATE hosted_room_replicas SET quarantined_at=?, quarantine_reason=? WHERE room_id=?",
                         (clock(None), reasons[0], room_id))


def _replica_read_envelope_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    """Whether current bounded validation can consume this copy, not its validity."""
    count, largest = conn.execute("""SELECT COUNT(*), COALESCE(MAX(
        COALESCE(LENGTH(CAST(event_id AS BLOB)),0) + COALESCE(LENGTH(CAST(kind AS BLOB)),0) +
        COALESCE(LENGTH(CAST(actor_json AS BLOB)),0) + COALESCE(LENGTH(CAST(payload_json AS BLOB)),0)),0)
        FROM hosted_room_replica_events WHERE room_id=?""", (room_id,)).fetchone()
    from gateway import hosted_rooms as limits
    return count <= limits.MAX_EVENTS_PER_ROOM + limits.CONTROL_EVENT_COUNT_RESERVE and largest <= MAX_LOG_PAGE_BYTES


def _load_replica_locked(conn: sqlite3.Connection, room_id: str):
    """Preflight metadata before SELECT*/decode; quota failure retains the copy."""
    bounded = conn.execute(
        f"SELECT ({_replica_header_bounds_sql()}) FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
    if bounded is None:
        return None
    if not bounded[0]:
        raise ReplicaError("stored replica metadata exceeds read bounds; history is preserved")
    if not _replica_read_envelope_locked(conn, room_id):
        raise ReplicaCapacityError("stored replica exceeds current read capacity; history is preserved")
    used = conn.execute("""SELECT
        (SELECT COALESCE(SUM(event_bytes),0) FROM hosted_rooms) +
        (SELECT COALESCE(SUM(event_bytes),0) FROM hosted_room_replicas)""").fetchone()[0]
    from gateway import hosted_rooms as limits
    if used > min(limits.MAX_GATEWAY_EVENT_BYTES, MAX_REPLICA_EVENT_BYTES):
        raise ReplicaCapacityError("stored replica exceeds current capacity; history is preserved")
    row = conn.execute(_SELECT_REPLICA, (room_id,)).fetchone()
    # The physical bound allows every valid UTF-8 character; apply the existing
    # logical character limits only after the bounded fetch.
    if (len(row["room_id"]) > MAX_ROOM_ID_CHARS or len(row["name"]) > MAX_ROOM_NAME_CHARS
            or len(row["authority_gateway_id"]) > MAX_ACTOR_ID_CHARS
            or len(row["quarantine_reason"] or "") > 256):
        raise ReplicaError("stored replica metadata exceeds read bounds; history is preserved")
    return row



@contextmanager
def _replica_transaction(db_path: DbPath) -> Iterator[sqlite3.Connection]:
    """IMMEDIATE transaction over a current replica schema whose stored copies were just re-audited.

    A still-running older process can write replica rows after migration, so the audit runs inside
    the same write transaction before every read, extension or takeover."""
    with _transaction(db_path, immediate=True) as conn:
        _initialize_replica_schema(conn)
        _audit_existing_replicas_locked(conn)
        yield conn


_positive_int = partial(bounded_int, error=ReplicaError, low=1)
_non_negative_int = partial(bounded_int, error=ReplicaError)


def _control_event(kind: str, epoch: int, payload: dict[str, Any]) -> tuple[str, str, str, str]:
    """(event_id, kind, actor_json, payload_json) of the system ``authority.<kind>`` control event for ``epoch``."""
    return f"system:authority-{kind}:{epoch}", f"authority.{kind}", _actor_json(_SYSTEM_ACTOR), _payload_json(payload)


def _append_control_event(
    conn: sqlite3.Connection, room_id: str, seq: int, epoch: int, event: tuple[str, str, str, str], now: float
) -> None:
    event_id, kind, actor_json, payload_json = event
    conn.execute(_INSERT_ROOM_EVENT, (room_id, seq, event_id, kind, actor_json, epoch, payload_json, now))


def _validate_page(page: Any) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Check one verbatim ``read_events()`` page; returns (events, authority, latest_seq)."""
    if not isinstance(page, dict):
        raise ReplicaError("page must be an object")
    _canonical_json(page, label="page", max_bytes=MAX_LOG_PAGE_BYTES)
    events, authority = page.get("events"), page.get("authority")
    if not isinstance(events, list):
        raise ReplicaError("page.events must be a list")
    if len(events) > MAX_LOG_LIMIT:
        raise ReplicaError(f"page.events cannot exceed {MAX_LOG_LIMIT} events")
    if not isinstance(authority, dict):
        raise ReplicaError("page.authority is required for replication")
    gateway_id = _validate_identifier(
        authority.get("gateway_id"), label="page.authority.gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
    epoch = _positive_int(authority.get("epoch"), message="page.authority.epoch must be a positive integer")
    cursor = _non_negative_int(page.get("cursor"), message="page.cursor must be a non-negative integer")
    latest_seq = _non_negative_int(page.get("latest_seq"), message="page.latest_seq must be a non-negative integer")
    has_more = page.get("has_more")
    if not isinstance(has_more, bool):
        raise ReplicaError("page.has_more must be a boolean")
    if cursor > latest_seq:
        raise ReplicaError("page.cursor cannot exceed page.latest_seq")
    if has_more != (cursor < latest_seq):
        raise ReplicaError("page.has_more does not match its replay cursor")
    normalized: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    previous_seq: int | None = None
    for event in events:
        if not isinstance(event, dict):
            raise ReplicaError("page events must be objects")
        seq = _positive_int(event.get("seq"), message="event.seq must be a positive integer")
        if previous_seq is not None and seq != previous_seq + 1:
            raise ReplicaGapError("page events must be contiguous")
        previous_seq = seq
        event_room_id = _validate_identifier(event.get("room_id"), label="event.room_id", max_chars=MAX_ROOM_ID_CHARS)
        event_id = _validate_identifier(event.get("event_id"), label="event.event_id", max_chars=MAX_EVENT_ID_CHARS)
        if event_id in event_ids:
            raise ReplicaError("page repeats an event_id")
        event_ids.add(event_id)
        kind = _validate_event_kind(event.get("kind"))
        actor, actor_json = _validate_actor(event.get("actor"), kind=kind)
        if actor["kind"] == "gateway" and actor["id"] != gateway_id:
            raise ReplicaError("gateway actor does not match page authority")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ReplicaError("event.payload must be an object")
        event_epoch = _positive_int(
            event.get("authority_epoch"), high=epoch, message="event.authority_epoch is outside the page lineage")
        created_at = event.get("created_at")
        if (isinstance(created_at, bool) or not isinstance(created_at, (int, float))
                or not math.isfinite(float(created_at))):
            raise ReplicaError("event.created_at must be a finite number")
        normalized.append({
            "room_id": event_room_id, "seq": seq, "event_id": event_id, "kind": kind, "actor_json": actor_json,
            "authority_epoch": event_epoch, "payload_json": _payload_json(payload), "created_at": float(created_at)})
    if normalized and normalized[-1]["seq"] != cursor:
        raise ReplicaError("page.cursor must equal the last returned sequence")
    if not normalized and cursor != latest_seq:
        raise ReplicaError("an incomplete replay page must include events")
    return normalized, {"gateway_id": gateway_id, "epoch": epoch}, latest_seq


def _event_row(room_id: str, event: dict[str, Any]) -> tuple[Any, ...]:
    return (room_id, event["seq"], event["event_id"], event["kind"], event["actor_json"], event["authority_epoch"],
            event["payload_json"], event["created_at"])


def ingest_page(
    db_path: DbPath, *, room_id: Any, room_name: Any, members: Any, page: Any, now: float | None = None
) -> dict[str, Any]:
    """Persist one verbatim ``read_events()`` page idempotently.

    Refuses sequence gaps, overlap that differs from stored history, authority changes without a
    verified lineage, and events after a terminal ``room.disbanded``.
    """
    room_id = _room_id(room_id)
    room_name = _validate_room_name(room_name)
    _, members_json = _validate_members(members)
    events, authority, latest_seq = _validate_page(page)
    if any(event["room_id"] != room_id for event in events):
        raise ReplicaError("page contains an event for a different room")
    now = clock(now)
    with _replica_transaction(db_path) as conn:
        _prune_disbanded_replicas_locked(conn, now=now)
        if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone():
            raise ReplicaError("room_id is already locally authoritative")
        if conn.execute("SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)).fetchone():
            raise ReplicaError("room_id is permanently retired on this gateway")
        row = _load_replica_locked(conn, room_id)
        if row is None:
            _prune_disbanded_replicas_locked(conn, now=None, max_replica_rooms=max(0, MAX_REPLICA_ROOMS - 1))
            reservation = conn.execute(
                "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?", (room_id,)).fetchone()
            if reservation is not None and reservation["owner_kind"] == "replica":
                raise ReplicaHistoryExpiredError("replica history expired; room_id remains permanently retired")
            if int(conn.execute("SELECT COUNT(*) FROM hosted_room_replicas").fetchone()[0]) >= MAX_REPLICA_ROOMS:
                raise ReplicaError("replica room capacity exhausted")
            if authority["epoch"] != 1:
                raise ReplicaLineageUnverifiedError(
                    "replica lineage is incomplete; the first authority epoch is required")
            stored_epoch, last_seq, disbanded_at = 0, 0, None
        else:
            stored_epoch, last_seq = int(row["authority_epoch"]), int(row["last_seq"])
            disbanded_at = row["disbanded_at"]
            if row["quarantine_reason"] is not None:
                raise ReplicaError("stored replica is quarantined: " + str(row["quarantine_reason"]))
            if row["name"] != room_name or row["members_json"] != members_json:
                raise ReplicaError("replica metadata conflicts with stored state")
            if row["authority_gateway_id"] != authority["gateway_id"] or stored_epoch != authority["epoch"]:
                raise ReplicaLineageUnverifiedError("replica authority changed without a verified takeover lineage")
            if latest_seq < int(row["latest_seq"]):
                raise ReplicaError("page.latest_seq regresses stored replica coverage")
        for event in events:
            if row is not None and event["authority_epoch"] != stored_epoch:
                raise ReplicaError("event authority conflicts with stored replica lineage")
            stored = conn.execute(
                """SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at
                     FROM hosted_room_replica_events WHERE room_id=? AND (seq=? OR event_id=?)""",
                (room_id, event["seq"], event["event_id"])).fetchall()
            if any(tuple(existing) != _event_row(room_id, event)[1:] for existing in stored):
                raise ReplicaError("replayed event conflicts with stored history")
            if event["seq"] <= last_seq and not stored:
                raise ReplicaError("stored replica history is incomplete")
        new_events = [event for event in events if event["seq"] > last_seq]
        if new_events and new_events[0]["seq"] != last_seq + 1:
            raise ReplicaGapError("page skips sequences the replica has not stored")
        if disbanded_at is not None and new_events:
            raise ReplicaError("a disbanded Group Chat cannot accept later events")
        disband_indexes = [index for index, event in enumerate(new_events) if event["kind"] == "room.disbanded"]
        if disband_indexes and disband_indexes != [len(new_events) - 1]:
            raise ReplicaError("room.disbanded must be the terminal event")
        if disband_indexes and new_events[-1]["seq"] != latest_seq:
            raise ReplicaError("room.disbanded must complete the source history")
        added_bytes = sum(
            utf8_len(event["event_id"], event["kind"], event["actor_json"], event["payload_json"])
            for event in new_events)
        _reserve_replica_bytes(conn, added_bytes)
        conn.executemany(_INSERT_REPLICA_EVENT, [_event_row(room_id, event) for event in new_events])
        new_last = new_events[-1]["seq"] if new_events else last_seq
        terminal_at = new_events[-1]["created_at"] if disband_indexes else disbanded_at
        if row is None:
            conn.execute("""INSERT INTO hosted_room_replicas (room_id, name, members_json,
                    authority_gateway_id, authority_epoch, last_seq, latest_seq, event_bytes,
                    created_at, updated_at, disbanded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (room_id, room_name, members_json, authority["gateway_id"], authority["epoch"], new_last,
                 latest_seq, added_bytes, now, now, terminal_at))
        else:
            conn.execute("""UPDATE hosted_room_replicas SET last_seq=?, latest_seq=?, event_bytes=event_bytes+?,
                    updated_at=?, disbanded_at=? WHERE room_id=?""",
                (new_last, latest_seq, added_bytes, now, terminal_at, room_id))
    return {
        "room_id": room_id, "stored_seq": new_last, "ingested": len(new_events), "authority": authority,
        "caught_up": new_last >= latest_seq}


def _reserve_replica_bytes(conn: sqlite3.Connection, added_bytes: int) -> None:
    """Fit ``added_bytes`` of replica history into the gateway budget, reclaiming terminal rooms first."""
    def used(table: str) -> int:
        return int(conn.execute(f"SELECT COALESCE(SUM(event_bytes), 0) FROM {table}").fetchone()[0])

    if used("hosted_rooms") + used("hosted_room_replicas") + added_bytes > MAX_REPLICA_EVENT_BYTES:
        _prune_disbanded_rooms_locked(conn, now=None, max_gateway_event_bytes=max(
            0, MAX_REPLICA_EVENT_BYTES - added_bytes - used("hosted_room_replicas")))
        _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=max(
            0, MAX_REPLICA_EVENT_BYTES - added_bytes - used("hosted_rooms")))
    if used("hosted_rooms") + used("hosted_room_replicas") + added_bytes > MAX_REPLICA_EVENT_BYTES:
        raise ReplicaError("replica event storage exhausted")


def replica_state(db_path: DbPath, *, room_id: Any) -> dict[str, Any]:
    """Return the stored replica's coverage, authority lineage and safety status."""
    room_id = _room_id(room_id)
    with _replica_transaction(db_path) as conn:
        row = _load_replica_locked(conn, room_id)
        reservation = None if row is not None else conn.execute(
            "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?", (room_id,)).fetchone()
    if row is None:
        if reservation is not None and reservation["owner_kind"] == "replica":
            raise ReplicaHistoryExpiredError("replica history expired; room_id remains permanently retired")
        raise ReplicaError("replica not found")
    return {
        "room_id": row["room_id"], "name": row["name"], "members": json.loads(row["members_json"]),
        "authority": {"gateway_id": row["authority_gateway_id"], "epoch": int(row["authority_epoch"])},
        "last_seq": int(row["last_seq"]), "latest_seq": int(row["latest_seq"]), "event_bytes": int(row["event_bytes"]),
        "created_at": float(row["created_at"]), "updated_at": float(row["updated_at"]),
        "disbanded_at": float(row["disbanded_at"]) if row["disbanded_at"] is not None else None,
        "safety_status": "quarantined" if row["quarantine_reason"] is not None else "passive",
        "safety_reason": row["quarantine_reason"]}


def promote_replica(
    db_path: DbPath, *, room_id: Any, reason: Any = "authority-unreachable", now: float | None = None
) -> dict[str, Any]:
    """Continue a replicated room on THIS gateway at ``epoch + 1``.

    Copies the replica log into the authoritative store and appends a lineage-proving ``authority.claimed``
    event, so wherever the claim replicates the old epoch is stale and every fenced primitive rejects it.
    The caller decides takeover is safe; this makes it atomic and provable. Until exclusive-authority
    recovery can prove it, the room store quarantines the claimed room: readable, but closed to new events.
    """
    room_id = _room_id(room_id)
    if not isinstance(reason, str) or not reason or len(reason) > 200:
        raise ReplicaError("reason must be a non-empty string of at most 200 chars")
    now = clock(now)
    local_gateway = local_authority_gateway_id()
    with _replica_transaction(db_path) as conn:
        replica = _load_replica_locked(conn, room_id)
        if replica is None:
            raise ReplicaError("replica not found")
        if replica["quarantine_reason"] is not None:
            raise ReplicaError("stored replica is quarantined: " + str(replica["quarantine_reason"]))
        if replica["disbanded_at"] is not None:
            raise RoomConflictError("room_id belongs to a disbanded room")
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
        # Move, not copy: the history leaves the replica tables before it enters the room tables, so the
        # shared byte budget never counts it twice, and the copy hands its room-id reservation to the
        # authority row it becomes (any other authority insert for a reserved id is still refused).
        history = conn.execute(
            f"SELECT {_EVENT_COLUMNS[1:-1]} FROM hosted_room_replica_events WHERE room_id=? ORDER BY seq",
            (room_id,)).fetchall()
        conn.execute("DELETE FROM hosted_room_replica_events WHERE room_id=?", (room_id,))
        conn.execute("DELETE FROM hosted_room_replicas WHERE room_id=?", (room_id,))
        conn.execute("DELETE FROM hosted_room_id_reservations WHERE room_id=? AND owner_kind='replica'", (room_id,))
        conn.execute("""INSERT INTO hosted_rooms
               (room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq, event_bytes,
                revision, created_at, updated_at, disbanded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)""",
            (
                room_id, replica["name"], replica["members_json"], local_gateway, target_epoch, claim_seq + 1,
                int(replica["event_bytes"]) + utf8_len(*claim), now, now))
        conn.executemany(_INSERT_ROOM_EVENT, [tuple(event) for event in history])
        _append_control_event(conn, room_id, claim_seq, target_epoch, claim, now)
    return {
        "room_id": room_id, "authority_gateway_id": local_gateway, "authority_epoch": target_epoch,
        "previous_gateway_id": previous_gateway, "previous_epoch": previous_epoch, "claim_seq": claim_seq,
        "latest_seq": claim_seq}


def demote_room(
    db_path: DbPath, *, room_id: Any, observed_gateway_id: Any, observed_epoch: Any, now: float | None = None
) -> dict[str, Any]:
    """Fence THIS gateway's stale room authority against a proven newer epoch.

    When a returning gateway observes (replicated ``authority.claimed`` or a transport rejection) that another
    gateway owns the room at a higher epoch, append ``authority.lost`` and adopt the observed lineage so no
    local send can commit at the stale epoch. Idempotent per lineage. The room store then quarantines the
    room, since its history may have diverged: it stays readable but accepts no new events.
    """
    room_id = _room_id(room_id)
    observed_gateway_id = _validate_identifier(
        observed_gateway_id, label="observed_gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
    observed_epoch = _positive_int(observed_epoch, message="observed_epoch must be a positive integer")
    now = clock(now)
    local_gateway = local_authority_gateway_id()
    with _transaction(db_path, immediate=True) as conn:
        row = conn.execute("""SELECT authority_gateway_id, authority_epoch, next_seq
                 FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL""", (room_id,)).fetchone()
        if row is None:
            raise ReplicaError("room not found in the local authoritative store")
        current_gateway, current_epoch = str(row["authority_gateway_id"]), int(row["authority_epoch"])
        if current_gateway == observed_gateway_id and current_epoch == observed_epoch:
            return {
                "room_id": room_id, "authority_gateway_id": current_gateway, "authority_epoch": current_epoch,
                "idempotent": True}
        _raise_if_quarantined(conn, room_id)
        if observed_epoch <= current_epoch:
            raise ReplicaEpochRegressionError("observed epoch does not supersede the stored authority")
        if current_gateway != local_gateway:
            raise ReplicaError("room is not locally authoritative; nothing to demote")
        lost = _control_event("lost", observed_epoch, {
            "previous_gateway_id": current_gateway, "authority_gateway_id": observed_gateway_id,
            "authority_epoch": observed_epoch})
        _append_control_event(conn, room_id, int(row["next_seq"]), observed_epoch, lost, now)
        conn.execute("""UPDATE hosted_rooms
                  SET authority_gateway_id=?, authority_epoch=?, next_seq=next_seq+1, event_bytes=event_bytes+?,
                      revision=revision+1, updated_at=?
                WHERE room_id=?""",
            (observed_gateway_id, observed_epoch, utf8_len(*lost), now, room_id))
    return {
        "room_id": room_id, "authority_gateway_id": observed_gateway_id, "authority_epoch": observed_epoch,
        "idempotent": False}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import Path  # noqa: F401,E402
import time  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'MAX_EVENT_JSON_BYTES': ('gateway.hosted_rooms', 'MAX_EVENT_JSON_BYTES'),
    'MAX_ROOM_ID_CHARS': ('gateway.hosted_rooms', 'MAX_ROOM_ID_CHARS'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
