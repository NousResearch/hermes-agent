"""Room-id reservations, quarantine and shared retention guards for hosted Group Chat stores.

Installed on every room store, the legacy ``shared-state.db`` and each canonical ``state.db``:

* ``hosted_room_quarantine`` keeps a room read-only, and out of pruning, once its history records a
  takeover no exclusive authority proved (an ``authority.claimed`` promoted from a replica, or an
  ``authority.lost``), or once one room id names both a local room and a stored copy.
* ``hosted_room_id_reservations`` binds every room id to the kind that first used it, even after its
  payload is pruned, so a copied room is never recreated as a local one.
* ``hosted_room_event_budget`` counts room and replica events against one byte budget.

A quarantined room never becomes writable again. The one change it accepts is an operator-confirmed
Disband on this gateway (``hosted_room_quarantine_disbands``): a tombstone that appends nothing to its
history and leaves its authority untouched. Its rows and events can't be deleted, so no pruning, not even
an older process's, removes that evidence.

The guards are SQLite triggers, so they also hold for writers that predate them, such as an older
gateway process still sharing the store.
"""

from __future__ import annotations

import sqlite3
import time

from gateway.hosted_rooms_common import table_columns, table_exists

_QUARANTINE_SCHEMA_COLUMNS = frozenset({"room_id", "reason", "detected_at"})

_ROOM_RESERVATION_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "owner_kind",
    "reserved_at",
})

_REPLICA_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "name",
    "members_json",
    "authority_gateway_id",
    "authority_epoch",
    "last_seq",
    "latest_seq",
    "event_bytes",
    "created_at",
    "updated_at",
    "disbanded_at",
    "quarantined_at",
    "quarantine_reason",
})

_REPLICA_EVENT_SCHEMA_COLUMNS = frozenset({
    "room_id", "seq", "event_id", "kind", "actor_json",
    "authority_epoch", "payload_json", "created_at",
})

_EVENT_BUDGET_SCHEMA_COLUMNS = frozenset({"singleton", "event_bytes"})

_QUARANTINE_DISBAND_SCHEMA_COLUMNS = frozenset({"room_id", "confirmed_at"})

_ROOM_SAFETY_TRIGGERS = frozenset({
    "trg_hosted_rooms_reject_reserved_insert",
    "trg_hosted_rooms_reserve_insert",
    "trg_hosted_replicas_reject_reserved_insert",
    "trg_hosted_replicas_reserve_insert",
    "trg_hosted_events_reject_quarantined_insert",
    "trg_hosted_events_quarantine_unsafe_lineage",
    "trg_hosted_events_shared_budget",
    "trg_hosted_replica_events_shared_budget",
    "trg_hosted_events_budget_account_insert",
    "trg_hosted_events_budget_account_delete",
    "trg_hosted_replica_events_budget_account_insert",
    "trg_hosted_replica_events_budget_account_delete",
    "trg_hosted_rooms_quarantined_tombstone",
    "trg_hosted_rooms_keep_quarantined",
    "trg_hosted_events_keep_quarantined",
})


def _quarantine_unsafe_authorities_locked(conn: sqlite3.Connection) -> None:
    """Derive missing fences after historical replay; retain original quarantine evidence."""
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_quarantine
           (room_id, reason, detected_at)
           SELECT room_id, 'unsafe_replica_promotion', MIN(created_at)
             FROM hosted_room_events
            WHERE kind='authority.claimed'
              AND payload_json LIKE '%"promoted_from_replica":true%'
            GROUP BY room_id"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_quarantine
           (room_id, reason, detected_at)
           SELECT room_id, 'unsafe_authority_demotion', MIN(created_at)
             FROM hosted_room_events
            WHERE kind='authority.lost'
            GROUP BY room_id"""
    )


def initialize_safety_schema(conn: sqlite3.Connection) -> None:
    from gateway.hosted_room_replicas import _initialize_replica_schema

    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_quarantine (
            room_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            detected_at REAL NOT NULL
        )"""
    )
    _initialize_replica_schema(conn)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_event_budget (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_quarantine_disbands (
            room_id TEXT PRIMARY KEY,
            confirmed_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_id_reservations (
            room_id TEXT PRIMARY KEY,
            owner_kind TEXT NOT NULL CHECK (owner_kind IN ('authority', 'replica')),
            reserved_at REAL NOT NULL
        )"""
    )
    _quarantine_unsafe_authorities_locked(conn)
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_quarantine
           (room_id, reason, detected_at)
           SELECT rooms.room_id, 'room_namespace_collision', rooms.updated_at
             FROM hosted_rooms AS rooms
             JOIN hosted_room_replicas AS replicas
               ON replicas.room_id=rooms.room_id"""
    )
    conn.execute(
        """UPDATE hosted_room_replicas
              SET quarantined_at=COALESCE(
                      quarantined_at,
                      (SELECT updated_at FROM hosted_rooms
                        WHERE hosted_rooms.room_id=hosted_room_replicas.room_id)
                  ),
                  quarantine_reason=COALESCE(
                      quarantine_reason,
                      'room_namespace_collision'
                  )
            WHERE EXISTS (
                SELECT 1 FROM hosted_rooms
                 WHERE hosted_rooms.room_id=hosted_room_replicas.room_id
            )"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_id_reservations
           (room_id, owner_kind, reserved_at)
           SELECT room_id, 'authority', created_at FROM hosted_rooms"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_id_reservations
           (room_id, owner_kind, reserved_at)
           SELECT room_id, 'replica', created_at FROM hosted_room_replicas"""
    )
    conn.execute(
        """INSERT INTO hosted_room_event_budget(singleton, event_bytes)
           VALUES (
               1,
               COALESCE((
                   SELECT SUM(
                       LENGTH(CAST(event_id AS BLOB)) +
                       LENGTH(CAST(kind AS BLOB)) +
                       LENGTH(CAST(actor_json AS BLOB)) +
                       LENGTH(CAST(payload_json AS BLOB))
                   ) FROM hosted_room_events
               ), 0) +
               COALESCE((
                   SELECT SUM(
                       LENGTH(CAST(event_id AS BLOB)) +
                       LENGTH(CAST(kind AS BLOB)) +
                       LENGTH(CAST(actor_json AS BLOB)) +
                       LENGTH(CAST(payload_json AS BLOB))
                   ) FROM hosted_room_replica_events
               ), 0)
           )
           ON CONFLICT(singleton) DO UPDATE SET event_bytes=excluded.event_bytes"""
    )
    from gateway import hosted_rooms as limits

    ordinary_event_budget = int(limits.MAX_GATEWAY_EVENT_BYTES)
    control_event_budget = ordinary_event_budget + int(
        limits.CONTROL_EVENT_BYTE_RESERVE
    )
    for trigger in (
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_reject_reserved_insert
           BEFORE INSERT ON hosted_rooms
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_id_reservations WHERE room_id=NEW.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'room_id is already reserved');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_reserve_insert
           AFTER INSERT ON hosted_rooms
           BEGIN
               INSERT INTO hosted_room_id_reservations
                   (room_id, owner_kind, reserved_at)
               VALUES (NEW.room_id, 'authority', NEW.created_at);
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replicas_reject_reserved_insert
           BEFORE INSERT ON hosted_room_replicas
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_id_reservations WHERE room_id=NEW.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'room_id is already reserved');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replicas_reserve_insert
           AFTER INSERT ON hosted_room_replicas
           BEGIN
               INSERT INTO hosted_room_id_reservations
                   (room_id, owner_kind, reserved_at)
               VALUES (NEW.room_id, 'replica', NEW.created_at);
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_reject_quarantined_insert
           BEFORE INSERT ON hosted_room_events
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_quarantine WHERE room_id=NEW.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'room authority is quarantined');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_quarantine_unsafe_lineage
           AFTER INSERT ON hosted_room_events
           WHEN NEW.kind='authority.lost'
             OR (
                 NEW.kind='authority.claimed'
                 AND NEW.payload_json LIKE '%"promoted_from_replica":true%'
             )
           BEGIN
               INSERT OR IGNORE INTO hosted_room_quarantine
                   (room_id, reason, detected_at)
               VALUES (
                   NEW.room_id,
                   CASE
                       WHEN NEW.kind='authority.lost'
                       THEN 'unsafe_authority_demotion'
                       ELSE 'unsafe_replica_promotion'
                   END,
                   NEW.created_at
               );
           END""",
        f"""CREATE TRIGGER IF NOT EXISTS trg_hosted_events_shared_budget
           BEFORE INSERT ON hosted_room_events
           WHEN NOT EXISTS (
               SELECT 1 FROM hosted_room_events
                WHERE room_id=NEW.room_id
                  AND (seq=NEW.seq OR event_id=NEW.event_id)
             )
             AND (
                 (SELECT event_bytes FROM hosted_room_event_budget WHERE singleton=1) +
                 LENGTH(CAST(NEW.event_id AS BLOB)) +
                 LENGTH(CAST(NEW.kind AS BLOB)) +
                 LENGTH(CAST(NEW.actor_json AS BLOB)) +
                 LENGTH(CAST(NEW.payload_json AS BLOB))
             ) > CASE
                 WHEN NEW.kind IN (
                     'authority.claimed', 'authority.lost',
                     'room.disbanded', 'room.stop_requested'
                 ) THEN {control_event_budget}
                 ELSE {ordinary_event_budget}
             END
           BEGIN
               SELECT RAISE(ABORT, 'hosted room event budget exceeded');
           END""",
        f"""CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_shared_budget
           BEFORE INSERT ON hosted_room_replica_events
           WHEN NOT EXISTS (
               SELECT 1 FROM hosted_room_replica_events
                WHERE room_id=NEW.room_id AND seq=NEW.seq
             )
             AND (
                 (SELECT event_bytes FROM hosted_room_event_budget WHERE singleton=1) +
                 LENGTH(CAST(NEW.event_id AS BLOB)) +
                 LENGTH(CAST(NEW.kind AS BLOB)) +
                 LENGTH(CAST(NEW.actor_json AS BLOB)) +
                 LENGTH(CAST(NEW.payload_json AS BLOB))
             ) > {ordinary_event_budget}
           BEGIN
               SELECT RAISE(ABORT, 'hosted room event budget exceeded');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_budget_account_insert
           AFTER INSERT ON hosted_room_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=event_bytes +
                      LENGTH(CAST(NEW.event_id AS BLOB)) +
                      LENGTH(CAST(NEW.kind AS BLOB)) +
                      LENGTH(CAST(NEW.actor_json AS BLOB)) +
                      LENGTH(CAST(NEW.payload_json AS BLOB))
                WHERE singleton=1;
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_budget_account_delete
           AFTER DELETE ON hosted_room_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=MAX(
                      0,
                      event_bytes -
                      LENGTH(CAST(OLD.event_id AS BLOB)) -
                      LENGTH(CAST(OLD.kind AS BLOB)) -
                      LENGTH(CAST(OLD.actor_json AS BLOB)) -
                      LENGTH(CAST(OLD.payload_json AS BLOB))
                  )
                WHERE singleton=1;
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_budget_account_insert
           AFTER INSERT ON hosted_room_replica_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=event_bytes +
                      LENGTH(CAST(NEW.event_id AS BLOB)) +
                      LENGTH(CAST(NEW.kind AS BLOB)) +
                      LENGTH(CAST(NEW.actor_json AS BLOB)) +
                      LENGTH(CAST(NEW.payload_json AS BLOB))
                WHERE singleton=1;
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_budget_account_delete
           AFTER DELETE ON hosted_room_replica_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=MAX(
                      0,
                      event_bytes -
                      LENGTH(CAST(OLD.event_id AS BLOB)) -
                      LENGTH(CAST(OLD.kind AS BLOB)) -
                      LENGTH(CAST(OLD.actor_json AS BLOB)) -
                      LENGTH(CAST(OLD.payload_json AS BLOB))
                  )
                WHERE singleton=1;
           END""",
        # A quarantined room takes one change only: the tombstone of a Disband an operator confirmed,
        # recorded in the same transaction, which leaves its authority and history untouched.
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_quarantined_tombstone
           BEFORE UPDATE OF disbanded_at ON hosted_rooms
           WHEN NEW.disbanded_at IS NOT OLD.disbanded_at
             AND EXISTS (
                 SELECT 1 FROM hosted_room_quarantine WHERE room_id=OLD.room_id
             )
             AND NOT (
                 OLD.disbanded_at IS NULL AND NEW.disbanded_at IS NOT NULL
                 AND EXISTS (
                     SELECT 1 FROM hosted_room_quarantine_disbands WHERE room_id=OLD.room_id
                 )
                 AND NEW.authority_gateway_id IS OLD.authority_gateway_id
                 AND NEW.authority_epoch IS OLD.authority_epoch
                 AND NEW.next_seq IS OLD.next_seq
                 AND NEW.event_bytes IS OLD.event_bytes
             )
           BEGIN
               SELECT RAISE(ABORT, 'room authority is quarantined');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_keep_quarantined
           BEFORE DELETE ON hosted_rooms
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_quarantine WHERE room_id=OLD.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'quarantined room history is kept');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_keep_quarantined
           BEFORE DELETE ON hosted_room_events
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_quarantine WHERE room_id=OLD.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'quarantined room history is kept');
           END""",
    ):
        conn.execute(trigger)
    # Audits every stored copy, re-deriving its byte count, before compacting any.
    _compact_over_budget_replicas_locked(conn)


def safety_schema_is_current(conn: sqlite3.Connection) -> bool:
    tables = {
        "hosted_room_quarantine": _QUARANTINE_SCHEMA_COLUMNS,
        "hosted_room_id_reservations": _ROOM_RESERVATION_SCHEMA_COLUMNS,
        "hosted_room_replicas": _REPLICA_SCHEMA_COLUMNS,
        "hosted_room_replica_events": _REPLICA_EVENT_SCHEMA_COLUMNS,
        "hosted_room_event_budget": _EVENT_BUDGET_SCHEMA_COLUMNS,
        "hosted_room_quarantine_disbands": _QUARANTINE_DISBAND_SCHEMA_COLUMNS,
    }
    triggers = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    return all(columns.issubset(table_columns(conn, table)) for table, columns in tables.items()) and (
        _ROOM_SAFETY_TRIGGERS.issubset(triggers))


def _compact_over_budget_replicas_locked(conn: sqlite3.Connection) -> int:
    """Reclaim only verified terminal copies; never erase an active prefix.

    Retained excess bytes keep the shared budget full and further writes refused.
    Capacity is not invented integrity failure: authorized retirement can still
    clean a valid copy, while genuinely quarantined evidence remains protected.
    """
    if not table_exists(conn, "hosted_room_replicas"):
        return 0
    from gateway import hosted_rooms as limits

    hosted_bytes = int(conn.execute("SELECT COALESCE(SUM(event_bytes),0) FROM hosted_rooms").fetchone()[0])
    # This helper re-audits before selecting only completed, non-quarantined
    # history under the existing explicit terminal-retention policy.
    return _prune_disbanded_replicas_locked(
        conn, now=None, max_replica_event_bytes=max(0, limits.MAX_GATEWAY_EVENT_BYTES - hosted_bytes))


def _quarantine_reason_locked(conn: sqlite3.Connection, room_id: str) -> str | None:
    row = conn.execute(
        "SELECT reason FROM hosted_room_quarantine WHERE room_id=?", (room_id,)
    ).fetchone()
    return str(row["reason"]) if row is not None else None


def _raise_if_quarantined(conn: sqlite3.Connection, room_id: str, *, hint: str = "") -> None:
    from gateway.hosted_rooms import RoomQuarantinedError

    reason = _quarantine_reason_locked(conn, room_id)
    if reason is not None:
        raise RoomQuarantinedError(
            "This Group Chat has an unverified authority takeover and is read-only "
            f"until its history is reconciled ({reason}).{hint}"
        )


def _record_quarantine_disband_locked(conn: sqlite3.Connection, room_id: str, now: float) -> None:
    """Record an operator's confirmation; the tombstone trigger accepts nothing else."""
    conn.execute(
        "INSERT OR IGNORE INTO hosted_room_quarantine_disbands (room_id, confirmed_at) VALUES (?, ?)",
        (room_id, now),
    )


def _replica_reserves_room_id_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    if not conn.execute(
        """SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='hosted_room_replicas'"""
    ).fetchone():
        return False
    return (
        conn.execute(
            "SELECT 1 FROM hosted_room_replicas WHERE room_id=?", (room_id,)
        ).fetchone()
        is not None
    )


def _room_id_reservation_kind_locked(
    conn: sqlite3.Connection, room_id: str
) -> str | None:
    row = conn.execute(
        "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?",
        (room_id,),
    ).fetchone()
    return str(row["owner_kind"]) if row is not None else None


def _replica_event_bytes_locked(conn: sqlite3.Connection) -> int:
    """Return passive-replica bytes when the optional replica table exists."""
    if not conn.execute(
        """SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='hosted_room_replicas'"""
    ).fetchone():
        return 0
    return int(
        conn.execute(
            "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_room_replicas"
        ).fetchone()[0]
    )


def _prune_disbanded_replicas_locked(
    conn: sqlite3.Connection,
    *,
    now: float | None,
    max_replica_event_bytes: int | None = None,
    max_replica_rooms: int | None = None,
) -> int:
    """Reclaim terminal replica payload while its room-ID reservation remains.

    A copy is terminal once it holds its room's complete history up to ``room.disbanded``, or once
    its owner-enrolled retirement covers exactly this copy. Quarantined copies are never reclaimed.
    """
    from gateway import hosted_rooms as limits
    from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
    from gateway.hosted_room_replicas import (
        _audit_existing_replicas_locked, _replica_header_bounds_sql, _replica_read_envelope_locked)

    # Re-audit even if an earlier observation/ingest audited then rolled back,
    # or an old writer committed new history since the last replica read.
    _audit_existing_replicas_locked(conn)
    canonical = "(disbanded_at IS NOT NULL AND last_seq=latest_seq)"
    eligible, terminal_at, authorized_retirement = canonical, "disbanded_at", "0"
    if table_exists(conn, RETIREMENT_TABLE):
        retired = f"""SELECT retired_at FROM {RETIREMENT_TABLE} AS retirement
            WHERE retirement.room_id=hosted_room_replicas.room_id
              AND retirement.authority_gateway_id=hosted_room_replicas.authority_gateway_id
              AND retirement.authority_epoch=hosted_room_replicas.authority_epoch"""
        authorized_retirement = f"EXISTS ({retired})"
        eligible = f"({canonical} OR {authorized_retirement})"
        terminal_at = f"CASE WHEN {canonical} THEN disbanded_at ELSE ({retired}) END"
    eligible += """ AND quarantine_reason IS NULL AND NOT EXISTS (
        SELECT 1 FROM hosted_room_quarantine WHERE hosted_room_quarantine.room_id=hosted_room_replicas.room_id)"""
    eligible += f" AND ({_replica_header_bounds_sql()})"

    def reclaimable(room_id: str) -> bool:
        # A verified owner notice authorizes releasing this exact retained copy
        # without requiring its opaque history to fit today's read envelope.
        # Automatic terminal pruning still requires bounded validation, and the
        # common SQL eligibility excludes genuine quarantine in both cases.
        retired = conn.execute(
            f"SELECT ({authorized_retirement}) FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
        return bool(retired and retired[0]) or _replica_read_envelope_locked(conn, room_id)

    candidates: set[str] = set()
    if now is not None:
        cutoff = now - limits.DISBANDED_REPLICA_RETENTION_SECONDS
        candidates.update(
            str(row["room_id"])
            for row in conn.execute(
                f"SELECT room_id FROM hosted_room_replicas WHERE {eligible} AND ({terminal_at})<=?", (cutoff,),
            ).fetchall()
        )
    if max_replica_event_bytes is not None:
        retained_bytes = int(
            conn.execute(
                "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_room_replicas"
            ).fetchone()[0]
        )
        if retained_bytes > max_replica_event_bytes:
            for row in conn.execute(
                f"""SELECT room_id, event_bytes FROM hosted_room_replicas WHERE {eligible}
                     ORDER BY ({terminal_at}) ASC, room_id ASC"""
            ).fetchall():
                room_id = str(row["room_id"])
                if not reclaimable(room_id):
                    continue
                candidates.add(room_id)
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_replica_event_bytes:
                    break
    if max_replica_rooms is not None:
        retained_rooms = int(
            conn.execute("SELECT COUNT(*) FROM hosted_room_replicas").fetchone()[0]
        )
        if retained_rooms > max_replica_rooms:
            for row in conn.execute(
                f"""SELECT room_id FROM hosted_room_replicas WHERE {eligible}
                     ORDER BY ({terminal_at}) ASC, room_id ASC"""
            ).fetchall():
                room_id = str(row["room_id"])
                if not reclaimable(room_id):
                    continue
                candidates.add(room_id)
                retained_rooms -= 1
                if retained_rooms <= max_replica_rooms:
                    break
    candidates = {room_id for room_id in candidates if reclaimable(room_id)}
    if not candidates:
        return 0
    placeholders = ",".join("?" for _ in candidates)
    room_ids = tuple(sorted(candidates))
    eligible = f"room_id IN ({placeholders}) AND {eligible}"
    conn.execute(
        f"""DELETE FROM hosted_room_replica_events WHERE room_id IN (
                SELECT room_id FROM hosted_room_replicas WHERE {eligible})""",
        room_ids,
    )
    deleted = conn.execute(
        f"DELETE FROM hosted_room_replicas WHERE {eligible}",
        room_ids,
    )
    return max(0, int(deleted.rowcount))
