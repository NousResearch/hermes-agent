"""Passive-replica namespace, quarantine and retention guards."""

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

_REPLICA_RESERVATION_COLUMNS = frozenset({
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
})


def initialize_safety_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_quarantine (
            room_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            detected_at REAL NOT NULL
        )"""
    )
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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_event_budget (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0)
        )"""
    )
    replica_columns = table_columns(conn, "hosted_room_replicas")
    for column, declaration in (
        ("disbanded_at", "REAL"),
        ("quarantined_at", "REAL"),
        ("quarantine_reason", "TEXT"),
    ):
        if column not in replica_columns:
            conn.execute(
                f"ALTER TABLE hosted_room_replicas ADD COLUMN {column} {declaration}"
            )
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
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_id_reservations (
            room_id TEXT PRIMARY KEY,
            owner_kind TEXT NOT NULL CHECK (owner_kind IN ('authority', 'replica')),
            reserved_at REAL NOT NULL
        )"""
    )
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
        """UPDATE hosted_room_replicas
              SET event_bytes=COALESCE((
                    SELECT SUM(
                        LENGTH(CAST(event_id AS BLOB)) +
                        LENGTH(CAST(kind AS BLOB)) +
                        LENGTH(CAST(actor_json AS BLOB)) +
                        LENGTH(CAST(payload_json AS BLOB))
                    ) FROM hosted_room_replica_events
                    WHERE hosted_room_replica_events.room_id=hosted_room_replicas.room_id
                  ), 0)"""
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
    ):
        conn.execute(trigger)
    _compact_over_budget_replicas_locked(conn)


def safety_schema_is_current(conn: sqlite3.Connection) -> bool:
    tables = {
        "hosted_room_quarantine": _QUARANTINE_SCHEMA_COLUMNS,
        "hosted_room_id_reservations": _ROOM_RESERVATION_SCHEMA_COLUMNS,
        "hosted_room_replicas": _REPLICA_RESERVATION_COLUMNS,
        "hosted_room_replica_events": _REPLICA_EVENT_SCHEMA_COLUMNS,
        "hosted_room_event_budget": _EVENT_BUDGET_SCHEMA_COLUMNS,
    }
    triggers = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    return all(columns.issubset(table_columns(conn, table)) for table, columns in tables.items()) and _ROOM_SAFETY_TRIGGERS.issubset(triggers)


def _compact_over_budget_replicas_locked(conn: sqlite3.Connection) -> int:
    """Bound legacy replica payload without dropping quarantined evidence."""
    if not table_exists(conn, "hosted_room_replicas"):
        return 0
    rows = conn.execute(
        """SELECT replicas.room_id, replicas.updated_at,
                  replicas.quarantine_reason,
                  COALESCE(SUM(
                      LENGTH(CAST(events.event_id AS BLOB)) +
                      LENGTH(CAST(events.kind AS BLOB)) +
                      LENGTH(CAST(events.actor_json AS BLOB)) +
                      LENGTH(CAST(events.payload_json AS BLOB))
                  ), 0) AS actual_bytes
             FROM hosted_room_replicas AS replicas
             LEFT JOIN hosted_room_replica_events AS events
               ON events.room_id=replicas.room_id
            GROUP BY replicas.room_id
            ORDER BY replicas.updated_at ASC, replicas.room_id ASC"""
    ).fetchall()
    replica_bytes = sum(int(row["actual_bytes"]) for row in rows)
    hosted_bytes = int(
        conn.execute(
            """SELECT COALESCE(SUM(
                       LENGTH(CAST(event_id AS BLOB)) +
                       LENGTH(CAST(kind AS BLOB)) +
                       LENGTH(CAST(actor_json AS BLOB)) +
                       LENGTH(CAST(payload_json AS BLOB))
                   ), 0) FROM hosted_room_events"""
        ).fetchone()[0]
    )
    from gateway import hosted_rooms as limits

    replica_budget = max(0, int(limits.MAX_GATEWAY_EVENT_BYTES) - hosted_bytes)
    removed = 0
    for row in rows:
        if replica_bytes <= replica_budget:
            break
        if row["quarantine_reason"] is not None:
            continue
        room_id = str(row["room_id"])
        conn.execute(
            """INSERT OR IGNORE INTO hosted_room_quarantine
               (room_id, reason, detected_at)
               VALUES (?, 'replica_storage_budget_exceeded', ?)""",
            (room_id, time.time()),
        )
        conn.execute(
            "DELETE FROM hosted_room_replica_events WHERE room_id=?", (room_id,)
        )
        conn.execute("DELETE FROM hosted_room_replicas WHERE room_id=?", (room_id,))
        replica_bytes -= int(row["actual_bytes"])
        removed += 1
    return removed


def _quarantine_reason_locked(conn: sqlite3.Connection, room_id: str) -> str | None:
    row = conn.execute(
        "SELECT reason FROM hosted_room_quarantine WHERE room_id=?", (room_id,)
    ).fetchone()
    return str(row["reason"]) if row is not None else None


def _raise_if_quarantined(conn: sqlite3.Connection, room_id: str) -> None:
    from gateway.hosted_rooms import RoomQuarantinedError

    reason = _quarantine_reason_locked(conn, room_id)
    if reason is not None:
        raise RoomQuarantinedError(
            "This Group Chat has an unverified authority takeover and is read-only "
            f"until its history is reconciled ({reason})."
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
    """Reclaim terminal replica payload while its room-ID reservation remains."""
    from gateway import hosted_rooms as limits
    candidates: set[str] = set()
    if now is not None:
        cutoff = now - limits.DISBANDED_REPLICA_RETENTION_SECONDS
        candidates.update(
            str(row["room_id"])
            for row in conn.execute(
                """SELECT room_id FROM hosted_room_replicas
                     WHERE disbanded_at IS NOT NULL AND disbanded_at<=?
                       AND last_seq=latest_seq AND quarantine_reason IS NULL""",
                (cutoff,),
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
                """SELECT room_id, event_bytes FROM hosted_room_replicas
                     WHERE disbanded_at IS NOT NULL AND last_seq=latest_seq
                       AND quarantine_reason IS NULL
                     ORDER BY disbanded_at ASC, room_id ASC"""
            ).fetchall():
                candidates.add(str(row["room_id"]))
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_replica_event_bytes:
                    break
    if max_replica_rooms is not None:
        retained_rooms = int(
            conn.execute("SELECT COUNT(*) FROM hosted_room_replicas").fetchone()[0]
        )
        if retained_rooms > max_replica_rooms:
            for row in conn.execute(
                """SELECT room_id FROM hosted_room_replicas
                     WHERE disbanded_at IS NOT NULL AND last_seq=latest_seq
                       AND quarantine_reason IS NULL
                     ORDER BY disbanded_at ASC, room_id ASC"""
            ).fetchall():
                candidates.add(str(row["room_id"]))
                retained_rooms -= 1
                if retained_rooms <= max_replica_rooms:
                    break
    if not candidates:
        return 0
    placeholders = ",".join("?" for _ in candidates)
    room_ids = tuple(sorted(candidates))
    conn.execute(
        f"DELETE FROM hosted_room_replica_events WHERE room_id IN ({placeholders})",
        room_ids,
    )
    deleted = conn.execute(
        f"""DELETE FROM hosted_room_replicas
             WHERE room_id IN ({placeholders}) AND disbanded_at IS NOT NULL
               AND last_seq=latest_seq AND quarantine_reason IS NULL""",
        room_ids,
    )
    return max(0, int(deleted.rowcount))
