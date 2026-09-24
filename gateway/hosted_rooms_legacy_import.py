"""One-shot import of the pre-isolation ``hosted_room*`` rows into ``shared-state.db``.

``0e422e0ece`` repointed the room store from the root ``state.db`` to ``shared-state.db`` without
moving the rows it already held, so an install that had rooms started with an empty coordination
set and every pre-existing room resolved to "hosted room not found" (#109775). The first open after
the upgrade copies the rows across once; the marker row keeps that a one-shot step, because a purge
in THIS store must never be undone by re-importing rows the legacy store still holds.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path

from gateway.hosted_rooms_common import clock, table_columns, table_exists

logger = logging.getLogger(__name__)

SOURCE_NAME = "state.db"
MARKER_TABLE = "hosted_room_legacy_imports"
# Liveness state, never copied: a lease is a ~15s heartbeat plus a process generation, so a copied
# lease names a process that is gone. The driver claims a fresh one instead.
_SKIP_TABLES = frozenset({"hosted_room_driver_leases"})
# Sources this process could not import (unreadable file, rows the target refused). Every store open
# re-checks readiness, so without this a broken legacy file would re-run the copy and re-warn on
# every poll; the retry happens on the next process start instead.
_failed_sources: set[Path] = set()


def source_path(db_path: Path) -> Path | None:
    """The pre-isolation store for ``db_path``, or ``None`` when this database has no predecessor.

    Only the shared coordination database has one: callers that pass any other path (older
    layouts, tests) own that file directly.
    """
    return db_path.with_name(SOURCE_NAME) if db_path.name == "shared-state.db" else None


def settled(conn: sqlite3.Connection, db_path: Path) -> bool:
    """True once the import for this database has been recorded, given up on for this process, or never applies."""
    source = source_path(db_path)
    if source is None or source in _failed_sources:
        return True
    return table_exists(conn, MARKER_TABLE) and conn.execute(
        f"SELECT 1 FROM {MARKER_TABLE} WHERE source=?", (SOURCE_NAME,)).fetchone() is not None


def _select_expressions(legacy: sqlite3.Connection, target: sqlite3.Connection, name: str) -> list[tuple[str, str]]:
    """``(target column, source expression)`` pairs for copying ``name``.

    Columns only the target has take the same default the in-place column migration applies, so a
    legacy layout from before the actor/authority columns imports instead of tripping NOT NULL.
    Columns only the source has are dropped; columns the target added without a default are left
    to its DDL default.
    """
    from gateway.hosted_rooms import _LEGACY_COLUMNS

    defaults = {(table, column): default for table, column, _, default in _LEGACY_COLUMNS if default is not None}
    source_columns = [str(row[1]) for row in legacy.execute(f"PRAGMA table_info({name})")]
    target_columns = table_columns(target, name)
    pairs = [(column, column) for column in source_columns if column in target_columns]
    pairs.extend((column, default) for (table, column), default in defaults.items()
                 if table == name and column in target_columns and column not in source_columns)
    return pairs


def _copy_rows(target: sqlite3.Connection, source: Path) -> int:
    """Copy every room ``target`` does not have yet, with all of its child rows; returns rooms copied.

    A room id present in both stores is skipped as a unit: grafting the legacy events under this
    store's room would leave ``next_seq`` behind ``MAX(seq)`` and every later append colliding.
    Room-scoped tables therefore use a plain INSERT — a row the target refuses raises and aborts
    the whole import rather than being dropped in silence. A table this store has not created yet
    (the driver, policy and replica schemas are initialized by their own modules) is created from
    the source's own DDL so its rows survive the upgrade too.
    """
    from gateway.hosted_rooms import _EVENT_BYTES_BACKFILL

    copied_rooms: list[str] = []
    existing = {str(row[0]) for row in target.execute("SELECT room_id FROM hosted_rooms")}
    with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=10)) as legacy:
        names = [str(row[0]) for row in legacy.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'hosted_room*'")]
        # Parents first; the safety triggers reserve IDs as each parent is inserted.
        # Copy orphan reservation tombstones last, after both authority and replica parents.
        order = {"hosted_rooms": 0, "hosted_room_replicas": 1,
                 "hosted_room_id_reservations": 3}
        for name in sorted(names, key=lambda name: (order.get(name, 2), name)):
            if name in _SKIP_TABLES or name == MARKER_TABLE or name.endswith(("_next", "_migrating")):
                continue
            if not table_exists(target, name):
                target.execute(str(legacy.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()[0]))
            pairs = _select_expressions(legacy, target, name)
            if not pairs:
                continue
            columns = [column for column, _ in pairs]
            rows = legacy.execute(f"SELECT {', '.join(expr for _, expr in pairs)} FROM {name}")
            if "room_id" in columns:
                room_index = columns.index("room_id")
                rows = (row for row in rows if str(row[room_index]) not in existing)
                verb = "INSERT"
            else:
                verb = "INSERT OR IGNORE"
            if name == "hosted_rooms":
                rows = list(rows)
                copied_rooms = [str(row[room_index]) for row in rows]
            elif name == "hosted_room_id_reservations":
                # Parent insert triggers already made the matching reservations. Preserve
                # source-only tombstones, but never mask a conflicting owner classification.
                rows = list(rows)
                owner_index = columns.index("owner_kind")
                reservation_room_index = columns.index("room_id")
                for row in rows:
                    held = target.execute(
                        "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?",
                        (row[reservation_room_index],),
                    ).fetchone()
                    if held is not None and held[0] != row[owner_index]:
                        raise sqlite3.IntegrityError("legacy room reservation owner differs from target")
                verb = "INSERT OR IGNORE"
            target.executemany(
                f"{verb} INTO {name} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", rows)
        if copied_rooms and "event_bytes" not in table_columns(legacy, "hosted_rooms"):
            target.execute(
                _EVENT_BYTES_BACKFILL.format(where=f"room_id IN ({', '.join('?' * len(copied_rooms))})"), copied_rooms)
    return len(copied_rooms)


def import_legacy_rooms(conn: sqlite3.Connection, db_path: Path) -> None:
    """Copy the pre-isolation rows in once, then record the marker; never fails the open.

    The copy runs inside the caller's schema transaction under its own savepoint, so a crash or a
    refused row leaves either both the copied rows and the marker or neither.
    """
    source = source_path(db_path)
    if source is None or settled(conn, db_path):
        return
    conn.execute("SAVEPOINT legacy_import")
    try:
        copied = 0
        if source.is_file():
            from gateway.hosted_room_safety import _quarantine_unsafe_authorities_locked

            # Historical replay is not a live append. Suspend only the two
            # quarantine event triggers inside this write-locked savepoint;
            # reservation and byte-accounting guards remain active. SQLite DDL
            # is transactional: rollback restores the triggers on any failure,
            # and no other writer can enter before they are restored on success.
            triggers = conn.execute(
                """SELECT name, sql FROM sqlite_master WHERE type='trigger'
                   AND name IN ('trg_hosted_events_reject_quarantined_insert',
                                'trg_hosted_events_quarantine_unsafe_lineage')"""
            ).fetchall()
            for name, _ in triggers:
                conn.execute(f'DROP TRIGGER "{name}"')
            copied = _copy_rows(conn, source)
            # Copy source quarantine rows without ignoring conflicts, then
            # derive only missing classifications from the complete history.
            _quarantine_unsafe_authorities_locked(conn)
            for _, ddl in triggers:
                conn.execute(ddl)
    except (OSError, sqlite3.Error) as exc:
        # A locked, unreadable or incompatible legacy store must not take hosted rooms down with
        # it: drop the partial copy, leave the marker unset, retry on the next process start.
        conn.execute("ROLLBACK TO legacy_import")
        conn.execute("RELEASE legacy_import")
        _failed_sources.add(source)
        logger.warning("hosted rooms: could not import the pre-isolation %s (%s); will retry on the next start",
                       source, exc)
        return
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {MARKER_TABLE} (source TEXT PRIMARY KEY, imported_at REAL NOT NULL, rooms INTEGER NOT NULL)")
    conn.execute(
        f"INSERT OR IGNORE INTO {MARKER_TABLE} (source, imported_at, rooms) VALUES (?, ?, ?)",
        (SOURCE_NAME, clock(None), copied))
    conn.execute("RELEASE legacy_import")
    if copied:
        logger.info("hosted rooms: imported %d pre-isolation Group Chat(s) from %s", copied, source)
