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
# lease names a process that is gone. Derived policy/event-budget rows are rebuilt from the durable log.
_SKIP_TABLES = frozenset({
    "hosted_room_driver_leases", "hosted_room_event_budget",
    "hosted_room_policy_cursors", "hosted_room_policy_threads", "hosted_room_policy_events",
    "hosted_room_policy_watermarks", "hosted_room_policy_publications", "hosted_room_policy_transcript",
    "hosted_room_policy_transcript_state",
})
# Current shipped durable schemas only. Source-owned DDL is never executed.
_TABLE_ORDER = (
    "hosted_rooms", "hosted_room_replicas", "hosted_room_events", "hosted_room_replica_events",
    "hosted_room_retired_ids", "hosted_room_quarantine", "hosted_room_links",
    "hosted_room_remote_runs", "hosted_room_peer_reservations", "hosted_room_history_imports",
    "hosted_room_history_source_reservations",
    "hosted_room_disband_fences", "hosted_room_id_reservations", "hosted_room_revoked_grants",
    "hosted_room_revoked_grant_ids", "hosted_room_revoked_grant_tokens", "hosted_room_driver_tasks",
)
_ALLOWED_TABLES = frozenset(_TABLE_ORDER)
_GLOBAL_IDEMPOTENT_TABLES = frozenset({
    "hosted_room_revoked_grants", "hosted_room_revoked_grant_ids", "hosted_room_revoked_grant_tokens"})
_AUTHORITY_ONLY_TABLES = frozenset({
    "hosted_room_events", "hosted_room_links", "hosted_room_remote_runs", "hosted_room_peer_reservations",
    "hosted_room_history_imports", "hosted_room_disband_fences", "hosted_room_driver_tasks"})
_REPLICA_ONLY_TABLES = frozenset({"hosted_room_replica_events"})
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
    """Copy recognized durable room units without executing source-owned DDL."""
    from gateway.hosted_rooms import _EVENT_BYTES_BACKFILL

    copied_rooms: list[str] = []
    with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=10)) as legacy:
        # The first schema read pins one source snapshot through ID/claim preflight
        # and every later copy SELECT. A separate WAL writer may commit meanwhile,
        # but its new rows cannot enter the copy after validation has finished.
        legacy.execute("BEGIN")
        names = {str(row[0]) for row in legacy.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'hosted_room*'")}
        unknown = sorted(name for name in names if name not in _ALLOWED_TABLES | _SKIP_TABLES | {MARKER_TABLE}
                         and not name.endswith(("_next", "_migrating")))
        if unknown:
            logger.warning(
                "hosted rooms: ignored unsupported legacy table(s) without executing source DDL: %s",
                ", ".join(unknown))

        def source_ids(table: str) -> set[str]:
            if table not in names or "room_id" not in table_columns(legacy, table):
                return set()
            return {str(row[0]) for row in legacy.execute(f"SELECT room_id FROM {table}")}

        authority_ids = source_ids("hosted_rooms")
        replica_ids = source_ids("hosted_room_replicas")
        if authority_ids & replica_ids:
            raise sqlite3.DatabaseError("legacy source reuses one room id as authority and replica")
        if "hosted_room_id_reservations" in names:
            for room_id, owner_kind in legacy.execute(
                "SELECT room_id, owner_kind FROM hosted_room_id_reservations"
            ):
                if ((room_id in authority_ids and owner_kind != "authority")
                        or (room_id in replica_ids and owner_kind != "replica")):
                    raise sqlite3.IntegrityError("legacy room reservation owner differs from source parent")
        # Runtime owns the compact table and initializes the destination before this
        # importer. A pruned source has no parent or marker left to qualify its claim.
        history_reservations = "hosted_room_history_source_reservations"
        claim_columns = ("source_id", "source_kind", "content_sha256", "room_id")
        source_claims: list[tuple[str, str, str, str]] = []
        matching_claim_rooms: set[str] = set()
        if history_reservations in names:
            if not set(claim_columns).issubset(table_columns(target, history_reservations)):
                raise sqlite3.DatabaseError("recognized legacy history reservations have no target schema")
            source_claims = [tuple(row) for row in legacy.execute(
                f"SELECT source_id, source_kind, content_sha256, room_id FROM {history_reservations}")]
            # Read destination namespace independently of the source; no source-supplied
            # DDL or conflict-ignore may silently rebind a durable source identity.
            target_occupied = {
                str(row[0]) for table in (
                    "hosted_rooms", "hosted_room_replicas", "hosted_room_retired_ids",
                    "hosted_room_id_reservations") if table_exists(target, table)
                for row in target.execute(f"SELECT room_id FROM {table}")}
            for claim in source_claims:
                source_id, _, _, room_id = claim
                existing = target.execute(
                    f"SELECT source_id, source_kind, content_sha256, room_id FROM {history_reservations} "
                    "WHERE source_id=? OR room_id=?", (source_id, room_id)).fetchall()
                if existing and (len(existing) != 1 or tuple(existing[0]) != claim):
                    raise sqlite3.IntegrityError("legacy history source reservation conflicts with target identity")
                if room_id in target_occupied and not existing:
                    raise sqlite3.IntegrityError("legacy history source reservation conflicts with target room namespace")
                if existing:
                    matching_claim_rooms.add(room_id)
        if table_exists(target, history_reservations) and "hosted_room_history_imports" in names:
            claims_by_source = {claim[0]: claim for claim in source_claims}
            claims_by_room = {claim[3]: claim for claim in source_claims}
            for source_id, source_kind, digest, room_id in legacy.execute(
                "SELECT source_id, source_kind, content_sha256, room_id FROM hosted_room_history_imports"
            ):
                marker = (source_id, source_kind, digest, room_id)
                if ((source_id in claims_by_source and claims_by_source[source_id] != marker)
                        or (room_id in claims_by_room and claims_by_room[room_id] != marker)):
                    raise sqlite3.IntegrityError("legacy history marker differs from source reservation")
                existing = target.execute(
                    f"SELECT source_id, source_kind, content_sha256, room_id FROM {history_reservations} "
                    "WHERE source_id=? OR room_id=?", (source_id, room_id)).fetchall()
                if existing and (len(existing) != 1 or tuple(existing[0]) != marker):
                    raise sqlite3.IntegrityError("legacy history marker conflicts with target source identity")
        source_namespaces = (
            authority_ids | replica_ids | source_ids("hosted_room_retired_ids")
            | source_ids("hosted_room_id_reservations"))
        target_namespaces = {
            str(row[0]) for row in target.execute("SELECT room_id FROM hosted_room_id_reservations")}
        target_namespaces.update(
            str(row[0]) for row in target.execute("SELECT room_id FROM hosted_room_retired_ids"))
        if table_exists(target, history_reservations):
            target_namespaces.update(
                str(row[0]) for row in target.execute(f"SELECT room_id FROM {history_reservations}")
                if str(row[0]) not in matching_claim_rooms)
        blocked = source_namespaces & target_namespaces
        admitted_authority = authority_ids - blocked
        admitted_replica = replica_ids - blocked
        admitted_namespaces = source_namespaces - blocked

        if "hosted_room_driver_tasks" in names and not table_exists(target, "hosted_room_driver_tasks"):
            from gateway import hosted_room_driver as driver
            driver._create_task_table(target)
            target.execute(driver._TASK_INDEX_SQL.format(if_not_exists="IF NOT EXISTS "))

        for name in _TABLE_ORDER:
            if name not in names:
                continue
            if not table_exists(target, name):
                raise sqlite3.DatabaseError(f"recognized legacy table has no current target schema: {name}")
            pairs = ([(column, column) for column in claim_columns] if name == history_reservations
                     else _select_expressions(legacy, target, name))
            if not pairs:
                continue
            columns = [column for column, _ in pairs]
            rows = legacy.execute(f"SELECT {', '.join(expr for _, expr in pairs)} FROM {name}")
            room_index = columns.index("room_id") if "room_id" in columns else None
            if room_index is not None and name != history_reservations:
                allowed = (admitted_authority if name == "hosted_rooms" or name in _AUTHORITY_ONLY_TABLES
                           else admitted_replica if name == "hosted_room_replicas" or name in _REPLICA_ONLY_TABLES
                           else admitted_namespaces)
                rows = (row for row in rows if str(row[room_index]) in allowed)
            if name == "hosted_rooms":
                if room_index is None:  # pragma: no cover - the canonical parent schema owns room_id
                    raise sqlite3.DatabaseError("hosted_rooms source has no room_id")
                rows = list(rows)
                copied_rooms = [str(row[room_index]) for row in rows]
            if name == "hosted_room_id_reservations" and room_index is not None:
                reserved = {
                    str(row[0]) for row in target.execute("SELECT room_id FROM hosted_room_id_reservations")}
                rows = (row for row in rows if str(row[room_index]) not in reserved)
            if name == history_reservations:
                # The preflight above compares all four immutable columns; only an
                # identical existing claim can be treated as an idempotent retry.
                rows = (row for row in rows if str(row[columns.index("room_id")]) not in matching_claim_rooms)
            verb = "INSERT OR IGNORE" if name in _GLOBAL_IDEMPOTENT_TABLES else "INSERT"
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
