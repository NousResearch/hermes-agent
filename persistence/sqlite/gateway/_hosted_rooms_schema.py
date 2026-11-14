"""The hosted-room schema step ``ROOMS_STEP``: the room, event, retired-id, link, remote-run, revocation, reservation and owed-revoke tables with their in-place migrations, then the one-shot import of pre-isolation rows into a ``shared-state.db``. Every verb on the room tables, read or write, runs it when it opens the file."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from persistence.gateway.hosted_rooms import HostedRoomError
from session_rules.hosted_rooms import _system_actor_json

from ._hosted_rooms_common import table_columns
from ._hosted_store import SchemaStep

__all__ = [
    "ROOMS_STEP",
    "_EVENT_BYTES_BACKFILL",
    "_LEGACY_COLUMNS",
    "_RETIRE_FROM_ROOMS",
    "_initialize_schema",
    "_schema_is_current",
]

# The remote-run receipt key in primary-key order, and the table body its DDL and its migration share.
_REMOTE_RUN_IDENTITY_COLUMNS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
    "target_profile", "task_id", "execution_generation")
_REMOTE_RUNS_BODY = """
            room_id TEXT NOT NULL,
            home_install_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            member_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL CHECK (execution_generation >= 1),
            target_install_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (
                room_id, home_install_id, authority_gateway_id, authority_epoch,
                member_id, target_install_id, target_profile, task_id,
                execution_generation
            )
        """
# Executed in this exact order on first open / migration.
_SCHEMA_DDL = (
    """CREATE TABLE IF NOT EXISTS hosted_rooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL DEFAULT 1 CHECK (authority_epoch >= 1),
            next_seq INTEGER NOT NULL DEFAULT 1 CHECK (next_seq >= 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0),
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            disbanded_at REAL
        )""",
    """CREATE TABLE IF NOT EXISTS hosted_room_events (
            room_id TEXT NOT NULL,
            seq INTEGER NOT NULL CHECK (seq >= 1),
            event_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor_json TEXT NOT NULL,
            authority_epoch INTEGER CHECK (authority_epoch IS NULL OR authority_epoch >= 1),
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq),
            UNIQUE (room_id, event_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        )""",
    """CREATE TABLE IF NOT EXISTS hosted_room_retired_ids (
            room_id TEXT PRIMARY KEY,
            retired_at REAL NOT NULL
        )""",
    """CREATE TABLE IF NOT EXISTS hosted_room_links (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_url TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            grant TEXT NOT NULL,
            catalog_json TEXT NOT NULL,
            cancellation_scope_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            transport_security TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id)
        )""", f"CREATE TABLE IF NOT EXISTS hosted_room_remote_runs ({_REMOTE_RUNS_BODY})",
    """CREATE TABLE IF NOT EXISTS hosted_room_revoked_grants (
            scope_key TEXT PRIMARY KEY,
            expires_at REAL NOT NULL,
            revoked_before REAL NOT NULL
        )""",
    """CREATE TABLE IF NOT EXISTS hosted_room_peer_reservations (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            expires_at REAL NOT NULL,
            revoked_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id, target_profile)
        )""",
    # A revoke the home still owes a member's gateway for a grant whose route it dropped. Not a room
    # dependent: it outlives the room's purge, and goes once revoked or past the grant's expiry. A held
    # row is a longer-lived grant a later one replaced: it waits for the room's end (Disband).
    """CREATE TABLE IF NOT EXISTS hosted_room_owed_revokes (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            grant_digest TEXT NOT NULL,
            target_url TEXT NOT NULL,
            grant TEXT NOT NULL,
            expires_at REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at REAL NOT NULL,
            created_at REAL NOT NULL,
            held INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (room_id, member_id, grant_digest)
        )""")
# (table, required columns) parsed from the DDL, in the order _schema_is_current probes them.
_REQUIRED_COLUMNS = tuple(
    (re.search(r"EXISTS (\w+)", ddl).group(1),
     frozenset(re.findall(r"^\s*(\w+) (?:TEXT|INTEGER|REAL)\b", ddl.split("(", 1)[1], re.MULTILINE))) for ddl in _SCHEMA_DDL)
_REMOTE_RUN_SCHEMA_COLUMNS = _REQUIRED_COLUMNS[4][1]

_RETIRE_FROM_ROOMS = (
    "INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at)"
    " SELECT room_id, disbanded_at FROM hosted_rooms WHERE {where}")


# Current when the table has every column and its primary key is the whole receipt key.
def _remote_run_schema_current(conn: sqlite3.Connection, columns: frozenset[str]) -> bool:
    if not _REMOTE_RUN_SCHEMA_COLUMNS.issubset(columns):
        return False
    pk_rows = [row for row in conn.execute("PRAGMA table_info(hosted_room_remote_runs)") if row[5]]
    return tuple(str(row[1]) for row in sorted(pk_rows, key=lambda row: int(row[5]))) == _REMOTE_RUN_IDENTITY_COLUMNS


def _migrate_remote_run_schema(conn: sqlite3.Connection) -> None:
    """Fence legacy receipts behind a complete authority-lineage key."""
    columns = table_columns(conn, "hosted_room_remote_runs")
    if _remote_run_schema_current(conn, columns):
        return
    conn.execute("DROP TABLE IF EXISTS hosted_room_remote_runs_migrating")
    conn.execute(f"CREATE TABLE hosted_room_remote_runs_migrating ({_REMOTE_RUNS_BODY})")
    if columns:
        fallbacks = (("home_install_id", "'legacy'"), ("authority_gateway_id", "'legacy'"), ("authority_epoch", "1"))
        home, gateway, epoch = (column if column in columns else default for column, default in fallbacks)
        conn.execute(
            f"""INSERT OR IGNORE INTO hosted_room_remote_runs_migrating(
                    room_id, home_install_id, authority_gateway_id,
                    authority_epoch, member_id, task_id,
                    execution_generation, target_install_id, target_profile,
                    run_id, session_id, created_at, updated_at
                )
                SELECT room_id, {home}, {gateway}, {epoch}, member_id, task_id,
                       execution_generation, target_install_id, target_profile,
                       run_id, session_id, created_at, updated_at
                  FROM hosted_room_remote_runs""")
    conn.execute("DROP TABLE hosted_room_remote_runs")
    conn.execute("ALTER TABLE hosted_room_remote_runs_migrating RENAME TO hosted_room_remote_runs")


# Draft builds before the actor contract carried no identity. Preserve their inert replay rows explicitly
# as legacy system events rather than guessing a user or Bot author.
_LEGACY_ACTOR_JSON = _system_actor_json("legacy").replace("'", "''")
# (table, column, declaration, default literal) applied in this exact order; each table's PRAGMA is read on
# first use. The default is also what the pre-isolation import selects for a source that predates the column.
_LEGACY_COLUMNS = (
    ("hosted_rooms", "authority_gateway_id", "TEXT NOT NULL", "'legacy'"),
    ("hosted_rooms", "authority_epoch", "INTEGER NOT NULL", "1"),
    ("hosted_rooms", "event_bytes", "INTEGER NOT NULL", "0"),
    ("hosted_room_events", "actor_json", "TEXT NOT NULL", f"'{_LEGACY_ACTOR_JSON}'"),
    ("hosted_room_events", "authority_epoch", "INTEGER", None))
_EVENT_BYTES_BACKFILL = """UPDATE hosted_rooms
                  SET event_bytes=COALESCE((
                      SELECT SUM(
                          length(CAST(event_id AS BLOB)) +
                          length(CAST(kind AS BLOB)) +
                          length(CAST(actor_json AS BLOB)) +
                          length(CAST(payload_json AS BLOB))
                      )
                      FROM hosted_room_events
                      WHERE hosted_room_events.room_id=hosted_rooms.room_id
                  ), 0) WHERE {where}"""


def _migrate_legacy_columns(conn: sqlite3.Connection) -> None:
    """Add columns draft schemas lacked; backfill event_bytes when first introduced."""
    columns: dict[str, frozenset[str]] = {}
    for table, column, declaration, default in _LEGACY_COLUMNS:
        if table not in columns:
            columns[table] = table_columns(conn, table)
        if column not in columns[table]:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                         + (f" DEFAULT {default}" if default is not None else ""))
    if "event_bytes" not in columns["hosted_rooms"]:
        conn.execute(_EVENT_BYTES_BACKFILL.format(where="1"))


_KIND_INDEX = "idx_hosted_room_events_kind"


# Creates the tables, upgrades older layouts in place, and refuses to finish unless the result is current.
def _initialize_schema(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_DDL:
        conn.execute(statement)
    _migrate_legacy_columns(conn)
    # Old schemas kept the final identity tombstone in hosted_rooms itself. Copy those identities before
    # bounded history pruning can remove their heavier room/event payloads. This compact registry is
    # intentionally permanent: a stale coordinate must never name a different Group Chat.
    conn.execute(_RETIRE_FROM_ROOMS.format(where="disbanded_at IS NOT NULL"))
    _migrate_remote_run_schema(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hosted_room_events_cursor ON hosted_room_events(room_id, seq)")
    # Latest-event-of-a-kind reads (room purpose/notes) seek here instead of walking the room's history.
    conn.execute(f"CREATE INDEX IF NOT EXISTS {_KIND_INDEX} ON hosted_room_events(room_id, kind, seq)")
    if not _schema_is_current(conn):
        raise HostedRoomError("hosted room schema migration did not complete")


# Every table has its columns, the remote-run key is complete and the event cursor and kind indexes exist.
def _schema_is_current(conn: sqlite3.Connection) -> bool:
    # Read every table first (fixed PRAGMA order), then compare.
    actual = [table_columns(conn, table) for table, _ in _REQUIRED_COLUMNS]
    return all(
        required.issubset(columns)
        and (table != "hosted_room_remote_runs" or _remote_run_schema_current(conn, columns))
        for (table, required), columns in zip(_REQUIRED_COLUMNS, actual, strict=True)) and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_hosted_room_events_cursor'"
    ).fetchone() is not None and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (_KIND_INDEX,)).fetchone() is not None


def _store_ready(conn: sqlite3.Connection, db_path: Path) -> bool:
    """The store can serve rooms once its schema is current and the pre-isolation import has run."""
    from ._hosted_rooms_legacy_import import settled

    return _schema_is_current(conn) and settled(conn, db_path)


def _initialize_store(conn: sqlite3.Connection, db_path: Path) -> None:
    """Create or migrate the schema, then copy the pre-isolation rows in."""
    from ._hosted_rooms_legacy_import import import_legacy_rooms

    _initialize_schema(conn)
    import_legacy_rooms(conn, db_path)


ROOMS_STEP: SchemaStep = (_store_ready, _initialize_store)
