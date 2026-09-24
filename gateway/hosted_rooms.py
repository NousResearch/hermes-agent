"""Durable state for gateway-hosted Bot Mode rooms.

Owns only hosted-room identity and its append-only event log; delivery, relay leasing and agent turns
belong to the relay and the hosted-room driver, so the log composes with a durable relay without a second
transport queue. Callers supply the database path (production handlers use the gateway's root ``state.db``).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from gateway.hosted_rooms_common import (
    DbPath, bounded_int, canonical_json, clock as _now, compact_json, connect, fenced_update as _fenced_update,
    identifier, open_sqlite, table_columns, table_exists, transaction, utf8_len)

PROTOCOL_VERSION = 2
MAX_ROOM_ID_CHARS = 128
MAX_EVENT_ID_CHARS = 128
MAX_ROOM_NAME_CHARS = 200
MAX_EVENT_KIND_CHARS = 64
MAX_ACTOR_ID_CHARS = 128
MAX_ACTOR_LABEL_CHARS = 200
MAX_MEMBERS = 128
MAX_MEMBERS_JSON_BYTES = 128 * 1024
MAX_EVENT_JSON_BYTES = 256 * 1024
MAX_LOG_LIMIT = 500
MAX_LOG_PAGE_BYTES = 2 * 1024 * 1024
MAX_ROOM_LIST_LIMIT = 500
MAX_ACTIVE_ROOMS = 256
MAX_DISBANDED_ROOM_TOMBSTONES = 512
DISBANDED_ROOM_RETENTION_SECONDS = 90 * 24 * 60 * 60
MAX_EVENTS_PER_ROOM = 50_000
MAX_ROOM_EVENT_BYTES = 256 * 1024 * 1024
# Leave substantial headroom below the pre-update state.db snapshot ceiling: event accounting excludes
# SQLite indexes and repeated room ids, so the logical budget must stay well below the physical-file limit.
MAX_GATEWAY_EVENT_BYTES = 16 * 1024 * 1024
CONTROL_EVENT_COUNT_RESERVE = 64
CONTROL_EVENT_BYTE_RESERVE = 1024 * 1024
_JOURNAL_MODE_LOCK_RETRIES = 8

_EVENT_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_CONTROL_EVENT_KINDS = frozenset({"authority.claimed", "authority.lost", "room.disbanded", "room.stop_requested"})
_EVENT_KINDS_BY_ACTOR = {
    "user": frozenset({"message.user"}), "member": frozenset({"message.member"}),
    "gateway": frozenset({
        "member.unavailable", "room.activity", "room.stop_requested", "turn.deferred", "turn.reassigned",
        "turn.cancelled", "turn.failed", "turn.settled", "turn.started"}),
    "system": frozenset({
        "authority.claimed", "authority.lost", "room.created", "room.disbanded", "room.members_changed", "room.renamed"
    })}
_OPTIONAL_ACTOR_FIELDS = (
    ("display_name", MAX_ACTOR_LABEL_CHARS), ("profile", MAX_ACTOR_ID_CHARS), ("connection_id", MAX_ACTOR_ID_CHARS))
_ACTOR_FIELDS = frozenset({"kind", "id", *(field for field, _ in _OPTIONAL_ACTOR_FIELDS)})

# --- schema -----------------------------------------------------------------
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
        )""")
# (table, required columns) parsed from the DDL, in the order _schema_is_current probes them.
_REQUIRED_COLUMNS = tuple(
    (re.search(r"EXISTS (\w+)", ddl).group(1),
     frozenset(re.findall(r"^\s*(\w+) (?:TEXT|INTEGER|REAL)\b", ddl.split("(", 1)[1], re.M))) for ddl in _SCHEMA_DDL)
_REMOTE_RUN_SCHEMA_COLUMNS = _REQUIRED_COLUMNS[4][1]

# --- SQL fragments (statement text must stay byte-stable after whitespace normalisation) ---
_EVENT_COLUMNS = ("room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at")
_SELECT_EVENT = f"SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND event_id=?"
_INSERT_EVENT = (f"INSERT INTO hosted_room_events ({_EVENT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
_ROOM_COLUMNS = (
    "room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq, revision,"
    " created_at, updated_at, disbanded_at")
_ROOM_COLUMNS_WITH_BYTES = (
    "room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq, event_bytes,"
    " revision, created_at, updated_at, disbanded_at")
_SELECT_ROOM = f"SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE room_id=?"
_SELECT_ROOM_WITH_BYTES = f"SELECT {_ROOM_COLUMNS_WITH_BYTES} FROM hosted_rooms WHERE room_id=?"
_SUM_EVENT_BYTES = "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
_INSERT_RETIRED = ("INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at) VALUES (?, ?)")
_RETIRE_FROM_ROOMS = (
    "INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at)"
    " SELECT room_id, disbanded_at FROM hosted_rooms WHERE {where}")
_LINK_COLUMNS = (
    "room_id", "member_id", "target_url", "target_profile", "grant", "catalog_json", "cancellation_scope_id",
    "trace_id", "transport_security", "status", "updated_at")
_REMOTE_RUN_WHERE = " AND ".join(f"{column}=?" for column in _REMOTE_RUN_IDENTITY_COLUMNS)
_SELECT_REMOTE_RUN = f"SELECT * FROM hosted_room_remote_runs WHERE {_REMOTE_RUN_WHERE}"
_LIVE_RESERVATION_WHERE = ("WHERE room_id=? AND target_profile=? AND expires_at>? AND revoked_at IS NULL")
_SELECT_LIVE_RESERVATION = (f"SELECT 1 FROM hosted_room_peer_reservations {_LIVE_RESERVATION_WHERE} LIMIT 1")


class HostedRoomError(ValueError): """Base class for invalid or conflicting hosted-room operations."""

class RoomNotFoundError(HostedRoomError): """Raised when a room does not exist or has been disbanded."""

class RoomHistoryExpiredError(RoomNotFoundError):
    """Raised when a retired room remains reserved after history compaction."""
    reason = "room_history_expired"

class RoomConflictError(HostedRoomError): """Raised when an idempotency key is reused for different room state."""

class RoomProbeUnavailableError(HostedRoomError):
    """Raised when a non-blocking ownership probe cannot read the room store."""

class EventConflictError(HostedRoomError): """Raised when an event id is reused with different immutable content."""

class AuthorityConflictError(HostedRoomError):
    """Raised when a stale room authority attempts to mutate hosted state."""
    reason = "authority_conflict"

class AuthoritySupersededError(AuthorityConflictError):
    """Raised when a successful authority claim was later superseded."""


# --- validation ---------------------------------------------------------------
_canonical_json = partial(canonical_json, error=HostedRoomError, ensure_ascii=False)
_validate_identifier = partial(identifier, error=HostedRoomError)
_room_id = partial(_validate_identifier, label="room_id", max_chars=MAX_ROOM_ID_CHARS)
_event_id = partial(_validate_identifier, label="event_id", max_chars=MAX_EVENT_ID_CHARS)
_actor_json = partial(_canonical_json, label="actor", max_bytes=4 * 1024)
_payload_json = partial(_canonical_json, label="payload", max_bytes=MAX_EVENT_JSON_BYTES)
_bounded_int = partial(bounded_int, error=HostedRoomError)
_validate_room_name = partial(
    _validate_identifier, label="name", max_chars=MAX_ROOM_NAME_CHARS, pattern=None, invalid="invalid room name")
_validate_event_kind = partial(
    _validate_identifier, label="kind", max_chars=MAX_EVENT_KIND_CHARS, pattern=_EVENT_KIND_RE,
    invalid="invalid event kind")


def _actor_id(value: Any, label: str) -> str:
    return _validate_identifier(value, label=label, max_chars=MAX_ACTOR_ID_CHARS)


def _require_positive_int(value: Any, label: str) -> int:
    return _bounded_int(value, message=f"{label} must be a positive integer", low=1)


def _bounded_limit(value: Any, maximum: int) -> int:
    return _bounded_int(value, message=f"limit must be between 1 and {maximum}", low=1, high=maximum)


def _non_negative(value: Any, label: str) -> int:
    return _bounded_int(value, message=f"{label} must be a non-negative integer")


def _system_actor_json(actor_id: str) -> str:
    return _actor_json({"kind": "system", "id": actor_id})


def _claim_payload_json(previous_gateway_id: str, new_gateway_id: str, epoch: int) -> str:
    return _payload_json(
        {"previous_gateway_id": previous_gateway_id, "authority_gateway_id": new_gateway_id, "authority_epoch": epoch})


def user_event_id(client_event_id: Any) -> str:
    """Map a client retry key into the server-owned user-event namespace."""
    return f"user:{hashlib.sha256(_event_id(client_event_id).encode('utf-8')).hexdigest()}"


def _validate_members(value: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list):
        raise HostedRoomError("members must be a list")
    if len(value) > MAX_MEMBERS:
        raise HostedRoomError("too many room members")
    if not all(isinstance(member, dict) for member in value):
        raise HostedRoomError("each room member must be an object")
    members = [dict(member) for member in value]
    return members, _canonical_json(members, label="members", max_bytes=MAX_MEMBERS_JSON_BYTES)


def _legacy_members_match(existing_json: str, proposed: list[dict[str, Any]]) -> bool:
    """Allow adoption to add routing metadata an older room could not store."""
    try:
        existing = json.loads(existing_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(existing, list) or len(existing) != len(proposed):
        return False
    for previous, current in zip(existing, proposed, strict=True):
        if not isinstance(previous, dict):
            return False
        previous, current = dict(previous), dict(current)
        previous_target, current_target = previous.pop("target", None), current.pop("target", None)
        if previous != current or (previous_target not in (None, {}) and previous_target != current_target):
            return False
    return True


def _validate_actor(value: Any, *, kind: str) -> tuple[dict[str, str], str]:
    if not isinstance(value, dict):
        raise HostedRoomError("actor must be an object")
    unknown = set(value) - _ACTOR_FIELDS
    if unknown:
        raise HostedRoomError(f"unknown actor fields: {', '.join(sorted(unknown))}")
    actor_kind = value.get("kind")
    if not isinstance(actor_kind, str) or actor_kind not in _EVENT_KINDS_BY_ACTOR:
        raise HostedRoomError("invalid actor.kind")
    if kind not in _EVENT_KINDS_BY_ACTOR[actor_kind]:
        raise HostedRoomError(f"actor kind '{actor_kind}' cannot append '{kind}'")
    actor = {"kind": actor_kind, "id": _actor_id(value.get("id"), "actor.id")}
    for field, max_chars in _OPTIONAL_ACTOR_FIELDS:
        field_value = value.get(field)
        if field_value is None:
            continue
        if not isinstance(field_value, str):
            raise HostedRoomError(f"actor.{field} must be a string")
        if len(field_value := field_value.strip()) > max_chars:
            raise HostedRoomError(f"actor.{field} is too long")
        if field_value:
            actor[field] = field_value
    return actor, _actor_json(actor)


# --- schema / connections -------------------------------------------------------
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
# (table, column, ddl) applied in this exact order; each table's PRAGMA is read on first use.
_LEGACY_COLUMN_DDL = (
    ("hosted_rooms", "authority_gateway_id",
     "ALTER TABLE hosted_rooms ADD COLUMN authority_gateway_id TEXT NOT NULL DEFAULT 'legacy'"),
    ("hosted_rooms", "authority_epoch",
     "ALTER TABLE hosted_rooms ADD COLUMN authority_epoch INTEGER NOT NULL DEFAULT 1"),
    ("hosted_rooms", "event_bytes", "ALTER TABLE hosted_rooms ADD COLUMN event_bytes INTEGER NOT NULL DEFAULT 0"),
    ("hosted_room_events", "actor_json",
     "ALTER TABLE hosted_room_events " f"ADD COLUMN actor_json TEXT NOT NULL DEFAULT '{_LEGACY_ACTOR_JSON}'"),
    ("hosted_room_events", "authority_epoch", "ALTER TABLE hosted_room_events ADD COLUMN authority_epoch INTEGER"))


def _migrate_legacy_columns(conn: sqlite3.Connection) -> None:
    """Add columns draft schemas lacked; backfill event_bytes when first introduced."""
    columns: dict[str, frozenset[str]] = {}
    for table, column, ddl in _LEGACY_COLUMN_DDL:
        if table not in columns:
            columns[table] = table_columns(conn, table)
        if column not in columns[table]:
            conn.execute(ddl)
    if "event_bytes" not in columns["hosted_rooms"]:
        conn.execute("""UPDATE hosted_rooms
                  SET event_bytes=COALESCE((
                      SELECT SUM(
                          length(CAST(event_id AS BLOB)) +
                          length(CAST(kind AS BLOB)) +
                          length(CAST(actor_json AS BLOB)) +
                          length(CAST(payload_json AS BLOB))
                      )
                      FROM hosted_room_events
                      WHERE hosted_room_events.room_id=hosted_rooms.room_id
                  ), 0)""")


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
    if not _schema_is_current(conn):
        raise HostedRoomError("hosted room schema migration did not complete")


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    # Read every table first (fixed PRAGMA order), then compare.
    actual = [table_columns(conn, table) for table, _ in _REQUIRED_COLUMNS]
    return all(
        required.issubset(columns)
        and (table != "hosted_room_remote_runs" or _remote_run_schema_current(conn, columns))
        for (table, required), columns in zip(_REQUIRED_COLUMNS, actual, strict=True)) and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_hosted_room_events_cursor'"
    ).fetchone() is not None


def default_db_path() -> Path:
    """Return the gateway-wide state database for the active install."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return (home.parent.parent if home.parent.name == "profiles" else home) / "state.db"


def local_authority_gateway_id() -> str:
    """Return the stable server-owned identity for hosted-room authority."""
    from hermes_cli.install_identity import get_install_id
    install_id = get_install_id()
    if not install_id:
        raise HostedRoomError("stable gateway install identity is unavailable")
    return _actor_id(f"install:{install_id}", "authority_gateway_id")


_connect = partial(
    connect, db_label="state.db (hosted_rooms)", ready=_schema_is_current,
    initialize=lambda conn: _initialize_schema(conn), lock_retries=_JOURNAL_MODE_LOCK_RETRIES)


def _read_connection(db_path: DbPath) -> sqlite3.Connection:
    """Open the room store without steady-state journal or migration writes."""
    path = Path(db_path)
    if not path.is_file():
        _connect(path).close()
    conn = open_sqlite(path)
    if not _schema_is_current(conn):
        conn.close()
        _connect(path).close()
        conn = open_sqlite(path)
    return conn


_transaction = partial(transaction, _connect, immediate=False)


# --- row helpers ------------------------------------------------------------------
def _is_retired(conn: sqlite3.Connection, room_id: str) -> bool:
    return conn.execute("SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)).fetchone() is not None


def _room_row(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...], room_id: str) -> sqlite3.Row:
    """Fetch one hosted_rooms row or raise the precise not-found/expired error."""
    row = conn.execute(sql, params).fetchone()
    if row is not None:
        return row
    # A retained disband tombstone still has replayable history; the caller did not opt into disbanded rooms.
    retained = conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if retained is None and _is_retired(conn, room_id):
        raise RoomHistoryExpiredError("Group Chat history expired; room_id remains permanently retired")
    raise RoomNotFoundError("hosted room not found")


def _require_authority(room: sqlite3.Row, gateway_id: str, epoch: int, message: str) -> None:
    if str(room["authority_gateway_id"]) != gateway_id or int(room["authority_epoch"]) != epoch:
        raise AuthorityConflictError(message)


def _reload(conn: sqlite3.Connection, sql: str, params: tuple, missing: str) -> sqlite3.Row:
    """Re-read a row this transaction just wrote; a miss is an invariant violation."""
    row = conn.execute(sql, params).fetchone()
    if row is None:  # pragma: no cover - guarded by the write above
        raise RuntimeError(missing)
    return row


def _room_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    keys = row.keys()  # sqlite3.Row: ``x in row`` scans values, so ``.keys()`` is load-bearing.
    return {
        "room_id": row["room_id"], "name": row["name"], "members": json.loads(row["members_json"]),
        "authority_gateway_id": row["authority_gateway_id"], "authority_epoch": int(row["authority_epoch"]),
        "revision": int(row["revision"]), "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]), "idempotent": idempotent,
        **({"disbanded_at": float(row["disbanded_at"])} if "disbanded_at" in keys and row["disbanded_at"] is not None
           else {}),
        **({"latest_seq": int(row["next_seq"]) - 1} if "next_seq" in keys else {})}


def _event_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    epoch = row["authority_epoch"]
    return {
        "room_id": row["room_id"], "seq": int(row["seq"]), "event_id": row["event_id"], "kind": row["kind"],
        "actor": json.loads(row["actor_json"]), "authority_epoch": int(epoch) if epoch is not None else None,
        "payload": json.loads(row["payload_json"]), "created_at": float(row["created_at"]), "idempotent": idempotent}


def _load_event(conn: sqlite3.Connection, room_id: str, event_id: str) -> sqlite3.Row | None:
    return conn.execute(_SELECT_EVENT, (room_id, event_id)).fetchone()


def _event_content(row: sqlite3.Row) -> tuple[Any, Any, Any, Any]:
    """The immutable (kind, actor_json, authority_epoch, payload_json) an event id is bound to."""
    return row["kind"], row["actor_json"], row["authority_epoch"], row["payload_json"]


def _gateway_event_bytes(conn: sqlite3.Connection) -> int:
    return int(conn.execute(_SUM_EVENT_BYTES).fetchone()[0])


def _insert_event(
    conn: sqlite3.Connection, room: sqlite3.Row, room_id: str, seq: int, event_id: str, kind: str, actor_json: str,
    epoch: int, payload_json: str, now: float, *, allow_control: bool = False) -> int:
    """Capacity-check then INSERT one event at ``seq``; returns its accounted bytes."""
    event_bytes = _prepare_event(conn, room, event_id, kind, actor_json, payload_json, allow_control=allow_control)
    conn.execute(_INSERT_EVENT, (room_id, seq, event_id, kind, actor_json, epoch, payload_json, now))
    return event_bytes


def _prepare_event(
    conn: sqlite3.Connection, room: sqlite3.Row, event_id: str, kind: str, actor_json: str, payload_json: str, *,
    allow_control: bool = False) -> int:
    """Size one pending event and enforce per-room and gateway capacity; returns its bytes."""
    additional_bytes = utf8_len(event_id, kind, actor_json, payload_json)
    count_reserve, byte_reserve = (CONTROL_EVENT_COUNT_RESERVE, CONTROL_EVENT_BYTE_RESERVE) if allow_control else (0, 0)
    gateway_byte_limit = MAX_GATEWAY_EVENT_BYTES + byte_reserve
    if int(room["next_seq"]) - 1 >= MAX_EVENTS_PER_ROOM + count_reserve:
        raise HostedRoomError("This Group Chat reached its history limit. Start a new Group Chat to continue.")
    if int(room["event_bytes"]) + additional_bytes > MAX_ROOM_EVENT_BYTES + byte_reserve:
        raise HostedRoomError("This Group Chat reached its storage limit. Start a new Group Chat to continue.")
    gateway_bytes = _gateway_event_bytes(conn)
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        _prune_disbanded_rooms_locked(
            conn, now=None, max_gateway_event_bytes=max(0, gateway_byte_limit - additional_bytes))
        gateway_bytes = _gateway_event_bytes(conn)
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        raise HostedRoomError("Group Chat storage is full on this host. Delete an old Group Chat and try again.")
    return additional_bytes


# --- retention -------------------------------------------------------------------
# Deleted in this order when a disbanded room's payload is purged.
_DEPENDENT_TABLES = (
    "hosted_room_policy_transcript_state", "hosted_room_policy_transcript", "hosted_room_policy_publications",
    "hosted_room_policy_watermarks", "hosted_room_policy_events", "hosted_room_policy_threads",
    "hosted_room_policy_cursors", "hosted_room_driver_tasks", "hosted_room_driver_leases", "hosted_room_remote_runs",
    "hosted_room_links", "hosted_room_peer_reservations", "hosted_room_events")


def _room_ids(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[str]:
    return [str(row["room_id"]) for row in conn.execute(sql, params).fetchall()]


def _prune_disbanded_rooms_locked(
    conn: sqlite3.Connection, *, now: float | None, max_gateway_event_bytes: int | None = None) -> int:
    candidates: set[str] = set()
    if now is not None:
        candidates.update(_room_ids(
            conn, """SELECT room_id FROM hosted_rooms
                     WHERE disbanded_at IS NOT NULL AND disbanded_at<=?""", (now - DISBANDED_ROOM_RETENTION_SECONDS,)))
    candidates.update(_room_ids(
        conn, """SELECT room_id FROM hosted_rooms WHERE disbanded_at IS NOT NULL
                ORDER BY disbanded_at DESC, room_id ASC LIMIT -1 OFFSET ?""", (MAX_DISBANDED_ROOM_TOMBSTONES,)))
    if max_gateway_event_bytes is not None:
        retained_bytes = _gateway_event_bytes(conn)
        if retained_bytes > max_gateway_event_bytes:
            for row in conn.execute("""SELECT room_id, event_bytes FROM hosted_rooms WHERE disbanded_at IS NOT NULL
                    ORDER BY disbanded_at ASC, room_id ASC"""
            ).fetchall():
                candidates.add(str(row["room_id"]))
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_gateway_event_bytes:
                    break
    if not candidates:
        return 0
    placeholders = ",".join("?" for _ in candidates)
    room_ids = tuple(sorted(candidates))
    conn.execute(_RETIRE_FROM_ROOMS.format(where=f"room_id IN ({placeholders}) AND disbanded_at IS NOT NULL"), room_ids)
    for table in _DEPENDENT_TABLES:
        if table_exists(conn, table):
            conn.execute(f"DELETE FROM {table} WHERE room_id IN ({placeholders})", room_ids)
    conn.execute(f"DELETE FROM hosted_rooms WHERE room_id IN ({placeholders})", room_ids)
    return len(room_ids)


def prune_disbanded_rooms(db_path: DbPath, *, now: float | None = None) -> int:
    """Purge deleted Group Chat payloads while reserving their identities."""
    with _transaction(db_path, immediate=True) as conn:
        return _prune_disbanded_rooms_locked(conn, now=_now(now))


# --- room links / grants / reservations / remote runs ---------------------------------
def list_room_link_records(db_path: DbPath) -> list[dict[str, Any]]:
    """Return private RoomLink records without logging or formatting grants."""
    with _transaction(db_path) as conn:
        rows = conn.execute("""SELECT room_id, member_id, target_url, target_profile, grant,
                      catalog_json, cancellation_scope_id, trace_id,
                      transport_security, status, updated_at
                 FROM hosted_room_links
             ORDER BY room_id, member_id""").fetchall()
    return [dict(row) for row in rows]


def upsert_room_link_record(db_path: DbPath, *, record: Mapping[str, Any], max_links: int) -> None:
    """Atomically insert or replace one private RoomLink record."""
    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            "SELECT 1 FROM hosted_room_links WHERE room_id=? AND member_id=?", (record["room_id"], record["member_id"])
        ).fetchone()
        if existing is None and int(conn.execute("SELECT COUNT(*) FROM hosted_room_links").fetchone()[0]) >= max_links:
            raise HostedRoomError("too many stored room links")
        conn.execute("""INSERT INTO hosted_room_links(
                   room_id, member_id, target_url, target_profile, grant,
                   catalog_json, cancellation_scope_id, trace_id,
                   transport_security, status, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(room_id, member_id) DO UPDATE SET
                   target_url=excluded.target_url,
                   target_profile=excluded.target_profile,
                   grant=excluded.grant,
                   catalog_json=excluded.catalog_json,
                   cancellation_scope_id=excluded.cancellation_scope_id,
                   trace_id=excluded.trace_id,
                   transport_security=excluded.transport_security,
                   status=excluded.status,
                   updated_at=excluded.updated_at""",
            tuple(record[column] for column in _LINK_COLUMNS))


def update_room_link_status(
    db_path: DbPath, *, room_id: str, member_id: str, status: str, now: float | None = None) -> bool:
    """Persist a non-secret route health classification."""
    with _transaction(db_path, immediate=True) as conn:
        return conn.execute(
            "UPDATE hosted_room_links SET status=?, updated_at=? WHERE room_id=? AND member_id=?",
            (status, _now(now), room_id, member_id)).rowcount == 1


def delete_room_link_records(db_path: DbPath, *, room_id: str) -> int:
    """Delete persisted peer routes after their target grants are revoked."""
    with _transaction(db_path, immediate=True) as conn:
        return conn.execute("DELETE FROM hosted_room_links WHERE room_id=?", (room_id,)).rowcount


def _claim_values(claims: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    return {key: str(claims.get(key) or "") for key in keys}


def _room_grant_scope_key(claims: Mapping[str, Any]) -> str:
    """Return a stable non-secret key for one room/home/target/profile scope."""
    fields = _claim_values(claims, (
        "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
        "target_profile"))
    if not all(fields.values()):
        raise HostedRoomError("room grant scope is incomplete")
    return hashlib.sha256(compact_json(fields).encode("utf-8")).hexdigest()


def revoke_room_grant_scope(
    db_path: DbPath, *, claims: Mapping[str, Any], expires_at: float, now: float | None = None) -> None:
    """Revoke every grant issued at or before now for one exact room scope."""
    scope_key = _room_grant_scope_key(claims)
    timestamp = _now(now)
    expiry = float(expires_at)
    if expiry <= timestamp:
        return
    with _transaction(db_path, immediate=True) as conn:
        conn.execute("DELETE FROM hosted_room_revoked_grants WHERE expires_at<=?", (timestamp,))
        conn.execute("""INSERT INTO hosted_room_revoked_grants(
                   scope_key, expires_at, revoked_before
               ) VALUES (?, ?, ?)
               ON CONFLICT(scope_key) DO UPDATE SET
                   expires_at=MAX(hosted_room_revoked_grants.expires_at,
                                  excluded.expires_at),
                   revoked_before=MAX(hosted_room_revoked_grants.revoked_before,
                                      excluded.revoked_before)""", (scope_key, expiry, timestamp))
        conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at=?, updated_at=? WHERE room_id=?
                AND member_id=? AND target_profile=? AND authority_gateway_id=?
                AND authority_epoch=?""",
            (
                timestamp, timestamp,
                *_claim_values(claims, ("room_id", "member_id", "target_profile", "authority_gateway_id")).values(),
                int(claims.get("authority_epoch") or 0)))


def _reservation_claims(claims: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    """Validate (room_id, member_id, target_profile, authority_gateway_id, authority_epoch)."""
    values = (
        _room_id(claims.get("room_id")),
        *(_actor_id(claims.get(key), key) for key in ("member_id", "target_profile", "authority_gateway_id")),
        int(claims.get("authority_epoch") or 0))
    if values[4] < 1:
        raise HostedRoomError("authority_epoch must be positive")
    return values


def _reservation_superseded(row: sqlite3.Row, gateway_id: str, epoch: int) -> bool:
    """A newer epoch, or the same epoch under another gateway, outranks this claim."""
    row_epoch = int(row["authority_epoch"])
    return row_epoch > epoch or (row_epoch == epoch and str(row["authority_gateway_id"]) != gateway_id)


def reserve_peer_room(
    db_path: DbPath, *, claims: Mapping[str, Any], expires_at: float, now: float | None = None) -> None:
    """Fence direct Desktop prompts before the first peer run is admitted."""
    timestamp = _now(now)
    expiry = float(expires_at)
    if expiry <= timestamp:
        raise HostedRoomError("peer room reservation must expire in the future")
    values = _reservation_claims(claims)
    room_id, _, target_profile, gateway_id, epoch = values
    with _transaction(db_path, immediate=True) as conn:
        conn.execute("DELETE FROM hosted_room_peer_reservations WHERE expires_at<=?", (timestamp,))
        authority_rows = conn.execute(
            f"""SELECT authority_gateway_id, authority_epoch
                FROM hosted_room_peer_reservations {_LIVE_RESERVATION_WHERE}""", (room_id, target_profile, timestamp)
        ).fetchall()
        if any(_reservation_superseded(row, gateway_id, epoch) for row in authority_rows):
            raise AuthorityConflictError("peer room reservation authority changed")
        conn.execute("""UPDATE hosted_room_peer_reservations SET revoked_at=?, updated_at=? WHERE room_id=?
                AND target_profile=? AND authority_epoch<? AND revoked_at IS NULL""",
            (timestamp, timestamp, room_id, target_profile, epoch))
        existing = conn.execute("""SELECT authority_gateway_id, authority_epoch FROM hosted_room_peer_reservations
                WHERE room_id=? AND member_id=? AND target_profile=?""", values[:3]).fetchone()
        if existing is not None and _reservation_superseded(existing, gateway_id, epoch):
            raise AuthorityConflictError("peer room reservation authority changed")
        conn.execute("""INSERT INTO hosted_room_peer_reservations(
                   room_id, member_id, target_profile, authority_gateway_id,
                   authority_epoch, expires_at, revoked_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
               ON CONFLICT(room_id, member_id, target_profile) DO UPDATE SET
                   authority_gateway_id=excluded.authority_gateway_id,
                   authority_epoch=excluded.authority_epoch,
                   expires_at=MAX(hosted_room_peer_reservations.expires_at,
                                  excluded.expires_at),
                   revoked_at=NULL,
                   updated_at=excluded.updated_at""", (*values, expiry, timestamp, timestamp))


def _read_one(db_path: DbPath, sql: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
    with _transaction(db_path) as conn:
        return conn.execute(sql, params).fetchone()


def peer_room_is_reserved(db_path: DbPath, *, room_id: str, target_profile: str, now: float | None = None) -> bool:
    """Return whether a live target-side RoomLink reservation fences Desktop."""
    params = (_room_id(room_id), _actor_id(target_profile, "target_profile"), _now(now))
    return _read_one(db_path, _SELECT_LIVE_RESERVATION, params) is not None


def peer_room_grant_is_current(db_path: DbPath, *, claims: Mapping[str, Any], now: float | None = None) -> bool:
    """Require a grant to match the target's current live reservation."""
    timestamp = _now(now)
    return _read_one(
        db_path, """SELECT 1 FROM hosted_room_peer_reservations WHERE room_id=? AND member_id=?
            AND target_profile=? AND authority_gateway_id=? AND authority_epoch=?
            AND expires_at>? AND revoked_at IS NULL LIMIT 1""", (*_reservation_claims(claims), timestamp)) is not None


def room_grant_is_revoked(db_path: DbPath, *, claims: Mapping[str, Any], now: float | None = None) -> bool:
    """Return whether a grant predates its exact scope's revocation fence."""
    timestamp = _now(now)
    scope_key = _room_grant_scope_key(claims)
    issued_at = float(claims.get("issued_at") or 0)
    row = _read_one(
        db_path, """SELECT revoked_before FROM hosted_room_revoked_grants
            WHERE scope_key=? AND expires_at>?""", (scope_key, timestamp))
    return row is not None and issued_at <= float(row["revoked_before"])


def _remote_run_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column] for column in _REMOTE_RUN_IDENTITY_COLUMNS)


def upsert_remote_run_receipt(db_path: DbPath, *, record: Mapping[str, Any], now: float | None = None) -> None:
    """Durably bind one logical peer task attempt to its remote run handle."""
    timestamp = _now(now)
    identity = _remote_run_identity(record)
    immutable = (*identity, record["run_id"], record["session_id"])
    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(_SELECT_REMOTE_RUN, identity).fetchone()
        if existing is not None:
            if (*_remote_run_identity(existing), existing["run_id"], existing["session_id"]) != immutable:
                raise HostedRoomError("remote run receipt conflicts with its logical task")
            conn.execute(
                f"UPDATE hosted_room_remote_runs SET updated_at=? WHERE {_REMOTE_RUN_WHERE}", (timestamp, *identity))
            return
        conn.execute("""INSERT INTO hosted_room_remote_runs(
                   room_id, home_install_id, authority_gateway_id,
                   authority_epoch, member_id, target_install_id,
                   target_profile, task_id, execution_generation, run_id,
                   session_id, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (*immutable, timestamp, timestamp))


def list_remote_run_receipts(
    db_path: DbPath, *, room_id: str | None = None, target_profile: str | None = None, session_id: str | None = None
) -> list[dict[str, Any]]:
    """Return remote run handles in durable task order."""
    candidates = (("room_id", room_id), ("target_profile", target_profile), ("session_id", session_id))
    filters = [(column, value) for column, value in candidates if value is not None]
    where = f" WHERE {' AND '.join(f'{column}=?' for column, _ in filters)}" if filters else ""
    with _transaction(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hosted_room_remote_runs" + where
            + " ORDER BY created_at, task_id, execution_generation", [value for _, value in filters]).fetchall()
    return [dict(row) for row in rows]


def remote_run_receipt(db_path: DbPath, *, record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the exact durable remote run handle for one task attempt."""
    row = _read_one(db_path, _SELECT_REMOTE_RUN, _remote_run_identity(record))
    return dict(row) if row is not None else None


# --- rooms and events -------------------------------------------------------------
def _adopt_legacy_room(
    conn: sqlite3.Connection, existing: sqlite3.Row, *, room_id: str, members_json: str, authority_gateway_id: str,
    now: float) -> dict[str, Any]:
    """Claim a 'legacy'-authority room for a real gateway with a fenced claim event."""
    target_epoch = int(existing["authority_epoch"]) + 1
    seq = int(existing["next_seq"])
    claim_bytes = _insert_event(
        conn, existing, room_id, seq, "system:authority-adopted", "authority.claimed",
        _system_actor_json("authority-control"), target_epoch,
        _claim_payload_json("legacy", authority_gateway_id, target_epoch), now, allow_control=True)
    _fenced_update(conn, """UPDATE hosted_rooms
            SET members_json=?, authority_gateway_id=?, authority_epoch=?,
                next_seq=next_seq+1, revision=revision+1, event_bytes=event_bytes+?, updated_at=?
            WHERE room_id=? AND authority_gateway_id='legacy' AND authority_epoch=? AND next_seq=?
            AND disbanded_at IS NULL""",
        (
            members_json, authority_gateway_id, target_epoch, claim_bytes, now, room_id,
            int(existing["authority_epoch"]), seq), AuthorityConflictError("legacy room adoption lost its fence"))
    existing = _reload(conn, _SELECT_ROOM, (room_id,), "adopted room could not be reloaded")
    claim_event = _reload(
        conn, _SELECT_EVENT, (room_id, "system:authority-adopted"), "legacy adoption event could not be reloaded")
    return {**_room_from_row(existing, idempotent=True), "adopted": True, "claim_event": _event_from_row(claim_event)}


def create_room(
    db_path: DbPath, *, room_id: Any, name: Any, members: Any, authority_gateway_id: Any, now: float | None = None
) -> dict[str, Any]:
    """Create a room, or return the identical existing room idempotently."""
    room_id = _room_id(room_id)
    name = _validate_room_name(name)
    normalized_members, members_json = _validate_members(members)
    authority_gateway_id = _actor_id(authority_gateway_id, "authority_gateway_id")
    now = _now(now)
    with _transaction(db_path, immediate=True) as conn:
        if _is_retired(conn, room_id):
            raise RoomConflictError("room_id belongs to a disbanded room")
        existing = conn.execute(_SELECT_ROOM_WITH_BYTES, (room_id,)).fetchone()
        if existing is not None:
            if existing["disbanded_at"] is not None:
                raise RoomConflictError("room_id belongs to a disbanded room")
            legacy_adoption = (existing["authority_gateway_id"] == "legacy" and authority_gateway_id != "legacy")
            members_match = existing["members_json"] == members_json or (
                legacy_adoption and _legacy_members_match(existing["members_json"], normalized_members))
            if existing["name"] != name or not members_match:
                raise RoomConflictError("room_id already exists with different state")
            if legacy_adoption:
                return _adopt_legacy_room(
                    conn, existing, room_id=room_id, members_json=members_json,
                    authority_gateway_id=authority_gateway_id, now=now)
            if existing["authority_gateway_id"] != authority_gateway_id:
                raise RoomConflictError("room_id already belongs to a different authority")
            return _room_from_row(existing, idempotent=True)
        active = conn.execute("SELECT COUNT(*) FROM hosted_rooms WHERE disbanded_at IS NULL").fetchone()[0]
        if int(active) >= MAX_ACTIVE_ROOMS:
            raise HostedRoomError("This host has too many active Group Chats. Delete one and try again.")
        conn.execute(
            f"""INSERT INTO hosted_rooms ({_ROOM_COLUMNS_WITH_BYTES})
                VALUES (?, ?, ?, ?, 1, 1, 0, 1, ?, ?, NULL)""",
            (room_id, name, members_json, authority_gateway_id, now, now))
        row = _reload(
            conn, """SELECT room_id, name, members_json, authority_gateway_id, authority_epoch, revision,
                created_at, updated_at FROM hosted_rooms WHERE room_id=?""", (room_id,),
            "created room could not be reloaded")
    return {**_room_from_row(row), "members": normalized_members}


def list_rooms(
    db_path: DbPath, *, include_disbanded: bool = False, limit: int = MAX_ROOM_LIST_LIMIT, offset: int = 0
) -> list[dict[str, Any]]:
    """Return one bounded read-only page ordered by most recent change."""
    limit = _bounded_limit(limit, MAX_ROOM_LIST_LIMIT)
    offset = _non_negative(offset, "offset")
    with closing(_read_connection(db_path)) as conn:
        rows = conn.execute(
            f"""SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE disbanded_at IS NULL OR ?
                ORDER BY updated_at DESC, room_id ASC LIMIT ? OFFSET ?""", (int(include_disbanded), limit, offset)
        ).fetchall()
    return [_room_from_row(row) for row in rows]


def rename_room(db_path: DbPath, *, room_id: Any, event_id: Any, name: Any, now: float | None = None) -> dict[str, Any]:
    """Rename a live room and append its replay event atomically."""
    room_id = _room_id(room_id)
    event_id = _event_id(event_id)
    name = _validate_room_name(name)
    now = _now(now)
    actor_json = _system_actor_json("room-control")
    payload_json = _payload_json({"name": name})
    with _transaction(db_path, immediate=True) as conn:
        room = _room_row(conn, _SELECT_ROOM_WITH_BYTES, (room_id,), room_id)
        if room["disbanded_at"] is not None:
            raise RoomNotFoundError("hosted room not found")
        existing = _load_event(conn, room_id, event_id)
        if existing is not None:
            if existing["kind"] != "room.renamed" or existing["payload_json"] != payload_json:
                raise EventConflictError("event_id already exists with different immutable content")
            return {**_room_from_row(room, idempotent=True), "event": _event_from_row(existing, idempotent=True)}
        seq = int(room["next_seq"])
        event_bytes = _prepare_event(conn, room, event_id, "room.renamed", actor_json, payload_json)
        # Rename updates the room row before inserting its event (order is load-bearing).
        conn.execute("""UPDATE hosted_rooms
                SET name=?, next_seq=?, event_bytes=event_bytes+?, revision=revision+1, updated_at=?
                WHERE room_id=?""", (name, seq + 1, event_bytes, now, room_id))
        conn.execute(_INSERT_EVENT, (
            room_id, seq, event_id, "room.renamed", actor_json, int(room["authority_epoch"]), payload_json, now))
        updated = conn.execute(_SELECT_ROOM, (room_id,)).fetchone()
        return {**_room_from_row(updated), "event": _event_from_row(_load_event(conn, room_id, event_id))}


def append_event(
    db_path: DbPath, *, room_id: Any, event_id: Any, kind: Any, actor: Any, payload: Any,
    authority_gateway_id: Any = None, authority_epoch: Any = None, now: float | None = None) -> dict[str, Any]:
    """Append one immutable event and allocate its per-room sequence atomically; repeating an ``event_id``
    with identical content returns the original, different content fails closed."""
    room_id = _room_id(room_id)
    event_id = _event_id(event_id)
    kind = _validate_event_kind(kind)
    normalized_actor, actor_json = _validate_actor(actor, kind=kind)
    # Every admitted actor kind is room-scoped, so authority fields are always required.
    authority_gateway_id = _actor_id(authority_gateway_id, "authority_gateway_id")
    if normalized_actor["kind"] == "gateway" and normalized_actor["id"] != authority_gateway_id:
        raise HostedRoomError("gateway actor.id must match authority_gateway_id")
    authority_epoch = _require_positive_int(authority_epoch, "authority_epoch")
    if not isinstance(payload, dict):
        raise HostedRoomError("payload must be an object")
    payload_json = _payload_json(payload)
    now = _now(now)
    with _transaction(db_path, immediate=True) as conn:
        existing = _load_event(conn, room_id, event_id)
        if existing is not None:
            if _event_content(existing) != (kind, actor_json, authority_epoch, payload_json):
                raise EventConflictError("event_id already exists with different content")
            return _event_from_row(existing, idempotent=True)
        room = _room_row(
            conn, """SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch
                FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL""", (room_id,), room_id)
        _require_authority(room, authority_gateway_id, authority_epoch, "stale hosted room authority")
        seq = int(room["next_seq"])
        event_bytes = _insert_event(
            conn, room, room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, now,
            allow_control=kind in _CONTROL_EVENT_KINDS)
        _fenced_update(conn, """UPDATE hosted_rooms SET next_seq=?, event_bytes=event_bytes+?, updated_at=?
                WHERE room_id=? AND next_seq=?""", (seq + 1, event_bytes, now, room_id, seq),
            RuntimeError("hosted room sequence advance lost its write fence"))
        row = _reload(
            conn, f"SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND seq=?", (room_id, seq),
            "appended event could not be reloaded")
    return {**_event_from_row(row), "actor": normalized_actor}


def _probe(path: Path, table: str, query: str, params: tuple[Any, ...], unavailable: str) -> bool:
    """Non-blocking existence probe: short timeout, no schema creation or migration."""
    if not path.is_file():
        return False
    try:
        with closing(sqlite3.connect(path, timeout=0.05)) as conn:
            table_row = conn.execute(
                f"SELECT 1 FROM sqlite_master WHERE type='table' AND name='{table}' LIMIT 1").fetchone()
            return table_row is not None and conn.execute(query, params).fetchone() is not None
    except sqlite3.Error as exc:
        raise RoomProbeUnavailableError(unavailable) from exc


def probe_hosted_room(db_path: DbPath, *, room_id: Any) -> bool:
    """Check room ownership without creating or migrating the shared store; runs on the synchronous
    prompt-admission path for older Desktop clients, so it fails fast under contention instead of blocking
    the WebSocket reader for SQLite's ten-second timeout."""
    return _probe(
        Path(db_path), "hosted_rooms", "SELECT 1 FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL LIMIT 1",
        (_room_id(room_id),), "hosted room ownership is temporarily unavailable")


def probe_peer_room_reservation(
    db_path: DbPath, *, room_id: Any, target_profile: Any, now: float | None = None) -> bool:
    """Check a peer reservation without creating or migrating shared state."""
    params = (_room_id(room_id), _actor_id(target_profile, "target_profile"), _now(now))
    return _probe(
        Path(db_path), "hosted_room_peer_reservations", _SELECT_LIVE_RESERVATION, params,
        "peer room ownership is temporarily unavailable")


def room_state(db_path: DbPath, *, room_id: Any, include_disbanded: bool = False) -> dict[str, Any]:
    """Return durable replay and authority state for one room."""
    room_id = _room_id(room_id)
    with _transaction(db_path) as conn:
        row = _room_row(
            conn,
            f"""SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE room_id=? AND (disbanded_at IS NULL
                OR ?)""", (room_id, int(include_disbanded)), room_id)
        claim_row = conn.execute(
            f"""SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=?
                AND kind='authority.claimed' AND authority_epoch=? ORDER BY seq DESC LIMIT 1""",
            (room_id, int(row["authority_epoch"]))).fetchone()
    return {**_room_from_row(row), **({"authority_claim": _event_from_row(claim_row)} if claim_row is not None else {})}


def request_room_stop(
    db_path: DbPath, *, room_id: Any, cancel_id: Any, expected_gateway_id: Any, expected_epoch: Any) -> dict[str, Any]:
    """Append an idempotent fence that supersedes earlier user turns."""
    cancel_id = _validate_identifier(cancel_id, label="cancel_id", max_chars=MAX_EVENT_ID_CHARS)
    return append_event(
        db_path, room_id=room_id, event_id=f"room-stop:{hashlib.sha256(cancel_id.encode()).hexdigest()[:32]}",
        kind="room.stop_requested", actor={"kind": "gateway", "id": expected_gateway_id},
        payload={"cancel_id": cancel_id}, authority_gateway_id=expected_gateway_id, authority_epoch=expected_epoch)


def claim_authority(
    db_path: DbPath, *, room_id: Any, expected_gateway_id: Any, expected_epoch: Any, new_gateway_id: Any, event_id: Any,
    now: float | None = None) -> dict[str, Any]:
    """Fence a verified authority transfer with a compare-and-swap epoch; does not decide *when* takeover is
    safe (a replicated driver calls it only after its lease/quorum policy established that the previous
    owner can no longer commit)."""
    room_id = _room_id(room_id)
    expected_gateway_id = _actor_id(expected_gateway_id, "expected_gateway_id")
    new_gateway_id = _actor_id(new_gateway_id, "new_gateway_id")
    event_id = _event_id(event_id)
    _require_positive_int(expected_epoch, "expected_epoch")
    now = _now(now)
    target_epoch = expected_epoch + 1
    claim_actor_json = _system_actor_json("authority-control")
    claim_payload_json = _claim_payload_json(expected_gateway_id, new_gateway_id, target_epoch)
    with _transaction(db_path, immediate=True) as conn:
        row = _room_row(
            conn, """SELECT authority_gateway_id, authority_epoch, next_seq, event_bytes
                FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL""", (room_id,), room_id)
        existing_event = _load_event(conn, room_id, event_id)
        idempotent = existing_event is not None
        if idempotent:
            if _event_content(existing_event) != (
                "authority.claimed", claim_actor_json, target_epoch, claim_payload_json):
                raise EventConflictError("event_id already exists with different content")
            if str(row["authority_gateway_id"]) != new_gateway_id or int(row["authority_epoch"]) != target_epoch:
                raise AuthoritySupersededError("authority claim succeeded but was later superseded")
        else:
            _require_authority(row, expected_gateway_id, expected_epoch, "hosted room authority changed")
            # Insert the claim event, then CAS the room's authority behind the epoch fence.
            claim_bytes = _insert_event(
                conn, row, room_id, int(row["next_seq"]), event_id, "authority.claimed", claim_actor_json, target_epoch,
                claim_payload_json, now, allow_control=True)
            _fenced_update(conn, """UPDATE hosted_rooms SET authority_gateway_id=?,
            authority_epoch=authority_epoch+1, next_seq=next_seq+1,
            event_bytes=event_bytes+?, revision=revision+1, updated_at=? WHERE room_id=?
            AND disbanded_at IS NULL AND authority_gateway_id=? AND authority_epoch=?""",
                (new_gateway_id, claim_bytes, now, room_id, expected_gateway_id, expected_epoch),
                AuthorityConflictError("hosted room authority changed"))
            existing_event = _load_event(conn, room_id, event_id)
        state_row = _reload(
            conn, """SELECT room_id, name, members_json, authority_gateway_id, authority_epoch, next_seq,
                revision, created_at, updated_at FROM hosted_rooms WHERE room_id=?""", (room_id,),
            "claimed room could not be reloaded")
    if existing_event is None:  # pragma: no cover - both claim paths set it
        raise RuntimeError("authority claim event could not be reloaded")
    return {
        **_room_from_row(state_row, idempotent=idempotent),
        "claim_event": _event_from_row(existing_event, idempotent=idempotent)}


def _disband_replay(conn: sqlite3.Connection, room_id: str, room: sqlite3.Row | None) -> dict[str, Any] | None:
    """Idempotent replay for a retired or already-disbanded room; None when the room is live."""
    if room is None:
        retired = conn.execute("SELECT retired_at FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)).fetchone()
        if retired is None:
            raise RoomNotFoundError("hosted room not found")
        return {
            "room_id": room_id, "disbanded_at": float(retired["retired_at"]), "idempotent": True,
            "history_expired": True}
    if room["disbanded_at"] is None:
        return None
    conn.execute(_INSERT_RETIRED, (room_id, float(room["disbanded_at"])))
    event = _load_event(conn, room_id, "system:room-disbanded")
    return {
        "room_id": room_id, "disbanded_at": float(room["disbanded_at"]), "idempotent": True,
        **({"event": _event_from_row(event, idempotent=True)} if event is not None else {})}


def disband_room(
    db_path: DbPath, *, room_id: Any, expected_gateway_id: Any, expected_epoch: Any, now: float | None = None
) -> dict[str, Any]:
    """Tombstone a room id permanently and idempotently."""
    room_id = _room_id(room_id)
    expected_gateway_id = _actor_id(expected_gateway_id, "expected_gateway_id")
    _require_positive_int(expected_epoch, "expected_epoch")
    now = _now(now)
    with _transaction(db_path, immediate=True) as conn:
        room = conn.execute("""SELECT authority_gateway_id, authority_epoch, next_seq, event_bytes, disbanded_at
                FROM hosted_rooms WHERE room_id=?""", (room_id,)).fetchone()
        if (replay := _disband_replay(conn, room_id, room)) is not None:
            return replay
        _require_authority(room, expected_gateway_id, expected_epoch, "stale hosted room authority")
        disband_bytes = _insert_event(
            conn, room, room_id, int(room["next_seq"]), "system:room-disbanded", "room.disbanded",
            _system_actor_json("room-control"), int(room["authority_epoch"]), _payload_json({"room_id": room_id}), now,
            allow_control=True)
        _fenced_update(conn, """UPDATE hosted_rooms
                SET disbanded_at=?, updated_at=?, revision=revision+1,
                    next_seq=next_seq+1, event_bytes=event_bytes+?
                WHERE room_id=? AND disbanded_at IS NULL AND authority_gateway_id=?
                AND authority_epoch=?""",
            (now, now, disband_bytes, room_id, expected_gateway_id, expected_epoch),
            RoomConflictError("hosted room disband lost its fence"))
        conn.execute(_INSERT_RETIRED, (room_id, now))
        event = _reload(
            conn, _SELECT_EVENT, (room_id, "system:room-disbanded"), "room disband event could not be reloaded")
        _prune_disbanded_rooms_locked(conn, now=now, max_gateway_event_bytes=MAX_GATEWAY_EVENT_BYTES)
    return {"room_id": room_id, "disbanded_at": now, "idempotent": False, "event": _event_from_row(event)}


def read_events(
    db_path: DbPath, *, room_id: Any, since_seq: Any = 0, limit: Any = 100, include_disbanded: bool = False
) -> dict[str, Any]:
    """Read a monotonic room-log delta after ``since_seq``."""
    room_id = _room_id(room_id)
    since_seq = _non_negative(since_seq, "since_seq")
    limit = _bounded_limit(limit, MAX_LOG_LIMIT)
    with _transaction(db_path) as conn:
        room = _room_row(
            conn, """SELECT next_seq, authority_gateway_id, authority_epoch FROM hosted_rooms
                WHERE room_id=? AND (disbanded_at IS NULL OR ?)""", (room_id, int(include_disbanded)), room_id)
        latest_seq = int(room["next_seq"]) - 1
        authority = {"gateway_id": str(room["authority_gateway_id"]), "epoch": int(room["authority_epoch"])}
        if since_seq > latest_seq:
            raise HostedRoomError("since_seq is ahead of the hosted room log")
        rows = conn.execute(
            f"""WITH candidates AS (
                   SELECT {_EVENT_COLUMNS},
                          SUM(
                              LENGTH(CAST(event_id AS BLOB)) +
                              LENGTH(CAST(kind AS BLOB)) +
                              LENGTH(CAST(actor_json AS BLOB)) +
                              LENGTH(CAST(payload_json AS BLOB))
                          ) OVER (ORDER BY seq ASC) AS cumulative_bytes
                     FROM hosted_room_events
                    WHERE room_id=? AND seq>?
                    ORDER BY seq ASC LIMIT ?
               )
               SELECT {_EVENT_COLUMNS}
                 FROM candidates
                WHERE cumulative_bytes<=?
                ORDER BY seq ASC""", (room_id, since_seq, limit, MAX_LOG_PAGE_BYTES)).fetchall()
    events = [_event_from_row(row) for row in rows]
    def build_page(page_events: list[dict[str, Any]]) -> dict[str, Any]:
        cursor = page_events[-1]["seq"] if page_events else since_seq
        return {"events": page_events, "cursor": cursor, "latest_seq": latest_seq, "has_more": cursor < latest_seq,
                "authority": authority}
    def fits(page_events: list[dict[str, Any]]) -> bool:
        page_json = json.dumps(build_page(page_events), ensure_ascii=False, separators=(",", ":"))
        return utf8_len(page_json) <= MAX_LOG_PAGE_BYTES
    if events and not fits(events):
        # Binary-search the largest prefix whose serialized page fits the budget.
        low, high = 1, len(events)
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if fits(events[:middle]) else (low, middle - 1)
        events = events[:low]
        if not fits(events):
            raise HostedRoomError("hosted room event exceeds replay page limit")
    return build_page(events)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Iterator  # noqa: F401,E402
from typing import NoReturn  # noqa: F401,E402
from contextlib import contextmanager  # noqa: F401,E402
import time  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
