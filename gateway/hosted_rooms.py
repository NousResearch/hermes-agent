"""Durable state for gateway-hosted Bot Mode rooms.

This module owns only hosted-room identity and its append-only event log. It
does not deliver events, lease relay work, or run agent turns; those concerns
belong to the relay and the future hosted-room driver. Keeping that boundary
explicit lets the room log compose with a durable relay without creating a
second transport queue.

The caller supplies the database path so tests and alternate gateway layouts
can isolate state. Production handlers use the gateway's root ``state.db``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NoReturn


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
# Leave substantial headroom below the pre-update state.db snapshot ceiling.
# Event accounting does not include SQLite indexes or repeated room ids, so the
# logical budget must stay well below the physical-file limit.
MAX_GATEWAY_EVENT_BYTES = 16 * 1024 * 1024
CONTROL_EVENT_COUNT_RESERVE = 64
CONTROL_EVENT_BYTE_RESERVE = 10 * 1024 * 1024
STOP_EVENT_COUNT_RESERVE = 16
STOP_EVENT_BYTE_RESERVE = 512 * 1024
# A driver task can publish a visible member event plus one terminal event.
# Thirty-two event slots therefore cover the atomic admission ceiling of
# sixteen unpublished task outcomes, while leaving control headroom for Stop
# and authority.lost.
TERMINAL_RECOVERY_COUNT_RESERVE = 32
TERMINAL_RECOVERY_BYTE_RESERVE = TERMINAL_RECOVERY_COUNT_RESERVE * (
    MAX_EVENT_JSON_BYTES + 4096
)
DEMOTION_CONTROL_EVENT_COUNT_RESERVE = 2
MAX_DEMOTION_CONTROL_EVENT_BYTES = MAX_EVENT_JSON_BYTES + 4096
DEMOTION_CONTROL_EVENT_BYTE_RESERVE = (
    DEMOTION_CONTROL_EVENT_COUNT_RESERVE * MAX_DEMOTION_CONTROL_EVENT_BYTES
)
MAX_TERMINAL_PUBLICATION_EVENTS = 2
MAX_TERMINAL_PUBLICATION_BYTES = 2 * (MAX_EVENT_JSON_BYTES + 4096)
if CONTROL_EVENT_COUNT_RESERVE < (
    STOP_EVENT_COUNT_RESERVE
    + TERMINAL_RECOVERY_COUNT_RESERVE
    + DEMOTION_CONTROL_EVENT_COUNT_RESERVE
):
    raise RuntimeError("hosted-room count reserves cannot close a demotion")
if CONTROL_EVENT_BYTE_RESERVE < (
    STOP_EVENT_BYTE_RESERVE
    + TERMINAL_RECOVERY_BYTE_RESERVE
    + DEMOTION_CONTROL_EVENT_BYTE_RESERVE
):
    raise RuntimeError("hosted-room byte reserves cannot close a demotion")
_JOURNAL_MODE_LOCK_RETRIES = 8

_DISCUSSION_LIABILITY_PREFIX = "\x00discussion:"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_EVENT_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ROOM_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "name",
    "members_json",
    "authority_gateway_id",
    "authority_epoch",
    "next_seq",
    "event_bytes",
    "revision",
    "created_at",
    "updated_at",
    "disbanded_at",
})
_EVENT_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "seq",
    "event_id",
    "kind",
    "actor_json",
    "authority_epoch",
    "payload_json",
    "created_at",
})
_RETIRED_ROOM_SCHEMA_COLUMNS = frozenset({"room_id", "retired_at"})

_EVENT_KINDS_BY_ACTOR = {
    "user": frozenset({"message.user"}),
    "member": frozenset({"message.member"}),
    "gateway": frozenset({
        "member.unavailable",
        "room.activity",
        "room.stop_requested",
        "turn.deferred",
        "turn.reassigned",
        "turn.cancelled",
        "turn.failed",
        "turn.settled",
        "turn.started",
    }),
    "system": frozenset({
        "authority.claimed",
        "authority.lost",
        "room.created",
        "room.disbanded",
        "room.members_changed",
        "room.renamed",
    }),
}
_CRITICAL_CONTROL_EVENT_KINDS = frozenset({
    "authority.claimed",
    "authority.lost",
    "room.disbanded",
})
_TERMINAL_COMPLETION_EVENT_KINDS = frozenset({
    "turn.settled",
    "turn.failed",
    "turn.cancelled",
    "turn.deferred",
})
_FINAL_TERMINAL_EVENT_KINDS = frozenset({
    "turn.settled",
    "turn.failed",
    "turn.cancelled",
})
_ACTOR_FIELDS = frozenset({"kind", "id", "display_name", "profile", "connection_id"})


class HostedRoomError(ValueError):
    """Base class for invalid or conflicting hosted-room operations."""


class RoomNotFoundError(HostedRoomError):
    """Raised when a room does not exist or has been disbanded."""


class RoomHistoryExpiredError(RoomNotFoundError):
    """Raised when a retired room remains reserved after history compaction."""

    reason = "room_history_expired"


class RoomConflictError(HostedRoomError):
    """Raised when an idempotency key is reused for different room state."""


class RoomProbeUnavailableError(HostedRoomError):
    """Raised when a non-blocking ownership probe cannot read the room store."""


class EventConflictError(HostedRoomError):
    """Raised when an event id is reused with different immutable content."""


class AuthorityConflictError(HostedRoomError):
    """Raised when a stale room authority attempts to mutate hosted state."""

    reason = "authority_conflict"


class RoomAdmissionBlockedError(HostedRoomError):
    """Raised when the current authority is fenced against new user work."""

    reason = "room_admissions_blocked"


class AuthoritySupersededError(AuthorityConflictError):
    """Raised when a successful authority claim was later superseded."""


def default_db_path() -> Path:
    """Return the gateway-wide state database for the active install."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    root = home.parent.parent if home.parent.name == "profiles" else home
    return root / "state.db"


def local_authority_gateway_id() -> str:
    """Return the stable server-owned identity for hosted-room authority."""
    from hermes_cli.install_identity import get_install_id

    install_id = get_install_id()
    if not install_id:
        raise HostedRoomError("stable gateway install identity is unavailable")
    return _validate_identifier(
        f"install:{install_id}",
        label="authority_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )


def _canonical_json(value: Any, *, label: str, max_bytes: int) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise HostedRoomError(f"{label} must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise HostedRoomError(f"{label} is too large")
    return encoded


def _validate_identifier(value: Any, *, label: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise HostedRoomError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > max_chars or not _IDENTIFIER_RE.fullmatch(value):
        raise HostedRoomError(f"invalid {label}")
    return value


def user_event_id(client_event_id: Any) -> str:
    """Map a client retry key into the server-owned user-event namespace."""
    normalized = _validate_identifier(
        client_event_id,
        label="event_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"user:{digest}"


def _validate_room_name(value: Any) -> str:
    if not isinstance(value, str):
        raise HostedRoomError("name must be a string")
    value = value.strip()
    if not value or len(value) > MAX_ROOM_NAME_CHARS:
        raise HostedRoomError("invalid room name")
    return value


def _validate_members(value: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list):
        raise HostedRoomError("members must be a list")
    if len(value) > MAX_MEMBERS:
        raise HostedRoomError("too many room members")
    members: list[dict[str, Any]] = []
    for member in value:
        if not isinstance(member, dict):
            raise HostedRoomError("each room member must be an object")
        members.append(dict(member))
    encoded = _canonical_json(
        members,
        label="members",
        max_bytes=MAX_MEMBERS_JSON_BYTES,
    )
    return members, encoded


def _validate_event_kind(value: Any) -> str:
    if not isinstance(value, str):
        raise HostedRoomError("kind must be a string")
    value = value.strip()
    if (
        not value
        or len(value) > MAX_EVENT_KIND_CHARS
        or not _EVENT_KIND_RE.fullmatch(value)
    ):
        raise HostedRoomError("invalid event kind")
    return value


def _optional_actor_field(actor: dict[str, Any], field: str, max_chars: int) -> str:
    value = actor.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HostedRoomError(f"actor.{field} must be a string")
    value = value.strip()
    if len(value) > max_chars:
        raise HostedRoomError(f"actor.{field} is too long")
    return value


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

    actor_id = _validate_identifier(
        value.get("id"),
        label="actor.id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    actor = {"kind": actor_kind, "id": actor_id}
    for field, max_chars in (
        ("display_name", MAX_ACTOR_LABEL_CHARS),
        ("profile", MAX_ACTOR_ID_CHARS),
        ("connection_id", MAX_ACTOR_ID_CHARS),
    ):
        field_value = _optional_actor_field(value, field, max_chars)
        if field_value:
            actor[field] = field_value
    encoded = _canonical_json(
        actor,
        label="actor",
        max_bytes=4 * 1024,
    )
    return actor, encoded


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
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
        )"""
    )
    conn.execute(
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
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_retired_ids (
            room_id TEXT PRIMARY KEY,
            retired_at REAL NOT NULL
        )"""
    )
    room_columns = {row[1] for row in conn.execute("PRAGMA table_info(hosted_rooms)")}
    if "authority_gateway_id" not in room_columns:
        conn.execute(
            "ALTER TABLE hosted_rooms "
            "ADD COLUMN authority_gateway_id TEXT NOT NULL DEFAULT 'legacy'"
        )
    if "authority_epoch" not in room_columns:
        conn.execute(
            "ALTER TABLE hosted_rooms "
            "ADD COLUMN authority_epoch INTEGER NOT NULL DEFAULT 1"
        )
    backfill_event_bytes = "event_bytes" not in room_columns
    if backfill_event_bytes:
        conn.execute(
            "ALTER TABLE hosted_rooms ADD COLUMN event_bytes INTEGER NOT NULL DEFAULT 0"
        )

    event_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_events)")
    }
    if "actor_json" not in event_columns:
        # Draft builds before the actor contract carried no identity. Preserve
        # their inert replay rows explicitly as legacy system events rather
        # than guessing a user or Bot author.
        legacy_actor = _canonical_json(
            {"kind": "system", "id": "legacy"},
            label="actor",
            max_bytes=4 * 1024,
        )
        escaped_actor = legacy_actor.replace("'", "''")
        conn.execute(
            "ALTER TABLE hosted_room_events "
            f"ADD COLUMN actor_json TEXT NOT NULL DEFAULT '{escaped_actor}'"
        )
    if "authority_epoch" not in event_columns:
        conn.execute(
            "ALTER TABLE hosted_room_events ADD COLUMN authority_epoch INTEGER"
        )
    if backfill_event_bytes:
        conn.execute(
            """UPDATE hosted_rooms
                  SET event_bytes=COALESCE((
                      SELECT SUM(
                          length(CAST(event_id AS BLOB)) +
                          length(CAST(kind AS BLOB)) +
                          length(CAST(actor_json AS BLOB)) +
                          length(CAST(payload_json AS BLOB))
                      )
                      FROM hosted_room_events
                      WHERE hosted_room_events.room_id=hosted_rooms.room_id
                  ), 0)"""
        )
    # Old schemas kept the final identity tombstone in hosted_rooms itself.
    # Copy those identities before bounded history pruning can remove their
    # heavier room/event payloads. This compact registry is intentionally
    # permanent: a stale coordinate must never name a different Group Chat.
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at)
           SELECT room_id, disbanded_at FROM hosted_rooms
            WHERE disbanded_at IS NOT NULL"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_events_cursor
           ON hosted_room_events(room_id, seq)"""
    )
    if not _schema_is_current(conn):
        raise HostedRoomError("hosted room schema migration did not complete")


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    room_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_rooms)")
    )
    event_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_events)")
    )
    retired_room_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_retired_ids)")
    )
    if not _ROOM_SCHEMA_COLUMNS.issubset(room_columns):
        return False
    if not _EVENT_SCHEMA_COLUMNS.issubset(event_columns):
        return False
    if not _RETIRED_ROOM_SCHEMA_COLUMNS.issubset(retired_room_columns):
        return False
    index = conn.execute(
        """SELECT 1 FROM sqlite_master
           WHERE type='index' AND name='idx_hosted_room_events_cursor'"""
    ).fetchone()
    return index is not None


def _connect(db_path: Path | str) -> sqlite3.Connection:
    from hermes_state import apply_wal_with_fallback

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        for attempt in range(_JOURNAL_MODE_LOCK_RETRIES):
            try:
                apply_wal_with_fallback(conn, db_label="state.db (hosted_rooms)")
                break
            except sqlite3.OperationalError as exc:
                if (
                    str(exc).lower() != "database is locked"
                    or attempt + 1 == _JOURNAL_MODE_LOCK_RETRIES
                ):
                    raise
                # SQLite's journal-mode pragma may not honor the connection's
                # busy timeout while another first opener initializes the DB,
                # especially on Windows. Retry only that transient lock class.
                time.sleep(0.01 * (2**attempt))
        conn.execute("PRAGMA foreign_keys=ON")
        if _schema_is_current(conn):
            return conn
        # Multiple profile gateways share this root database. Serialize every
        # draft-schema transition in SQLite itself so a crash rolls back the
        # whole DDL/data migration and another process can safely retry it.
        conn.execute("BEGIN IMMEDIATE")
        _initialize_schema(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


def _read_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open the room store without steady-state journal or migration writes."""

    path = Path(db_path)
    if not path.is_file():
        initialized = _connect(path)
        initialized.close()
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if _schema_is_current(conn):
        return conn
    conn.close()
    migrated = _connect(path)
    migrated.close()
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _transaction(
    db_path: Path | str, *, immediate: bool = False
) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _raise_room_not_found(conn: sqlite3.Connection, room_id: str) -> NoReturn:
    retained = conn.execute(
        "SELECT 1 FROM hosted_rooms WHERE room_id=?",
        (room_id,),
    ).fetchone()
    if retained is not None:
        # A retained disband tombstone still has replayable history. The
        # caller simply did not opt into reading disbanded rooms.
        raise RoomNotFoundError("hosted room not found")
    retired = conn.execute(
        "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?",
        (room_id,),
    ).fetchone()
    if retired is not None:
        raise RoomHistoryExpiredError(
            "Group Chat history expired; room_id remains permanently retired"
        )
    raise RoomNotFoundError("hosted room not found")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def _room_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    room = {
        "room_id": row["room_id"],
        "name": row["name"],
        "members": json.loads(row["members_json"]),
        "authority_gateway_id": row["authority_gateway_id"],
        "authority_epoch": int(row["authority_epoch"]),
        "revision": int(row["revision"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "idempotent": idempotent,
    }
    if "disbanded_at" in row.keys() and row["disbanded_at"] is not None:
        room["disbanded_at"] = float(row["disbanded_at"])
    if "next_seq" in row.keys():
        room["latest_seq"] = int(row["next_seq"]) - 1
    return room


def _event_storage_bytes(
    *, event_id: str, kind: str, actor_json: str, payload_json: str
) -> int:
    return len((event_id + kind + actor_json + payload_json).encode("utf-8"))


def _stop_event_id(cancel_id: str) -> str:
    digest = hashlib.sha256(cancel_id.encode()).hexdigest()[:32]
    return f"room-stop:{digest}"


def _remaining_demotion_control_events(
    conn: sqlite3.Connection,
    *,
    room_id: str,
) -> int:
    """Reserve Stop plus authority.lost, or only authority.lost after Stop."""

    if not _table_exists(
        conn, "hosted_room_driver_demotion_intents"
    ) or not _table_exists(conn, "hosted_room_driver_admission_barriers"):
        return DEMOTION_CONTROL_EVENT_COUNT_RESERVE
    intent = conn.execute(
        """SELECT intent.cancel_id, intent.gateway_id, intent.authority_epoch
             FROM hosted_room_driver_demotion_intents AS intent
             JOIN hosted_rooms AS room ON room.room_id=intent.room_id
            WHERE intent.room_id=? AND room.disbanded_at IS NULL
              AND room.authority_gateway_id=intent.gateway_id
              AND room.authority_epoch=intent.authority_epoch""",
        (room_id,),
    ).fetchone()
    if intent is None:
        return DEMOTION_CONTROL_EVENT_COUNT_RESERVE
    cancel_id = str(intent["cancel_id"])
    stop = conn.execute(
        """SELECT kind, actor_json, authority_epoch, payload_json
             FROM hosted_room_events
            WHERE room_id=? AND event_id=?""",
        (room_id, _stop_event_id(cancel_id)),
    ).fetchone()
    if stop is None:
        return DEMOTION_CONTROL_EVENT_COUNT_RESERVE
    expected_actor_json = _canonical_json(
        {"kind": "gateway", "id": str(intent["gateway_id"])},
        label="actor",
        max_bytes=4 * 1024,
    )
    expected_payload_json = _canonical_json(
        {"cancel_id": cancel_id},
        label="payload",
        max_bytes=MAX_EVENT_JSON_BYTES,
    )
    try:
        stop_epoch = int(stop["authority_epoch"])
    except (TypeError, ValueError):
        return DEMOTION_CONTROL_EVENT_COUNT_RESERVE
    if (
        str(stop["kind"]) != "room.stop_requested"
        or str(stop["actor_json"]) != expected_actor_json
        or stop_epoch != int(intent["authority_epoch"])
        or str(stop["payload_json"]) != expected_payload_json
    ):
        return DEMOTION_CONTROL_EVENT_COUNT_RESERVE
    return DEMOTION_CONTROL_EVENT_COUNT_RESERVE - 1


def _discussion_liability_key(room_id: str, thread_id: str) -> tuple[str, str]:
    return room_id, f"{_DISCUSSION_LIABILITY_PREFIX}{thread_id}"


def _discussion_source_state(
    conn: sqlite3.Connection,
) -> tuple[
    dict[tuple[str, str], tuple[int, str]],
    dict[str, int],
    set[tuple[str, str, str]],
]:
    """Return latest sources, Stop fences, and durably closed discussions."""

    latest_by_thread: dict[tuple[str, str], tuple[int, str]] = {}
    stopped_through: dict[str, int] = {}
    completed: set[tuple[str, str, str]] = set()
    rows = conn.execute(
        """SELECT event.room_id, event.seq, event.event_id, event.kind,
                  event.payload_json
             FROM hosted_room_events AS event
             JOIN hosted_rooms AS room ON room.room_id=event.room_id
            WHERE room.disbanded_at IS NULL
              AND event.kind IN (
                  'message.user', 'room.activity', 'room.stop_requested'
              )
            ORDER BY event.room_id, event.seq"""
    ).fetchall()
    for row in rows:
        room_id = str(row["room_id"])
        seq = int(row["seq"])
        kind = str(row["kind"])
        if kind == "room.stop_requested":
            stopped_through[room_id] = max(stopped_through.get(room_id, 0), seq)
            continue
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if kind == "room.activity":
            discussion_event_id = payload.get("discussion_event_id")
            thread_id = payload.get("thread_id")
            if (
                payload.get("status") in {"settled", "bounded"}
                and isinstance(discussion_event_id, str)
                and discussion_event_id
                and isinstance(thread_id, str)
                and thread_id
            ):
                completed.add((room_id, discussion_event_id, thread_id))
            continue
        thread_id = payload.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            latest_by_thread[(room_id, thread_id)] = (
                seq,
                str(row["event_id"]),
            )
    return latest_by_thread, stopped_through, completed


def _pending_discussion_sources(
    conn: sqlite3.Connection,
    *,
    consumed_task_sources: set[tuple[str, int, str]],
    source_state: tuple[
        dict[tuple[str, str], tuple[int, str]],
        dict[str, int],
        set[tuple[str, str, str]],
    ]
    | None = None,
) -> dict[tuple[str, str], int]:
    latest_by_thread, stopped_through, completed = (
        source_state if source_state is not None else _discussion_source_state(conn)
    )

    pending: dict[tuple[str, str], int] = {}
    for (room_id, thread_id), (seq, event_id) in latest_by_thread.items():
        if seq <= stopped_through.get(room_id, 0):
            continue
        if (room_id, event_id, thread_id) in completed:
            continue
        if (room_id, seq, thread_id) in consumed_task_sources:
            continue
        pending[_discussion_liability_key(room_id, thread_id)] = seq
    return pending


def _terminal_event_matches_driver_task(
    task: sqlite3.Row,
    *,
    kind: str,
    payload: dict[str, Any],
) -> bool:
    """Fail closed unless a room-log terminal exactly identifies its driver task."""

    task_id = str(task["task_id"])
    thread_id = str(task["thread_id"])
    turn_id = str(task["turn_id"])
    source_event_id = task["source_event_id"]
    if (
        source_event_id is None
        or str(task["source_kind"]) != "message.user"
        or payload.get("task_id") != task_id
        or payload.get("thread_id") != thread_id
        or payload.get("turn_id") != turn_id
        or payload.get("discussion_event_id") != str(source_event_id)
    ):
        return False

    try:
        source_payload = json.loads(str(task["source_payload_json"]))
        task_payload = json.loads(str(task["task_payload_json"]))
        members = json.loads(str(task["members_json"]))
    except (TypeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(source_payload, dict)
        or source_payload.get("thread_id") != thread_id
        or not isinstance(task_payload, dict)
        or not isinstance(members, list)
    ):
        return False

    member_id = payload.get("member_id")
    member_index = payload.get("member_index")
    round_index = payload.get("round_index")
    seen_through_seq = payload.get("seen_through_seq")
    source_event_seq = int(task["source_event_seq"])
    if (
        not isinstance(member_id, str)
        or not member_id
        or isinstance(member_index, bool)
        or not isinstance(member_index, int)
        or member_index < 0
        or isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index < 0
        or isinstance(seen_through_seq, bool)
        or not isinstance(seen_through_seq, int)
        or seen_through_seq < source_event_seq
    ):
        return False

    target_profile = task_payload.get("target_profile")
    if not isinstance(target_profile, str) or not target_profile:
        return False
    expected_member_ids: set[str] = set()
    for member in members:
        if not isinstance(member, dict):
            continue
        profile = member.get("profile")
        legacy_id = member.get("id")
        if profile != target_profile and not (
            profile is None and legacy_id == target_profile
        ):
            continue
        candidate = member.get("member_id", legacy_id)
        if isinstance(candidate, str) and candidate:
            expected_member_ids.add(candidate)
    if member_id not in expected_member_ids:
        return False

    status = str(task["status"])
    if kind == "turn.deferred":
        generation = payload.get("execution_generation")
        return (
            status == "deferred"
            and isinstance(generation, int)
            and not isinstance(generation, bool)
            and generation == int(task["execution_generation"])
            and generation > 0
        )
    allowed_final_kinds = {
        "settled": frozenset({"turn.settled", "turn.cancelled"}),
        "failed": frozenset({"turn.failed", "turn.cancelled"}),
        "cancelled": frozenset({"turn.cancelled"}),
    }
    return kind in allowed_final_kinds.get(status, frozenset())


def _published_terminal_task_outcomes(
    conn: sqlite3.Connection,
) -> tuple[set[tuple[str, str]], set[tuple[str, str, int]]]:
    """Return only terminal outcomes correlated to durable driver task state."""

    if not _table_exists(conn, "hosted_room_driver_tasks"):
        return set(), set()
    task_rows = conn.execute(
        """SELECT task.room_id, task.task_id, task.thread_id, task.turn_id,
                  task.source_event_seq, task.payload_json AS task_payload_json,
                  task.status, task.execution_generation, room.members_json,
                  source.event_id AS source_event_id,
                  source.kind AS source_kind,
                  source.payload_json AS source_payload_json
             FROM hosted_room_driver_tasks AS task
             JOIN hosted_rooms AS room ON room.room_id=task.room_id
             LEFT JOIN hosted_room_events AS source
               ON source.room_id=task.room_id
              AND source.seq=task.source_event_seq
            WHERE room.disbanded_at IS NULL"""
    ).fetchall()
    tasks = {
        (str(row["room_id"]), str(row["task_id"])): row for row in task_rows
    }
    final: set[tuple[str, str]] = set()
    deferred: set[tuple[str, str, int]] = set()
    for row in conn.execute(
        """SELECT room_id, kind, payload_json FROM hosted_room_events
           WHERE kind IN (
               'turn.settled', 'turn.failed', 'turn.cancelled', 'turn.deferred'
           )"""
    ).fetchall():
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        room_id = str(row["room_id"])
        task = tasks.get((room_id, task_id))
        kind = str(row["kind"])
        if task is None or not _terminal_event_matches_driver_task(
            task,
            kind=kind,
            payload=payload,
        ):
            continue
        if kind == "turn.deferred":
            deferred.add((room_id, task_id, int(payload["execution_generation"])))
        else:
            final.add((room_id, task_id))
    return final, deferred


def _correlated_terminal_task_ids(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    events: list[dict[str, Any]],
) -> frozenset[str]:
    """Return pending final events that exactly identify durable driver tasks."""

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    task_ids: set[str] = set()
    for event in events:
        kind = event.get("kind")
        payload = event.get("payload")
        if payload is None:
            try:
                payload = json.loads(str(event.get("payload_json", "")))
            except json.JSONDecodeError:
                continue
        if kind not in _FINAL_TERMINAL_EVENT_KINDS or not isinstance(payload, dict):
            continue
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        candidates.append((task_id, str(kind), payload))
        task_ids.add(task_id)
    if not candidates or not _table_exists(conn, "hosted_room_driver_tasks"):
        return frozenset()

    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        f"""SELECT task.room_id, task.task_id, task.thread_id, task.turn_id,
                   task.source_event_seq, task.payload_json AS task_payload_json,
                   task.status, task.execution_generation, room.members_json,
                   source.event_id AS source_event_id,
                   source.kind AS source_kind,
                   source.payload_json AS source_payload_json
              FROM hosted_room_driver_tasks AS task
              JOIN hosted_rooms AS room ON room.room_id=task.room_id
              LEFT JOIN hosted_room_events AS source
                ON source.room_id=task.room_id
               AND source.seq=task.source_event_seq
             WHERE task.room_id=? AND room.disbanded_at IS NULL
               AND task.task_id IN ({placeholders})""",
        (room_id, *sorted(task_ids)),
    ).fetchall()
    tasks = {str(row["task_id"]): row for row in rows}
    return frozenset(
        task_id
        for task_id, kind, payload in candidates
        if (task := tasks.get(task_id)) is not None
        and _terminal_event_matches_driver_task(task, kind=kind, payload=payload)
    )


def _terminal_task_discussion_liability_keys(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    task_ids: frozenset[str],
) -> frozenset[tuple[str, str]]:
    """Transfer final task reserves to their still-open discussions."""

    if not task_ids or not _table_exists(conn, "hosted_room_driver_tasks"):
        return frozenset()
    pending = _pending_discussion_sources(conn, consumed_task_sources=set())
    placeholders = ",".join("?" for _ in task_ids)
    keys: set[tuple[str, str]] = set()
    for row in conn.execute(
        f"""SELECT thread_id, source_event_seq
              FROM hosted_room_driver_tasks
             WHERE room_id=? AND task_id IN ({placeholders})""",
        (room_id, *sorted(task_ids)),
    ).fetchall():
        key = _discussion_liability_key(room_id, str(row["thread_id"]))
        if pending.get(key) == int(row["source_event_seq"]):
            keys.add(key)
    return frozenset(keys)


def _closing_discussion_liability_keys(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    events: list[dict[str, Any]],
) -> frozenset[tuple[str, str]]:
    """Return exact open-discussion reserves closed by room.activity events."""

    pending = _pending_discussion_sources(conn, consumed_task_sources=set())
    keys: set[tuple[str, str]] = set()
    for event in events:
        if event.get("kind") != "room.activity":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload_json = event.get("payload_json")
            if not isinstance(payload_json, str):
                continue
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict) or payload.get("status") not in {
            "settled",
            "bounded",
        }:
            continue
        thread_id = payload.get("thread_id")
        discussion_event_id = payload.get("discussion_event_id")
        if not isinstance(thread_id, str) or not thread_id:
            continue
        if not isinstance(discussion_event_id, str) or not discussion_event_id:
            continue
        source = conn.execute(
            """SELECT seq FROM hosted_room_events
               WHERE room_id=? AND event_id=? AND kind='message.user'""",
            (room_id, discussion_event_id),
        ).fetchone()
        key = _discussion_liability_key(room_id, thread_id)
        if source is not None and pending.get(key) == int(source["seq"]):
            keys.add(key)
    return frozenset(keys)


def _pending_discussion_liability_key_for_source(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    source_event_seq: int,
    thread_id: str,
) -> tuple[str, str] | None:
    key = _discussion_liability_key(room_id, thread_id)
    pending = _pending_discussion_sources(conn, consumed_task_sources=set())
    return key if pending.get(key) == source_event_seq else None


def _terminal_publication_liabilities(
    conn: sqlite3.Connection,
) -> set[tuple[str, str]]:
    published, published_deferred = _published_terminal_task_outcomes(conn)
    source_state = _discussion_source_state(conn)
    completed = source_state[2]
    liabilities: set[tuple[str, str]] = set()
    consumed_task_sources: set[tuple[str, int, str]] = set()
    if _table_exists(conn, "hosted_room_driver_tasks"):
        for row in conn.execute(
            """SELECT task.room_id, task.task_id, task.thread_id,
                       task.source_event_seq, task.status,
                       task.execution_generation, source.event_id AS source_event_id
                 FROM hosted_room_driver_tasks AS task
                 JOIN hosted_rooms AS room ON room.room_id=task.room_id
                 LEFT JOIN hosted_room_events AS source
                   ON source.room_id=task.room_id
                  AND source.seq=task.source_event_seq
                WHERE room.disbanded_at IS NULL"""
        ).fetchall():
            room_id = str(row["room_id"])
            key = (room_id, str(row["task_id"]))
            source_event_id = row["source_event_id"]
            deferred_is_closed = (
                str(row["status"]) == "deferred"
                and (
                    room_id,
                    str(row["task_id"]),
                    int(row["execution_generation"]),
                )
                in published_deferred
                and source_event_id is not None
                and (
                    room_id,
                    str(source_event_id),
                    str(row["thread_id"]),
                )
                in completed
            )
            if deferred_is_closed:
                continue
            if str(row["status"]) not in {"settled", "failed", "cancelled"}:
                liabilities.add(key)
            elif key not in published:
                liabilities.add(key)
            else:
                # The final outcome is durable, but its discussion still needs
                # one reserved closing transition until room.activity lands.
                continue
            consumed_task_sources.add(
                (room_id, int(row["source_event_seq"]), str(row["thread_id"]))
            )
    liabilities.update(
        _pending_discussion_sources(
            conn,
            consumed_task_sources=consumed_task_sources,
            source_state=source_state,
        )
    )
    return liabilities


def _assert_terminal_recovery_headroom(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    additional_events: int = 0,
    additional_bytes: int = 0,
    released_task_ids: frozenset[str] = frozenset(),
    released_liability_keys: frozenset[tuple[str, str]] = frozenset(),
    prospective_liability_keys: frozenset[tuple[str, str]] = frozenset(),
    remaining_demotion_control_events: int | None = None,
) -> None:
    liabilities = _terminal_publication_liabilities(conn)
    liabilities.difference_update((room_id, task_id) for task_id in released_task_ids)
    liabilities.difference_update(released_liability_keys)
    liabilities.update(prospective_liability_keys)
    room_liabilities = sum(key[0] == room_id for key in liabilities)
    gateway_liabilities = len(liabilities)
    if remaining_demotion_control_events is None:
        remaining_demotion_control_events = _remaining_demotion_control_events(
            conn,
            room_id=room_id,
        )
    if not 0 <= remaining_demotion_control_events <= (
        DEMOTION_CONTROL_EVENT_COUNT_RESERVE
    ):
        raise HostedRoomError("invalid demotion control reserve")
    room_liability_events = (
        room_liabilities * MAX_TERMINAL_PUBLICATION_EVENTS
    )
    gateway_liability_bytes = (
        gateway_liabilities * MAX_TERMINAL_PUBLICATION_BYTES
    )
    room_liability_bytes = room_liabilities * MAX_TERMINAL_PUBLICATION_BYTES
    if room_liability_events > TERMINAL_RECOVERY_COUNT_RESERVE:
        raise HostedRoomError(
            "This Group Chat has more unpublished terminal work than its "
            "recovery reserve can guarantee."
        )
    if (
        room_liability_bytes > TERMINAL_RECOVERY_BYTE_RESERVE
        or gateway_liability_bytes > TERMINAL_RECOVERY_BYTE_RESERVE
    ):
        raise HostedRoomError(
            "This host has more unpublished terminal work than its recovery "
            "storage reserve can guarantee."
        )
    room = conn.execute(
        """SELECT next_seq, event_bytes FROM hosted_rooms
           WHERE room_id=? AND disbanded_at IS NULL""",
        (room_id,),
    ).fetchone()
    if room is None:
        _raise_room_not_found(conn, room_id)
    control_event_limit = MAX_EVENTS_PER_ROOM + CONTROL_EVENT_COUNT_RESERVE
    control_room_byte_limit = MAX_ROOM_EVENT_BYTES + CONTROL_EVENT_BYTE_RESERVE
    control_gateway_byte_limit = (
        MAX_GATEWAY_EVENT_BYTES + CONTROL_EVENT_BYTE_RESERVE
    )
    remaining_demotion_bytes = (
        remaining_demotion_control_events * MAX_DEMOTION_CONTROL_EVENT_BYTES
    )
    used_events = int(room["next_seq"]) - 1 + additional_events
    if (
        used_events
        + room_liability_events
        + remaining_demotion_control_events
        > control_event_limit
    ):
        raise HostedRoomError(
            "This Group Chat must preserve terminal recovery headroom. "
            "Finish pending member turns before continuing."
        )
    room_bytes = int(room["event_bytes"]) + additional_bytes
    if (
        room_bytes
        + room_liability_bytes
        + remaining_demotion_bytes
        > control_room_byte_limit
    ):
        raise HostedRoomError(
            "This Group Chat must preserve terminal recovery storage. "
            "Finish pending member turns before continuing."
        )
    gateway_bytes = int(
        conn.execute(
            "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
        ).fetchone()[0]
    ) + additional_bytes
    if (
        gateway_bytes
        + gateway_liability_bytes
        + remaining_demotion_bytes
        > control_gateway_byte_limit
    ):
        raise HostedRoomError(
            "This host must preserve terminal recovery storage. "
            "Finish pending member turns before continuing."
        )


def _assert_event_capacity(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    room: sqlite3.Row,
    additional_bytes: int,
    additional_events: int = 1,
    allow_control: bool = False,
    allow_stop: bool = False,
    allow_terminal_recovery: bool = False,
    released_task_ids: frozenset[str] = frozenset(),
    released_liability_keys: frozenset[tuple[str, str]] = frozenset(),
    prospective_liability_keys: frozenset[tuple[str, str]] = frozenset(),
    remaining_demotion_control_events: int | None = None,
) -> None:
    if allow_control:
        count_reserve = CONTROL_EVENT_COUNT_RESERVE
        byte_reserve = CONTROL_EVENT_BYTE_RESERVE
    elif allow_stop:
        count_reserve = STOP_EVENT_COUNT_RESERVE
        byte_reserve = STOP_EVENT_BYTE_RESERVE
    elif allow_terminal_recovery:
        # Stop fences consume the first reserve tier. Terminal recovery owns
        # the next tier so cancellation traffic cannot crowd out completion.
        count_reserve = STOP_EVENT_COUNT_RESERVE + TERMINAL_RECOVERY_COUNT_RESERVE
        byte_reserve = STOP_EVENT_BYTE_RESERVE + TERMINAL_RECOVERY_BYTE_RESERVE
    else:
        count_reserve = 0
        byte_reserve = 0
    event_limit = MAX_EVENTS_PER_ROOM + count_reserve
    room_byte_limit = MAX_ROOM_EVENT_BYTES + byte_reserve
    gateway_byte_limit = MAX_GATEWAY_EVENT_BYTES + byte_reserve
    if int(room["next_seq"]) - 1 + additional_events > event_limit:
        raise HostedRoomError(
            "This Group Chat reached its history limit. Start a new Group Chat to continue."
        )
    room_bytes = int(room["event_bytes"])
    if room_bytes + additional_bytes > room_byte_limit:
        raise HostedRoomError(
            "This Group Chat reached its storage limit. Start a new Group Chat to continue."
        )
    gateway_bytes = int(
        conn.execute(
            "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
        ).fetchone()[0]
    )
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        _prune_disbanded_rooms_locked(
            conn,
            now=None,
            max_gateway_event_bytes=max(0, gateway_byte_limit - additional_bytes),
        )
        gateway_bytes = int(
            conn.execute(
                "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
            ).fetchone()[0]
        )
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        raise HostedRoomError(
            "Group Chat storage is full on this host. Delete an old Group Chat and try again."
        )
    _assert_terminal_recovery_headroom(
        conn,
        room_id=room_id,
        additional_events=additional_events,
        additional_bytes=additional_bytes,
        released_task_ids=released_task_ids,
        released_liability_keys=released_liability_keys,
        prospective_liability_keys=prospective_liability_keys,
        remaining_demotion_control_events=remaining_demotion_control_events,
    )


def _is_terminal_recovery_plan(
    plan: list[tuple[int, dict[str, Any]]],
) -> bool:
    """Return whether one complete batch is a bounded terminal publication."""

    if all(
        event["kind"] in _TERMINAL_COMPLETION_EVENT_KINDS
        for _, event in plan
    ):
        return True
    if len(plan) != 2:
        return False
    member = plan[0][1]
    terminal = plan[1][1]
    if member["kind"] != "message.member" or terminal["kind"] != "turn.settled":
        return False
    try:
        member_payload = json.loads(member["payload_json"])
        terminal_payload = json.loads(terminal["payload_json"])
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(member_payload, dict) or not isinstance(
        terminal_payload, dict
    ):
        return False

    for field in (
        "task_id",
        "discussion_event_id",
        "member_id",
        "thread_id",
        "turn_id",
    ):
        member_value = member_payload.get(field)
        terminal_value = terminal_payload.get(field)
        if (
            not isinstance(member_value, str)
            or not member_value.strip()
            or not isinstance(terminal_value, str)
            or not terminal_value.strip()
            or terminal_value != member_value
        ):
            return False

    member_event_id = member.get("event_id")
    message_event_id = terminal_payload.get("message_event_id")
    return (
        terminal_payload.get("passed") is False
        and isinstance(member_event_id, str)
        and bool(member_event_id.strip())
        and isinstance(message_event_id, str)
        and bool(message_event_id.strip())
        and message_event_id == member_event_id
    )


def _prune_disbanded_rooms_locked(
    conn: sqlite3.Connection,
    *,
    now: float | None,
    max_gateway_event_bytes: int | None = None,
) -> int:
    candidates: set[str] = set()
    if now is not None:
        cutoff = now - DISBANDED_ROOM_RETENTION_SECONDS
        candidates.update(
            str(row["room_id"])
            for row in conn.execute(
                """SELECT room_id FROM hosted_rooms
                     WHERE disbanded_at IS NOT NULL AND disbanded_at<=?""",
                (cutoff,),
            ).fetchall()
        )
    candidates.update(
        str(row["room_id"])
        for row in conn.execute(
            """SELECT room_id FROM hosted_rooms
                 WHERE disbanded_at IS NOT NULL
                 ORDER BY disbanded_at DESC, room_id ASC
                 LIMIT -1 OFFSET ?""",
            (MAX_DISBANDED_ROOM_TOMBSTONES,),
        ).fetchall()
    )
    if max_gateway_event_bytes is not None:
        retained_bytes = int(
            conn.execute(
                "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
            ).fetchone()[0]
        )
        if retained_bytes > max_gateway_event_bytes:
            for row in conn.execute(
                """SELECT room_id, event_bytes FROM hosted_rooms
                     WHERE disbanded_at IS NOT NULL
                     ORDER BY disbanded_at ASC, room_id ASC"""
            ).fetchall():
                room_id = str(row["room_id"])
                if room_id not in candidates:
                    candidates.add(room_id)
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_gateway_event_bytes:
                    break
    if not candidates:
        return 0

    placeholders = ",".join("?" for _ in candidates)
    room_ids = tuple(sorted(candidates))
    conn.execute(
        f"""INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at)
            SELECT room_id, disbanded_at FROM hosted_rooms
             WHERE room_id IN ({placeholders}) AND disbanded_at IS NOT NULL""",
        room_ids,
    )
    dependent_tables = (
        "hosted_room_policy_transcript_state",
        "hosted_room_policy_transcript",
        "hosted_room_policy_publications",
        "hosted_room_policy_watermarks",
        "hosted_room_policy_events",
        "hosted_room_policy_threads",
        "hosted_room_policy_cursors",
        "hosted_room_approval_requests",
        "hosted_room_terminal_receipts",
        "hosted_room_driver_tasks",
        "hosted_room_driver_leases",
        "hosted_room_events",
    )
    for table in dependent_tables:
        if _table_exists(conn, table):
            conn.execute(
                f"DELETE FROM {table} WHERE room_id IN ({placeholders})",
                room_ids,
            )
    conn.execute(
        f"DELETE FROM hosted_rooms WHERE room_id IN ({placeholders})",
        room_ids,
    )
    return len(room_ids)


def prune_disbanded_rooms(
    db_path: Path | str,
    *,
    now: float | None = None,
) -> int:
    """Purge deleted Group Chat payloads while reserving their identities."""

    timestamp = time.time() if now is None else float(now)
    with _transaction(db_path, immediate=True) as conn:
        return _prune_disbanded_rooms_locked(conn, now=timestamp)


def _event_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    return {
        "room_id": row["room_id"],
        "seq": int(row["seq"]),
        "event_id": row["event_id"],
        "kind": row["kind"],
        "actor": json.loads(row["actor_json"]),
        "authority_epoch": (
            int(row["authority_epoch"]) if row["authority_epoch"] is not None else None
        ),
        "payload": json.loads(row["payload_json"]),
        "created_at": float(row["created_at"]),
        "idempotent": idempotent,
    }


def create_room(
    db_path: Path | str,
    *,
    room_id: Any,
    name: Any,
    members: Any,
    authority_gateway_id: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Create a room, or return the identical existing room idempotently."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    name = _validate_room_name(name)
    normalized_members, members_json = _validate_members(members)
    authority_gateway_id = _validate_identifier(
        authority_gateway_id,
        label="authority_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    now = time.time() if now is None else float(now)

    with _transaction(db_path, immediate=True) as conn:
        if conn.execute(
            "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?",
            (room_id,),
        ).fetchone():
            raise RoomConflictError("room_id belongs to a disbanded room")
        existing = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, event_bytes, revision,
                      created_at, updated_at, disbanded_at
               FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if existing is not None:
            if existing["disbanded_at"] is not None:
                raise RoomConflictError("room_id belongs to a disbanded room")
            if existing["name"] != name or existing["members_json"] != members_json:
                raise RoomConflictError("room_id already exists with different state")
            if (
                existing["authority_gateway_id"] == "legacy"
                and authority_gateway_id != "legacy"
            ):
                target_epoch = int(existing["authority_epoch"]) + 1
                seq = int(existing["next_seq"])
                claim_actor_json = _canonical_json(
                    {"kind": "system", "id": "authority-control"},
                    label="actor",
                    max_bytes=4 * 1024,
                )
                claim_payload_json = _canonical_json(
                    {
                        "previous_gateway_id": "legacy",
                        "authority_gateway_id": authority_gateway_id,
                        "authority_epoch": target_epoch,
                    },
                    label="payload",
                    max_bytes=MAX_EVENT_JSON_BYTES,
                )
                claim_bytes = _event_storage_bytes(
                    event_id="system:authority-adopted",
                    kind="authority.claimed",
                    actor_json=claim_actor_json,
                    payload_json=claim_payload_json,
                )
                _assert_event_capacity(
                    conn,
                    room_id=room_id,
                    room=existing,
                    additional_bytes=claim_bytes,
                    allow_control=True,
                )
                conn.execute(
                    """INSERT INTO hosted_room_events
                       (room_id, seq, event_id, kind, actor_json,
                        authority_epoch, payload_json, created_at)
                       VALUES (?, ?, 'system:authority-adopted',
                               'authority.claimed', ?, ?, ?, ?)""",
                    (
                        room_id,
                        seq,
                        claim_actor_json,
                        target_epoch,
                        claim_payload_json,
                        now,
                    ),
                )
                adopted = conn.execute(
                    """UPDATE hosted_rooms
                          SET authority_gateway_id=?, authority_epoch=?,
                              next_seq=next_seq+1, revision=revision+1,
                              event_bytes=event_bytes+?, updated_at=?
                        WHERE room_id=? AND authority_gateway_id='legacy'
                          AND authority_epoch=? AND next_seq=?
                          AND disbanded_at IS NULL""",
                    (
                        authority_gateway_id,
                        target_epoch,
                        claim_bytes,
                        now,
                        room_id,
                        int(existing["authority_epoch"]),
                        seq,
                    ),
                )
                if adopted.rowcount != 1:
                    raise AuthorityConflictError("legacy room adoption lost its fence")
                existing = conn.execute(
                    """SELECT room_id, name, members_json, authority_gateway_id,
                              authority_epoch, next_seq, revision, created_at,
                              updated_at, disbanded_at
                         FROM hosted_rooms WHERE room_id=?""",
                    (room_id,),
                ).fetchone()
                if existing is None:  # pragma: no cover - row updated above
                    raise RuntimeError("adopted room could not be reloaded")
                result = _room_from_row(existing, idempotent=True)
                result["adopted"] = True
                claim_event = conn.execute(
                    """SELECT room_id, seq, event_id, kind, actor_json,
                              authority_epoch, payload_json, created_at
                         FROM hosted_room_events
                        WHERE room_id=? AND event_id='system:authority-adopted'""",
                    (room_id,),
                ).fetchone()
                if claim_event is None:  # pragma: no cover - inserted above
                    raise RuntimeError("legacy adoption event could not be reloaded")
                result["claim_event"] = _event_from_row(claim_event)
                return result
            if existing["authority_gateway_id"] != authority_gateway_id:
                raise RoomConflictError(
                    "room_id already belongs to a different authority"
                )
            return _room_from_row(existing, idempotent=True)

        active_rooms = int(
            conn.execute(
                "SELECT COUNT(*) FROM hosted_rooms WHERE disbanded_at IS NULL"
            ).fetchone()[0]
        )
        if active_rooms >= MAX_ACTIVE_ROOMS:
            raise HostedRoomError(
                "This host has too many active Group Chats. Delete one and try again."
            )

        conn.execute(
            """INSERT INTO hosted_rooms
               (room_id, name, members_json, authority_gateway_id,
                authority_epoch, next_seq, event_bytes, revision,
                created_at, updated_at, disbanded_at)
               VALUES (?, ?, ?, ?, 1, 1, 0, 1, ?, ?, NULL)""",
            (room_id, name, members_json, authority_gateway_id, now, now),
        )
        row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, revision, created_at, updated_at
               FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the insert above
            raise RuntimeError("created room could not be reloaded")
    result = _room_from_row(row)
    result["members"] = normalized_members
    return result


def list_rooms(
    db_path: Path | str,
    *,
    include_disbanded: bool = False,
    limit: int = MAX_ROOM_LIST_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return one bounded read-only page ordered by most recent change."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_ROOM_LIST_LIMIT
    ):
        raise HostedRoomError(f"limit must be between 1 and {MAX_ROOM_LIST_LIMIT}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise HostedRoomError("offset must be a non-negative integer")
    conn = _read_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at, updated_at,
                      disbanded_at
               FROM hosted_rooms
               WHERE disbanded_at IS NULL OR ?
               ORDER BY updated_at DESC, room_id ASC
               LIMIT ? OFFSET ?""",
            (int(include_disbanded), limit, offset),
        ).fetchall()
    finally:
        conn.close()
    return [_room_from_row(row) for row in rows]


def append_event(
    db_path: Path | str,
    *,
    room_id: Any,
    event_id: Any,
    kind: Any,
    actor: Any,
    payload: Any,
    authority_gateway_id: Any = None,
    authority_epoch: Any = None,
    require_open_admissions: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Append one immutable event and allocate its per-room sequence atomically.

    Repeating the same ``event_id`` and immutable content returns the original
    event. Reusing the id for different content fails closed.
    """
    if not isinstance(require_open_admissions, bool):
        raise HostedRoomError("require_open_admissions must be a boolean")
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    event_id = _validate_identifier(
        event_id,
        label="event_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    kind = _validate_event_kind(kind)
    normalized_actor, actor_json = _validate_actor(actor, kind=kind)
    if require_open_admissions and kind != "message.user":
        raise HostedRoomError(
            "require_open_admissions is only valid for message.user events"
        )
    authority_scoped = normalized_actor["kind"] in {
        "user",
        "member",
        "gateway",
        "system",
    }
    normalized_authority_gateway_id: str | None = None
    normalized_authority_epoch: int | None = None
    if authority_scoped:
        normalized_authority_gateway_id = _validate_identifier(
            authority_gateway_id,
            label="authority_gateway_id",
            max_chars=MAX_ACTOR_ID_CHARS,
        )
        if (
            normalized_actor["kind"] == "gateway"
            and normalized_actor["id"] != normalized_authority_gateway_id
        ):
            raise HostedRoomError("gateway actor.id must match authority_gateway_id")
        if (
            isinstance(authority_epoch, bool)
            or not isinstance(authority_epoch, int)
            or authority_epoch < 1
        ):
            raise HostedRoomError("authority_epoch must be a positive integer")
        normalized_authority_epoch = authority_epoch
    elif authority_gateway_id is not None or authority_epoch is not None:
        raise HostedRoomError(
            "authority fields are only valid for room-scoped events"
        )
    if not isinstance(payload, dict):
        raise HostedRoomError("payload must be an object")
    admission_thread_id: str | None = None
    if require_open_admissions:
        admission_thread_id = _validate_identifier(
            payload.get("thread_id"),
            label="payload.thread_id",
            max_chars=MAX_EVENT_ID_CHARS,
        )
    payload_json = _canonical_json(
        payload,
        label="payload",
        max_bytes=MAX_EVENT_JSON_BYTES,
    )
    now = time.time() if now is None else float(now)

    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
               FROM hosted_room_events WHERE room_id=? AND event_id=?""",
            (room_id, event_id),
        ).fetchone()
        if existing is not None:
            if (
                existing["kind"] != kind
                or existing["actor_json"] != actor_json
                or existing["authority_epoch"] != normalized_authority_epoch
                or existing["payload_json"] != payload_json
            ):
                raise EventConflictError(
                    "event_id already exists with different content"
                )
            return _event_from_row(existing, idempotent=True)

        room = conn.execute(
            """SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch
                  FROM hosted_rooms
               WHERE room_id=? AND disbanded_at IS NULL""",
            (room_id,),
        ).fetchone()
        if room is None:
            _raise_room_not_found(conn, room_id)
        if authority_scoped and (
            room["authority_gateway_id"] != normalized_authority_gateway_id
            or int(room["authority_epoch"]) != normalized_authority_epoch
        ):
            raise AuthorityConflictError("stale hosted room authority")
        if require_open_admissions and _table_exists(
            conn, "hosted_room_driver_admission_barriers"
        ):
            barrier = conn.execute(
                """SELECT 1 FROM hosted_room_driver_admission_barriers
                   WHERE room_id=? AND gateway_id=? AND authority_epoch=?""",
                (
                    room_id,
                    str(room["authority_gateway_id"]),
                    int(room["authority_epoch"]),
                ),
            ).fetchone()
            if barrier is not None:
                raise RoomAdmissionBlockedError(
                    "new user events are blocked while room authority is demoting"
                )
        seq = int(room["next_seq"])
        event_bytes = _event_storage_bytes(
            event_id=event_id,
            kind=kind,
            actor_json=actor_json,
            payload_json=payload_json,
        )
        released_task_ids = _correlated_terminal_task_ids(
            conn,
            room_id=room_id,
            events=[{"kind": kind, "payload": payload}],
        )
        prospective_discussion_keys = _terminal_task_discussion_liability_keys(
            conn,
            room_id=room_id,
            task_ids=released_task_ids,
        )
        closing_discussion_keys = _closing_discussion_liability_keys(
            conn,
            room_id=room_id,
            events=[{"kind": kind, "payload": payload}],
        )
        _assert_event_capacity(
            conn,
            room_id=room_id,
            room=room,
            additional_bytes=event_bytes,
            allow_control=kind in _CRITICAL_CONTROL_EVENT_KINDS,
            allow_stop=kind == "room.stop_requested",
            allow_terminal_recovery=(
                kind in _TERMINAL_COMPLETION_EVENT_KINDS
                or bool(closing_discussion_keys)
            ),
            released_task_ids=released_task_ids,
            released_liability_keys=closing_discussion_keys,
            prospective_liability_keys=(
                prospective_discussion_keys
                | (
                    frozenset(
                        {_discussion_liability_key(room_id, admission_thread_id)}
                    )
                    if admission_thread_id is not None
                    else frozenset()
                )
            ),
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                room_id,
                seq,
                event_id,
                kind,
                actor_json,
                normalized_authority_epoch,
                payload_json,
                now,
            ),
        )
        advanced = conn.execute(
            """UPDATE hosted_rooms
               SET next_seq=?, event_bytes=event_bytes+?, updated_at=?
               WHERE room_id=? AND next_seq=?""",
            (seq + 1, event_bytes, now, room_id, seq),
        )
        if advanced.rowcount != 1:
            raise RuntimeError("hosted room sequence advance lost its write fence")
        row = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
               FROM hosted_room_events WHERE room_id=? AND seq=?""",
            (room_id, seq),
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the insert above
            raise RuntimeError("appended event could not be reloaded")
    result = _event_from_row(row)
    result["actor"] = normalized_actor
    return result


def append_events(
    db_path: Path | str,
    *,
    events: list[dict[str, Any]],
    allow_terminal_recovery: bool = False,
    expected_latest_seq: int | None = None,
) -> list[dict[str, Any]]:
    """Append one same-room publication plan in a single transaction.

    Capacity is reserved for every new event before the first insert. Existing
    events retain the same idempotency and conflict semantics as
    :func:`append_event`.
    """

    if not isinstance(events, list) or not events:
        raise HostedRoomError("events must be a non-empty list")
    if expected_latest_seq is not None and (
        isinstance(expected_latest_seq, bool)
        or not isinstance(expected_latest_seq, int)
        or expected_latest_seq < 0
    ):
        raise HostedRoomError("expected_latest_seq must be a non-negative integer")
    required = frozenset({"room_id", "event_id", "kind", "actor", "payload"})
    optional = frozenset({"authority_gateway_id", "authority_epoch", "now"})
    prepared: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    room_ids: set[str] = set()

    for raw in events:
        if not isinstance(raw, dict):
            raise HostedRoomError("each event must be an object")
        fields = frozenset(raw)
        missing = required - fields
        unknown = fields - required - optional
        if missing:
            raise HostedRoomError(
                f"event is missing fields: {', '.join(sorted(missing))}"
            )
        if unknown:
            raise HostedRoomError(
                f"event has unknown fields: {', '.join(sorted(unknown))}"
            )

        room_id = _validate_identifier(
            raw["room_id"],
            label="room_id",
            max_chars=MAX_ROOM_ID_CHARS,
        )
        event_id = _validate_identifier(
            raw["event_id"],
            label="event_id",
            max_chars=MAX_EVENT_ID_CHARS,
        )
        if event_id in seen_event_ids:
            raise HostedRoomError("event ids must be unique within a batch")
        seen_event_ids.add(event_id)
        room_ids.add(room_id)

        kind = _validate_event_kind(raw["kind"])
        normalized_actor, actor_json = _validate_actor(raw["actor"], kind=kind)
        authority_scoped = normalized_actor["kind"] in {
            "user",
            "member",
            "gateway",
            "system",
        }
        normalized_gateway_id: str | None = None
        normalized_epoch: int | None = None
        authority_gateway_id = raw.get("authority_gateway_id")
        authority_epoch = raw.get("authority_epoch")
        if authority_scoped:
            normalized_gateway_id = _validate_identifier(
                authority_gateway_id,
                label="authority_gateway_id",
                max_chars=MAX_ACTOR_ID_CHARS,
            )
            if (
                normalized_actor["kind"] == "gateway"
                and normalized_actor["id"] != normalized_gateway_id
            ):
                raise HostedRoomError(
                    "gateway actor.id must match authority_gateway_id"
                )
            if (
                isinstance(authority_epoch, bool)
                or not isinstance(authority_epoch, int)
                or authority_epoch < 1
            ):
                raise HostedRoomError("authority_epoch must be a positive integer")
            normalized_epoch = authority_epoch
        elif authority_gateway_id is not None or authority_epoch is not None:
            raise HostedRoomError(
                "authority fields are only valid for room-scoped events"
            )
        payload = raw["payload"]
        if not isinstance(payload, dict):
            raise HostedRoomError("payload must be an object")
        payload_json = _canonical_json(
            payload,
            label="payload",
            max_bytes=MAX_EVENT_JSON_BYTES,
        )
        created_at = (
            time.time() if raw.get("now") is None else float(raw["now"])
        )
        prepared.append(
            {
                "room_id": room_id,
                "event_id": event_id,
                "kind": kind,
                "normalized_actor": normalized_actor,
                "actor_json": actor_json,
                "payload": payload,
                "payload_json": payload_json,
                "authority_scoped": authority_scoped,
                "authority_gateway_id": normalized_gateway_id,
                "authority_epoch": normalized_epoch,
                "created_at": created_at,
                "event_bytes": _event_storage_bytes(
                    event_id=event_id,
                    kind=kind,
                    actor_json=actor_json,
                    payload_json=payload_json,
                ),
            }
        )

    if len(room_ids) != 1:
        raise HostedRoomError("an event batch must target exactly one room")

    results: list[dict[str, Any] | None] = [None for _ in prepared]
    with _transaction(db_path, immediate=True) as conn:
        pending: list[tuple[int, dict[str, Any]]] = []
        for index, event in enumerate(prepared):
            existing = conn.execute(
                """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                          payload_json, created_at
                   FROM hosted_room_events WHERE room_id=? AND event_id=?""",
                (event["room_id"], event["event_id"]),
            ).fetchone()
            if existing is None:
                pending.append((index, event))
                continue
            if (
                existing["kind"] != event["kind"]
                or existing["actor_json"] != event["actor_json"]
                or existing["authority_epoch"] != event["authority_epoch"]
                or existing["payload_json"] != event["payload_json"]
            ):
                raise EventConflictError(
                    "event_id already exists with different content"
                )
            result = _event_from_row(existing, idempotent=True)
            result["actor"] = event["normalized_actor"]
            results[index] = result

        if pending:
            existing_indices = [
                index for index, result in enumerate(results) if result is not None
            ]
            if existing_indices != list(range(len(existing_indices))):
                raise EventConflictError(
                    "existing batch events must form an ordered prefix"
                )
            room_id = next(iter(room_ids))
            room = conn.execute(
                """SELECT next_seq, event_bytes, authority_gateway_id,
                          authority_epoch
                     FROM hosted_rooms
                    WHERE room_id=? AND disbanded_at IS NULL""",
                (room_id,),
            ).fetchone()
            if room is None:
                _raise_room_not_found(conn, room_id)
            if (
                expected_latest_seq is not None
                and int(room["next_seq"]) - 1 != expected_latest_seq
            ):
                raise RoomConflictError(
                    "hosted room latest sequence changed before event publication"
                )
            for _, event in pending:
                if event["authority_scoped"] and (
                    room["authority_gateway_id"]
                    != event["authority_gateway_id"]
                    or int(room["authority_epoch"]) != event["authority_epoch"]
                ):
                    raise AuthorityConflictError("stale hosted room authority")

            additional_bytes = sum(event["event_bytes"] for _, event in pending)
            terminal_recovery = allow_terminal_recovery and _is_terminal_recovery_plan(
                list(enumerate(prepared))
            )
            released_task_ids = (
                _correlated_terminal_task_ids(
                    conn,
                    room_id=room_id,
                    events=[event for _, event in pending],
                )
                if terminal_recovery
                else frozenset()
            )
            prospective_discussion_keys = (
                _terminal_task_discussion_liability_keys(
                    conn,
                    room_id=room_id,
                    task_ids=released_task_ids,
                )
                if terminal_recovery
                else frozenset()
            )
            closing_discussion_keys = _closing_discussion_liability_keys(
                conn,
                room_id=room_id,
                events=[event for _, event in pending],
            )
            _assert_event_capacity(
                conn,
                room_id=room_id,
                room=room,
                additional_bytes=additional_bytes,
                additional_events=len(pending),
                allow_control=all(
                    event["kind"] in _CRITICAL_CONTROL_EVENT_KINDS
                    for _, event in pending
                ),
                allow_stop=all(
                    event["kind"] == "room.stop_requested" for _, event in pending
                ),
                allow_terminal_recovery=(
                    terminal_recovery or bool(closing_discussion_keys)
                ),
                released_task_ids=released_task_ids,
                released_liability_keys=closing_discussion_keys,
                prospective_liability_keys=prospective_discussion_keys,
            )
            first_seq = int(room["next_seq"])
            next_seq = first_seq
            for index, event in pending:
                conn.execute(
                    """INSERT INTO hosted_room_events
                       (room_id, seq, event_id, kind, actor_json,
                        authority_epoch, payload_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        room_id,
                        next_seq,
                        event["event_id"],
                        event["kind"],
                        event["actor_json"],
                        event["authority_epoch"],
                        event["payload_json"],
                        event["created_at"],
                    ),
                )
                row = conn.execute(
                    """SELECT room_id, seq, event_id, kind, actor_json,
                              authority_epoch, payload_json, created_at
                       FROM hosted_room_events WHERE room_id=? AND seq=?""",
                    (room_id, next_seq),
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by the insert above
                    raise RuntimeError("appended event could not be reloaded")
                result = _event_from_row(row)
                result["actor"] = event["normalized_actor"]
                results[index] = result
                next_seq += 1

            advanced = conn.execute(
                """UPDATE hosted_rooms
                   SET next_seq=?, event_bytes=event_bytes+?, updated_at=?
                   WHERE room_id=? AND next_seq=?""",
                (
                    next_seq,
                    additional_bytes,
                    max(event["created_at"] for _, event in pending),
                    room_id,
                    first_seq,
                ),
            )
            if advanced.rowcount != 1:
                raise RuntimeError("hosted room sequence advance lost its write fence")

    if any(result is None for result in results):  # pragma: no cover - internal fence
        raise RuntimeError("event batch did not produce every requested result")
    return [result for result in results if result is not None]


def probe_hosted_room(db_path: Path | str, *, room_id: Any) -> bool:
    """Check room ownership without creating or migrating the shared store.

    This runs on the synchronous prompt-admission path for older Desktop
    clients, so it fails quickly under contention instead of blocking the
    WebSocket reader for SQLite's normal ten-second timeout.
    """

    checked_room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    path = Path(db_path)
    if not path.is_file():
        return False
    try:
        conn = sqlite3.connect(path, timeout=0.05)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='hosted_rooms' LIMIT 1"
            ).fetchone()
            if table is None:
                return False
            return (
                conn.execute(
                    "SELECT 1 FROM hosted_rooms WHERE room_id=? "
                    "AND disbanded_at IS NULL LIMIT 1",
                    (checked_room_id,),
                ).fetchone()
                is not None
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RoomProbeUnavailableError(
            "hosted room ownership is temporarily unavailable"
        ) from exc


def room_state(
    db_path: Path | str,
    *,
    room_id: Any,
    include_disbanded: bool = False,
) -> dict[str, Any]:
    """Return durable replay and authority state for one room."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at, updated_at,
                      disbanded_at
                 FROM hosted_rooms
                WHERE room_id=? AND (disbanded_at IS NULL OR ?)""",
            (room_id, int(include_disbanded)),
        ).fetchone()
        if row is None:
            _raise_room_not_found(conn, room_id)
        claim_row = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
                 FROM hosted_room_events
                WHERE room_id=? AND kind='authority.claimed'
                  AND authority_epoch=?
                ORDER BY seq DESC LIMIT 1""",
            (room_id, int(row["authority_epoch"])),
        ).fetchone()
    state = _room_from_row(row)
    state["latest_seq"] = int(row["next_seq"]) - 1
    if claim_row is not None:
        state["authority_claim"] = _event_from_row(claim_row)
    return state


def _request_room_stop_locked(
    conn: sqlite3.Connection,
    *,
    room_id: Any,
    cancel_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    now: float,
    demotion_control: bool = False,
) -> dict[str, Any]:
    """Append one Stop while participating in the caller's write transaction."""

    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    cancel_id = _validate_identifier(
        cancel_id,
        label="cancel_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    gateway_id = _validate_identifier(
        expected_gateway_id,
        label="expected_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    if (
        isinstance(expected_epoch, bool)
        or not isinstance(expected_epoch, int)
        or expected_epoch < 1
    ):
        raise HostedRoomError("expected_epoch must be a positive integer")
    event_id = _stop_event_id(cancel_id)
    actor_json = _canonical_json(
        {"kind": "gateway", "id": gateway_id},
        label="actor",
        max_bytes=4 * 1024,
    )
    payload_json = _canonical_json(
        {"cancel_id": cancel_id},
        label="payload",
        max_bytes=MAX_EVENT_JSON_BYTES,
    )
    existing = conn.execute(
        """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                  payload_json, created_at
             FROM hosted_room_events WHERE room_id=? AND event_id=?""",
        (room_id, event_id),
    ).fetchone()
    if existing is not None:
        if (
            str(existing["kind"]) != "room.stop_requested"
            or str(existing["actor_json"]) != actor_json
            or int(existing["authority_epoch"]) != expected_epoch
            or str(existing["payload_json"]) != payload_json
        ):
            raise EventConflictError(
                "event_id already exists with different content"
            )
        return _event_from_row(existing, idempotent=True)

    room = conn.execute(
        """SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch
             FROM hosted_rooms
            WHERE room_id=? AND disbanded_at IS NULL""",
        (room_id,),
    ).fetchone()
    if room is None:
        _raise_room_not_found(conn, room_id)
    if (
        str(room["authority_gateway_id"]) != gateway_id
        or int(room["authority_epoch"]) != expected_epoch
    ):
        raise AuthorityConflictError("stale hosted room authority")
    event_bytes = _event_storage_bytes(
        event_id=event_id,
        kind="room.stop_requested",
        actor_json=actor_json,
        payload_json=payload_json,
    )
    _assert_event_capacity(
        conn,
        room_id=room_id,
        room=room,
        additional_bytes=event_bytes,
        allow_control=demotion_control,
        allow_stop=not demotion_control,
        remaining_demotion_control_events=(1 if demotion_control else None),
    )
    seq = int(room["next_seq"])
    conn.execute(
        """INSERT INTO hosted_room_events
           (room_id, seq, event_id, kind, actor_json, authority_epoch,
            payload_json, created_at)
           VALUES (?, ?, ?, 'room.stop_requested', ?, ?, ?, ?)""",
        (
            room_id,
            seq,
            event_id,
            actor_json,
            expected_epoch,
            payload_json,
            now,
        ),
    )
    advanced = conn.execute(
        """UPDATE hosted_rooms
              SET next_seq=?, event_bytes=event_bytes+?, updated_at=?
            WHERE room_id=? AND next_seq=?""",
        (seq + 1, event_bytes, now, room_id, seq),
    )
    if advanced.rowcount != 1:
        raise RuntimeError("hosted room sequence advance lost its write fence")
    row = conn.execute(
        """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                  payload_json, created_at
             FROM hosted_room_events WHERE room_id=? AND seq=?""",
        (room_id, seq),
    ).fetchone()
    if row is None:  # pragma: no cover - guarded by the insert above
        raise RuntimeError("Stop event could not be reloaded")
    return _event_from_row(row)


def request_room_stop(
    db_path: Path | str,
    *,
    room_id: Any,
    cancel_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
) -> dict[str, Any]:
    """Append an idempotent fence that supersedes earlier user turns."""

    with _transaction(db_path, immediate=True) as conn:
        return _request_room_stop_locked(
            conn,
            room_id=room_id,
            cancel_id=cancel_id,
            expected_gateway_id=expected_gateway_id,
            expected_epoch=expected_epoch,
            now=time.time(),
        )


def claim_authority(
    db_path: Path | str,
    *,
    room_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    new_gateway_id: Any,
    event_id: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence a verified authority transfer with a compare-and-swap epoch.

    This storage primitive does not decide *when* takeover is safe. A future
    replicated driver must call it only after its lease/quorum policy has
    established that the previous owner can no longer commit.
    """
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    expected_gateway_id = _validate_identifier(
        expected_gateway_id,
        label="expected_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    new_gateway_id = _validate_identifier(
        new_gateway_id,
        label="new_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    event_id = _validate_identifier(
        event_id,
        label="event_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    if (
        isinstance(expected_epoch, bool)
        or not isinstance(expected_epoch, int)
        or expected_epoch < 1
    ):
        raise HostedRoomError("expected_epoch must be a positive integer")
    now = time.time() if now is None else float(now)
    target_epoch = expected_epoch + 1
    claim_actor = {"kind": "system", "id": "authority-control"}
    claim_actor_json = _canonical_json(
        claim_actor,
        label="actor",
        max_bytes=4 * 1024,
    )
    claim_payload = {
        "previous_gateway_id": expected_gateway_id,
        "authority_gateway_id": new_gateway_id,
        "authority_epoch": target_epoch,
    }
    claim_payload_json = _canonical_json(
        claim_payload,
        label="payload",
        max_bytes=MAX_EVENT_JSON_BYTES,
    )

    with _transaction(db_path, immediate=True) as conn:
        row = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, next_seq, event_bytes
                 FROM hosted_rooms
                WHERE room_id=? AND disbanded_at IS NULL""",
            (room_id,),
        ).fetchone()
        if row is None:
            _raise_room_not_found(conn, room_id)
        current_gateway = str(row["authority_gateway_id"])
        current_epoch = int(row["authority_epoch"])
        existing_event = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
                 FROM hosted_room_events WHERE room_id=? AND event_id=?""",
            (room_id, event_id),
        ).fetchone()
        if existing_event is not None:
            if (
                existing_event["kind"] != "authority.claimed"
                or existing_event["actor_json"] != claim_actor_json
                or existing_event["authority_epoch"] != target_epoch
                or existing_event["payload_json"] != claim_payload_json
            ):
                raise EventConflictError(
                    "event_id already exists with different content"
                )
            if current_gateway != new_gateway_id or current_epoch != target_epoch:
                raise AuthoritySupersededError(
                    "authority claim succeeded but was later superseded"
                )
            idempotent = True
        elif current_gateway != expected_gateway_id or current_epoch != expected_epoch:
            raise AuthorityConflictError("hosted room authority changed")
        else:
            seq = int(row["next_seq"])
            claim_bytes = _event_storage_bytes(
                event_id=event_id,
                kind="authority.claimed",
                actor_json=claim_actor_json,
                payload_json=claim_payload_json,
            )
            _assert_event_capacity(
                conn,
                room_id=room_id,
                room=row,
                additional_bytes=claim_bytes,
                allow_control=True,
            )
            conn.execute(
                """INSERT INTO hosted_room_events
                   (room_id, seq, event_id, kind, actor_json, authority_epoch,
                    payload_json, created_at)
                   VALUES (?, ?, ?, 'authority.claimed', ?, ?, ?, ?)""",
                (
                    room_id,
                    seq,
                    event_id,
                    claim_actor_json,
                    target_epoch,
                    claim_payload_json,
                    now,
                ),
            )
            updated = conn.execute(
                """UPDATE hosted_rooms
                      SET authority_gateway_id=?, authority_epoch=authority_epoch+1,
                          next_seq=next_seq+1, event_bytes=event_bytes+?,
                          revision=revision+1, updated_at=?
                    WHERE room_id=? AND disbanded_at IS NULL
                      AND authority_gateway_id=? AND authority_epoch=?""",
                (
                    new_gateway_id,
                    claim_bytes,
                    now,
                    room_id,
                    expected_gateway_id,
                    expected_epoch,
                ),
            )
            if updated.rowcount != 1:
                raise AuthorityConflictError("hosted room authority changed")
            idempotent = False
            existing_event = conn.execute(
                """SELECT room_id, seq, event_id, kind, actor_json,
                          authority_epoch, payload_json, created_at
                     FROM hosted_room_events WHERE room_id=? AND event_id=?""",
                (room_id, event_id),
            ).fetchone()
        state_row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at, updated_at
                 FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if state_row is None:  # pragma: no cover - room exists in this transaction
            raise RuntimeError("claimed room could not be reloaded")
    state = _room_from_row(state_row, idempotent=idempotent)
    state["latest_seq"] = int(state_row["next_seq"]) - 1
    if existing_event is None:  # pragma: no cover - both claim paths set it
        raise RuntimeError("authority claim event could not be reloaded")
    state["claim_event"] = _event_from_row(
        existing_event,
        idempotent=idempotent,
    )
    return state


def disband_room(
    db_path: Path | str,
    *,
    room_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Tombstone a room id permanently and idempotently."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    expected_gateway_id = _validate_identifier(
        expected_gateway_id,
        label="expected_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    if (
        isinstance(expected_epoch, bool)
        or not isinstance(expected_epoch, int)
        or expected_epoch < 1
    ):
        raise HostedRoomError("expected_epoch must be a positive integer")
    now = time.time() if now is None else float(now)

    with _transaction(db_path, immediate=True) as conn:
        room = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, next_seq,
                      event_bytes, disbanded_at
                 FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if room is None:
            retired = conn.execute(
                "SELECT retired_at FROM hosted_room_retired_ids WHERE room_id=?",
                (room_id,),
            ).fetchone()
            if retired is None:
                raise RoomNotFoundError("hosted room not found")
            return {
                "room_id": room_id,
                "disbanded_at": float(retired["retired_at"]),
                "idempotent": True,
                "history_expired": True,
            }
        if room["disbanded_at"] is not None:
            conn.execute(
                """INSERT OR IGNORE INTO hosted_room_retired_ids
                   (room_id, retired_at) VALUES (?, ?)""",
                (room_id, float(room["disbanded_at"])),
            )
            event = conn.execute(
                """SELECT room_id, seq, event_id, kind, actor_json,
                          authority_epoch, payload_json, created_at
                     FROM hosted_room_events
                    WHERE room_id=? AND event_id='system:room-disbanded'""",
                (room_id,),
            ).fetchone()
            return {
                "room_id": room_id,
                "disbanded_at": float(room["disbanded_at"]),
                "idempotent": True,
                **(
                    {"event": _event_from_row(event, idempotent=True)}
                    if event is not None
                    else {}
                ),
            }
        if (
            str(room["authority_gateway_id"]) != expected_gateway_id
            or int(room["authority_epoch"]) != expected_epoch
        ):
            raise AuthorityConflictError("stale hosted room authority")
        seq = int(room["next_seq"])
        actor_json = _canonical_json(
            {"kind": "system", "id": "room-control"},
            label="actor",
            max_bytes=4 * 1024,
        )
        payload_json = _canonical_json(
            {"room_id": room_id},
            label="payload",
            max_bytes=MAX_EVENT_JSON_BYTES,
        )
        disband_bytes = _event_storage_bytes(
            event_id="system:room-disbanded",
            kind="room.disbanded",
            actor_json=actor_json,
            payload_json=payload_json,
        )
        _assert_event_capacity(
            conn,
            room_id=room_id,
            room=room,
            additional_bytes=disband_bytes,
            allow_control=True,
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES (?, ?, 'system:room-disbanded', 'room.disbanded', ?, ?, ?, ?)""",
            (
                room_id,
                seq,
                actor_json,
                int(room["authority_epoch"]),
                payload_json,
                now,
            ),
        )
        updated = conn.execute(
            """UPDATE hosted_rooms
               SET disbanded_at=?, updated_at=?, revision=revision+1,
                   next_seq=next_seq+1, event_bytes=event_bytes+?
               WHERE room_id=? AND disbanded_at IS NULL
                 AND authority_gateway_id=? AND authority_epoch=?""",
            (
                now,
                now,
                disband_bytes,
                room_id,
                expected_gateway_id,
                expected_epoch,
            ),
        )
        if updated.rowcount != 1:
            raise RoomConflictError("hosted room disband lost its fence")
        conn.execute(
            """INSERT OR IGNORE INTO hosted_room_retired_ids
               (room_id, retired_at) VALUES (?, ?)""",
            (room_id, now),
        )
        event = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM hosted_room_events
                WHERE room_id=? AND event_id='system:room-disbanded'""",
            (room_id,),
        ).fetchone()
        if event is None:  # pragma: no cover - inserted in this transaction
            raise RuntimeError("room disband event could not be reloaded")
        _prune_disbanded_rooms_locked(
            conn,
            now=now,
            max_gateway_event_bytes=MAX_GATEWAY_EVENT_BYTES,
        )
    return {
        "room_id": room_id,
        "disbanded_at": now,
        "idempotent": False,
        "event": _event_from_row(event),
    }


def read_events(
    db_path: Path | str,
    *,
    room_id: Any,
    since_seq: Any = 0,
    limit: Any = 100,
    include_disbanded: bool = False,
) -> dict[str, Any]:
    """Read a monotonic room-log delta after ``since_seq``."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    if isinstance(since_seq, bool) or not isinstance(since_seq, int) or since_seq < 0:
        raise HostedRoomError("since_seq must be a non-negative integer")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_LOG_LIMIT
    ):
        raise HostedRoomError(f"limit must be between 1 and {MAX_LOG_LIMIT}")

    with _transaction(db_path) as conn:
        room = conn.execute(
            """SELECT next_seq, authority_gateway_id, authority_epoch
               FROM hosted_rooms
               WHERE room_id=? AND (disbanded_at IS NULL OR ?)""",
            (room_id, int(include_disbanded)),
        ).fetchone()
        if room is None:
            _raise_room_not_found(conn, room_id)
        latest_seq = int(room["next_seq"]) - 1
        authority_gateway = str(room["authority_gateway_id"])
        authority_epoch = int(room["authority_epoch"])
        if since_seq > latest_seq:
            raise HostedRoomError("since_seq is ahead of the hosted room log")
        rows = conn.execute(
            """WITH candidates AS (
                   SELECT room_id, seq, event_id, kind, actor_json,
                          authority_epoch, payload_json, created_at,
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
               SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM candidates
                WHERE cumulative_bytes<=?
                ORDER BY seq ASC""",
            (room_id, since_seq, limit, MAX_LOG_PAGE_BYTES),
        ).fetchall()
    events = [_event_from_row(row) for row in rows]

    def build_page(page_events: list[dict[str, Any]]) -> dict[str, Any]:
        cursor = page_events[-1]["seq"] if page_events else since_seq
        return {
            "events": page_events,
            "cursor": cursor,
            "latest_seq": latest_seq,
            "has_more": cursor < latest_seq,
            "authority": {
                "gateway_id": authority_gateway,
                "epoch": authority_epoch,
            },
        }

    def page_bytes(page: dict[str, Any]) -> int:
        return len(
            json.dumps(
                page,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    page = build_page(events)
    if events and page_bytes(page) > MAX_LOG_PAGE_BYTES:
        low, high = 1, len(events)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = build_page(events[:middle])
            if page_bytes(candidate) <= MAX_LOG_PAGE_BYTES:
                low = middle
            else:
                high = middle - 1
        page = build_page(events[:low])
        if page_bytes(page) > MAX_LOG_PAGE_BYTES:
            raise HostedRoomError("hosted room event exceeds replay page limit")
    return page
