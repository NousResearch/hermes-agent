"""``HostedRoomRepository`` on SQLite: hosted room owners, transport bindings and member sessions in the bound
profile's ``state.db`` through the store holder, and the rooms and their event log in the file ``placement`` names,
each write one transaction."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from persistence import AuthorityRefusal, JsonValue, SessionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.sessions import TitleConflict, sanitize_title
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError, EventConflictError, EventCursorConflictError, HostedMemberGuard, HostedOwnerClaim,
    HostedPlacement, HostedRoomError, HostedRoomRefusal, RoomConflictError, RoomHistoryExpiredError, RoomNotFoundError,
    RoomProbeUnavailableError,
)
from session_rules.hosted_rooms import (
    CONTEXT_EVENT_KIND, CONTROL_EVENT_BYTE_RESERVE, CONTROL_EVENT_COUNT_RESERVE, DISBANDED_ROOM_RETENTION_SECONDS,
    IDENTIFIER_RE, MAX_ACTIVE_ROOMS, MAX_DISBANDED_ROOM_TOMBSTONES, MAX_EVENTS_PER_ROOM, MAX_GATEWAY_EVENT_BYTES,
    MAX_LOG_PAGE_BYTES, MAX_ROOM_EVENT_BYTES, _CONTEXT_ACTOR, _CONTROL_EVENT_KINDS, _EMPTY_CONTEXT, _actor_id,
    _actor_json, _claim_payload_json, _control_event, _legacy_members_match, _payload_json, _room_id,
    _system_actor_json, _validate_members, _validate_room_name, clock as _now, context_from_payload, retry_fields,
    utf8_len, validate_context,
)
from session_rules.legacy_local import same_owner

from .._backend import SQLiteStore, register_repository, translated
from ._hosted_rooms_common import _epoch, fenced_update as _fenced_update, table_exists
from ._hosted_rooms_legacy_import import _copy_rows
from ._hosted_rooms_schema import ROOMS_STEP, _RETIRE_FROM_ROOMS
from ._hosted_store import HostedFiles
from .hosted_room_replicas import _append_control_event

__all__ = ["SQLiteHostedRoomRepository", "_event_from_row", "_member_authorized_locked", "_room_from_row",
           "read_events_page"]

# The ``state_meta`` key prefixes of room owners and transport bindings, as main's gateway writes them.
OWNER_PREFIX = "gateway.hosted.owner.v1:"
TRANSPORT_PREFIX = "gateway.hosted.transport.v1:"

_ROOM_SESSION_CONFLICT = "room session identity conflicts with existing data"
_ROOM_TITLE_CONFLICT = (
    "Another group already uses this room title on the target gateway. Rename or migrate that group before retrying.")

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


# Refuses with ``message`` unless the room row names this authority and epoch.
def _require_authority(room: sqlite3.Row, gateway_id: str, epoch: int, message: str) -> None:
    if str(room["authority_gateway_id"]) != gateway_id or int(room["authority_epoch"]) != epoch:
        raise AuthorityConflictError(message)


def _reload(conn: sqlite3.Connection, sql: str, params: tuple, missing: str) -> sqlite3.Row:
    """Re-read a row this transaction just wrote; a miss is an invariant violation."""
    row = conn.execute(sql, params).fetchone()
    if row is None:  # pragma: no cover - guarded by the write above
        raise RuntimeError(missing)
    return row


# The room view of a row; ``disbanded_at`` and ``latest_seq`` appear only when the row carries them.
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


# The event view of a row, with actor and payload decoded.
def _event_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    epoch = row["authority_epoch"]
    return {
        "room_id": row["room_id"], "seq": int(row["seq"]), "event_id": row["event_id"], "kind": row["kind"],
        "actor": json.loads(row["actor_json"]), "authority_epoch": int(epoch) if epoch is not None else None,
        "payload": json.loads(row["payload_json"]), "created_at": float(row["created_at"]), "idempotent": idempotent}


# The event row bound to ``event_id`` in the room, or None.
def _load_event(conn: sqlite3.Connection, room_id: str, event_id: str) -> sqlite3.Row | None:
    return conn.execute(_SELECT_EVENT, (room_id, event_id)).fetchone()


# The payload of the room's newest event of ``kind``, of the newest below ``before_seq`` when given, or None.
def _latest_payload(conn: sqlite3.Connection, room_id: str, kind: str, before_seq: int | None) -> Any:
    sql = "SELECT payload_json FROM hosted_room_events WHERE room_id=? AND kind=?"
    params: tuple[Any, ...] = (room_id, kind)
    if before_seq is not None:
        sql, params = sql + " AND seq<?", (*params, int(before_seq))
    row = conn.execute(sql + " ORDER BY seq DESC LIMIT 1", params).fetchone()
    return None if row is None else json.loads(row["payload_json"])


def _event_content(row: sqlite3.Row) -> tuple[Any, Any, Any, Any]:
    """The immutable (kind, actor_json, authority_epoch, payload_json) an event id is bound to."""
    return row["kind"], row["actor_json"], row["authority_epoch"], row["payload_json"]


# The event bytes of every room in the file.
def _gateway_event_bytes(conn: sqlite3.Connection) -> int:
    return int(conn.execute(_SUM_EVENT_BYTES).fetchone()[0])


def _insert_event(
    conn: sqlite3.Connection, room: sqlite3.Row, room_id: str, seq: int, event_id: str, kind: str, actor_json: str,
    epoch: int, payload_json: str, now: float, *, allow_control: bool = False) -> int:
    """Capacity-check then INSERT one event at ``seq``; returns its accounted bytes."""
    event_bytes = _prepare_event(conn, room, event_id, kind, actor_json, payload_json, allow_control=allow_control)
    conn.execute(_INSERT_EVENT, (room_id, seq, event_id, kind, actor_json, epoch, payload_json, now))
    return event_bytes


def _append_locked(
    conn: sqlite3.Connection, *, room_id: str, event_id: str, kind: str, actor: dict[str, Any], actor_json: str,
    payload_json: str, authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None,
    control_reserve: bool, now: float) -> dict[str, Any]:
    """Append one event in the caller's write: a stored ``event_id`` replays or conflicts, then the room, authority,
    cursor, attachment and capacity checks, the insert and the sequence advance."""
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
    if expected_latest_seq is not None and seq - 1 != expected_latest_seq:
        raise EventCursorConflictError("room changed before event publication")
    if kind in {"message.user", "message.member"}:
        from .hosted_room_attachments import retain_message_attachments
        retain_message_attachments(conn, room_id=room_id, event_id=event_id,
                                   manifest=json.loads(payload_json).get("attachments", []), now=now)
    event_bytes = _insert_event(
        conn, room, room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, now,
        allow_control=kind in _CONTROL_EVENT_KINDS or (control_reserve and actor["kind"] == "gateway"))
    _fenced_update(conn, """UPDATE hosted_rooms SET next_seq=?, event_bytes=event_bytes+?, updated_at=?
            WHERE room_id=? AND next_seq=?""", (seq + 1, event_bytes, now, room_id, seq),
        RuntimeError("hosted room sequence advance lost its write fence"))
    row = _reload(
        conn, f"SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND seq=?", (room_id, seq),
        "appended event could not be reloaded")
    return {**_event_from_row(row), "actor": actor}


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


# Purges disbanded rooms past retention or beyond the tombstone cap, then the oldest while the file's event bytes
# exceed ``max_gateway_event_bytes``; each purged id stays retired.
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


# The caller validates the arguments first.
def read_events_page(
    conn: sqlite3.Connection, *, room_id: str, since_seq: int, limit: int, include_disbanded: bool = False
) -> dict[str, Any]:
    """Read a monotonic room-log delta after ``since_seq`` on the caller's connection: the largest prefix whose
    serialized page fits the page budget."""
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


# --- owners and the member check ----------------------------------------------------
def _claim_owner_locked(conn: sqlite3.Connection, epoch: int, room_id: str, actor: str, rule: OwnerRule | None, *,
                        create: bool) -> None:
    """Check (or, with ``create``, claim) room ownership inside the caller's write on ``state.db``."""
    if (not isinstance(actor, str) or not actor
            or not isinstance(room_id, str) or len(room_id) > 128
            or not IDENTIFIER_RE.fullmatch(room_id)):
        raise AuthorityRefusal('invalid_params')
    _epoch(conn, epoch)
    key = OWNER_PREFIX + room_id
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if row is None and create:
        historical = conn.execute(
            'SELECT 1 FROM hosted_rooms WHERE room_id=? UNION ALL '
            'SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?',
            (room_id, room_id)).fetchone()
        if historical is None:
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, actor))
            return
    if row is None or not same_owner(row[0], actor, rule):
        raise AuthorityRefusal('permission_denied')


# The caller runs the epoch fence first.
def _member_authorized_locked(conn: sqlite3.Connection, guard: HostedMemberGuard, *, operation: str) -> bool:
    """The hosted member check on the caller's ``state.db`` connection; False on any mismatch."""
    from .hosted_room_driver import _task_from_row

    owned = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (OWNER_PREFIX + guard.room_id,)).fetchone()
    if owned is None or owned[0] != guard.owner:
        return False
    stored = conn.execute('SELECT * FROM hosted_rooms WHERE room_id=?',
                          (guard.room_id,)).fetchone()
    if stored is None or stored['disbanded_at'] is not None:
        return False
    room = _room_from_row(stored)
    if (room['authority_gateway_id'], room['authority_epoch']) != (guard.gateway_id, guard.authority_epoch):
        return False
    members = room['members']
    if not any(m.get('member_id') == guard.member_id and m.get('profile') == guard.profile
               and (m.get('target') is None or (
                   isinstance(m.get('target'), dict)
                   and m['target'].get('kind', 'local') == 'local'))
               for m in members):
        return False
    if not guard.home_current:
        return False
    if guard.task is not None:
        stored = conn.execute('SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                              (guard.room_id, guard.task.task_id)).fetchone()
        if stored is None:
            return False
        current = _task_from_row(stored)
        return (current['identity'] == guard.task and current['execution_generation'] == guard.generation
                and current['payload'].get('target_profile') == guard.profile
                and current['payload'].get('target_member_id', guard.profile) == guard.member_id
                and current['status'] in ({'running'} if operation in {'submit', 'execute'}
                                           else {'running', 'stopping'}))
    return True


# --- the hand-off of a standalone backend's rooms -----------------------------------
# The rooms a ``shared-state.db`` handed to the gateway store: never offered again. Created by the first hand-off.
HANDOFF_TABLE = "hosted_room_gateway_handoffs"


def _ensure_handoff_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {HANDOFF_TABLE} (room_id TEXT PRIMARY KEY, handed_at REAL NOT NULL)")


def _fenced_room_ids(conn: sqlite3.Connection, handoff_gateway_id: str) -> frozenset[str]:
    """Fenced live rooms not yet recorded as handed over."""
    _ensure_handoff_table(conn)
    return frozenset(str(row[0]) for row in conn.execute(
        f"SELECT room_id FROM hosted_rooms WHERE authority_gateway_id=? AND disbanded_at IS NULL"
        f" AND room_id NOT IN (SELECT room_id FROM {HANDOFF_TABLE})", (handoff_gateway_id,)))


def _record_handoff(conn: sqlite3.Connection, room_ids: Collection[str], now: float) -> None:
    _ensure_handoff_table(conn)
    conn.executemany(f"INSERT OR IGNORE INTO {HANDOFF_TABLE} (room_id, handed_at) VALUES (?, ?)",
                     [(room_id, now) for room_id in sorted(room_ids)])


def _handed_room_ids(conn: sqlite3.Connection, handoff_gateway_id: str) -> frozenset[str]:
    """Every room handed to the gateway store, recorded or still only fenced."""
    return frozenset(str(row[0]) for row in conn.execute(
        "SELECT room_id FROM hosted_rooms WHERE authority_gateway_id=?", (handoff_gateway_id,)))


def _claim(conn: sqlite3.Connection, room_id: str, local_id: str, handoff_id: str, owner: str, now: float) -> None:
    """Claim a copied room for ``local_id`` at the next epoch and own it; an id that was not copied is left alone."""
    row = conn.execute("SELECT authority_epoch, next_seq FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if row is None:
        return
    epoch = int(row["authority_epoch"]) + 1
    _append_control_event(conn, room_id, int(row["next_seq"]), epoch, _control_event("claimed", epoch, {
        "previous_gateway_id": handoff_id, "authority_gateway_id": local_id, "authority_epoch": epoch}), now)
    conn.execute("""UPDATE hosted_rooms SET authority_gateway_id=?, authority_epoch=?, next_seq=next_seq+1,
                 revision=revision+1, updated_at=? WHERE room_id=?""", (local_id, epoch, now, room_id))
    conn.execute("INSERT OR IGNORE INTO state_meta(key, value) VALUES (?, ?)", (OWNER_PREFIX + room_id, owner))


def _import_fenced_rooms(conn: sqlite3.Connection, source: Path, *, epoch: int | None, room_ids: Collection[str],
                         owner: str, local_gateway_id: str, handoff_gateway_id: str, now: float) -> int:
    """Copy the fenced rooms this store has never known from ``source``, then claim them; returns rooms copied."""
    if epoch is not None:
        _epoch(conn, epoch)
    # Every id this store has ever known: a disbanded room's row is pruned later, its id stays.
    known = {str(row[0]) for row in conn.execute(
        "SELECT room_id FROM hosted_rooms UNION SELECT room_id FROM hosted_room_retired_ids")}
    missing = frozenset(set(room_ids) - known)
    copied = _copy_rows(conn, source, only=missing) if missing else 0
    for room_id in sorted(missing):
        _claim(conn, room_id, local_gateway_id, handoff_gateway_id, owner, now)
    return copied


class SQLiteHostedRoomRepository:
    """Hosted room owners, transport bindings, member sessions, rooms and the room event log."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store
        self._files = HostedFiles(store, (ROOMS_STEP,))

    # ── writes on state.db ──────────────────────────────────────────────────

    def claim_owner(self, *, epoch: int, room_id: str, actor: str, owner: OwnerRule | None) -> None:
        with translated():
            self._store.holder()._execute_write(
                lambda conn: _claim_owner_locked(conn, epoch, room_id, actor, owner, create=False))

    # Binds the session's transport once; a different binding for the same session is refused.
    def bind_transport(self, *, epoch: int, session_id: str, binding: str) -> None:
        key = TRANSPORT_PREFIX + session_id

        def persist(conn: sqlite3.Connection) -> None:
            _epoch(conn, epoch)
            old = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
            if old is not None and old[0] != binding:
                raise AuthorityRefusal('permission_denied')
            conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)', (key, binding))
        with translated():
            self._store.holder()._execute_write(persist)

    # Titles and hides a room session; a title another session holds raises ``TitleConflict``.
    def label_room_session(self, *, epoch: int, session_id: str, title: str) -> None:
        def label(conn: sqlite3.Connection) -> None:
            _epoch(conn, epoch)
            try:
                conn.execute('UPDATE sessions SET title=?, hidden=1 WHERE id=?',
                             (title, session_id))
            except sqlite3.IntegrityError as exc:
                # The unique title index refused the update: report the session that holds the title.
                holder = conn.execute('SELECT id FROM sessions WHERE title=? AND id!=?', (title, session_id)).fetchone()
                if holder is None:
                    raise
                raise TitleConflict(title, SessionId(holder[0])) from exc
        with translated():
            self._store.holder()._execute_write(label)

    # Creates the hidden ``bot_room`` session of a room member, or accepts the one there with the same title.
    def ensure_room_member_session(self, *, session_id: str, title: str) -> str:
        def atomic(conn: sqlite3.Connection) -> str:
            row = conn.execute("SELECT id, title, source FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is not None:
                if row["title"] != title or row["source"] != "bot_room":
                    raise HostedRoomRefusal(_ROOM_SESSION_CONFLICT)
                return session_id
            clean_title = sanitize_title(title)
            conflict = conn.execute(
                "SELECT id FROM sessions WHERE title=? AND id!=?", (clean_title, session_id)).fetchone()
            if conflict:
                raise HostedRoomRefusal(_ROOM_TITLE_CONFLICT)
            conn.execute(
                "INSERT INTO sessions(id, source, title, hidden, started_at) VALUES(?, 'bot_room', ?, 1, ?)",
                (session_id, clean_title, time.time()))
            return session_id
        with translated():
            return self._store.holder()._execute_write(atomic)

    # ── writes on the placement's file ──────────────────────────────────────

    # Creates a room or replays an identical one, adopting a room whose authority is ``legacy``; with ``owner``,
    # the owner claim commits in the same transaction.
    def create_room(self, *, placement: HostedPlacement, room_id: str, name: str, members: Sequence[Mapping[str, JsonValue]], authority_gateway_id: str, owner: HostedOwnerClaim | None = None, now: float | None = None) -> Mapping[str, JsonValue]:
        room_id = _room_id(room_id)
        name = _validate_room_name(name)
        normalized_members, members_json = _validate_members(members)
        authority_gateway_id = _actor_id(authority_gateway_id, "authority_gateway_id")
        now = _now(now)
        if owner is not None and placement != "profile":
            raise ValueError("an owner claim requires placement='profile'")

        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            if owner is not None:
                _claim_owner_locked(conn, owner.epoch, room_id, owner.actor, owner.rule, create=True)
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
        with translated():
            return self._files._hosted_write(placement, write, immediate=True)

    # Renames the room and logs ``room.renamed``; the same event id with the same name replays.
    def rename_room(self, *, placement: HostedPlacement, room_id: str, event_id: str, name: str, now: float | None = None) -> Mapping[str, JsonValue]:
        now = _now(now)
        actor_json = _system_actor_json("room-control")
        payload_json = _payload_json({"name": name})

        def write(conn: sqlite3.Connection) -> dict[str, Any]:
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
        with translated():
            return self._files._hosted_write(placement, write, immediate=True)

    # The caller validates the event first. Appends it under the room's authority; the same id and content replays.
    def append_event(self, *, placement: HostedPlacement, room_id: str, event_id: str, kind: str, actor: Mapping[str, str], payload: Mapping[str, JsonValue], authority_gateway_id: str, authority_epoch: int, expected_latest_seq: int | None = None, control_reserve: bool = False, now: float | None = None) -> Mapping[str, JsonValue]:
        normalized_actor = dict(actor)
        actor_json = _actor_json(normalized_actor)
        payload_json = _payload_json(dict(payload))
        now = _now(now)
        with translated():
            return self._files._hosted_write(placement, lambda conn: _append_locked(
                conn, room_id=room_id, event_id=event_id, kind=kind, actor=normalized_actor, actor_json=actor_json,
                payload_json=payload_json, authority_gateway_id=authority_gateway_id, authority_epoch=authority_epoch,
                expected_latest_seq=expected_latest_seq, control_reserve=control_reserve, now=now), immediate=True)

    # The caller validates the room, event id and authority first; the omitted routing, the record's rules, the lead
    # check and the retry comparison all resolve inside the append's write.
    def save_room_settings(self, *, placement: HostedPlacement, room_id: str, event_id: str, purpose: str, notes: str, routing: str | None, lead_member_id: str | None, seated_member_ids: Collection[str], authority_gateway_id: str, authority_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        actor = dict(_CONTEXT_ACTOR)
        actor_json = _actor_json(actor)
        seated = frozenset(seated_member_ids)
        now = _now(now)

        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            first = _load_event(conn, room_id, event_id)
            original = (json.loads(first["payload_json"])
                        if first is not None and first["kind"] == CONTEXT_EVENT_KIND else None)
            current = ({**_EMPTY_CONTEXT, **original} if original is not None
                       else context_from_payload(_latest_payload(conn, room_id, CONTEXT_EVENT_KIND, None)))
            context = validate_context(
                purpose, notes, current["routing"] if routing is None else routing,
                current["lead_member_id"] if lead_member_id is None else lead_member_id)
            if first is None and context["routing"] == "lead" and context["lead_member_id"] not in seated:
                raise HostedRoomError("the lead must be a member of this Group Chat")
            payload = context if original is None else retry_fields(context, original)
            event = _append_locked(
                conn, room_id=room_id, event_id=event_id, kind=CONTEXT_EVENT_KIND, actor=actor, actor_json=actor_json,
                payload_json=_payload_json(payload), authority_gateway_id=authority_gateway_id,
                authority_epoch=authority_epoch, expected_latest_seq=None, control_reserve=False, now=now)
            return {"context": context, "event": event}
        with translated():
            return self._files._hosted_write(placement, write, immediate=True)

    # The caller validates the arguments first. Disbands the room under its authority, retires its id and prunes;
    # a room already disbanded or retired replays.
    def disband_room(self, *, placement: HostedPlacement, room_id: str, expected_gateway_id: str, expected_epoch: int, now: float | None = None) -> Mapping[str, JsonValue]:
        now = _now(now)

        def write(conn: sqlite3.Connection) -> dict[str, Any]:
            room = conn.execute("""SELECT authority_gateway_id, authority_epoch, next_seq, event_bytes, disbanded_at
                    FROM hosted_rooms WHERE room_id=?""", (room_id,)).fetchone()
            if (replay := _disband_replay(conn, room_id, room)) is not None:
                return replay
            _require_authority(room, expected_gateway_id, expected_epoch, "stale hosted room authority")
            disband_bytes = _insert_event(
                conn, room, room_id, int(room["next_seq"]), "system:room-disbanded", "room.disbanded",
                _system_actor_json("room-control"), int(room["authority_epoch"]), _payload_json({"room_id": room_id}),
                now, allow_control=True)
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
        with translated():
            return self._files._hosted_write(placement, write, immediate=True)

    # Purges disbanded rooms past retention or beyond the tombstone cap.
    def prune_disbanded_rooms(self, *, placement: HostedPlacement, now: float | None = None) -> int:
        timestamp = _now(now)
        with translated():
            return self._files._hosted_write(
                placement, lambda conn: _prune_disbanded_rooms_locked(conn, now=timestamp), immediate=True)

    # ── reads on state.db ───────────────────────────────────────────────────

    # The room's recorded owner, or None.
    def owner(self, room_id: str) -> str | None:
        with translated(), self._store.holder()._read_ctx() as conn:
            row = conn.execute('SELECT value FROM state_meta WHERE key=?', (OWNER_PREFIX + room_id,)).fetchone()
        return None if row is None else row[0]

    # The ids of every room with a recorded owner.
    def owned_rooms(self) -> frozenset[str]:
        with translated(), self._store.holder()._read_ctx() as conn:
            return frozenset(r[0][len(OWNER_PREFIX):] for r in conn.execute(
                'SELECT key FROM state_meta WHERE key LIKE ?', (OWNER_PREFIX + '%',)))

    # Probes the owner key with a short lock wait and no schema step.
    def probe_room_owner(self, room_id: str) -> bool:
        return _probe(self._files.path("profile"), "state_meta", "SELECT 1 FROM state_meta WHERE key=? LIMIT 1",
                      (OWNER_PREFIX + room_id,), "native room ownership is temporarily unavailable")

    # The session's transport binding, or None.
    def transport_binding(self, session_id: str) -> str | None:
        with translated(), self._store.holder()._read_ctx() as conn:
            stored = conn.execute('SELECT value FROM state_meta WHERE key=?',
                                  (TRANSPORT_PREFIX + session_id,)).fetchone()
        return None if stored is None else stored[0]

    # The member check after the epoch fence, on one read of ``state.db``.
    def member_authorized(self, guard: HostedMemberGuard, *, epoch: int, operation: str) -> bool:
        with translated(), self._store.holder()._read_ctx() as conn:
            _epoch(conn, epoch)
            return _member_authorized_locked(conn, guard, operation=operation)

    # ── reads on the placement's file ───────────────────────────────────────

    # The caller validates the arguments first; rooms by most recent update.
    def list_rooms(self, *, placement: HostedPlacement, include_disbanded: bool = False, limit: int = 500, offset: int = 0) -> list[Mapping[str, JsonValue]]:
        def read(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return conn.execute(
                f"""SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE disbanded_at IS NULL OR ?
                    ORDER BY updated_at DESC, room_id ASC LIMIT ? OFFSET ?""", (int(include_disbanded), limit, offset)
            ).fetchall()
        with translated():
            rows = self._files._hosted_read(placement, read)
        return [_room_from_row(row) for row in rows]

    # The caller validates the arguments first; the room view with the claim event of its current epoch, if any.
    def room_state(self, *, placement: HostedPlacement, room_id: str, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        def read(conn: sqlite3.Connection) -> tuple[sqlite3.Row, sqlite3.Row | None]:
            row = _room_row(
                conn,
                f"""SELECT {_ROOM_COLUMNS} FROM hosted_rooms WHERE room_id=? AND (disbanded_at IS NULL
                    OR ?)""", (room_id, int(include_disbanded)), room_id)
            claim_row = conn.execute(
                f"""SELECT {_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=?
                    AND kind='authority.claimed' AND authority_epoch=? ORDER BY seq DESC LIMIT 1""",
                (room_id, int(row["authority_epoch"]))).fetchone()
            return row, claim_row
        with translated():
            row, claim_row = self._files._hosted_read(placement, read)
        return {**_room_from_row(row), **({"authority_claim": _event_from_row(claim_row)} if claim_row is not None else {})}

    def read_events(self, *, placement: HostedPlacement, room_id: str, since_seq: int = 0, limit: int = 100, include_disbanded: bool = False) -> Mapping[str, JsonValue]:
        with translated():
            return self._files._hosted_read(placement, lambda conn: read_events_page(
                conn, room_id=room_id, since_seq=since_seq, limit=limit, include_disbanded=include_disbanded))

    # The caller validates the arguments first; probes for a live room with a short lock wait and no schema step.
    def probe_hosted_room(self, *, placement: HostedPlacement, room_id: str) -> bool:
        return _probe(
            self._files.path(placement), "hosted_rooms",
            "SELECT 1 FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL LIMIT 1",
            (room_id,), "hosted room ownership is temporarily unavailable")

    # The caller validates the arguments first; one read of the log on the kind index.
    def latest_event_payload(self, *, placement: HostedPlacement, room_id: str, kind: str, before_seq: int | None = None) -> Mapping[str, JsonValue] | None:
        with translated():
            return self._files._hosted_read(placement, lambda conn: _latest_payload(conn, room_id, kind, before_seq))

    # The caller validates the arguments first; the stored event, or None.
    def event_by_id(self, *, placement: HostedPlacement, room_id: str, event_id: str) -> Mapping[str, JsonValue] | None:
        with translated():
            row = self._files._hosted_read(placement, lambda conn: _load_event(conn, room_id, event_id))
        return None if row is None else _event_from_row(row)

    # ── the hand-off: shared-state.db, and the copy into state.db ──────────────

    # A write, as main's: it creates the hand-off table before it reads.
    def fenced_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        with translated():
            return self._files._hosted_write(
                "shared", lambda conn: _fenced_room_ids(conn, handoff_gateway_id), immediate=True)

    def record_room_handoff(self, *, room_ids: Collection[str], now: float | None = None) -> None:
        timestamp = _now(now)
        with translated():
            self._files._hosted_write(
                "shared", lambda conn: _record_handoff(conn, room_ids, timestamp), immediate=True)

    def handed_room_ids(self, *, handoff_gateway_id: str) -> frozenset[str]:
        with translated():
            return self._files._hosted_read("shared", lambda conn: _handed_room_ids(conn, handoff_gateway_id))

    def import_fenced_rooms(self, *, epoch: int | None, room_ids: Collection[str], owner: str, local_gateway_id: str, handoff_gateway_id: str, now: float | None = None) -> int:
        timestamp = _now(now)
        source = self._files.path("shared")
        with translated():
            return self._files._hosted_write("profile", lambda conn: _import_fenced_rooms(
                conn, source, epoch=epoch, room_ids=room_ids, owner=owner, local_gateway_id=local_gateway_id,
                handoff_gateway_id=handoff_gateway_id, now=timestamp), immediate=True)


register_repository("hosted_rooms", SQLiteHostedRoomRepository)
