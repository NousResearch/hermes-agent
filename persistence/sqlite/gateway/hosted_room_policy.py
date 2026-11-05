"""``HostedRoomPolicyRepository`` on SQLite: the bounded room policy checkpoint
in the hosted-room tables of either file.

The append-only room log stays the source of truth; the checkpoint materializes only what the next active discussion
needs, by durable cursor. ``snapshot`` runs a fixed sequence: the cursor write, the cursor bound read, then per log
page one read of the page and one write applying it, then the snapshot read. Where main raises a bare
``RuntimeError``, this raises ``PolicyProjectionError`` with the same message.

The first checkpoint verb on a file in a process opens it with the rooms step and creates the checkpoint tables and indexes in their own transaction, again when a larger active-event budget arrives. The page reads of ``snapshot`` go through the room log's read, which opens the file with the rooms step.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from functools import partial
from collections.abc import Mapping
from typing import Any, Callable

from persistence import JsonValue
from persistence.gateway.hosted_rooms import HostedPlacement, PolicyProjectionError, PolicySnapshot, RoomNotFoundError
from session_rules.hosted_rooms import MAX_LOG_LIMIT, compact_json
from session_rules.hosted_tasks import validate_task_input

from .._backend import SQLiteStore, register_repository, translated
from ._hosted_rooms_common import fenced_update
from ._hosted_rooms_schema import ROOMS_STEP
from ._hosted_store import HostedFiles, file_identity

__all__ = ["SQLiteHostedRoomPolicyRepository"]

# The projection bounds; a caller's active-event budget can raise the first, never lower it.
MAX_ACTIVE_POLICY_EVENTS = 64
# ``turn.deferred`` receipts come from explicit user retries, not from the configured reply
# budget, so they are bounded on their own: sizing the reply budget exactly must leave room.
MAX_ACTIVE_RETRY_RECEIPTS = 64
MAX_THREAD_TRANSCRIPT_EVENTS = 24
_TRANSCRIPT_SCHEMA_VERSION = 5  # 5: rebuild threads an old settle closed under a newer message
_TERMINAL_KINDS = frozenset({"turn.settled", "turn.failed", "turn.cancelled", "turn.deferred"})
# What a Continue restores into a reopened discussion's active projection (the cap's activity is not one).
_REOPENED_KINDS = (*sorted(_TERMINAL_KINDS), "message.member", "room.assignment", "room.routed", "room.continued")

_SCHEMA_DDL = (
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_capacity (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), max_active_events INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_cursors (
        room_id TEXT PRIMARY KEY, through_seq INTEGER NOT NULL DEFAULT 0,
        stopped_through_seq INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_threads (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        latest_user_seq INTEGER NOT NULL, completed INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(room_id, thread_id))""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_policy_pending
        ON hosted_room_policy_threads(room_id, completed, latest_user_seq, thread_id)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_events (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        seq INTEGER NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(room_id, seq))""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_policy_events_active
        ON hosted_room_policy_events(room_id, discussion_event_id, seq)""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_watermarks (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, member_id TEXT NOT NULL,
        seen_through_seq INTEGER NOT NULL, PRIMARY KEY(room_id, thread_id, member_id))""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_publications (
        room_id TEXT NOT NULL, task_id TEXT NOT NULL, kind TEXT NOT NULL,
        execution_generation INTEGER NOT NULL DEFAULT 0, seq INTEGER NOT NULL,
        PRIMARY KEY(room_id, task_id, kind, execution_generation))""",
    # Transcript stores only references into the already bounded room log, so
    # prompt payloads are never duplicated outside room byte limits.
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_transcript (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, seq INTEGER NOT NULL,
        kind TEXT NOT NULL, settled_seq INTEGER, PRIMARY KEY(room_id, thread_id, seq))""",
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_transcript_state (
        room_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_user_thread
       ON hosted_room_events(room_id, json_extract(payload_json, '$.thread_id'), seq)
       WHERE kind='message.user'""",
)
# Identities of the files whose checkpoint tables this process created, with each file's replay capacity.
_PREPARED: dict[tuple[str, int, int] | None, int] = {}
_PREPARED_LOCK = threading.Lock()

# The room event columns every event read here selects.
_ROOM_EVENT_COLUMNS = "room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at"
_DELETE_ACTIVE_EVENTS_SQL = ("DELETE FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=?")
_TRANSCRIPT_EVENTS_SQL = f"""WITH transcript_events(seq) AS (
        SELECT seq FROM hosted_room_policy_transcript WHERE room_id=? AND thread_id=?
        UNION ALL
        SELECT settled_seq FROM hosted_room_policy_transcript
         WHERE room_id=? AND thread_id=? AND settled_seq IS NOT NULL)
    SELECT {", ".join("events." + column for column in _ROOM_EVENT_COLUMNS.split(", "))}
    FROM transcript_events
    JOIN hosted_room_events AS events ON events.room_id=? AND events.seq=transcript_events.seq
    ORDER BY events.seq LIMIT ?"""


def _initialize_schema(conn: sqlite3.Connection, capacity: int) -> int:
    """Create the tables, record ``capacity`` unless the file holds a larger one, and return the file's replay
    capacity: the largest active-event budget any checkpoint on the file has recorded."""
    for ddl in _SCHEMA_DDL:
        conn.execute(ddl)
    # Preserve replay capacity across restarts with a lower scheduling budget.
    conn.execute("""INSERT INTO hosted_room_policy_capacity(singleton, max_active_events)
        VALUES (1, ?) ON CONFLICT(singleton) DO UPDATE SET
        max_active_events=MAX(max_active_events, excluded.max_active_events)""", (capacity,))
    return int(conn.execute(
        "SELECT max_active_events FROM hosted_room_policy_capacity WHERE singleton=1").fetchone()[0])


def _event_from_room_row(row: sqlite3.Row) -> dict[str, Any]:
    """The room event view of one ``hosted_room_events`` row."""
    from .hosted_rooms import _event_from_row

    return _event_from_row(row)


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "")


def _require_room(conn: sqlite3.Connection, room_id: str) -> None:
    if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone() is None:
        raise RoomNotFoundError("hosted room not found")


def _settled_message(
    conn: sqlite3.Connection, room_id: str, discussion_event_id: str, message_event_id: Any) -> dict[str, Any] | None:
    """Read the committed message even after its active discussion was compacted."""
    row = conn.execute(
        f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND event_id=?",
        (room_id, message_event_id)).fetchone()
    if row is None:
        return None
    event = _event_from_room_row(row)
    if event["kind"] != "message.member" or _text(event["payload"], "discussion_event_id") != discussion_event_id:
        return None
    return event


def _store_active_event(
    conn: sqlite3.Connection, *, event: Mapping[str, Any], thread_id: str, discussion_event_id: str) -> None:
    conn.execute("""INSERT OR IGNORE INTO hosted_room_policy_events(
               room_id, thread_id, discussion_event_id, seq, event_json
           ) VALUES (?, ?, ?, ?, ?)""",
        (event["room_id"], thread_id, discussion_event_id, int(event["seq"]), compact_json(dict(event))))


def _store_transcript_event(
    conn: sqlite3.Connection, *, event: Mapping[str, Any], thread_id: str, settled_seq: int | None = None) -> None:
    conn.execute("""INSERT INTO hosted_room_policy_transcript(
               room_id, thread_id, seq, kind, settled_seq
           ) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(room_id, thread_id, seq) DO UPDATE SET
               settled_seq=COALESCE(excluded.settled_seq, hosted_room_policy_transcript.settled_seq)""",
        (event["room_id"], thread_id, int(event["seq"]), str(event["kind"]), settled_seq))
    if event["kind"] in {"message.user", "message.member"}:
        cutoff = conn.execute("""SELECT seq FROM hosted_room_policy_transcript
               WHERE room_id=? AND thread_id=? AND kind IN ('message.user', 'message.member')
               ORDER BY seq DESC LIMIT 1 OFFSET ?""",
            (event["room_id"], thread_id, MAX_THREAD_TRANSCRIPT_EVENTS - 1)).fetchone()
        if cutoff is not None:
            conn.execute(
                "DELETE FROM hosted_room_policy_transcript WHERE room_id=? AND thread_id=? AND seq<?",
                (event["room_id"], thread_id, int(cutoff["seq"])))


def _discussion_events(
    conn: sqlite3.Connection, *, room_id: str, thread_id: str, discussion_event_id: str, bound_error: str,
    max_active_events: int,
) -> list[dict[str, Any]]:
    """Merge the thread transcript with the active projection, ordered by seq."""
    active_rows = []
    for receipts, bound in ((0, max_active_events), (1, MAX_ACTIVE_RETRY_RECEIPTS)):
        rows = conn.execute("""SELECT event_json FROM hosted_room_policy_events
               WHERE room_id=? AND discussion_event_id=?
                 AND (json_extract(event_json, '$.kind') = 'turn.deferred') = ?
               ORDER BY seq LIMIT ?""",
            (room_id, discussion_event_id, receipts, bound + 1)).fetchall()
        if len(rows) > bound:
            raise PolicyProjectionError(bound_error)
        active_rows.extend(rows)
    # LIMIT also reads one overflow sentinel; keep its parameter inside SQLite's integer range.
    max_transcript_policy_events = min(
        2**63 - 2, MAX_THREAD_TRANSCRIPT_EVENTS * (max_active_events + MAX_ACTIVE_RETRY_RECEIPTS + 2))
    rows = conn.execute(_TRANSCRIPT_EVENTS_SQL,
                        (room_id, thread_id, room_id, thread_id, room_id,
                         max_transcript_policy_events + 1)).fetchall()
    if len(rows) > max_transcript_policy_events:
        raise PolicyProjectionError("thread policy transcript exceeded its bound")
    events_by_seq = {
        int(event["seq"]): event
        for event in (*map(_event_from_room_row, rows), *(json.loads(row["event_json"]) for row in active_rows))}
    return [events_by_seq[seq] for seq in sorted(events_by_seq)]


# -- per-kind projection handlers (dispatched by _apply_event) -----------


def _apply_user_message(conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    room_id = str(event["room_id"])
    thread_id, event_id = _text(payload, "thread_id"), _text(event, "event_id")
    if not thread_id or not event_id:
        return
    conn.execute("""INSERT INTO hosted_room_policy_threads(
               room_id, thread_id, discussion_event_id, latest_user_seq, completed
           ) VALUES (?, ?, ?, ?, 0)
           ON CONFLICT(room_id, thread_id) DO UPDATE SET
               discussion_event_id=excluded.discussion_event_id,
               latest_user_seq=excluded.latest_user_seq, completed=0""",
        (room_id, thread_id, event_id, int(event["seq"])))
    _store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=event_id)
    _store_transcript_event(conn, event=event, thread_id=thread_id)


def _apply_discussion_event(conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Index member messages and terminal turn outcomes of a known discussion."""
    room_id, seq, kind = str(event["room_id"]), int(event["seq"]), _text(event, "kind")
    thread_id, discussion_event_id = _text(payload, "thread_id"), _text(payload, "discussion_event_id")
    if conn.execute(
        "SELECT 1 FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=? LIMIT 1",
        (room_id, discussion_event_id)).fetchone() is not None:
        _store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=discussion_event_id)
    # Late outcomes still need publication receipts and transcript commits,
    # but must not resurrect a completed discussion's active projection.
    if kind not in _TERMINAL_KINDS:
        return
    task_id = _text(payload, "task_id")
    execution_generation = int(payload.get("execution_generation") or 0) if kind == "turn.deferred" else 0
    if task_id:
        conn.execute("""INSERT OR IGNORE INTO hosted_room_policy_publications(
                   room_id, task_id, kind, execution_generation, seq
               ) VALUES (?, ?, ?, ?, ?)""",
            (room_id, task_id, kind, execution_generation, seq))
    member_id = _text(payload, "member_id")
    seen_through_seq = int(payload.get("seen_through_seq") or 0)
    if kind == "turn.settled" and payload.get("message_event_id"):
        committed = _settled_message(conn, room_id, discussion_event_id, payload["message_event_id"])
        if committed is not None:
            seen_through_seq = max(seen_through_seq, int(committed["seq"]))
            _store_transcript_event(conn, event=committed, thread_id=thread_id, settled_seq=seq)
    else:
        # Non-visible receipts still supply historical reconstruction watermarks.
        _store_transcript_event(conn, event=event, thread_id=thread_id)
    if member_id and seen_through_seq > 0:
        conn.execute("""INSERT INTO hosted_room_policy_watermarks(
                   room_id, thread_id, member_id, seen_through_seq
               ) VALUES (?, ?, ?, ?)
               ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET
                   seen_through_seq=MAX(hosted_room_policy_watermarks.seen_through_seq, excluded.seen_through_seq)""",
            (room_id, thread_id, member_id, seen_through_seq))


def _apply_room_activity(conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    room_id, thread_id = str(event["room_id"]), _text(payload, "thread_id")
    discussion_event_id = _text(payload, "discussion_event_id")
    conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, discussion_event_id))
    # A newer message on this thread may already have reopened it under its own discussion: the
    # old discussion's completion (planned before that message) must not close the new one.
    conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id=? AND thread_id=? AND discussion_event_id=?",
                 (room_id, thread_id, discussion_event_id))


def _apply_continued(conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Continue reopens a discussion its cap completed: restore its thread row and active projection
    from the room log (the cap's ``room.activity`` dropped both)."""
    room_id, discussion_event_id = str(event["room_id"]), _text(payload, "discussion_event_id")
    source = conn.execute(f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND event_id=?",
                          (room_id, discussion_event_id)).fetchone()
    if source is None or source["kind"] != "message.user":
        return
    source_event = _event_from_room_row(source)
    thread_id = _text(source_event["payload"], "thread_id")
    # A Continue written by an older build after a newer message on the thread (the router now
    # refuses those) must not hand the thread back: the newer discussion owns it.
    if conn.execute("""SELECT 1 FROM hosted_room_events WHERE room_id=? AND seq>? AND seq<=?
            AND kind='message.user' AND json_extract(payload_json, '$.thread_id')=? LIMIT 1""",
            (room_id, int(source_event["seq"]), int(event["seq"]), thread_id)).fetchone():
        return
    conn.execute("""INSERT INTO hosted_room_policy_threads(
               room_id, thread_id, discussion_event_id, latest_user_seq, completed
           ) VALUES (?, ?, ?, ?, 0)
           ON CONFLICT(room_id, thread_id) DO UPDATE SET
               discussion_event_id=excluded.discussion_event_id,
               latest_user_seq=excluded.latest_user_seq, completed=0""",
        (room_id, thread_id, discussion_event_id, int(source_event["seq"])))
    _store_active_event(conn, event=source_event, thread_id=thread_id, discussion_event_id=discussion_event_id)
    for row in conn.execute(f"""SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events
            WHERE room_id=? AND seq>? AND seq<=? AND kind IN ({",".join("?" * len(_REOPENED_KINDS))})
              AND json_extract(payload_json, '$.discussion_event_id')=? ORDER BY seq""",
            (room_id, int(source_event["seq"]), int(event["seq"]), *_REOPENED_KINDS,
             discussion_event_id)).fetchall():
        _store_active_event(conn, event=_event_from_room_row(row), thread_id=thread_id,
                            discussion_event_id=discussion_event_id)


def _apply_stop_requested(conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    conn.execute("""UPDATE hosted_room_policy_cursors
           SET stopped_through_seq=MAX(stopped_through_seq, ?) WHERE room_id=?""",
        (int(event["seq"]), str(event["room_id"])))


_APPLY_BY_KIND: dict[str, Callable[..., None]] = {
    "message.user": _apply_user_message, "message.member": _apply_discussion_event,
    **dict.fromkeys(_TERMINAL_KINDS, _apply_discussion_event), "room.activity": _apply_room_activity,
    # The lead's assignment joins its discussion's projection so the planner sees who takes round 0.
    "room.assignment": _apply_discussion_event,
    "room.routed": _apply_discussion_event, "room.continued": _apply_continued,
    "room.stop_requested": _apply_stop_requested}


def _apply_event(conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
    handler = _APPLY_BY_KIND.get(_text(event, "kind"))
    if handler is not None:
        payload = event.get("payload")
        handler(conn, event, payload if isinstance(payload, Mapping) else {})


def _ensure_cursor_and_transcript(conn: sqlite3.Connection, room_id: str) -> int:
    """Create the cursor and migrate derived policy caches by bounded replay."""
    _require_room(conn, room_id)
    conn.execute("""INSERT OR IGNORE INTO hosted_room_policy_cursors(
               room_id, through_seq, stopped_through_seq, updated_at
           ) VALUES (?, 0, 0, 0)""", (room_id,))
    cursor = int(
        conn.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()[
            "through_seq"])
    transcript_state = conn.execute(
        "SELECT schema_version FROM hosted_room_policy_transcript_state WHERE room_id=?", (room_id,)).fetchone()
    if transcript_state is None or int(transcript_state["schema_version"]) < _TRANSCRIPT_SCHEMA_VERSION:
        # Rebuild derived history without rewriting admissions or the room log.
        for table in ("hosted_room_policy_events", "hosted_room_policy_threads",
                      "hosted_room_policy_watermarks", "hosted_room_policy_transcript"):
            conn.execute(f"DELETE FROM {table} WHERE room_id=?", (room_id,))
        conn.execute("UPDATE hosted_room_policy_cursors SET through_seq=0, stopped_through_seq=0 WHERE room_id=?",
                     (room_id,))
        cursor = 0
        conn.execute("""INSERT INTO hosted_room_policy_transcript_state(room_id, schema_version)
               VALUES (?, ?)
               ON CONFLICT(room_id) DO UPDATE SET schema_version=excluded.schema_version""",
            (room_id, _TRANSCRIPT_SCHEMA_VERSION))
    return cursor


# The message and terminal events published for ``task_id``; refuses one whose payload names another task.
def _published_task_events(conn: sqlite3.Connection, *, room_id: str, task_id: str | None) -> list[dict[str, Any]]:
    if not task_id:
        return []
    digest = task_id.removeprefix("dtask:")
    rows = conn.execute(
        """SELECT * FROM hosted_room_events
           WHERE room_id=? AND event_id IN (?, ?) ORDER BY seq""",
        (room_id, f"dmessage:{digest}", f"dterminal:{digest}"),
    ).fetchall()
    events = [_event_from_room_row(row) for row in rows]
    if any(event["payload"].get("task_id") != task_id for event in events):
        raise PolicyProjectionError("published task receipt identity changed")
    return events


class SQLiteHostedRoomPolicyRepository:
    """The room policy checkpoint on either hosted-room file."""

    def __init__(self, store: SQLiteStore) -> None:
        self._files = HostedFiles(store)
        self._room_files = HostedFiles(store, (ROOMS_STEP,))

    def _prepare(self, placement: HostedPlacement, max_active_events: int) -> int:
        """Check the budget, then create the checkpoint tables and record the budget's capacity once per file per
        process, again when a larger budget arrives; return the file's replay capacity as read then."""
        if isinstance(max_active_events, bool) or not isinstance(max_active_events, int) or max_active_events < 1:
            raise ValueError("max_active_events must be a positive integer")
        # LIMIT also reads one overflow sentinel; keep its parameter inside SQLite's integer range.
        budget = min(2**63 - 2, max(MAX_ACTIVE_POLICY_EVENTS, max_active_events))
        path = self._files.path(placement)
        with _PREPARED_LOCK:
            capacity = _PREPARED.get(file_identity(path))
            if capacity is None or capacity < budget:
                capacity = self._room_files._hosted_write(
                    placement, partial(_initialize_schema, capacity=budget), immediate=False)
                _PREPARED[file_identity(path)] = capacity
            return capacity

    def _sync(self, placement: HostedPlacement, *, room_id: str, latest_seq: int) -> int:
        """Materialize each unseen event exactly once by durable cursor."""
        from .hosted_rooms import read_events_page

        def _cursor(conn: sqlite3.Connection) -> int:
            return _ensure_cursor_and_transcript(conn, room_id)
        cursor = self._files._hosted_write(placement, _cursor, immediate=True)
        if cursor > latest_seq:
            def _bound(conn: sqlite3.Connection) -> None:
                room = conn.execute("SELECT next_seq FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
                if room is None:
                    raise RoomNotFoundError("hosted room not found")
                if cursor >= int(room["next_seq"]):
                    raise PolicyProjectionError("room policy cursor is ahead of the durable log")
            self._files._hosted_read(placement, _bound)
        while cursor < latest_seq:
            page = self._room_files._hosted_read(placement, lambda conn: read_events_page(
                conn, room_id=room_id, since_seq=cursor, limit=MAX_LOG_LIMIT))
            rows = [event for event in page.get("events", []) if isinstance(event, Mapping)]
            next_cursor = int(page.get("cursor") or cursor)
            if not rows or next_cursor <= cursor:
                raise PolicyProjectionError("hosted room policy cursor did not advance")

            def _apply(conn: sqlite3.Connection) -> int | None:
                _require_room(conn, room_id)
                current = conn.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?",
                                       (room_id,)).fetchone()
                if current is None:
                    raise PolicyProjectionError("room policy cursor disappeared during replay")
                if int(current["through_seq"]) != cursor:
                    # Another worker applied the page: reload before any projection effects.
                    return int(current["through_seq"])
                for event in rows:
                    _apply_event(conn, event)
                fenced_update(
                    conn, "UPDATE hosted_room_policy_cursors SET through_seq=?, updated_at=? WHERE room_id=?",
                    (next_cursor, float(rows[-1].get("created_at") or 0), room_id),
                    PolicyProjectionError("room policy cursor disappeared during replay"))
                return None
            moved = self._files._hosted_write(placement, _apply, immediate=True)
            cursor = next_cursor if moved is None else moved
        return cursor

    def snapshot(self, *, placement: HostedPlacement, room_id: str, latest_seq: int,
                 max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> PolicySnapshot:
        """Return only the oldest active discussion and its watermark set."""
        def _read(conn: sqlite3.Connection) -> PolicySnapshot:
            cursor = conn.execute(
                "SELECT through_seq, stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()
            if cursor is None:
                raise RoomNotFoundError("hosted room checkpoint not found")
            through_seq = int(cursor["through_seq"])
            stopped_through_seq = int(cursor["stopped_through_seq"])
            thread = conn.execute("""SELECT thread_id, discussion_event_id FROM hosted_room_policy_threads
                   WHERE room_id=? AND completed=0 AND latest_user_seq>?
                   ORDER BY latest_user_seq, thread_id LIMIT 1""", (room_id, stopped_through_seq)).fetchone()
            if thread is None:
                return PolicySnapshot(
                    through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=(), watermarks={})
            thread_id = str(thread["thread_id"])
            events = _discussion_events(
                conn, room_id=room_id, thread_id=thread_id, discussion_event_id=str(thread["discussion_event_id"]),
                bound_error="active room policy projection exceeded its bound", max_active_events=capacity)
            watermark_rows = conn.execute("""SELECT member_id, seen_through_seq FROM hosted_room_policy_watermarks
                   WHERE room_id=? AND thread_id=?""", (room_id, thread_id)).fetchall()
            return PolicySnapshot(
                through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=tuple(events),
                watermarks={(thread_id, str(row["member_id"])): int(row["seen_through_seq"]) for row in watermark_rows})
        with translated():
            capacity = self._prepare(placement, max_active_events)
            self._sync(placement, room_id=room_id, latest_seq=latest_seq)
            return self._files._hosted_read(placement, _read)

    def publication_exists(self, *, placement: HostedPlacement, room_id: str, task_id: str, status: str,
                           execution_generation: int, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> bool:
        """Return whether one exact driver outcome is already in the room log."""
        sql, params = (
            ("""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id=? AND task_id=? AND kind=? AND execution_generation=?""",
             (room_id, task_id, f"turn.{status}", execution_generation))
            if status == "deferred" else
            ("""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id=? AND task_id=? AND kind IN ('turn.settled', 'turn.failed', 'turn.cancelled')""",
             (room_id, task_id)))

        def _read(conn: sqlite3.Connection) -> bool:
            return conn.execute(sql, params).fetchone() is not None
        with translated():
            self._prepare(placement, max_active_events)
            return self._files._hosted_read(placement, _read)

    def events_for_task(self, *, placement: HostedPlacement, room_id: str, source_event_seq: int,
                        input_context: Mapping[str, JsonValue] | None = None, task_id: str | None = None,
                        max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> tuple[Mapping[str, JsonValue], ...]:
        """Load one bounded discussion projection for terminal reconstruction."""
        def _read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            published = _published_task_events(conn, room_id=room_id, task_id=task_id)
            if input_context is not None:
                context = validate_task_input(input_context)
                seqs = sorted({source_event_seq, *context["event_seqs"]})
                placeholders = ",".join("?" for _ in seqs)
                rows = conn.execute(f"SELECT * FROM hosted_room_events WHERE room_id=? AND seq IN ({placeholders}) ORDER BY seq",
                                    (room_id, *seqs)).fetchall()
                if len(rows) != len(seqs):
                    raise PolicyProjectionError("admitted task input event is missing")
                events = {int(row["seq"]): _event_from_room_row(row) for row in rows}
                source_event = events[source_event_seq]
                thread_id = _text(source_event["payload"], "thread_id")
                if source_event["kind"] != "message.user" or not thread_id:
                    raise PolicyProjectionError("admitted task source is not a user discussion")
                # Supersession remains authoritative after the display projection compacts.
                latest = conn.execute("""SELECT * FROM hosted_room_events
                    WHERE room_id=? AND kind='message.user' AND json_extract(payload_json, '$.thread_id')=?
                    ORDER BY seq DESC LIMIT 1""", (room_id, thread_id)).fetchone()
                if latest is not None:
                    events[int(latest["seq"])] = _event_from_room_row(latest)
                for event in published:
                    events[int(event["seq"])] = event
                return [events[seq] for seq in sorted(events)]
            row = conn.execute(
                f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND seq=?",
                (room_id, source_event_seq)).fetchone()
            if row is None or row["kind"] != "message.user":
                return published
            source = _event_from_room_row(row)
            events = _discussion_events(
                conn, room_id=room_id, thread_id=_text(source["payload"], "thread_id"),
                discussion_event_id=str(source["event_id"]),
                bound_error="task policy projection exceeded its bound", max_active_events=capacity)
            # The source can age out of BOTH bounded projections while a
            # deferred task remains retryable. Its frozen prompt lives in the task.
            by_seq = {event["seq"]: event for event in (*events, *published)}
            by_seq[source_event_seq] = source
            return [by_seq[seq] for seq in sorted(by_seq)]
        with translated():
            capacity = self._prepare(placement, max_active_events)
            return tuple(self._files._hosted_read(placement, _read))

    def compact_completed(self, *, placement: HostedPlacement, room_id: str,
                          max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> None:
        """Drop any completed projections left by an interrupted sync."""
        def _txn(conn: sqlite3.Connection) -> None:
            for row in conn.execute(
                "SELECT discussion_event_id FROM hosted_room_policy_threads WHERE room_id=? AND completed=1", (room_id,)
            ).fetchall():
                conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, str(row["discussion_event_id"])))
            conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id=? AND completed=1", (room_id,))
        with translated():
            self._prepare(placement, max_active_events)
            self._files._hosted_write(placement, _txn, immediate=False)


register_repository("hosted_room_policy", SQLiteHostedRoomPolicyRepository)
