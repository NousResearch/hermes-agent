"""Durable bounded policy projection for hosted Group Chat preparation.

The append-only room log remains the user-visible source of truth. This module materializes only the state
needed to choose and reconstruct the next active discussion, so a busy room does not replay its complete
history every poll.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway import hosted_rooms
from gateway.hosted_rooms_common import DbPath, compact_json, fenced_update


MAX_ACTIVE_POLICY_EVENTS = 64
MAX_THREAD_TRANSCRIPT_EVENTS = 24
_TRANSCRIPT_SCHEMA_VERSION = 1
_TERMINAL_KINDS = frozenset({"turn.settled", "turn.failed", "turn.cancelled", "turn.deferred"})

_SCHEMA_DDL = (
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
)

_ROOM_EVENT_COLUMNS = hosted_rooms._EVENT_COLUMNS
_DELETE_ACTIVE_EVENTS_SQL = ("DELETE FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=?")
_TRANSCRIPT_EVENTS_SQL = f"""WITH transcript_events(seq) AS (
        SELECT seq FROM hosted_room_policy_transcript WHERE room_id=? AND thread_id=?
        UNION ALL
        SELECT settled_seq FROM hosted_room_policy_transcript
         WHERE room_id=? AND thread_id=? AND settled_seq IS NOT NULL)
    SELECT {", ".join("events." + column for column in _ROOM_EVENT_COLUMNS.split(", "))}
    FROM transcript_events
    JOIN hosted_room_events AS events ON events.room_id=? AND events.seq=transcript_events.seq
    ORDER BY events.seq"""


@dataclass(frozen=True)
class PolicySnapshot:
    """Bounded active policy input at one durable room-log cursor."""
    through_seq: int
    stopped_through_seq: int
    events: tuple[dict[str, Any], ...]
    watermarks: Mapping[tuple[str, str], int]


_event_from_room_row = hosted_rooms._event_from_row


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "")


def _require_room(conn: sqlite3.Connection, room_id: str) -> None:
    if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone() is None:
        raise hosted_rooms.RoomNotFoundError("hosted room not found")


def _settled_message(
    conn: sqlite3.Connection, room_id: str, discussion_event_id: str, message_event_id: Any) -> dict[str, Any] | None:
    """Return the indexed member message a ``turn.settled`` event committed, if it is in the projection."""
    rows = conn.execute(
        "SELECT seq, event_json FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=?",
        (room_id, discussion_event_id)).fetchall()
    return next(
        (m for m in (json.loads(row["event_json"]) for row in rows) if m.get("event_id") == message_event_id), None)


class HostedRoomPolicyCheckpoint:
    """Incrementally index room policy without compacting visible history."""
    def __init__(self, db_path: DbPath) -> None:
        self.db_path = Path(db_path)
        with self._connect() as conn:
            for ddl in _SCHEMA_DDL:
                conn.execute(ddl)

    def _connect(self) -> sqlite3.Connection:
        from hermes_state_wal import apply_wal_with_fallback
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(conn, db_label="state.db (room policy checkpoint)")
        return conn

    @staticmethod
    def _store_active_event(
        conn: sqlite3.Connection, *, event: Mapping[str, Any], thread_id: str, discussion_event_id: str) -> None:
        conn.execute("""INSERT OR IGNORE INTO hosted_room_policy_events(
                   room_id, thread_id, discussion_event_id, seq, event_json
               ) VALUES (?, ?, ?, ?, ?)""",
            (event["room_id"], thread_id, discussion_event_id, int(event["seq"]), compact_json(dict(event))))

    @staticmethod
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

    def _backfill_transcript(self, conn: sqlite3.Connection, *, room_id: str, through_seq: int) -> None:
        """Migrate bounded committed thread history from the durable room log."""
        if through_seq <= 0:
            return
        settled_seq_by_message = {
            message_event_id: int(row["seq"])
            for row in conn.execute("""SELECT seq, payload_json FROM hosted_room_events
               WHERE room_id=? AND seq<=? AND kind='turn.settled' ORDER BY seq""", (room_id, through_seq))
            if (message_event_id := _text(json.loads(row["payload_json"]), "message_event_id"))}
        for row in conn.execute(
            f"""SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events
               WHERE room_id=? AND seq<=? AND kind IN ('message.user', 'message.member')
               ORDER BY seq""", (room_id, through_seq)):
            if row["kind"] == "message.member" and row["event_id"] not in settled_seq_by_message:
                continue
            event = _event_from_room_row(row)
            thread_id = _text(event["payload"], "thread_id")
            if thread_id:
                self._store_transcript_event(
                    conn, event=event, thread_id=thread_id, settled_seq=settled_seq_by_message.get(str(row["event_id"]))
                )

    def _discussion_events(
        self, conn: sqlite3.Connection, *, room_id: str, thread_id: str, discussion_event_id: str, bound_error: str
    ) -> list[dict[str, Any]]:
        """Merge the thread transcript with the active projection, ordered by seq."""
        active_rows = conn.execute("""SELECT event_json FROM hosted_room_policy_events
               WHERE room_id=? AND discussion_event_id=? ORDER BY seq LIMIT ?""",
            (room_id, discussion_event_id, MAX_ACTIVE_POLICY_EVENTS + 1)).fetchall()
        if len(active_rows) > MAX_ACTIVE_POLICY_EVENTS:
            raise RuntimeError(bound_error)
        rows = conn.execute(_TRANSCRIPT_EVENTS_SQL, (room_id, thread_id, room_id, thread_id, room_id)).fetchall()
        events_by_seq = {
            int(event["seq"]): event
            for event in (*map(_event_from_room_row, rows), *(json.loads(row["event_json"]) for row in active_rows))}
        return [events_by_seq[seq] for seq in sorted(events_by_seq)]

    # -- per-kind projection handlers (dispatched by _apply_event) -----------

    def _apply_user_message(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
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
        self._store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=event_id)
        self._store_transcript_event(conn, event=event, thread_id=thread_id)

    def _apply_discussion_event(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        """Index member messages and terminal turn outcomes of a known discussion."""
        room_id, seq, kind = str(event["room_id"]), int(event["seq"]), _text(event, "kind")
        thread_id, discussion_event_id = _text(payload, "thread_id"), _text(payload, "discussion_event_id")
        if conn.execute(
            "SELECT 1 FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=? LIMIT 1",
            (room_id, discussion_event_id)).fetchone() is None:
            return
        self._store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=discussion_event_id)
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
                self._store_transcript_event(conn, event=committed, thread_id=thread_id, settled_seq=seq)
        if member_id and seen_through_seq > 0:
            conn.execute("""INSERT INTO hosted_room_policy_watermarks(
                       room_id, thread_id, member_id, seen_through_seq
                   ) VALUES (?, ?, ?, ?)
                   ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET
                       seen_through_seq=MAX(hosted_room_policy_watermarks.seen_through_seq, excluded.seen_through_seq)""",
                (room_id, thread_id, member_id, seen_through_seq))

    def _apply_room_activity(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        room_id, thread_id = str(event["room_id"]), _text(payload, "thread_id")
        conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, _text(payload, "discussion_event_id")))
        conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id=? AND thread_id=?", (room_id, thread_id))

    def _apply_stop_requested(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        conn.execute("""UPDATE hosted_room_policy_cursors
               SET stopped_through_seq=MAX(stopped_through_seq, ?) WHERE room_id=?""",
            (int(event["seq"]), str(event["room_id"])))

    _APPLY_BY_KIND: dict[str, Callable[..., None]] = {
        "message.user": _apply_user_message, "message.member": _apply_discussion_event,
        **dict.fromkeys(_TERMINAL_KINDS, _apply_discussion_event), "room.activity": _apply_room_activity,
        "room.stop_requested": _apply_stop_requested}

    def _apply_event(self, conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
        handler = self._APPLY_BY_KIND.get(_text(event, "kind"))
        if handler is not None:
            payload = event.get("payload")
            handler(self, conn, event, payload if isinstance(payload, Mapping) else {})

    def _ensure_cursor_and_transcript(self, conn: sqlite3.Connection, room_id: str) -> int:
        """Create the room cursor if absent, backfill the transcript once, return through_seq."""
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
            self._backfill_transcript(conn, room_id=room_id, through_seq=cursor)
            conn.execute("""INSERT INTO hosted_room_policy_transcript_state(room_id, schema_version)
                   VALUES (?, ?)
                   ON CONFLICT(room_id) DO UPDATE SET schema_version=excluded.schema_version""",
                (room_id, _TRANSCRIPT_SCHEMA_VERSION))
        return cursor

    def sync(self, *, room_id: str, latest_seq: int) -> int:
        """Materialize each unseen event exactly once by durable cursor."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = self._ensure_cursor_and_transcript(conn, room_id)
        if cursor > latest_seq:
            raise RuntimeError("room policy cursor is ahead of the durable log")
        while cursor < latest_seq:
            page = hosted_rooms.read_events(
                self.db_path, room_id=room_id, since_seq=cursor, limit=hosted_rooms.MAX_LOG_LIMIT)
            rows = [event for event in page.get("events", []) if isinstance(event, Mapping)]
            next_cursor = int(page.get("cursor") or cursor)
            if not rows or next_cursor <= cursor:
                raise RuntimeError("hosted room policy cursor did not advance")
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _require_room(conn, room_id)
                for event in rows:
                    self._apply_event(conn, event)
                fenced_update(
                    conn, "UPDATE hosted_room_policy_cursors SET through_seq=?, updated_at=? WHERE room_id=?",
                    (next_cursor, float(rows[-1].get("created_at") or 0), room_id),
                    RuntimeError("room policy cursor disappeared during replay"))
            cursor = next_cursor
        return cursor

    def snapshot(self, *, room_id: str, latest_seq: int) -> PolicySnapshot:
        """Return only the oldest active discussion and its watermark set."""
        through_seq = self.sync(room_id=room_id, latest_seq=latest_seq)
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()
            stopped_through_seq = int(cursor["stopped_through_seq"])
            thread = conn.execute("""SELECT thread_id, discussion_event_id FROM hosted_room_policy_threads
                   WHERE room_id=? AND completed=0 AND latest_user_seq>?
                   ORDER BY latest_user_seq, thread_id LIMIT 1""", (room_id, stopped_through_seq)).fetchone()
            if thread is None:
                return PolicySnapshot(
                    through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=(), watermarks={})
            thread_id = str(thread["thread_id"])
            events = self._discussion_events(
                conn, room_id=room_id, thread_id=thread_id, discussion_event_id=str(thread["discussion_event_id"]),
                bound_error="active room policy projection exceeded its bound")
            watermark_rows = conn.execute("""SELECT member_id, seen_through_seq FROM hosted_room_policy_watermarks
                   WHERE room_id=? AND thread_id=?""", (room_id, thread_id)).fetchall()
        return PolicySnapshot(
            through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=tuple(events),
            watermarks={(thread_id, str(row["member_id"])): int(row["seen_through_seq"]) for row in watermark_rows})

    def publication_exists(self, *, room_id: str, task_id: str, status: str, execution_generation: int) -> bool:
        """Return whether one exact driver outcome is already in the room log."""
        sql, params = (
            ("""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id=? AND task_id=? AND kind=? AND execution_generation=?""",
             (room_id, task_id, f"turn.{status}", execution_generation))
            if status == "deferred" else
            ("""SELECT 1 FROM hosted_room_policy_publications
                     WHERE room_id=? AND task_id=? AND kind IN ('turn.settled', 'turn.failed', 'turn.cancelled')""",
             (room_id, task_id)))
        with self._connect() as conn:
            return conn.execute(sql, params).fetchone() is not None

    def events_for_task(self, *, room_id: str, source_event_seq: int) -> list[dict[str, Any]]:
        """Load one bounded discussion projection for terminal reconstruction."""
        with self._connect() as conn:
            source = conn.execute(
                "SELECT discussion_event_id, thread_id FROM hosted_room_policy_events WHERE room_id=? AND seq=?",
                (room_id, source_event_seq)).fetchone()
            return [] if source is None else self._discussion_events(
                conn, room_id=room_id, thread_id=str(source["thread_id"]),
                discussion_event_id=str(source["discussion_event_id"]),
                bound_error="task policy projection exceeded its bound")

    def compact_completed(self, *, room_id: str) -> None:
        """Drop any completed projections left by an interrupted sync."""
        with self._connect() as conn:
            for row in conn.execute(
                "SELECT discussion_event_id FROM hosted_room_policy_threads WHERE room_id=? AND completed=1", (room_id,)
            ).fetchall():
                conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, str(row["discussion_event_id"])))
            conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id=? AND completed=1", (room_id,))
