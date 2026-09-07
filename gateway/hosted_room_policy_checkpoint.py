"""Durable bounded policy projection for hosted Group Chat preparation.

The append-only room log remains the user-visible source of truth. This module materializes only the state
needed to choose and reconstruct the next active discussion, so a busy room does not replay its complete
history every poll.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_task_input import validate_task_input
from gateway.hosted_rooms_common import DbPath, compact_json, fenced_update


MAX_ACTIVE_POLICY_EVENTS = 64
MAX_THREAD_TRANSCRIPT_EVENTS = 24
# Versions every derived projection in this module (transcript + manual holds). Bumping it
# replays the durable room log once per room, up to the cursor the room already reached, so a
# checkpoint written by an older build gains the new projection without losing its history.
# Version 3 is the combined upgrade: local (holds) and upstream (transcript) both shipped a
# version 2 for DIFFERENT projections, so a store at 2 must still be replayed for the other one.
# 4: watermarks are rebuilt because the previous projection advanced every held member on every
# threaded event, discarding input addressed to a peer. Holds, threads, transcript and citation
# state rebuild with them from the canonical log, which is unchanged.
_PROJECTION_SCHEMA_VERSION = 4
MAX_TRANSCRIPT_POLICY_EVENTS = MAX_THREAD_TRANSCRIPT_EVENTS * (MAX_ACTIVE_POLICY_EVENTS + 2)
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
    # Manual member holds are room-scoped: one row per member the user paused, holding the log
    # position of the hold still in force (0 once released). The delta a held member skips is
    # consumed in the thread-scoped watermark table, exactly like an executed turn, so a pause in
    # one thread never eats another thread's unread context.
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_holds (
        room_id TEXT NOT NULL, member_id TEXT NOT NULL, held_at_seq INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(room_id, member_id))""",
    # Unresolved peer citations, per thread and member: where a member was last named BY NAME by
    # a peer, and where it last posted. Silent-round recovery needs both after the bounded
    # transcript has trimmed the messages carrying them, and one row per member per thread keeps
    # that memory bounded rather than rescanning history.
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_citations (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, member_id TEXT NOT NULL,
        cited_at_seq INTEGER NOT NULL DEFAULT 0, last_post_seq INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(room_id, thread_id, member_id))""",
    # The citation state as it stood when one discussion STARTED. Its phases are frozen, so they
    # must not read the live aggregate above: a later reply can resolve an old citation and make a
    # new one, and neither may retroactively change an earlier phase's responder set, member index
    # or task identity. Captured once per discussion and dropped with its thread.
    """CREATE TABLE IF NOT EXISTS hosted_room_policy_citation_baseline (
        room_id TEXT NOT NULL, thread_id TEXT NOT NULL, discussion_event_id TEXT NOT NULL,
        member_id TEXT NOT NULL, cited_at_seq INTEGER NOT NULL DEFAULT 0,
        last_post_seq INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(room_id, thread_id, member_id))""",
    """CREATE INDEX IF NOT EXISTS idx_hosted_room_user_thread
       ON hosted_room_events(room_id, json_extract(payload_json, '$.thread_id'), seq)
       WHERE kind='message.user'""",
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
    ORDER BY events.seq LIMIT ?"""


@dataclass(frozen=True)
class MemberHold:
    """One member's durable manual-hold state in a room (``held_at_seq`` 0 means released)."""
    member_id: str
    held_at_seq: int

    @property
    def held(self) -> bool:
        return self.held_at_seq > 0


@dataclass(frozen=True)
class PolicySnapshot:
    """Bounded active policy input at one durable room-log cursor."""
    through_seq: int
    stopped_through_seq: int
    events: tuple[dict[str, Any], ...]
    watermarks: Mapping[tuple[str, str], int]
    holds: tuple[MemberHold, ...] = ()
    #: ``member_id -> (cited_at_seq, last_post_seq)`` as of this discussion's START, so a handoff
    #: older than the bounded transcript is still owed a turn without any phase reading state its
    #: own replies produced.
    citations: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def held_member_ids(self) -> tuple[str, ...]:
        return tuple(hold.member_id for hold in self.holds if hold.held)


_event_from_room_row = hosted_rooms._event_from_row


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "")


def _require_room(conn: sqlite3.Connection, room_id: str) -> None:
    if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone() is None:
        raise hosted_rooms.RoomNotFoundError("hosted room not found")


def _canonical_event(
    conn: sqlite3.Connection, room_id: str, *, seq: int | None = None, event_id: str | None = None
) -> dict[str, Any] | None:
    """Point-read one event from the durable room log by its primary or unique key.

    The room log is the source of truth and is never trimmed while the room lives, so an event
    the bounded projection has already compacted is still readable by exact coordinates. This is
    an indexed single-row lookup (``PRIMARY KEY(room_id, seq)`` / ``UNIQUE(room_id, event_id)``),
    never a scan of the room.
    """
    column, value = ("seq", seq) if event_id is None else ("event_id", event_id)
    row = conn.execute(
        f"SELECT {_ROOM_EVENT_COLUMNS} FROM hosted_room_events WHERE room_id=? AND {column}=?",
        (room_id, value)).fetchone()
    return None if row is None else _event_from_room_row(row)


def _room_members(conn: sqlite3.Connection, room_id: str) -> tuple[discussion.DiscussionMember, ...]:
    """Return the durable roster as policy members: ids, handles and frozen friendly labels.

    ``display_name`` comes from the room's own frozen ``members_json``, never from live profile
    metadata. Dropping it here silently narrowed the durable control path: the resolver would
    accept ``@research-buddy`` in the planner while this projection -- the one that actually
    holds and releases members -- could not see the alias at all.

    Holds are keyed by durable member id, so a malformed row without one is skipped rather than
    failing the whole projection; full roster validation stays with the planner.
    """
    row = conn.execute("SELECT members_json FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if row is None:
        return ()
    members = json.loads(row["members_json"])
    return tuple(
        discussion.DiscussionMember(
            member_id=str(member["member_id"]), profile=_text(member, "profile"),
            handle=_text(member, "handle"), display_name=_text(member, "display_name"))
        for member in (members if isinstance(members, list) else [])
        if isinstance(member, Mapping) and member.get("member_id"))


def _settled_message(
    conn: sqlite3.Connection, room_id: str, discussion_event_id: str, message_event_id: Any) -> dict[str, Any] | None:
    """Return the member message a ``turn.settled`` event committed.

    Normally it is already in the active projection. A turn that settles after its discussion was
    completed and compacted (a late exact receipt) has no projection left, so the committed
    message is read from the durable log by its exact event id.
    """
    rows = conn.execute(
        "SELECT seq, event_json FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=?",
        (room_id, discussion_event_id)).fetchall()
    indexed = next(
        (m for m in (json.loads(row["event_json"]) for row in rows) if m.get("event_id") == message_event_id), None)
    if indexed is not None or not isinstance(message_event_id, str) or not message_event_id:
        return indexed
    committed = _canonical_event(conn, room_id, event_id=message_event_id)
    return committed if committed is not None and committed.get("kind") == "message.member" else None


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

    def _backfill_holds(self, conn: sqlite3.Connection, *, room_id: str, through_seq: int) -> None:
        """Rebuild manual holds for a checkpoint written before this projection existed.

        The room log is the source of truth, so the holds a room already carries are replayed from
        it in sequence order -- exactly the live path -- rather than starting an existing room with
        no holds and silently resuming members the user had paused.
        """
        if through_seq <= 0:
            return
        members = _room_members(conn, room_id)
        for row in conn.execute(
            """SELECT seq, kind, payload_json FROM hosted_room_events
               WHERE room_id=? AND seq<=? ORDER BY seq""", (room_id, through_seq)):
            seq, kind = int(row["seq"]), str(row["kind"])
            payload = json.loads(row["payload_json"])
            payload = payload if isinstance(payload, Mapping) else {}
            if kind == "message.user":
                self._apply_holds(
                    conn, room_id,
                    directive=discussion.resolve_hold_directive(payload.get("text"), members), seq=seq)
            elif kind == "room.stop_requested":
                self._apply_holds(
                    conn, room_id,
                    directive=discussion.HoldDirective(hold=tuple(m.member_id for m in members)), seq=seq)

    def _discussion_events(
        self, conn: sqlite3.Connection, *, room_id: str, thread_id: str, discussion_event_id: str, bound_error: str
    ) -> list[dict[str, Any]]:
        """Merge the thread transcript with the active projection, ordered by seq."""
        active_rows = conn.execute("""SELECT event_json FROM hosted_room_policy_events
               WHERE room_id=? AND discussion_event_id=? ORDER BY seq LIMIT ?""",
            (room_id, discussion_event_id, MAX_ACTIVE_POLICY_EVENTS + 1)).fetchall()
        if len(active_rows) > MAX_ACTIVE_POLICY_EVENTS:
            raise RuntimeError(bound_error)
        rows = conn.execute(_TRANSCRIPT_EVENTS_SQL,
                            (room_id, thread_id, room_id, thread_id, room_id, MAX_TRANSCRIPT_POLICY_EVENTS + 1)).fetchall()
        if len(rows) > MAX_TRANSCRIPT_POLICY_EVENTS:
            raise RuntimeError("thread policy transcript exceeded its bound")
        events_by_seq = {
            int(event["seq"]): event
            for event in (*map(_event_from_room_row, rows), *(json.loads(row["event_json"]) for row in active_rows))}
        # A retained message and its settlement stay in the bounded transcript after the user
        # message they answer has aged out of the window. Those replies are accepted context a
        # resumed member still needs, so the missing SOURCE is reattached by a bounded canonical
        # point read on this same connection rather than dropping the pair to satisfy validation.
        # A source that is absent, or that belongs to another room, thread or kind, or that does
        # not precede what references it, stays an error.
        referenced: dict[str, int] = {}
        for seq, event in events_by_seq.items():
            source_id = str((event.get("payload") or {}).get("discussion_event_id") or "")
            if source_id:
                referenced[source_id] = min(referenced.get(source_id, seq), seq)
        for source_id in {
            str(event.get("event_id") or "") for event in events_by_seq.values()
            if str(event.get("kind") or "") == "message.user"
        }:
            referenced.pop(source_id, None)
        for source_id, first_seq in sorted(referenced.items()):
            source = _canonical_event(conn, room_id, event_id=source_id)
            if (source is None or str(source.get("kind") or "") != "message.user"
                    or str(source.get("room_id") or "") != room_id
                    or str((source.get("payload") or {}).get("thread_id") or "") != thread_id
                    or int(source["seq"]) >= first_seq):
                raise RuntimeError("retained thread transcript references an unusable discussion source")
            events_by_seq.setdefault(int(source["seq"]), source)
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
        # Freeze the citation state this discussion starts from, before any of its own replies
        # move the live aggregate. Everything newer reaches its phases through the committed
        # events they already carry.
        conn.execute(
            "DELETE FROM hosted_room_policy_citation_baseline WHERE room_id=? AND thread_id=?",
            (room_id, thread_id))
        conn.execute("""INSERT INTO hosted_room_policy_citation_baseline(
                   room_id, thread_id, discussion_event_id, member_id, cited_at_seq, last_post_seq)
               SELECT room_id, thread_id, ?, member_id, cited_at_seq, last_post_seq
                   FROM hosted_room_policy_citations WHERE room_id=? AND thread_id=?""",
            (event_id, room_id, thread_id))
        self._store_active_event(conn, event=event, thread_id=thread_id, discussion_event_id=event_id)
        self._store_transcript_event(conn, event=event, thread_id=thread_id)
        # User text is the only input that changes manual holds (see resolve_hold_directive).
        self._apply_holds(
            conn, room_id,
            directive=discussion.resolve_hold_directive(
                payload.get("text"), _room_members(conn, room_id)),
            seq=int(event["seq"]))

    def _apply_discussion_event(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        """Index member messages and terminal turn outcomes of a discussion.

        A turn can still publish after its discussion completed and was compacted (a late exact
        receipt for deferred or uncertain work). The active projection is deliberately not
        re-inflated for such a discussion, but its exactly-once publication marker, watermark and
        bounded transcript entry are still recorded -- those are what later preparations read to
        know the outcome is already in the room.
        """
        room_id, seq, kind = str(event["room_id"]), int(event["seq"]), _text(event, "kind")
        thread_id, discussion_event_id = _text(payload, "thread_id"), _text(payload, "discussion_event_id")
        source = conn.execute(
            "SELECT seq FROM hosted_room_policy_events WHERE room_id=? AND discussion_event_id=? ORDER BY seq LIMIT 1",
            (room_id, discussion_event_id)).fetchone()
        source_seq = int(source["seq"]) if source is not None else None
        if source is not None:
            # Only an ACTIVE discussion keeps an inflated projection.
            self._store_active_event(
                conn, event=event, thread_id=thread_id, discussion_event_id=discussion_event_id)
        elif (canonical := _canonical_event(conn, room_id, event_id=discussion_event_id)) is not None:
            # Compacted discussion: no early return. The projection is gone, but the durable log
            # still holds the source event, and the exactly-once publication marker, watermark and
            # bounded transcript entry below are what later preparations read to know the outcome
            # is already in the room.
            source_seq = int(canonical["seq"])
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
                if source_seq is not None and seen_through_seq >= source_seq:
                    seen_through_seq = max(seen_through_seq, int(committed["seq"]))
                self._store_transcript_event(conn, event=committed, thread_id=thread_id, settled_seq=seq)
                self._record_citations(conn, room_id=room_id, thread_id=thread_id, message=committed)
        else:
            # Non-visible receipts still supply historical reconstruction watermarks.
            self._store_transcript_event(conn, event=event, thread_id=thread_id)
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
        conn.execute(
            "DELETE FROM hosted_room_policy_citation_baseline WHERE room_id=? AND thread_id=?",
            (room_id, thread_id))

    def _apply_stop_requested(
        self, conn: sqlite3.Connection, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        room_id, seq = str(event["room_id"]), int(event["seq"])
        conn.execute("""UPDATE hosted_room_policy_cursors
               SET stopped_through_seq=MAX(stopped_through_seq, ?) WHERE room_id=?""", (seq, room_id))
        # A room Stop is the native UI Stop: it fences earlier turns AND holds every roster member
        # at room scope until the user releases them, so reopening a frontend or sending an
        # unaddressed message does not silently restart the work that was stopped.
        self._apply_holds(
            conn, room_id,
            directive=discussion.HoldDirective(
                hold=tuple(member.member_id for member in _room_members(conn, room_id))),
            seq=seq)

    def _apply_holds(
        self, conn: sqlite3.Connection, room_id: str, *, directive: discussion.HoldDirective,
        seq: int) -> None:
        """Hold and release exactly the members one control event names."""
        for member_id in directive.hold:
            # An already held member keeps its original hold position (native keeps the stamp).
            conn.execute("""INSERT INTO hosted_room_policy_holds(room_id, member_id, held_at_seq)
                   VALUES (?, ?, ?)
                   ON CONFLICT(room_id, member_id) DO UPDATE SET held_at_seq=CASE
                       WHEN hosted_room_policy_holds.held_at_seq > 0
                       THEN hosted_room_policy_holds.held_at_seq ELSE excluded.held_at_seq END""",
                (room_id, member_id, seq))
        for member_id in directive.release:
            conn.execute(
                "UPDATE hosted_room_policy_holds SET held_at_seq=0 WHERE room_id=? AND member_id=?",
                (room_id, member_id))

    def _record_citations(
        self, conn: sqlite3.Connection, *, room_id: str, thread_id: str, message: Mapping[str, Any]) -> None:
        """Remember who one committed member message named, and that its author has now posted.

        Only committed content reaches here (the settlement boundary above), and only explicit
        citations count: ``@all`` addresses the room, so it owes nobody a turn. Bounded to one row
        per member per thread, which is what lets silent-round recovery outlive the trimmed
        transcript without rescanning the log.
        """
        payload = message.get("payload") if isinstance(message.get("payload"), Mapping) else {}
        speaker_id, seq = _text(payload, "member_id"), int(message["seq"])
        if not thread_id or not speaker_id:
            return
        conn.execute("""INSERT INTO hosted_room_policy_citations(
                   room_id, thread_id, member_id, cited_at_seq, last_post_seq)
               VALUES (?, ?, ?, 0, ?)
               ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET
                   last_post_seq=MAX(hosted_room_policy_citations.last_post_seq, excluded.last_post_seq)""",
            (room_id, thread_id, speaker_id, seq))
        for member in discussion.explicit_mentions(payload.get("text"), _room_members(conn, room_id)):
            if member.member_id == speaker_id:
                continue
            conn.execute("""INSERT INTO hosted_room_policy_citations(
                       room_id, thread_id, member_id, cited_at_seq, last_post_seq)
                   VALUES (?, ?, ?, ?, 0)
                   ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET
                       cited_at_seq=MAX(hosted_room_policy_citations.cited_at_seq, excluded.cited_at_seq)""",
                (room_id, thread_id, member.member_id, seq))

    @staticmethod
    def _canonical_room(conn: sqlite3.Connection, room_id: str) -> dict[str, Any] | None:
        """Point-read the room's own frozen identity and roster on this connection."""
        row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id, authority_epoch, disbanded_at
               FROM hosted_rooms WHERE room_id=?""", (room_id,)).fetchone()
        if row is None or row["disbanded_at"] is not None:
            return None
        members = json.loads(row["members_json"])
        return {
            "room_id": str(row["room_id"]), "name": str(row["name"]),
            "members": members if isinstance(members, list) else [],
            "authority_gateway_id": str(row["authority_gateway_id"]),
            "authority_epoch": int(row["authority_epoch"])}

    def _record_walk_consumptions(self, conn: sqlite3.Connection, room_id: str) -> None:
        """Charge the held skips the policy walk actually reaches at this point in the log.

        The same walk, on the same connection, in the same chronology as every other projection
        effect: the planner runs on the bounded snapshot this projection currently holds --
        committed content only, stop fence and FIFO thread order included -- and only what that
        walk reports for itself is written. Nothing is inferred from terminal counts or invented
        rounds, and a validation failure fails this transaction rather than certifying a partial
        migration.
        """
        room = self._canonical_room(conn, room_id)
        if room is None:
            return
        snapshot = self._snapshot_on(conn, room_id)
        if not snapshot.events:
            return
        # Canonical storage accepts any roster up to its own maximum, including an empty one, so
        # a room below the Discussion minimum is a supported storage state that can never carry a
        # discussion: there is no walk to run. ONLY that cardinality is skipped -- a duplicate
        # handle, a reserved one, a bad target or a malformed event still fails this transaction.
        if len(room["members"]) < discussion.MIN_DISCUSSION_MEMBERS:
            return
        profiles = tuple({
            str(member.get("profile") or "") for member in room["members"]
            if isinstance(member, Mapping)} - {""})
        decision = discussion.plan_next_task(
            room, list(snapshot.events), local_profiles=profiles,
            initial_watermarks=snapshot.watermarks, held_member_ids=snapshot.held_member_ids,
            initial_citations=snapshot.citations)
        if decision.thread_id and decision.held_consumptions:
            self._record_held_consumptions(
                conn, room_id=room_id, thread_id=decision.thread_id,
                consumptions=decision.held_consumptions)

    @staticmethod
    def _record_held_consumptions(
        conn: sqlite3.Connection, *, room_id: str, thread_id: str,
        consumptions: Sequence[tuple[str, int]]) -> None:
        """Connection-level recorder, shared by the live drive and the chronological rebuild so
        neither opens a writer inside the other's transaction."""
        for member_id, seen_through_seq in consumptions:
            if not str(member_id) or int(seen_through_seq) <= 0:
                continue
            conn.execute("""INSERT INTO hosted_room_policy_watermarks(
                       room_id, thread_id, member_id, seen_through_seq)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(room_id, thread_id, member_id) DO UPDATE SET
                       seen_through_seq=MAX(hosted_room_policy_watermarks.seen_through_seq,
                                            excluded.seen_through_seq)""",
                (room_id, thread_id, str(member_id), int(seen_through_seq)))

    def apply_held_consumptions(
        self, *, room_id: str, thread_id: str, consumptions: Sequence[tuple[str, int]],
        expected_through_seq: int) -> bool:
        """Persist the held skips one policy walk actually reached.

        The durable half of the native held skip (``group-rounds.ts``): a paused member's
        watermark for THIS thread advances past the entries it was silent for, so releasing it
        does not replay a conversation it can no longer answer usefully. Which skips those are is
        decided by the planner's own slot walk (``DiscussionDecision.held_consumptions``) and only
        recorded here, so the projection can never drift into consuming input addressed to a peer
        at a slot the round never reached. Thread-scoped like every other watermark, and
        monotonic, so re-applying the same walk is a no-op.

        Fenced on the exact cursor the walk read. Consuming a member's context cannot be undone by
        the caller's later cancellation check, so a release or a newer message that committed
        between the snapshot and this write must void the decision instead: ``False`` means
        nothing was written and the room has to be planned again.
        """
        if not thread_id or not consumptions:
            return True
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()
            room = conn.execute("SELECT next_seq FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
            if cursor is None or room is None:
                return False
            # Both halves: the projection must still be at the cursor the walk read, and the
            # canonical log must carry nothing newer than that cursor.
            if int(cursor["through_seq"]) != int(expected_through_seq):
                return False
            if int(room["next_seq"]) != int(expected_through_seq) + 1:
                return False
            self._record_held_consumptions(
                conn, room_id=room_id, thread_id=thread_id, consumptions=consumptions)
        return True

    _APPLY_BY_KIND: dict[str, Callable[..., None]] = {
        "message.user": _apply_user_message, "message.member": _apply_discussion_event,
        **dict.fromkeys(_TERMINAL_KINDS, _apply_discussion_event), "room.activity": _apply_room_activity,
        "room.stop_requested": _apply_stop_requested}

    def _apply_event(self, conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        kind, room_id = _text(event, "kind"), str(event["room_id"])
        # A completing discussion's own walk must run while its projection still exists: the
        # activity handler deletes the active rows this snapshot is built from.
        if kind == "room.activity":
            self._record_walk_consumptions(conn, room_id)
        handler = self._APPLY_BY_KIND.get(kind)
        if handler is not None:
            handler(self, conn, event, payload)
        # Again after the handler: completing one discussion exposes the next oldest pending
        # thread, whose own walk may already owe a held skip. The stop fence and FIFO order come
        # from the snapshot itself, so a stopped room still exposes no work here.
        self._record_walk_consumptions(conn, room_id)
        # No unconditional consumption here. Advancing every held member on every event carrying
        # a thread id discarded input addressed to somebody else: a member paused while its peer
        # was asked a question came back having never seen that question. A held member consumes
        # its delta only at the ordinary slot the round walk actually reached (see
        # `_record_walk_consumptions`), which is the native contract.

    def _ensure_cursor_and_projections(self, conn: sqlite3.Connection, room_id: str) -> int:
        """Create the room cursor if absent, migrate derived projections by bounded replay."""
        _require_room(conn, room_id)
        conn.execute("""INSERT OR IGNORE INTO hosted_room_policy_cursors(
                   room_id, through_seq, stopped_through_seq, updated_at
               ) VALUES (?, 0, 0, 0)""", (room_id,))
        cursor = int(
            conn.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()[
                "through_seq"])
        projection_state = conn.execute(
            "SELECT schema_version FROM hosted_room_policy_transcript_state WHERE room_id=?", (room_id,)).fetchone()
        if projection_state is None or int(projection_state["schema_version"]) < _PROJECTION_SCHEMA_VERSION:
            # Replay corrected partial-batch watermarks without rewriting admissions or the room
            # log. DERIVED state only: the canonical log, driver rows, task identities, receipts
            # and the publications table are untouched. Holds are rebuilt from the same replay as
            # every other projection, so a hold recorded now is never applied to older history.
            for table in ("hosted_room_policy_events", "hosted_room_policy_threads",
                          "hosted_room_policy_watermarks", "hosted_room_policy_transcript",
                          "hosted_room_policy_holds", "hosted_room_policy_citations",
                          "hosted_room_policy_citation_baseline"):
                conn.execute(f"DELETE FROM {table} WHERE room_id=?", (room_id,))
            conn.execute("UPDATE hosted_room_policy_cursors SET through_seq=0, stopped_through_seq=0 WHERE room_id=?",
                         (room_id,))
            # Returning 0 makes the existing `sync` machinery replay this room in bounded pages;
            # the old backfills are deliberately NOT run on top of that replay.
            cursor = 0
            conn.execute("""INSERT INTO hosted_room_policy_transcript_state(room_id, schema_version)
                   VALUES (?, ?)
                   ON CONFLICT(room_id) DO UPDATE SET schema_version=excluded.schema_version""",
                (room_id, _PROJECTION_SCHEMA_VERSION))
        return cursor

    def sync(self, *, room_id: str, latest_seq: int) -> int:
        """Materialize each unseen event exactly once by durable cursor."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = self._ensure_cursor_and_projections(conn, room_id)
        if cursor > latest_seq:
            with self._connect() as conn:
                room = conn.execute("SELECT next_seq FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
                if room is None:
                    raise hosted_rooms.RoomNotFoundError("hosted room not found")
                if cursor >= int(room["next_seq"]):
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
                current = conn.execute("SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?",
                                       (room_id,)).fetchone()
                if current is None:
                    raise RuntimeError("room policy cursor disappeared during replay")
                if int(current["through_seq"]) != cursor:
                    # Another worker applied the page: reload before any projection effects.
                    cursor = int(current["through_seq"])
                    continue
                for event in rows:
                    self._apply_event(conn, event)
                fenced_update(
                    conn, "UPDATE hosted_room_policy_cursors SET through_seq=?, updated_at=? WHERE room_id=?",
                    (next_cursor, float(rows[-1].get("created_at") or 0), room_id),
                    RuntimeError("room policy cursor disappeared during replay"))
            cursor = next_cursor
        return cursor

    @staticmethod
    def _holds(conn: sqlite3.Connection, room_id: str) -> tuple[MemberHold, ...]:
        return tuple(
            MemberHold(str(row["member_id"]), int(row["held_at_seq"]))
            for row in conn.execute("""SELECT member_id, held_at_seq
                   FROM hosted_room_policy_holds WHERE room_id=? ORDER BY member_id""", (room_id,)))

    def member_holds(self, *, room_id: str, latest_seq: int | None = None) -> tuple[MemberHold, ...]:
        """Return durable hold state; ``latest_seq`` first materializes the log up to it."""
        if latest_seq is not None:
            self.sync(room_id=room_id, latest_seq=latest_seq)
        with self._connect() as conn:
            return self._holds(conn, room_id)

    def snapshot(self, *, room_id: str, latest_seq: int) -> PolicySnapshot:
        """Return only the oldest active discussion, its watermark set and manual holds."""
        self.sync(room_id=room_id, latest_seq=latest_seq)
        with self._connect() as conn:
            conn.execute("BEGIN")
            return self._snapshot_on(conn, room_id)

    def _snapshot_on(self, conn: sqlite3.Connection, room_id: str) -> PolicySnapshot:
        """The bounded policy snapshot on an OPEN connection.

        Factored so the projection can hand the planner exactly the input a live drive gets --
        oldest active thread, its committed events, watermarks and holds -- without opening a
        second connection inside the replay transaction.
        """
        cursor = conn.execute(
            "SELECT through_seq, stopped_through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (room_id,)).fetchone()
        if cursor is None:
            raise hosted_rooms.RoomNotFoundError("hosted room checkpoint not found")
        through_seq = int(cursor["through_seq"])
        stopped_through_seq = int(cursor["stopped_through_seq"])
        holds = self._holds(conn, room_id)
        thread = conn.execute("""SELECT thread_id, discussion_event_id FROM hosted_room_policy_threads
               WHERE room_id=? AND completed=0 AND latest_user_seq>?
               ORDER BY latest_user_seq, thread_id LIMIT 1""", (room_id, stopped_through_seq)).fetchone()
        if thread is None:
            return PolicySnapshot(
                through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=(), watermarks={},
                holds=holds)
        thread_id = str(thread["thread_id"])
        events = self._discussion_events(
            conn, room_id=room_id, thread_id=thread_id, discussion_event_id=str(thread["discussion_event_id"]),
            bound_error="active room policy projection exceeded its bound")
        watermark_rows = conn.execute("""SELECT member_id, seen_through_seq FROM hosted_room_policy_watermarks
               WHERE room_id=? AND thread_id=?""", (room_id, thread_id)).fetchall()
        # Matched on the discussion id as well as the thread: a baseline left by a previous
        # discussion of this thread is never read by a newer one.
        citation_rows = conn.execute("""SELECT member_id, cited_at_seq, last_post_seq
               FROM hosted_room_policy_citation_baseline
               WHERE room_id=? AND thread_id=? AND discussion_event_id=?""",
            (room_id, thread_id, str(thread["discussion_event_id"]))).fetchall()
        return PolicySnapshot(
            through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=tuple(events),
            watermarks={
                (thread_id, str(row["member_id"])): int(row["seen_through_seq"])
                for row in watermark_rows},
            holds=holds,
            citations={
                str(row["member_id"]): (int(row["cited_at_seq"]), int(row["last_post_seq"]))
                for row in citation_rows})

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

    def events_for_task(self, *, room_id: str, source_event_seq: int,
                        input_context: Mapping[str, Any] | None = None,
                        task_id: str | None = None) -> list[dict[str, Any]]:
        """Load one bounded discussion projection for terminal reconstruction.

        A turn can outlive its discussion: deferred and uncertain work is not live work, so the
        discussion may settle and be compacted while that task is still unpublished. Its source
        message is nevertheless still in the durable log at this exact sequence, so it is read back
        by key and merged with the bounded thread transcript. Nothing is fabricated and nothing
        unbounded is retained: an outcome whose source event no longer exists still returns
        nothing beyond this task's own published events, and reconstruction still refuses it.
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            published = self._published_task_events(conn, room_id=room_id, task_id=task_id)
            if input_context is not None:
                context = validate_task_input(input_context)
                seqs = sorted({source_event_seq, *context["event_seqs"]})
                placeholders = ",".join("?" for _ in seqs)
                rows = conn.execute(f"SELECT * FROM hosted_room_events WHERE room_id=? AND seq IN ({placeholders}) ORDER BY seq",
                                    (room_id, *seqs)).fetchall()
                if len(rows) != len(seqs):
                    raise RuntimeError("admitted task input event is missing")
                events = {int(row["seq"]): _event_from_room_row(row) for row in rows}
                source_event = events[source_event_seq]
                thread_id = _text(source_event["payload"], "thread_id")
                if source_event["kind"] != "message.user" or not thread_id:
                    raise RuntimeError("admitted task source is not a user discussion")
                # Supersession remains authoritative after the display projection compacts.
                latest = conn.execute("""SELECT * FROM hosted_room_events
                    WHERE room_id=? AND kind='message.user' AND json_extract(payload_json, '$.thread_id')=?
                    ORDER BY seq DESC LIMIT 1""", (room_id, thread_id)).fetchone()
                if latest is not None:
                    events[int(latest["seq"])] = _event_from_room_row(latest)
                for event in published:
                    events[int(event["seq"])] = event
                return [events[seq] for seq in sorted(events)]
            source = conn.execute(
                "SELECT discussion_event_id, thread_id FROM hosted_room_policy_events WHERE room_id=? AND seq=?",
                (room_id, source_event_seq)).fetchone()
            projection: list[dict[str, Any]] = []
            if source is not None:
                projection = list(self._discussion_events(
                    conn, room_id=room_id, thread_id=str(source["thread_id"]),
                    discussion_event_id=str(source["discussion_event_id"]),
                    bound_error="task policy projection exceeded its bound"))
            elif ((committed := _canonical_event(conn, room_id, seq=source_event_seq)) is not None
                    and committed.get("kind") == "message.user"
                    and (thread_id := _text(committed["payload"], "thread_id"))):
                # Compacted discussion: the durable log still holds the source message, so the
                # bounded thread transcript is merged with it. The transcript window may have
                # rolled past that message, and the task still needs exactly it to reconstruct
                # its own coordinates.
                projection = [
                    *self._discussion_events(
                        conn, room_id=room_id, thread_id=thread_id,
                        discussion_event_id=str(committed["event_id"]),
                        bound_error="task policy projection exceeded its bound"),
                    committed,
                ]
            # This task's own published events survive BOTH branches, including the one where the
            # source event is gone: dropping them would lose the outcome already in the room.
            events = {int(event["seq"]): event for event in (*published, *projection)}
            return [events[seq] for seq in sorted(events)]

    def compact_completed(self, *, room_id: str) -> None:
        """Drop any completed projections left by an interrupted sync."""
        with self._connect() as conn:
            for row in conn.execute(
                "SELECT discussion_event_id FROM hosted_room_policy_threads WHERE room_id=? AND completed=1", (room_id,)
            ).fetchall():
                conn.execute(_DELETE_ACTIVE_EVENTS_SQL, (room_id, str(row["discussion_event_id"])))
            conn.execute("DELETE FROM hosted_room_policy_threads WHERE room_id=? AND completed=1", (room_id,))


    def _published_task_events(
        self,
        conn: sqlite3.Connection,
        *,
        room_id: str,
        task_id: str | None,
    ) -> list[dict[str, Any]]:
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
            raise RuntimeError("published task receipt identity changed")
        return events
