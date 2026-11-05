"""In-memory reference ``HostedRoomPolicyRepository``: the room policy checkpoint over the placement's hosted-room tables.

The checkpoint tables live under ``hosted_room_tables``: ``hosted_room_policy_cursors`` and ``hosted_room_policy_transcript_state`` map ``room_id`` to one row; ``hosted_room_policy_threads[room_id][thread_id]``, ``hosted_room_policy_events[room_id][seq]``, ``hosted_room_policy_watermarks[room_id][(thread_id, member_id)]``, ``hosted_room_policy_publications[room_id][(task_id, kind, execution_generation)]`` and ``hosted_room_policy_transcript[room_id][(thread_id, seq)]`` hold the per-room rows under their primary keys. ``hosted_room_policy_capacity[1]`` is the file's recorded active-event capacity, which every verb raises to its budget first and never lowers. ``snapshot`` runs this sequence: the cursor write, the ahead-of-log read, one room-log page read (``HostedRoomRepository.read_events``) and one projection write per page, then the snapshot read.

Reference limit: every verb holds the backend lock for its whole sequence, so no other writer moves the cursor between a page read and its write; the adopt-and-reread branch is kept but never taken.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from persistence import JsonValue
from persistence.gateway.hosted_rooms import HostedPlacement, PolicyProjectionError, PolicySnapshot, RoomNotFoundError
from session_rules.hosted_tasks import compact_json, validate_task_input

from ..backend import register_repository
from .hosted_room_tables import HostedTables, event_from_row, find_event, read_tables, room_events, write_tables

__all__ = ["InMemoryHostedRoomPolicyRepository"]

# Projection bounds.
MAX_ACTIVE_POLICY_EVENTS = 64
# Retry receipts (``turn.deferred``) are bounded apart from the discussion's other events.
MAX_ACTIVE_RETRY_RECEIPTS = 64
MAX_THREAD_TRANSCRIPT_EVENTS = 24
_TRANSCRIPT_SCHEMA_VERSION = 5
_TERMINAL_KINDS = frozenset({"turn.settled", "turn.failed", "turn.cancelled", "turn.deferred"})
# What a Continue restores into a reopened discussion's active projection (the cap's activity is not one).
_REOPENED_KINDS = frozenset({*_TERMINAL_KINDS, "message.member", "room.assignment", "room.routed", "room.continued"})
# The room log's page limit.
MAX_LOG_LIMIT = 500
_MESSAGE_KINDS = frozenset({"message.user", "message.member"})
_DERIVED_TABLES = ("hosted_room_policy_events", "hosted_room_policy_threads", "hosted_room_policy_watermarks", "hosted_room_policy_transcript")

Row = dict[str, Any]


def _text(mapping: Mapping[str, Any], key: str) -> str:
    return str(mapping.get(key) or "")


def _require_room(tables: HostedTables, room_id: str) -> None:
    """Any room row, disbanded or not."""
    if room_id not in tables["hosted_rooms"]:
        raise RoomNotFoundError("hosted room not found")


def _settled_message(tables: HostedTables, room_id: str, discussion_event_id: str, message_event_id: Any) -> dict[str, Any] | None:
    """The committed member message of this discussion, even after compaction."""
    row = find_event(tables, room_id, message_event_id)
    if row is None:
        return None
    event = event_from_row(row)
    if event["kind"] != "message.member" or _text(event["payload"], "discussion_event_id") != discussion_event_id:
        return None
    return event


def _store_active_event(tables: HostedTables, *, event: Mapping[str, Any], thread_id: str, discussion_event_id: str) -> None:
    """``INSERT OR IGNORE`` of the event's JSON into the active projection."""
    tables.rows("hosted_room_policy_events", event["room_id"]).setdefault(int(event["seq"]), {
        "room_id": event["room_id"], "thread_id": thread_id, "discussion_event_id": discussion_event_id,
        "seq": int(event["seq"]), "event_json": compact_json(dict(event))})


def _store_transcript_event(tables: HostedTables, *, event: Mapping[str, Any], thread_id: str, settled_seq: int | None = None) -> None:
    """Upsert the reference (keeping a stored ``settled_seq`` when none is given), then keep only the thread's newest 24 messages and what follows the oldest of them."""
    rows = tables.rows("hosted_room_policy_transcript", event["room_id"])
    key = (thread_id, int(event["seq"]))
    stored = rows.get(key)
    if stored is None:
        rows[key] = {"room_id": event["room_id"], "thread_id": thread_id, "seq": int(event["seq"]), "kind": str(event["kind"]), "settled_seq": settled_seq}
    elif settled_seq is not None:
        stored["settled_seq"] = settled_seq
    if event["kind"] not in _MESSAGE_KINDS:
        return
    messages = sorted((row["seq"] for row in rows.values() if row["thread_id"] == thread_id and row["kind"] in _MESSAGE_KINDS), reverse=True)
    if len(messages) >= MAX_THREAD_TRANSCRIPT_EVENTS:
        cutoff = messages[MAX_THREAD_TRANSCRIPT_EVENTS - 1]
        for stale in [key for key, row in rows.items() if row["thread_id"] == thread_id and row["seq"] < cutoff]:
            del rows[stale]


def _discussion_events(tables: HostedTables, *, room_id: str, thread_id: str, discussion_event_id: str, bound_error: str, capacity: int) -> list[dict[str, Any]]:
    """The thread transcript's room events merged with the active projection, by seq, each bounded; ``capacity`` bounds the active events other than retry receipts."""
    active = sorted((row for row in tables["hosted_room_policy_events"].get(room_id, {}).values() if row["discussion_event_id"] == discussion_event_id), key=lambda row: row["seq"])
    receipts = [row for row in active if json.loads(row["event_json"]).get("kind") == "turn.deferred"]
    if len(active) - len(receipts) > capacity or len(receipts) > MAX_ACTIVE_RETRY_RECEIPTS:
        raise PolicyProjectionError(bound_error)
    transcript_bound = min(2**63 - 2, MAX_THREAD_TRANSCRIPT_EVENTS * (capacity + MAX_ACTIVE_RETRY_RECEIPTS + 2))
    transcript = [row for row in tables["hosted_room_policy_transcript"].get(room_id, {}).values() if row["thread_id"] == thread_id]
    seqs = [row["seq"] for row in transcript] + [row["settled_seq"] for row in transcript if row["settled_seq"] is not None]
    log = tables["hosted_room_events"].get(room_id, {})
    joined = sorted((log[seq] for seq in seqs if seq in log), key=lambda row: row["seq"])[:transcript_bound + 1]
    if len(joined) > transcript_bound:
        raise PolicyProjectionError("thread policy transcript exceeded its bound")
    by_seq = {int(event["seq"]): event for event in (*map(event_from_row, joined), *(json.loads(row["event_json"]) for row in active))}
    return [by_seq[seq] for seq in sorted(by_seq)]


def _apply_user_message(tables: HostedTables, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """A user message opens its thread's discussion and enters the active projection and the transcript."""
    room_id = str(event["room_id"])
    thread_id, event_id = _text(payload, "thread_id"), _text(event, "event_id")
    if not thread_id or not event_id:
        return
    tables.rows("hosted_room_policy_threads", room_id)[thread_id] = {
        "room_id": room_id, "thread_id": thread_id, "discussion_event_id": event_id, "latest_user_seq": int(event["seq"]), "completed": 0}
    _store_active_event(tables, event=event, thread_id=thread_id, discussion_event_id=event_id)
    _store_transcript_event(tables, event=event, thread_id=thread_id)


def _apply_discussion_event(tables: HostedTables, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Member messages join a known active discussion; terminal outcomes record their publication receipt, transcript reference and member watermark."""
    room_id, seq, kind = str(event["room_id"]), int(event["seq"]), _text(event, "kind")
    thread_id, discussion_event_id = _text(payload, "thread_id"), _text(payload, "discussion_event_id")
    if any(row["discussion_event_id"] == discussion_event_id for row in tables["hosted_room_policy_events"].get(room_id, {}).values()):
        _store_active_event(tables, event=event, thread_id=thread_id, discussion_event_id=discussion_event_id)
    if kind not in _TERMINAL_KINDS:
        return
    task_id = _text(payload, "task_id")
    execution_generation = int(payload.get("execution_generation") or 0) if kind == "turn.deferred" else 0
    if task_id:
        tables.rows("hosted_room_policy_publications", room_id).setdefault((task_id, kind, execution_generation), {
            "room_id": room_id, "task_id": task_id, "kind": kind, "execution_generation": execution_generation, "seq": seq})
    member_id = _text(payload, "member_id")
    seen_through_seq = int(payload.get("seen_through_seq") or 0)
    if kind == "turn.settled" and payload.get("message_event_id"):
        committed = _settled_message(tables, room_id, discussion_event_id, payload["message_event_id"])
        if committed is not None:
            seen_through_seq = max(seen_through_seq, int(committed["seq"]))
            _store_transcript_event(tables, event=committed, thread_id=thread_id, settled_seq=seq)
    else:
        _store_transcript_event(tables, event=event, thread_id=thread_id)
    if member_id and seen_through_seq > 0:
        watermarks = tables.rows("hosted_room_policy_watermarks", room_id)
        stored = watermarks.get((thread_id, member_id))
        watermarks[(thread_id, member_id)] = {
            "room_id": room_id, "thread_id": thread_id, "member_id": member_id,
            "seen_through_seq": seen_through_seq if stored is None else max(int(stored["seen_through_seq"]), seen_through_seq)}


def _apply_room_activity(tables: HostedTables, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Drop the discussion's active events, and its thread row while the thread still belongs to that discussion (a newer message may have reopened it under its own)."""
    room_id, discussion_event_id = str(event["room_id"]), _text(payload, "discussion_event_id")
    active = tables["hosted_room_policy_events"].get(room_id, {})
    for seq in [seq for seq, row in active.items() if row["discussion_event_id"] == discussion_event_id]:
        del active[seq]
    threads, thread_id = tables["hosted_room_policy_threads"].get(room_id, {}), _text(payload, "thread_id")
    if threads.get(thread_id, {}).get("discussion_event_id") == discussion_event_id:
        del threads[thread_id]


def _apply_continued(tables: HostedTables, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Continue reopens a discussion its cap completed: the source user message's thread row and active event come back, with the discussion's reopened-kind events up to the Continue; a newer user message on the thread before the Continue owns the thread, and nothing comes back."""
    room_id, discussion_event_id = str(event["room_id"]), _text(payload, "discussion_event_id")
    source = find_event(tables, room_id, discussion_event_id)
    if source is None or source["kind"] != "message.user":
        return
    source_event = event_from_row(source)
    thread_id = _text(source_event["payload"], "thread_id")
    if any(int(source_event["seq"]) < int(row["seq"]) <= int(event["seq"]) and row["kind"] == "message.user"
           and event_from_row(row)["payload"].get("thread_id") == thread_id for row in room_events(tables, room_id)):
        return
    tables.rows("hosted_room_policy_threads", room_id)[thread_id] = {
        "room_id": room_id, "thread_id": thread_id, "discussion_event_id": discussion_event_id, "latest_user_seq": int(source_event["seq"]), "completed": 0}
    _store_active_event(tables, event=source_event, thread_id=thread_id, discussion_event_id=discussion_event_id)
    for row in room_events(tables, room_id):
        restored = event_from_row(row)
        if (int(source_event["seq"]) < int(row["seq"]) <= int(event["seq"]) and row["kind"] in _REOPENED_KINDS
                and restored["payload"].get("discussion_event_id") == discussion_event_id):
            _store_active_event(tables, event=restored, thread_id=thread_id, discussion_event_id=discussion_event_id)


def _apply_stop_requested(tables: HostedTables, event: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """A stop request advances the cursor's ``stopped_through_seq`` to the event."""
    cursor = tables["hosted_room_policy_cursors"].get(str(event["room_id"]))
    if cursor is not None:
        cursor["stopped_through_seq"] = max(int(cursor["stopped_through_seq"]), int(event["seq"]))


# Event kind -> projection step.
_APPLY_BY_KIND = {
    "message.user": _apply_user_message, "message.member": _apply_discussion_event,
    **dict.fromkeys(_TERMINAL_KINDS, _apply_discussion_event), "room.activity": _apply_room_activity,
    "room.assignment": _apply_discussion_event,
    "room.routed": _apply_discussion_event, "room.continued": _apply_continued,
    "room.stop_requested": _apply_stop_requested}


def _apply_event(tables: HostedTables, event: Mapping[str, Any]) -> None:
    handler = _APPLY_BY_KIND.get(_text(event, "kind"))
    if handler is not None:
        payload = event.get("payload")
        handler(tables, event, payload if isinstance(payload, Mapping) else {})


def _ensure_cursor_and_transcript(tables: HostedTables, room_id: str) -> int:
    """Create the cursor; below transcript schema 4, clear the derived rows and replay from 0."""
    _require_room(tables, room_id)
    cursor = tables["hosted_room_policy_cursors"].setdefault(room_id, {"room_id": room_id, "through_seq": 0, "stopped_through_seq": 0, "updated_at": 0})
    state = tables["hosted_room_policy_transcript_state"].get(room_id)
    if state is None or int(state["schema_version"]) < _TRANSCRIPT_SCHEMA_VERSION:
        for name in _DERIVED_TABLES:
            tables[name].pop(room_id, None)
        cursor.update(through_seq=0, stopped_through_seq=0)
        tables["hosted_room_policy_transcript_state"][room_id] = {"room_id": room_id, "schema_version": _TRANSCRIPT_SCHEMA_VERSION}
    return int(cursor["through_seq"])


def _published_task_events(tables: HostedTables, *, room_id: str, task_id: str | None) -> list[dict[str, Any]]:
    """The task's ``dmessage:`` and ``dterminal:`` receipts, which must name the task."""
    if not task_id:
        return []
    digest = task_id.removeprefix("dtask:")
    names = {f"dmessage:{digest}", f"dterminal:{digest}"}
    events = [event_from_row(row) for row in room_events(tables, room_id) if row["event_id"] in names]
    if any(event["payload"].get("task_id") != task_id for event in events):
        raise PolicyProjectionError("published task receipt identity changed")
    return events


def _latest_user_message(tables: HostedTables, room_id: str, thread_id: str) -> Row | None:
    """The newest ``message.user`` row whose payload ``thread_id`` is ``thread_id``."""
    rows = [row for row in room_events(tables, room_id) if row["kind"] == "message.user"]
    return next((row for row in reversed(rows) if (payload := json.loads(row["payload_json"])) and isinstance(payload, dict) and payload.get("thread_id") == thread_id), None)


class InMemoryHostedRoomPolicyRepository:
    """The room policy checkpoint on either placement."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def _capacity(self, placement: HostedPlacement, max_active_events: int) -> int:
        """Check the budget, raise the file's recorded capacity to it, and return the recorded capacity."""
        if isinstance(max_active_events, bool) or not isinstance(max_active_events, int) or max_active_events < 1:
            raise ValueError("max_active_events must be a positive integer")
        budget = min(2**63 - 2, max(MAX_ACTIVE_POLICY_EVENTS, max_active_events))
        with write_tables(self._store, placement) as tables:
            recorded = tables["hosted_room_policy_capacity"]
            recorded[1] = max(recorded.get(1, budget), budget)
            capacity: int = recorded[1]
        return capacity

    def _sync(self, placement: HostedPlacement, room_id: str, latest_seq: int) -> None:
        """Bring the projection up to ``latest_seq``: the cursor, the ahead-of-log check, then one projection write per room-log page."""
        with write_tables(self._store, placement) as tables:
            cursor = _ensure_cursor_and_transcript(tables, room_id)
        if cursor > latest_seq:
            room = read_tables(self._store, placement)["hosted_rooms"].get(room_id)
            if room is None:
                raise RoomNotFoundError("hosted room not found")
            if cursor >= int(room["next_seq"]):
                raise PolicyProjectionError("room policy cursor is ahead of the durable log")
        while cursor < latest_seq:
            page = self._store.hosted_rooms.read_events(placement=placement, room_id=room_id, since_seq=cursor, limit=MAX_LOG_LIMIT)
            rows = [event for event in page.get("events", []) if isinstance(event, Mapping)]
            next_cursor = int(page.get("cursor") or cursor)
            if not rows or next_cursor <= cursor:
                raise PolicyProjectionError("hosted room policy cursor did not advance")
            with write_tables(self._store, placement) as tables:
                _require_room(tables, room_id)
                current = tables["hosted_room_policy_cursors"].get(room_id)
                if current is None:
                    raise PolicyProjectionError("room policy cursor disappeared during replay")
                if int(current["through_seq"]) != cursor:
                    cursor = int(current["through_seq"])
                    continue
                for event in rows:
                    _apply_event(tables, event)
                current.update(through_seq=next_cursor, updated_at=float(rows[-1].get("created_at") or 0))
            cursor = next_cursor

    def snapshot(self, *, placement: HostedPlacement, room_id: str, latest_seq: int, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> PolicySnapshot:
        capacity = self._capacity(placement, max_active_events)
        self._sync(placement, room_id, latest_seq)
        tables = read_tables(self._store, placement)
        cursor = tables["hosted_room_policy_cursors"].get(room_id)
        if cursor is None:
            raise RoomNotFoundError("hosted room checkpoint not found")
        through_seq, stopped_through_seq = int(cursor["through_seq"]), int(cursor["stopped_through_seq"])
        pending = [row for row in tables["hosted_room_policy_threads"].get(room_id, {}).values()
                   if row["completed"] == 0 and row["latest_user_seq"] > stopped_through_seq]
        if not pending:
            return PolicySnapshot(through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=(), watermarks={})
        thread = min(pending, key=lambda row: (row["latest_user_seq"], row["thread_id"]))
        thread_id = str(thread["thread_id"])
        events = _discussion_events(
            tables, room_id=room_id, thread_id=thread_id, discussion_event_id=str(thread["discussion_event_id"]),
            bound_error="active room policy projection exceeded its bound", capacity=capacity)
        watermarks = {(thread_id, str(row["member_id"])): int(row["seen_through_seq"])
                      for row in tables["hosted_room_policy_watermarks"].get(room_id, {}).values() if row["thread_id"] == thread_id}
        return PolicySnapshot(through_seq=through_seq, stopped_through_seq=stopped_through_seq, events=tuple(events), watermarks=watermarks)

    def compact_completed(self, *, placement: HostedPlacement, room_id: str, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> None:
        self._capacity(placement, max_active_events)
        with write_tables(self._store, placement) as tables:
            threads = tables["hosted_room_policy_threads"].get(room_id, {})
            completed = [thread_id for thread_id, row in threads.items() if row["completed"] == 1]
            active = tables["hosted_room_policy_events"].get(room_id, {})
            for thread_id in completed:
                discussion_event_id = str(threads[thread_id]["discussion_event_id"])
                for seq in [seq for seq, row in active.items() if row["discussion_event_id"] == discussion_event_id]:
                    del active[seq]
            for thread_id in completed:
                del threads[thread_id]

    def publication_exists(self, *, placement: HostedPlacement, room_id: str, task_id: str, status: str, execution_generation: int, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> bool:
        self._capacity(placement, max_active_events)
        keys = read_tables(self._store, placement)["hosted_room_policy_publications"].get(room_id, {})
        if status == "deferred":
            return (task_id, f"turn.{status}", execution_generation) in keys
        return any(key[0] == task_id and key[1] in {"turn.settled", "turn.failed", "turn.cancelled"} for key in keys)

    def events_for_task(self, *, placement: HostedPlacement, room_id: str, source_event_seq: int, input_context: Mapping[str, JsonValue] | None = None, task_id: str | None = None, max_active_events: int = MAX_ACTIVE_POLICY_EVENTS) -> tuple[Mapping[str, JsonValue], ...]:
        capacity = self._capacity(placement, max_active_events)
        tables = read_tables(self._store, placement)
        published = _published_task_events(tables, room_id=room_id, task_id=task_id)
        log = tables["hosted_room_events"].get(room_id, {})
        if input_context is not None:
            context = validate_task_input(input_context)
            seqs = sorted({source_event_seq, *context["event_seqs"]})
            if any(seq not in log for seq in seqs):
                raise PolicyProjectionError("admitted task input event is missing")
            events = {seq: event_from_row(log[seq]) for seq in seqs}
            source = events[source_event_seq]
            thread_id = _text(source["payload"], "thread_id")
            if source["kind"] != "message.user" or not thread_id:
                raise PolicyProjectionError("admitted task source is not a user discussion")
            latest = _latest_user_message(tables, room_id, thread_id)
            if latest is not None:
                events[int(latest["seq"])] = event_from_row(latest)
            for event in published:
                events[int(event["seq"])] = event
            return tuple(events[seq] for seq in sorted(events))
        row = log.get(source_event_seq)
        if row is None or row["kind"] != "message.user":
            return tuple(published)
        source = event_from_row(row)
        discussion = _discussion_events(
            tables, room_id=room_id, thread_id=_text(source["payload"], "thread_id"), discussion_event_id=str(source["event_id"]),
            bound_error="task policy projection exceeded its bound", capacity=capacity)
        by_seq = {event["seq"]: event for event in (*discussion, *published)}
        by_seq[source_event_seq] = source
        return tuple(by_seq[seq] for seq in sorted(by_seq))


register_repository("hosted_room_policy", InMemoryHostedRoomPolicyRepository)
