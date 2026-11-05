"""Contract cases for ``persistence.gateway.hosted_rooms.HostedRoomPolicyRepository``, the room policy checkpoint.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs: room events go in through ``HostedRoomRepository.append_event`` and the checkpoint reads them back through its own cursor. The cases pin these rules: each event is applied once by a durable cursor; the snapshot is the oldest active discussion after the stop fence, merged from the thread transcript and the active projection, with its member watermarks; terminal outcomes leave publication receipts.
"""

from __future__ import annotations

import pytest

from persistence.gateway.hosted_rooms import PolicyProjectionError, PolicySnapshot, RoomNotFoundError

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
ROOM = "room"
USER = {"kind": "user", "id": "u1"}
MEMBER = {"kind": "member", "id": "m1"}


def _room(store, placement: str = "profile") -> None:
    store.hosted_rooms.create_room(placement=placement, room_id=ROOM, name="Room", members=MEMBERS, authority_gateway_id="gw", now=100.0)


def _append(store, event_id: str, kind: str, payload: dict, *, actor: dict | None = None, placement: str = "profile") -> int:
    event = store.hosted_rooms.append_event(
        placement=placement, room_id=ROOM, event_id=event_id, kind=kind, actor=actor or USER, payload=payload,
        authority_gateway_id="gw", authority_epoch=1, now=100.0)
    return int(event["seq"])


def _user(store, event_id: str, thread_id: str = "t1", **extra) -> int:
    return _append(store, event_id, "message.user", {"thread_id": thread_id, "text": event_id, **extra})


def _logged(store, *seqs: int) -> tuple:
    events = store.hosted_rooms.read_events(placement="profile", room_id=ROOM, since_seq=0, limit=500)["events"]
    return tuple(event for event in events if event["seq"] in seqs)


def _snapshot(store, latest_seq: int, placement: str = "profile") -> PolicySnapshot:
    return store.hosted_room_policy.snapshot(placement=placement, room_id=ROOM, latest_seq=latest_seq)


# ── snapshot ──────────────────────────────────────────────────────────────────


def test_snapshot_projects_the_oldest_active_discussion_with_its_transcript(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    assert _snapshot(store, 0) == PolicySnapshot(through_seq=0, stopped_through_seq=0, events=(), watermarks={})
    _user(store, "u1")
    _user(store, "u2")
    _user(store, "v1", thread_id="t2")
    # CONTRACT: the newest user message opens the thread's active discussion; earlier ones come from the thread transcript.
    assert _snapshot(store, 3) == PolicySnapshot(through_seq=3, stopped_through_seq=0, events=_logged(store, 1, 2), watermarks={})
    _append(store, "a1", "room.activity", {"thread_id": "t1", "discussion_event_id": "u2"})
    assert _snapshot(store, 4) == PolicySnapshot(through_seq=4, stopped_through_seq=0, events=_logged(store, 3), watermarks={})


def test_snapshot_projects_the_leads_assignment_into_its_discussion(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _user(store, "u1")
    _append(store, "assign-1", "room.assignment", {"thread_id": "t1", "discussion_event_id": "u1", "task_id": "task-1", "lead_member_id": "m1", "member_ids": ["m1"], "fallback": False}, actor={"kind": "gateway", "id": "gw"})
    # CONTRACT: a room.assignment of a known discussion joins its active events, so the planner sees who takes round 0.
    assert _snapshot(store, 2).events == _logged(store, 1, 2)


def test_snapshot_applies_each_event_once_by_its_durable_cursor(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _user(store, "u1")
    assert _snapshot(store, 1).through_seq == 1
    _append(store, "m1", "message.member", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1"}, actor=MEMBER)
    # CONTRACT: a latest_seq behind the cursor reads the cursor without replaying (the log is not behind it).
    assert _snapshot(store, 0).events == _logged(store, 1)
    assert _snapshot(store, 2).events == _logged(store, 1, 2)
    assert _snapshot(store, 2) == _snapshot(store, 1)


def test_snapshot_watermarks_and_the_stop_fence(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _user(store, "u1")
    member = _append(store, "m1", "message.member", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1"}, actor=MEMBER)
    _append(store, "s1", "turn.settled", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1", "task_id": "task-1", "message_event_id": "m1", "seen_through_seq": 1})
    _append(store, "f1", "turn.failed", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m2", "task_id": "task-2", "seen_through_seq": 2})
    snapshot = _snapshot(store, 4)
    # CONTRACT: a settled turn's committed member message raises its member's watermark past the receipt's own seen_through_seq.
    assert snapshot.watermarks == {("t1", "m1"): member, ("t1", "m2"): 2}
    assert snapshot.events == _logged(store, 1, 2, 3, 4)
    stop = _append(store, "stop", "room.stop_requested", {"cancel_id": "c1"}, actor={"kind": "system", "id": "authority-control"})
    assert _snapshot(store, stop) == PolicySnapshot(through_seq=stop, stopped_through_seq=stop, events=(), watermarks={})
    _user(store, "u2")
    assert _snapshot(store, stop + 1).events == _logged(store, 1, 2, 3, 4, stop + 1)


def test_snapshot_refuses_a_missing_room_a_stalled_cursor_and_an_overflowing_projection(backend, scope_a):
    store = backend.open(scope_a)
    with pytest.raises(RoomNotFoundError, match="^hosted room not found$"):
        _snapshot(store, 0)
    _room(store)
    _user(store, "u1")
    # CONTRACT: a latest_seq past the durable log stops at the first empty page.
    with pytest.raises(PolicyProjectionError, match="^hosted room policy cursor did not advance$"):
        _snapshot(store, 5)
    assert _snapshot(store, 1).through_seq == 1
    for n in range(64):
        _append(store, f"m{n}", "message.member", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1"}, actor=MEMBER)
    with pytest.raises(PolicyProjectionError, match="^active room policy projection exceeded its bound$"):
        _snapshot(store, 65)
    with pytest.raises(PolicyProjectionError, match="^task policy projection exceeded its bound$"):
        store.hosted_room_policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=1)


def _discussion(store, placement: str = "profile") -> None:
    """A room whose discussion holds 65 active events: the user message and 64 member replies."""
    _room(store, placement)
    _append(store, "u1", "message.user", {"thread_id": "t1"}, placement=placement)
    for n in range(64):
        _append(store, f"m{n}", "message.member", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1"},
                actor=MEMBER, placement=placement)


def test_a_configured_budget_above_the_default_is_recorded_and_read_back(backend, scope_a):
    store = backend.open(scope_a)
    policy = store.hosted_room_policy
    _discussion(store)
    for budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="^max_active_events must be a positive integer$"):
            policy.snapshot(placement="profile", room_id=ROOM, latest_seq=65, max_active_events=budget)
    # CONTRACT: a budget above the default bounds the active discussion by that budget.
    assert len(policy.snapshot(placement="profile", room_id=ROOM, latest_seq=65, max_active_events=65).events) == 65
    # CONTRACT: the file keeps the largest budget given: verbs called with a lower budget read it back.
    assert len(_snapshot(store, 65).events) == 65
    assert len(policy.snapshot(placement="profile", room_id=ROOM, latest_seq=65, max_active_events=1).events) == 65
    assert len(policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=1)) == 65
    # CONTRACT: each placement's file records its own capacity.
    _discussion(store, "shared")
    with pytest.raises(PolicyProjectionError, match="^active room policy projection exceeded its bound$"):
        _snapshot(store, 65, placement="shared")


def test_snapshot_keeps_the_placements_apart(backend, scope_a, scope_b):
    store, other = backend.open(scope_a), backend.open(scope_b)
    _room(store, placement="shared")
    _append(store, "u1", "message.user", {"thread_id": "t1"}, placement="shared")
    with pytest.raises(RoomNotFoundError, match="^hosted room not found$"):
        _snapshot(store, 1)
    assert _snapshot(store, 1, placement="shared").through_seq == 1
    # CONTRACT: the shared checkpoint is one per install: another profile reads the cursor the first one advanced.
    assert _snapshot(other, 0, placement="shared").through_seq == 1
    _room(store)
    assert _snapshot(store, 0) == PolicySnapshot(through_seq=0, stopped_through_seq=0, events=(), watermarks={})


# ── compact_completed and publication_exists ──────────────────────────────────


def test_compact_completed_keeps_active_discussions_and_needs_no_room(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_room_policy.compact_completed(placement="profile", room_id="missing")
    _room(store)
    _user(store, "u1")
    before = _snapshot(store, 1)
    store.hosted_room_policy.compact_completed(placement="profile", room_id=ROOM)
    assert _snapshot(store, 1) == before
    assert before.events == _logged(store, 1)


def test_publication_exists_matches_terminal_outcomes_and_exact_deferrals(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _append(store, "o1", "turn.cancelled", {"task_id": "task-1", "thread_id": "t1"})
    _append(store, "o2", "turn.deferred", {"task_id": "task-2", "thread_id": "t1", "execution_generation": 3})
    _snapshot(store, 2)

    def exists(task_id: str, status: str, generation: int) -> bool:
        return store.hosted_room_policy.publication_exists(placement="profile", room_id=ROOM, task_id=task_id, status=status, execution_generation=generation)

    # CONTRACT: any terminal kind answers for any non-deferred status, whatever the generation.
    assert exists("task-1", "settled", 7) is True
    assert exists("task-1", "deferred", 0) is False
    assert exists("task-2", "deferred", 3) is True
    assert exists("task-2", "deferred", 2) is False
    assert exists("task-2", "failed", 3) is False


# ── events_for_task ───────────────────────────────────────────────────────────


def test_events_for_task_reads_the_discussion_and_the_published_receipts(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _user(store, "u1")
    _append(store, "m1", "message.member", {"thread_id": "t1", "discussion_event_id": "u1"}, actor=MEMBER)
    _snapshot(store, 2)
    _append(store, "dmessage:abc", "message.member", {"thread_id": "t1", "discussion_event_id": "other", "task_id": "dtask:abc"}, actor=MEMBER)
    _append(store, "dterminal:abc", "turn.settled", {"thread_id": "t1", "task_id": "dtask:abc"})
    policy = store.hosted_room_policy
    assert policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=1) == _logged(store, 1, 2)
    assert policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=1, task_id="dtask:abc") == _logged(store, 1, 2, 3, 4)
    # CONTRACT: a source that is not a user message reads only the published receipts.
    assert policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=2, task_id="dtask:abc") == _logged(store, 3, 4)
    assert policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=9) == ()
    _append(store, "dmessage:bad", "message.member", {"thread_id": "t1", "task_id": "dtask:other"}, actor=MEMBER)
    with pytest.raises(PolicyProjectionError, match="^published task receipt identity changed$"):
        policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=1, task_id="bad")


def test_events_for_task_with_an_input_context_reads_the_admitted_events(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _user(store, "u1")
    _append(store, "m1", "message.member", {"thread_id": "t1", "discussion_event_id": "u1"}, actor=MEMBER)
    _user(store, "u2")
    _user(store, "v1", thread_id="t2")

    def read(source: int, context: dict, task_id: str | None = None):
        return store.hosted_room_policy.events_for_task(placement="profile", room_id=ROOM, source_event_seq=source, input_context=context, task_id=task_id)

    # CONTRACT: the context's events and the thread's newest user message, read from the log without any snapshot.
    assert read(1, {"watermark": 0, "event_seqs": [2]}) == _logged(store, 1, 2, 3)
    with pytest.raises(ValueError, match="^invalid task input context$"):
        read(1, {"event_seqs": [2]})
    with pytest.raises(PolicyProjectionError, match="^admitted task input event is missing$"):
        read(1, {"watermark": 0, "event_seqs": [9]})
    with pytest.raises(PolicyProjectionError, match="^admitted task source is not a user discussion$"):
        read(2, {"watermark": 0, "event_seqs": [3]})


def test_snapshot_transcript_keeps_the_latest_settlement_of_a_committed_message(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    _user(store, "u1")
    _append(store, "m1", "message.member", {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1"}, actor=MEMBER)
    settled = {"thread_id": "t1", "discussion_event_id": "u1", "member_id": "m1", "message_event_id": "m1"}
    _append(store, "s1", "turn.settled", {**settled, "task_id": "task-1"})
    _append(store, "s2", "turn.settled", {**settled, "task_id": "task-2"})
    _append(store, "a1", "room.activity", {"thread_id": "t1", "discussion_event_id": "u1"})
    _user(store, "u2")
    # CONTRACT: the transcript stores the committed message with the newest settlement that named it; the closed discussion's active events are gone.
    assert _snapshot(store, 6).events == _logged(store, 1, 2, 4, 6)
