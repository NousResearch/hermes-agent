"""The bounded policy transcript window must stay referentially closed.

Regression: a thread longer than ``MAX_THREAD_TRANSCRIPT_EVENTS`` message rows trimmed the
``message.user`` event that a surviving ``turn.settled`` names, so every poll raised
``turn.settled references no matching user discussion`` and the room never dispatched again —
no member reply, no activity line, and the group chat looked dead. The count trim is the one
authority for what the policy projection can see, so it now keeps each retained settlement's
discussion; a window written by the old trim reads as a compacted prefix instead of
corruption, and an already-wedged cache rebuilds on the next prepare.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import (
    MAX_THREAD_TRANSCRIPT_EVENTS,
    HostedRoomPolicyCheckpoint,
)


ROOM_ID = "room-window"
GATEWAY_ID = "gateway-a"
LOCAL_PROFILES = ("writer", "reviewer")
MEMBERS = [
    {"member_id": f"member-{profile}", "profile": profile, "handle": profile}
    for profile in LOCAL_PROFILES
]


@pytest.fixture
def room_db(tmp_path: Path) -> tuple[Path, dict]:
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Audit", members=MEMBERS,
        authority_gateway_id=GATEWAY_ID, now=1)
    return db, room


def _events(db: Path) -> list[dict]:
    return hosted_rooms.read_events(
        db, room_id=ROOM_ID, since_seq=0, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]


def _append_user(db: Path, *, event_id: str, text: str) -> dict:
    return hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "local-user"}, authority_gateway_id=GATEWAY_ID,
        authority_epoch=1, payload={"text": text, "thread_id": "thread-1"}, now=time.time())


def _plan(room: dict, db: Path, events: list[dict] | None = None,
          watermarks=None) -> discussion.DiscussionDecision:
    return discussion.plan_next_task(
        room, _events(db) if events is None else events, local_profiles=LOCAL_PROFILES,
        initial_watermarks=watermarks)


def _settle_next(room: dict, db: Path, *, text: str) -> discussion.DiscussionTaskPlan:
    decision = _plan(room, db)
    assert decision.status == "task", decision
    publication = discussion.plan_publication(
        room, _events(db), decision.task, status="settled", result={"text": text},
        local_profiles=LOCAL_PROFILES)
    for event in publication.events:
        hosted_rooms.append_event(db, **event.append_kwargs(ROOM_ID), now=time.time())
    return decision.task


def _long_thread(room: dict, db: Path, *, discussions: int) -> dict:
    """One thread with more than the transcript window's worth of message rows."""
    for index in range(discussions):
        _append_user(db, event_id=f"user-{index}", text=f"Request {index}.")
        _settle_next(room, db, text=f"Answer {index}.")
    return _append_user(db, event_id="latest", text="Continue")


def _snapshot(db: Path):
    latest_seq = _events(db)[-1]["seq"]
    return HostedRoomPolicyCheckpoint(db).snapshot(room_id=ROOM_ID, latest_seq=latest_seq)


def _message_rows(events) -> list[dict]:
    return [event for event in events if event["kind"] in {"message.user", "message.member"}]


def _compacted_anchors(events) -> list[dict]:
    """Discussion user rows older than the window's oldest member row: the trim's boundary."""
    member_seqs = [event["seq"] for event in events if event["kind"] == "message.member"]
    if not member_seqs:
        return []
    return [event for event in events
            if event["kind"] == "message.user" and event["seq"] < min(member_seqs)]


def test_a_long_thread_keeps_the_discussion_of_every_retained_settlement(room_db):
    db, room = room_db
    latest = _long_thread(room, db, discussions=13)
    log_messages = _message_rows(_events(db))
    assert len(log_messages) > MAX_THREAD_TRANSCRIPT_EVENTS  # the window really did trim

    snapshot = _snapshot(db)
    window = list(snapshot.events)
    assert _message_rows(window)[0]["seq"] > log_messages[0]["seq"]
    assert _compacted_anchors(window), "a trimmed settlement lost the discussion it answered"
    window_ids = {event["event_id"] for event in window}
    retained = [event for event in window if event["kind"] == "message.member"]
    assert retained
    assert [event["payload"]["discussion_event_id"] for event in retained
            if event["payload"]["discussion_event_id"] not in window_ids] == []

    decision = _plan(room, db, window, snapshot.watermarks)
    assert decision.status == "task", decision
    assert decision.task is not None
    assert decision.source_event_seq == latest["seq"]


def test_an_aged_out_discussion_anchor_is_a_compacted_prefix(room_db):
    """A window written by the old trim (anchor gone) reads, it does not raise."""
    db, room = room_db
    latest = _long_thread(room, db, discussions=13)
    snapshot = _snapshot(db)
    anchors = {event["seq"] for event in _compacted_anchors(snapshot.events)}
    assert anchors
    compacted = [event for event in snapshot.events if event["seq"] not in anchors]
    assert len(compacted) < len(snapshot.events)

    decision = _plan(room, db, compacted, snapshot.watermarks)
    assert decision.status == "task", decision
    assert decision.source_event_seq == latest["seq"]


def test_an_already_wedged_cache_rebuilds_and_plans_the_next_turn(room_db):
    """No manual DB repair: the stored pre-fix cache is a stale schema and rebuilds."""
    db, room = room_db
    latest = _long_thread(room, db, discussions=13)
    snapshot = _snapshot(db)
    anchors = [event["seq"] for event in _compacted_anchors(snapshot.events)]
    assert anchors, "the trim kept no discussion anchor to rebuild from"

    # Reproduce the cache the wedged room had: schema 4, anchor trimmed away.
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM hosted_room_policy_transcript WHERE room_id=? AND seq=?",
                     (ROOM_ID, anchors[0]))
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=4 WHERE room_id=?",
                     (ROOM_ID,))

    rebuilt = _snapshot(db)
    assert anchors[0] in {event["seq"] for event in rebuilt.events}
    decision = _plan(room, db, list(rebuilt.events), rebuilt.watermarks)
    assert decision.status == "task", decision
    assert decision.source_event_seq == latest["seq"]


def test_a_settlement_whose_message_is_missing_still_raises(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Request.")
    task = _settle_next(room, db, text="Answer.")
    terminal = next(event for event in _events(db) if event["kind"] == "turn.settled")
    hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id="dterminal:missing-message", kind="turn.settled",
        actor={"kind": "gateway", "id": GATEWAY_ID}, authority_gateway_id=GATEWAY_ID,
        authority_epoch=1, now=time.time(),
        payload={**terminal["payload"], "task_id": "dtask:missing",
                 "turn_id": "d1.r0.p0.s1.m000000000000000000000000",
                 "message_event_id": "dmessage:missing"})

    with pytest.raises(discussion.DiscussionValidationError,
                       match="turn.settled references no matching member message"):
        discussion.derive_member_watermarks(room, _events(db), local_profiles=LOCAL_PROFILES)
    assert task.identity.task_id != "dtask:missing"


def test_a_settlement_naming_a_foreign_discussion_still_raises(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Request.")
    _settle_next(room, db, text="Answer.")
    terminal = next(event for event in _events(db) if event["kind"] == "turn.settled")
    hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id="dterminal:foreign-discussion", kind="turn.settled",
        actor={"kind": "gateway", "id": GATEWAY_ID}, authority_gateway_id=GATEWAY_ID,
        authority_epoch=1, now=time.time(),
        payload={**terminal["payload"], "task_id": "dtask:foreign",
                 "turn_id": "d1.r0.p0.s1.m000000000000000000000000",
                 "discussion_event_id": "user:never-existed"})

    with pytest.raises(discussion.DiscussionValidationError,
                       match="turn.settled references no matching member message"):
        discussion.derive_member_watermarks(room, _events(db), local_profiles=LOCAL_PROFILES)


def test_a_duplicate_terminal_still_raises(room_db):
    db, room = room_db
    _append_user(db, event_id="user-1", text="Request.")
    _settle_next(room, db, text="Answer.")
    terminal = next(event for event in _events(db) if event["kind"] == "turn.settled")
    hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id="dterminal:duplicate", kind="turn.settled",
        actor={"kind": "gateway", "id": GATEWAY_ID}, authority_gateway_id=GATEWAY_ID,
        authority_epoch=1, now=time.time(), payload=terminal["payload"])

    with pytest.raises(discussion.DiscussionValidationError):
        discussion.derive_member_watermarks(room, _events(db), local_profiles=LOCAL_PROFILES)
