"""Manual member holds: native directive parity and the durable hold projection."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


ROOM_ID = "hold-room"
GATEWAY_ID = "gateway-holds"
VECTORS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "hosted_room_hold_directives.json").read_text())
MEMBERS = [
    {"member_id": member["member_id"], "profile": member["handle"], "handle": member["handle"]}
    for member in VECTORS["members"]]
LOCAL_PROFILES = tuple(member["profile"] for member in MEMBERS)
ROSTER = discussion.validate_roster(MEMBERS, local_profiles=LOCAL_PROFILES)


@pytest.fixture
def room_db(tmp_path: Path) -> tuple[Path, dict]:
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db, room_id=ROOM_ID, name="Holds", members=MEMBERS,
        authority_gateway_id=GATEWAY_ID, now=1)
    return db, room


def _user(db: Path, *, event_id: str, text: str, thread_id: str = "thread-1") -> dict:
    return hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "local-user"}, payload={"text": text, "thread_id": thread_id},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1, now=time.time())


def _activity(db: Path, *, event_id: str, discussion_event_id: str, thread_id: str) -> dict:
    """Complete one discussion exactly as the service does when the policy settles it."""
    return hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id=event_id, kind="room.activity",
        actor={"kind": "gateway", "id": GATEWAY_ID},
        payload={
            "status": "settled", "reason_code": "members_held", "thread_id": thread_id,
            "discussion_event_id": discussion_event_id},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1)


def _stop(db: Path, *, cancel_id: str) -> dict:
    return hosted_rooms.request_room_stop(
        db, room_id=ROOM_ID, cancel_id=cancel_id, expected_gateway_id=GATEWAY_ID, expected_epoch=1)


def _snapshot(db: Path, room: dict):
    checkpoint = HostedRoomPolicyCheckpoint(db)
    latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
    return checkpoint.snapshot(room_id=ROOM_ID, latest_seq=latest)


def _held(db: Path, room: dict) -> set[str]:
    return set(_snapshot(db, room).held_member_ids)


# -- directive semantics (shared vectors, also run by the native Vitest suite) --

@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_hold_directive_matches_the_shared_native_vectors(case):
    directive = discussion.resolve_hold_directive(case["text"], ROSTER)

    assert sorted(directive.hold) == sorted(case["hold"])
    assert sorted(directive.release) == sorted(case["release"])


def test_member_prose_and_terminal_events_never_reach_the_directive(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@all stop")
    assert _held(db, room) == {member.member_id for member in ROSTER}

    # A Bot saying "resume" in the room, and a late terminal event, are not user control input.
    hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id="member-1", kind="message.member",
        actor={"kind": "member", "id": ROSTER[0].member_id, "profile": ROSTER[0].profile},
        payload={
            "text": "@all resume, I am done", "thread_id": "thread-1", "member_id": ROSTER[0].member_id,
            "member_index": 0, "round_index": 0, "task_id": "dtask:x", "turn_id": "turn-x",
            "discussion_event_id": "user-1"},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1)
    hosted_rooms.append_event(
        db, room_id=ROOM_ID, event_id="terminal-1", kind="turn.cancelled",
        actor={"kind": "gateway", "id": GATEWAY_ID},
        payload={
            "reason": "superseded_by_newer_user_event", "seen_through_seq": 1, "thread_id": "thread-1",
            "member_id": ROSTER[0].member_id, "member_index": 0, "round_index": 0, "task_id": "dtask:x",
            "turn_id": "turn-x", "discussion_event_id": "user-1"},
        authority_gateway_id=GATEWAY_ID, authority_epoch=1)

    assert _held(db, room) == {member.member_id for member in ROSTER}


# -- durable projection --------------------------------------------------------

def test_room_stop_holds_every_member_and_a_later_message_finds_them_held(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@all get to work")
    _stop(db, cancel_id="desktop-stop")
    assert _held(db, room) == {member.member_id for member in ROSTER}

    _user(db, event_id="user-2", text="any progress?", thread_id="thread-2")
    snapshot = _snapshot(db, room)
    decision = discussion.plan_next_task(
        room, list(snapshot.events), local_profiles=LOCAL_PROFILES,
        initial_watermarks=snapshot.watermarks, held_member_ids=snapshot.held_member_ids)

    assert set(snapshot.held_member_ids) == {member.member_id for member in ROSTER}
    assert (decision.status, decision.reason) == ("settled", "members_held")
    assert decision.task is None


def test_release_gestures_follow_native_precedence_across_threads(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@all stop")
    ids = [member.member_id for member in ROSTER]

    _user(db, event_id="user-2", text="@all standup in five")
    assert _held(db, room) == set(ids), "plain @all is not a release"

    _user(db, event_id="user-3", text="@impl what does the log say?", thread_id="thread-2")
    assert _held(db, room) == set(ids[1:]), "a direct mention releases only that member"

    _user(db, event_id="user-4", text="thanks everyone")
    assert _held(db, room) == set(ids[1:]), "unaddressed prose releases nobody"

    _user(db, event_id="user-5", text="@all resume")
    assert _held(db, room) == set()


def test_simultaneous_stop_and_resume_holds_every_addressed_member(room_db):
    """Native precedence: one stop word makes the whole message a hold for everyone it names."""
    db, room = room_db
    _user(db, event_id="user-1", text="@impl resume but @research stop")

    assert _held(db, room) == {"member-impl", "member-research"}


def test_a_held_member_keeps_its_original_hold_position(room_db):
    db, room = room_db
    first = _user(db, event_id="user-1", text="@impl stop")
    _user(db, event_id="user-2", text="@impl stop again")
    holds = {hold.member_id: hold for hold in _snapshot(db, room).holds}

    assert holds["member-impl"].held_at_seq == int(first["seq"])


def test_repeated_stop_and_input_stay_idempotent(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@all stop")
    _stop(db, cancel_id="stop-1")
    _stop(db, cancel_id="stop-1")  # same cancel id: one durable fence event
    _user(db, event_id="user-1", text="@all stop")  # same event id: one durable message

    events = hosted_rooms.read_events(db, room_id=ROOM_ID)["events"]
    assert [event["kind"] for event in events] == ["message.user", "room.stop_requested"]
    assert _held(db, room) == {member.member_id for member in ROSTER}


def test_held_skip_consumes_only_its_own_thread_and_keeps_other_context(room_db):
    """Native `group-rounds.ts` advances only the (thread, member) watermark it skipped."""
    db, room = room_db
    # Thread A carries context this member has never read.
    _user(db, event_id="user-a1", text="@impl here is the failing trace", thread_id="thread-a")
    _user(db, event_id="user-a2", text="@impl and the second half of it", thread_id="thread-a")
    # The pause, and the delta it consumes, happen entirely in thread B.
    _user(db, event_id="user-b1", text="@impl stop", thread_id="thread-b")
    _user(db, event_id="user-b2", text="@impl pause on this one too", thread_id="thread-b")
    _activity(db, event_id="activity-b", discussion_event_id="user-b2", thread_id="thread-b")
    assert _held(db, room) == {"member-impl"}

    release = _user(db, event_id="user-a3", text="@impl what do you make of it?", thread_id="thread-a")
    snapshot = _snapshot(db, room)
    decision = discussion.plan_next_task(
        room, list(snapshot.events), local_profiles=LOCAL_PROFILES,
        initial_watermarks=snapshot.watermarks, held_member_ids=snapshot.held_member_ids)

    assert decision.status == "task" and decision.task is not None
    assert decision.task.member.member_id == "member-impl"
    prompt = decision.task.payload["prompt"]
    # Thread A's unread context survives a pause consumed in thread B; a room-wide consumption
    # floor would have skipped both earlier lines.
    assert "here is the failing trace" in prompt
    assert "and the second half of it" in prompt
    assert "what do you make of it?" in prompt
    assert decision.task.seen_through_seq == int(release["seq"])


def test_hold_skip_consumes_its_thread_delta_exactly_once(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@impl stop")
    _user(db, event_id="user-2", text="@research keep going")
    settled = _user(db, event_id="user-3", text="@impl resume")

    first = _snapshot(db, room)
    second = _snapshot(db, room)  # a second poll must not move anything

    assert first.watermarks == second.watermarks
    assert first.watermarks[("thread-1", "member-impl")] == int(settled["seq"]) - 1


def test_holds_survive_a_reconstructed_checkpoint_on_the_same_database(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@impl stop")
    _user(db, event_id="user-2", text="@research keep going")
    before = _snapshot(db, room)

    reopened = HostedRoomPolicyCheckpoint(db)  # a fresh process reading the same store
    after = reopened.snapshot(
        room_id=ROOM_ID, latest_seq=int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"]))

    assert after.held_member_ids == before.held_member_ids == ("member-impl",)
    assert after.watermarks == before.watermarks


def test_an_existing_checkpoint_backfills_holds_from_its_own_log(room_db):
    """A checkpoint written before this projection existed must not resume held members."""
    db, room = room_db
    _user(db, event_id="user-1", text="@all get to work")
    _user(db, event_id="user-2", text="@impl stop")
    _user(db, event_id="user-3", text="@research and you pause too")
    checkpoint = HostedRoomPolicyCheckpoint(db)
    latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
    checkpoint.sync(room_id=ROOM_ID, latest_seq=latest)

    # Exactly what an older build leaves behind: a cursor at the current log position, the
    # transcript/watermark projections it knew about, and no hold rows at all.
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM hosted_room_policy_holds")
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=1")
        cursor = conn.execute(
            "SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?", (ROOM_ID,)).fetchone()
    assert cursor[0] == latest

    migrated = HostedRoomPolicyCheckpoint(db).snapshot(room_id=ROOM_ID, latest_seq=latest)

    assert set(migrated.held_member_ids) == {"member-impl", "member-research"}
    assert discussion.plan_next_task(
        room, list(migrated.events), local_profiles=LOCAL_PROFILES,
        initial_watermarks=migrated.watermarks, held_member_ids=migrated.held_member_ids).task is None


def test_backfill_preserves_the_thread_transcript_it_already_had(room_db):
    db, room = room_db
    _user(db, event_id="user-1", text="@impl first request")
    checkpoint = HostedRoomPolicyCheckpoint(db)
    latest = int(hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"])
    before = checkpoint.snapshot(room_id=ROOM_ID, latest_seq=latest)

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=1")
    after = HostedRoomPolicyCheckpoint(db).snapshot(room_id=ROOM_ID, latest_seq=latest)

    assert [event["event_id"] for event in after.events] == [event["event_id"] for event in before.events]
    assert after.watermarks == before.watermarks
