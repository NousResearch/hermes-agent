"""A discussion's completion never ends a newer message on the same thread.

When a member's reply lands, the room plans the next round. If the user sends a new
message on that thread before the round settles, the settled status for the OLD
discussion must not close the thread the newer message just reopened: the newer
message would be silently dropped (the room shows idle and nobody answers).
"""
import pytest

from gateway import hosted_rooms as rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from gateway.hosted_room_router import paused_discussion
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture(name="db")
def db_fixture(tmp_path):
    """A store handle on the test home; the room lives in its shared-state file."""
    handle = open_persistence(profile_scope(home=tmp_path))
    yield handle
    handle.close()


def _append(db, event_id, kind, payload, actor="user"):
    return rooms.append_event(db, placement="shared", room_id="room", event_id=event_id, kind=kind,
        actor={"kind": actor, "id": "home"}, authority_gateway_id="home", authority_epoch=1, payload=payload)


def test_old_discussion_settling_after_a_newer_message_keeps_the_newer_one_active(db):
    rooms.create_room(db, placement="shared", room_id="room", name="Room", authority_gateway_id="home",
                      members=[{"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"}])
    _append(db, "first", "message.user", {"text": "@reviewer one", "thread_id": "thread"})
    newer = _append(db, "second", "message.user", {"text": "@reviewer two", "thread_id": "thread"})
    # The first discussion's settle was planned before the second message, appended after it.
    last = _append(db, "dactivity:first:silent_round", "room.activity", {
        "status": "settled", "reason_code": "silent_round", "thread_id": "thread",
        "discussion_event_id": "first"}, "gateway")
    snapshot = HostedRoomPolicyCheckpoint(db, placement="shared").snapshot(room_id="room", latest_seq=int(last["seq"]))
    # The thread stays open for the newer message (earlier messages ride along as its transcript).
    assert any(e["seq"] == newer["seq"] for e in snapshot.events)


def test_a_cap_notice_landing_after_a_newer_message_cannot_be_continued(db):
    """Continue would hand the thread back to the capped discussion and strand the newer message."""
    rooms.create_room(db, placement="shared", room_id="room", name="Room", authority_gateway_id="home",
                      members=[{"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"}])
    _append(db, "first", "message.user", {"text": "@reviewer one", "thread_id": "thread"})
    _append(db, "second", "message.user", {"text": "@reviewer two", "thread_id": "thread"})
    _append(db, "dactivity:first:max_messages", "room.activity", {
        "status": "bounded", "reason_code": "max_messages", "thread_id": "thread",
        "discussion_event_id": "first"}, "gateway")
    events = rooms.read_events(db, placement="shared", room_id="room")["events"]
    assert paused_discussion(events, "first") is None


def test_a_continue_an_older_build_wrote_after_a_newer_message_does_not_strand_it(db):
    """Logs from before the router fence can hold that Continue; replaying it must not re-point the thread."""
    rooms.create_room(db, placement="shared", room_id="room", name="Room", authority_gateway_id="home",
                      members=[{"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"}])
    _append(db, "first", "message.user", {"text": "@reviewer one", "thread_id": "thread"})
    _append(db, "second", "message.user", {"text": "@reviewer two", "thread_id": "thread"})
    _append(db, "dactivity:first:max_messages", "room.activity", {
        "status": "bounded", "reason_code": "max_messages", "thread_id": "thread",
        "discussion_event_id": "first"}, "gateway")
    _append(db, "dcontinue:first:1", "room.continued",
            {"discussion_event_id": "first", "thread_id": "thread", "grant": 1}, "gateway")
    _append(db, "dactivity:second:silent_round", "room.activity", {
        "status": "settled", "reason_code": "silent_round", "thread_id": "thread",
        "discussion_event_id": "second"}, "gateway")
    other = _append(db, "third", "message.user", {"text": "@reviewer other", "thread_id": "other"})
    snapshot = HostedRoomPolicyCheckpoint(db, placement="shared").snapshot(room_id="room", latest_seq=int(other["seq"]))
    assert {(e.get("payload") or {}).get("thread_id") for e in snapshot.events} == {"other"}
