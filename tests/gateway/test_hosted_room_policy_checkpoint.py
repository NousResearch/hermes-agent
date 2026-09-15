"""Storage-level checkpoint races and reconstruction across bounded projections."""

import sqlite3

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint, MAX_THREAD_TRANSCRIPT_EVENTS


def _room(tmp_path):
    db = tmp_path / "state.db"
    room = rooms.create_room(
        db, room_id="checkpoint-room", name="Checkpoint",
        members=[
            {"member_id": profile, "profile": profile, "handle": profile}
            for profile in ("ops", "review")],
        authority_gateway_id="gateway-a", now=1)
    return db, room


def _user(db, event_id, *, gateway="gateway-a", epoch=1):
    return rooms.append_event(
        db, room_id="checkpoint-room", event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "user"}, payload={"text": "@ops answer", "thread_id": "thread"},
        authority_gateway_id=gateway, authority_epoch=epoch, now=2)


def test_sync_cas_does_not_reapply_a_page_or_mistake_stale_snapshot_for_corruption(tmp_path, monkeypatch):
    db, room = _room(tmp_path)
    first, second = _user(db, "first"), _user(db, "second")
    checkpoint = HostedRoomPolicyCheckpoint(db)
    competitor = HostedRoomPolicyCheckpoint(db)
    read_events = rooms.read_events
    applied = []
    apply_event = checkpoint._apply_event
    raced = False

    def race(*args, **kwargs):
        nonlocal raced
        page = read_events(*args, **kwargs)
        if not raced:
            raced = True
            competitor.sync(room_id=room["room_id"], latest_seq=second["seq"])
        return page

    def record(conn, event):
        applied.append(event["seq"])
        apply_event(conn, event)

    monkeypatch.setattr(rooms, "read_events", race)
    monkeypatch.setattr(checkpoint, "_apply_event", record)
    assert checkpoint.sync(room_id=room["room_id"], latest_seq=first["seq"]) == second["seq"]
    assert applied == []
    assert checkpoint.sync(room_id=room["room_id"], latest_seq=first["seq"]) == second["seq"]
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT discussion_event_id, latest_user_seq FROM hosted_room_policy_threads WHERE room_id=?",
            (room["room_id"],)).fetchone() == ("second", second["seq"])
        conn.execute(
            "UPDATE hosted_room_policy_cursors SET through_seq=? WHERE room_id=?",
            (second["seq"] + 1, room["room_id"]))
    with pytest.raises(RuntimeError, match="ahead of the durable log"):
        checkpoint.sync(room_id=room["room_id"], latest_seq=second["seq"])


@pytest.mark.parametrize("close_status", ["settled", "bounded"])
@pytest.mark.parametrize("sync_before_close", [False, True])
def test_old_discussion_close_preserves_newer_same_thread_replay(tmp_path, close_status, sync_before_close):
    db, room = _room(tmp_path)
    first, second = _user(db, "first"), _user(db, "second")
    checkpoint = HostedRoomPolicyCheckpoint(db)
    if sync_before_close:
        checkpoint.sync(room_id=room["room_id"], latest_seq=second["seq"])
    closed = rooms.append_event(
        db, room_id=room["room_id"], event_id="close-first", kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={"status": close_status, "reason_code": "complete", "thread_id": "thread",
                 "discussion_event_id": first["event_id"]},
        authority_gateway_id="gateway-a", authority_epoch=1)
    current = rooms.room_state(db, room_id=room["room_id"])
    complete_log = rooms.read_events(db, room_id=room["room_id"])["events"]
    expected = discussion.plan_next_task(current, complete_log, local_profiles=("ops", "review"))
    assert expected.task is not None
    assert expected.discussion_event_id == second["event_id"]

    # Reopen the checkpoint as well as replaying it incrementally: both must
    # agree with the complete durable log, not merely retain a cursor value.
    snapshot = HostedRoomPolicyCheckpoint(db).snapshot(room_id=room["room_id"], latest_seq=closed["seq"])
    actual = discussion.plan_next_task(
        current, snapshot.events, local_profiles=("ops", "review"), initial_watermarks=snapshot.watermarks)
    assert actual == expected
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT discussion_event_id, latest_user_seq FROM hosted_room_policy_threads WHERE room_id=?",
            (room["room_id"],)).fetchone() == (second["event_id"], second["seq"])

    final = rooms.append_event(
        db, room_id=room["room_id"], event_id="close-second", kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={"status": close_status, "reason_code": "complete", "thread_id": "thread",
                 "discussion_event_id": second["event_id"]},
        authority_gateway_id="gateway-a", authority_epoch=1)
    assert checkpoint.snapshot(room_id=room["room_id"], latest_seq=final["seq"]).events == ()


def test_deferred_task_restores_expired_source_and_lineage_with_its_frozen_prompt(tmp_path):
    db, room = _room(tmp_path)
    source = _user(db, "old-source")
    events = rooms.read_events(db, room_id=room["room_id"])["events"]
    plan = discussion.plan_next_task(room, events, local_profiles=("ops", "review")).task
    assert plan is not None
    driver.admit_task(db, plan.identity, payload=dict(plan.payload), clock=lambda: 10)
    lease = driver.acquire_lease(
        db, room_id=room["room_id"], gateway_id="gateway-a", authority_epoch=1,
        process_generation="first-process", ttl_seconds=1, clock=lambda: 10)
    attempt = driver.start_task(db, plan.identity, lease, expected_cancel_generation=0, clock=lambda: 10)
    recovered_lease = driver.acquire_lease(
        db, room_id=room["room_id"], gateway_id="gateway-a", authority_epoch=1,
        process_generation="next-process", ttl_seconds=100, clock=lambda: 20)
    driver.recover_room(db, recovered_lease, clock=lambda: 20)
    task = driver.defer_indeterminate_task(
        db, plan.identity, recovered_lease, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation, reason="member-unavailable", clock=lambda: 20)
    assert task["status"] == "deferred"
    rooms.append_event(
        db, room_id=room["room_id"], event_id="closed-old", kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"}, authority_gateway_id="gateway-a", authority_epoch=1,
        payload={"status": "settled", "reason_code": "silent_round", "thread_id": "thread",
                 "discussion_event_id": source["event_id"]})
    rooms.claim_authority(
        db, room_id=room["room_id"], expected_gateway_id="gateway-a", expected_epoch=1,
        new_gateway_id="gateway-b", event_id="promoted")
    for index in range(MAX_THREAD_TRANSCRIPT_EVENTS + 1):
        _user(db, f"new-{index}", gateway="gateway-b", epoch=2)
    current = rooms.room_state(db, room_id=room["room_id"])
    checkpoint = HostedRoomPolicyCheckpoint(db)
    checkpoint.sync(room_id=room["room_id"], latest_seq=current["latest_seq"])
    with sqlite3.connect(db) as conn:
        for table in ("hosted_room_policy_events", "hosted_room_policy_transcript"):
            assert conn.execute(
                f"SELECT 1 FROM {table} WHERE room_id=? AND seq=?", (room["room_id"], source["seq"])).fetchone() is None
    restored = checkpoint.events_for_task(room_id=room["room_id"], source_event_seq=source["seq"])
    assert source["event_id"] in {event["event_id"] for event in restored}
    assert "promoted" in {event["event_id"] for event in restored}
    reconstructed = discussion.reconstruct_task_plan(current, restored, task, local_profiles=())
    assert reconstructed.identity == plan.identity
    assert reconstructed.payload["prompt"] == plan.payload["prompt"]
