"""Recovered history remains context, never an implicit retry of old requests."""

import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms, hosted_room_work_records as records
from gateway import hosted_room_discussion as discussion
from gateway.hosted_room_manual_recovery import prepare_recovery
from gateway.hosted_room_manual_promotion_schema import TABLE
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from tests.gateway.test_hosted_room_replica_ingress import HOME, TARGET, SECRET, ingest, pair
from tests.gateway.test_manual_group_promotion_staging import saved, stage


def request(source, event_id, text, gateway_id, epoch):
    return rooms.append_event(source, room_id="room", event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": text, "thread_id": "shared-thread"},
        authority_gateway_id=gateway_id, authority_epoch=epoch)


@pytest.fixture
def staged(saved):
    source, target, token, _preview = saved
    old = request(source, "before-loss", "@reviewer Review the old request.", HOME, 1)
    ingest((source, target), token)
    record = records.capture(source, room_id="room", local_gateway_id=HOME)
    records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    preview = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    pending = stage(saved, snapshot_id=preview["snapshot_id"])
    return target, old, pending


def expose_to_planner(path, *, active=True):
    # Simulate only the later controller's committed activation boundary.
    # No production activation method or public takeover RPC is enabled.
    with sqlite3.connect(path) as conn:
        if active:
            conn.execute(f"UPDATE {TABLE} SET status='active' WHERE room_id='room'")
        conn.execute("DELETE FROM hosted_room_quarantine WHERE room_id='room'")


def test_old_requests_are_not_replanned_but_new_work_keeps_saved_context(staged):
    target, old, pending = staged
    expose_to_planner(target)
    room = rooms.room_state(target, room_id="room")
    snapshot = HostedRoomPolicyCheckpoint(target).snapshot(room_id="room", latest_seq=room["latest_seq"])
    assert snapshot.events == ()
    assert snapshot.stopped_through_seq > old["seq"]
    assert any(event["event_id"] == old["event_id"] for event in rooms.read_events(target, room_id="room")["events"])
    with sqlite3.connect(target) as conn:
        conn.execute("UPDATE hosted_room_policy_cursors SET stopped_through_seq=0 WHERE room_id='room'")
    rebuilt = HostedRoomPolicyCheckpoint(target).snapshot(room_id="room", latest_seq=room["latest_seq"])
    assert rebuilt.events == ()
    assert rebuilt.stopped_through_seq > old["seq"]
    newer = request(target, "after-recovery", "@reviewer Review this new request.", TARGET, pending["authority_epoch"])
    room = rooms.room_state(target, room_id="room")
    snapshot = HostedRoomPolicyCheckpoint(target).snapshot(room_id="room", latest_seq=newer["seq"])
    decision = discussion.plan_next_task(room, list(snapshot.events), local_profiles=["default"], initial_watermarks=snapshot.watermarks)
    assert decision.source_event_seq == newer["seq"]
    assert decision.discussion_event_id == newer["event_id"]
    assert any(event["event_id"] == old["event_id"] for event in snapshot.events)


@pytest.mark.parametrize("damage", ["pending", "missing", "changed"])
def test_claim_without_its_active_exact_record_cannot_plan(staged, damage):
    target, _old, _pending = staged
    expose_to_planner(target, active=damage != "pending")
    with sqlite3.connect(target) as conn:
        if damage == "missing":
            conn.execute(f"DELETE FROM {TABLE} WHERE room_id='room'")
        elif damage == "changed":
            row = conn.execute("SELECT payload_json FROM hosted_room_events WHERE kind='authority.claimed'").fetchone()
            payload = {**json.loads(row[0]), "saved_recovery_point": "a" * 64}
            conn.execute("UPDATE hosted_room_events SET payload_json=? WHERE kind='authority.claimed'", (json.dumps(payload),))
    with pytest.raises(RuntimeError, match="recovery record"):
        HostedRoomPolicyCheckpoint(target).snapshot(room_id="room", latest_seq=rooms.room_state(target, room_id="room")["latest_seq"])


def test_ordinary_authority_claim_does_not_discard_pending_work(saved):
    source = saved[0]
    old = request(source, "existing-work", "@reviewer Keep this request.", HOME, 1)
    claimed = rooms.claim_authority(source, room_id="room", expected_gateway_id=HOME, expected_epoch=1,
        new_gateway_id=TARGET, event_id="ordinary-claim")
    snapshot = HostedRoomPolicyCheckpoint(source).snapshot(room_id="room", latest_seq=claimed["latest_seq"])
    assert snapshot.stopped_through_seq < old["seq"]
    assert any(event["event_id"] == old["event_id"] for event in snapshot.events)
