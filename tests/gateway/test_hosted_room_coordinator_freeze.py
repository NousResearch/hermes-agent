"""A local recovery fence preserves evidence without inventing shared history."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms as rooms
from gateway import hosted_room_coordinator_freeze as freeze


@pytest.fixture
def room(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "origin")
    path = tmp_path / "state.db"
    rooms.create_room(path, room_id="room", name="Shared project", authority_gateway_id="origin", members=[
        {"member_id": "writer", "profile": "default", "handle": "writer"},
        {"member_id": "reviewer", "profile": "review", "handle": "reviewer", "target": {
            "kind": "peer", "peer_id": "review-peer", "installation_id": "successor", "profile": "review",
            "capability_digest": "a" * 64}},
    ], now=90)
    rooms.append_event(path, room_id="room", event_id="request", kind="message.user",
        actor={"kind": "user", "id": "owner"}, authority_gateway_id="origin", authority_epoch=1,
        payload={"text": "PRIVATE_SHARED_CONTENT"}, now=91)
    return path


def request(path, **overrides):
    with sqlite3.connect(path) as conn:
        seq = conn.execute("SELECT next_seq-1 FROM hosted_rooms WHERE room_id='room'").fetchone()[0]
    return freeze.freeze_local_coordinator(path, **{
        "room_id": "room", "recovery_id": "owner-decision", "expected_gateway_id": "origin",
        "expected_epoch": 1, "expected_history_seq": seq, "successor_gateway_id": "successor",
        "confirm": True, "now": 100, **overrides,
    })


def raw_history(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT * FROM hosted_room_events ORDER BY room_id,seq").fetchall()


def test_local_freeze_preserves_history_identity_tasks_and_claims_no_remote_stop(room):
    task = driver.TaskIdentity("room", "task", "thread", "turn")
    driver.admit_task(room, task, payload={"target_profile": "default", "prompt": "PRIVATE_PROMPT",
        "source_event_seq": rooms.read_events(room, room_id="room")["latest_seq"]}, clock=lambda: 95)
    lease = driver.acquire_lease(room, room_id="room", gateway_id="origin", authority_epoch=1,
        process_generation="worker", ttl_seconds=100, clock=lambda: 95)
    before = raw_history(room)
    first = request(room)
    assert first["local_commits_fenced"] is True
    assert first["accepted_work_stopped"] is False
    assert first["successor_authorized"] is False
    assert "PRIVATE_" not in json.dumps(first)
    assert raw_history(room) == before
    assert request(room, now=200) == {**first, "idempotent": True}
    with rooms._transaction(room) as conn:
        saved = conn.execute("SELECT authority_gateway_id,authority_epoch,next_seq FROM hosted_rooms WHERE room_id='room'").fetchone()
        assert tuple(saved) == ("origin", 1, first["history_seq"] + 1)
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "authority"
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.append_event(room, room_id="room", event_id="late", kind="message.user", actor={"kind": "user", "id": "owner"},
            payload={"text": "late"}, authority_gateway_id="origin", authority_epoch=1)
    with pytest.raises(driver.RoomUnavailableError):
        driver.start_task(room, task, lease, expected_cancel_generation=0, clock=lambda: 101)
    assert driver.get_task(room, task)["status"] == "queued"
    assert driver.cancel_task(room, task, cancel_id="owner-stop", expected_cancel_generation=0, clock=lambda: 101)["status"] == "cancelled"
    assert raw_history(room) == before


@pytest.mark.parametrize("change", [
    {"confirm": False}, {"confirm": "true"}, {"expected_history_seq": 0}, {"expected_history_seq": True},
    {"expected_epoch": 2}, {"expected_gateway_id": "other"}, {"successor_gateway_id": "origin"},
    {"successor_gateway_id": "unrelated"},
])
def test_unconfirmed_stale_or_unrelated_decisions_do_not_fence(room, change):
    before = raw_history(room)
    with pytest.raises(rooms.HostedRoomError):
        request(room, **change)
    assert rooms.room_state(room, room_id="room")["authority_gateway_id"] == "origin"
    assert raw_history(room) == before


def test_two_deliveries_commit_one_local_freeze(room):
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: request(room), range(2)))
    assert sorted(reply["idempotent"] for reply in replies) == [False, True]
    assert {reply["history_sha256"] for reply in replies} == {replies[0]["history_sha256"]}
    with sqlite3.connect(room) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_coordinator_freezes").fetchone()[0] == 1


def test_reused_decision_cannot_change_destination_or_restore_missing_fence(room):
    request(room)
    with pytest.raises(freeze.CoordinatorFreezeError):
        request(room, recovery_id="another")
    with sqlite3.connect(room) as conn:
        conn.execute("DELETE FROM hosted_room_quarantine WHERE room_id='room'")
    with pytest.raises(freeze.CoordinatorFreezeError, match="unavailable or changed"):
        request(room)


def test_existing_quarantine_is_not_reclassified_or_cleared(room):
    with sqlite3.connect(room) as conn:
        conn.execute("INSERT INTO hosted_room_quarantine VALUES('room','unsafe_replica_promotion',95)")
    with pytest.raises(rooms.RoomQuarantinedError):
        request(room)
    with sqlite3.connect(room) as conn:
        assert conn.execute("SELECT reason FROM hosted_room_quarantine WHERE room_id='room'").fetchone()[0] == "unsafe_replica_promotion"
