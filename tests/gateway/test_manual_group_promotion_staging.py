"""Atomic saved-room conversion stays quarantined until reconciliation."""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms, hosted_room_driver as driver, hosted_room_replicas as replicas
from gateway import hosted_room_manual_promotion as promotion, hosted_room_work_records as records
from gateway.hosted_room_manual_recovery import prepare_recovery
from gateway.hosted_room_manual_promotion_schema import TABLE
from tests.gateway.test_hosted_room_replica_ingress import HOME, TARGET, SECRET, MEMBERS, grant, ingest, pair


@pytest.fixture
def saved(pair, monkeypatch):
    return save_pair(pair, monkeypatch)


def save_pair(pair, monkeypatch):
    source, target = pair
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: TARGET)
    driver.list_tasks(source, room_id="room")
    token, _ = grant(target, permissions=("replicate", "work_records"))
    ingest(pair, token)
    record = records.capture(source, room_id="room", local_gateway_id=HOME)
    records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    preview = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert preview["blockers"] == []
    return source, target, token, preview


def stage(saved, **overrides):
    return promotion.stage_manual_recovery(saved[1], **{
        "room_id": "room", "recovery_id": "owner-decision", "snapshot_id": saved[3]["snapshot_id"],
        "confirm_previous_host_fenced": True, "confirm_saved_point": True, **overrides,
    })


def rows(path, table):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT * FROM {table} WHERE room_id='room' ORDER BY rowid").fetchall()


def test_saved_history_moves_once_without_executing_or_discarding_work_records(saved):
    source, target, _token, preview = saved
    source_history = rows(source, "hosted_room_events")
    copied_history = rows(target, "hosted_room_replica_events")
    copied_work = rows(target, records.TARGET_TABLE)
    result = stage(saved)
    assert result["room_id"] == preview["room_id"]
    assert result["authority_gateway_id"] == TARGET
    assert result["authority_epoch"] == preview["source_authority"]["epoch"] + 1
    assert result["execution_authorized"] is False
    assert result["accepted_tail"] == "unverified"
    assert result["status"] == "pending_reconciliation"
    assert stage(saved) == {**result, "idempotent": True}
    assert rows(source, "hosted_room_events") == source_history
    moved = rows(target, "hosted_room_events")
    assert moved[:-1] == copied_history
    assert moved[-1][3] == "authority.claimed"
    with rooms._transaction(target) as conn:
        record = conn.execute(f"SELECT * FROM {TABLE} WHERE room_id='room'").fetchone()
        assert json.loads(record["work_record_json"]) == json.loads(copied_work[0][3])
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replicas WHERE room_id='room'").fetchone()[0] == 0
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "authority"
        expected_bytes = sum(len(str(value).encode("utf-8")) for event in moved for value in (event[2], event[3], event[4], event[6]))
        assert conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] == expected_bytes
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(target, room_id="room")
    with pytest.raises(driver.RoomUnavailableError):
        driver.acquire_lease(target, room_id="room", gateway_id=TARGET, authority_epoch=result["authority_epoch"],
            process_generation="must-not-start", ttl_seconds=30, clock=lambda: 100)
    assert driver.list_tasks(target, room_id="room") == []


@pytest.mark.parametrize("overrides", [
    {"confirm_previous_host_fenced": False}, {"confirm_saved_point": False},
    {"confirm_previous_host_fenced": "true"}, {"snapshot_id": "a" * 64},
])
def test_no_implicit_or_stale_confirmation_can_move_a_copy(saved, overrides):
    target = saved[1]
    before = replicas.replica_state(target, room_id="room")
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved, **overrides)
    assert replicas.replica_state(target, room_id="room") == before
    assert rows(target, "hosted_rooms") == []


def test_new_copied_task_invalidates_the_selected_snapshot(saved):
    source, target, token, _preview = saved
    task = driver.TaskIdentity("room", "new-task", "thread", "turn")
    driver.admit_task(source, task, payload={"target_profile": "default", "target_member_id": "writer",
        "source_event_seq": rooms.read_events(source, room_id="room")["latest_seq"], "prompt": "PRIVATE_TASK"}, clock=lambda: 100)
    record = records.capture(source, room_id="room", local_gateway_id=HOME)
    records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved)
    assert rows(target, "hosted_rooms") == []
    refreshed = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert stage(saved, snapshot_id=refreshed["snapshot_id"])["execution_authorized"] is False
    assert driver.list_tasks(target, room_id="room") == []


def test_failed_transfer_restores_original_copy_and_schema(saved, monkeypatch):
    target = saved[1]
    before = replicas.replica_state(target, room_id="room")
    history, work = rows(target, "hosted_room_replica_events"), rows(target, records.TARGET_TABLE)
    original = promotion._move_events

    def fail_after_move(conn, room_id):
        original(conn, room_id)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(promotion, "_move_events", fail_after_move)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        stage(saved)
    assert replicas.replica_state(target, room_id="room") == before
    assert rows(target, "hosted_room_replica_events") == history
    assert rows(target, records.TARGET_TABLE) == work
    assert rows(target, "hosted_rooms") == []


def test_simultaneous_deliveries_create_one_pending_decision(saved):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: stage(saved), range(2)))
    assert sorted(result["idempotent"] for result in results) == [False, True]
    assert len(rows(saved[1], TABLE)) == 1


def test_completed_transition_cannot_be_reused_by_ordinary_writers(saved):
    target = saved[1]
    stage(saved)
    before = rows(target, "hosted_rooms")
    with rooms._transaction(target, immediate=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="reserved"):
            conn.execute("INSERT OR REPLACE INTO hosted_rooms SELECT * FROM hosted_rooms WHERE room_id='room'")
    assert rows(target, "hosted_rooms") == before
    rooms.create_room(target, room_id="unrelated", name="Another group", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    assert rooms.room_state(target, room_id="unrelated")["authority_gateway_id"] == TARGET


@pytest.mark.parametrize("state", ["partial", "quarantined", "disbanded", "closing"])
def test_unavailable_or_closing_copy_cannot_be_transferred(saved, state):
    source, target, token, _preview = saved
    if state == "partial":
        with sqlite3.connect(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET latest_seq=last_seq+1 WHERE room_id='room'")
    elif state == "quarantined":
        with sqlite3.connect(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET quarantine_reason='unverified' WHERE room_id='room'")
    elif state == "disbanded":
        rooms.disband_room(source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
        ingest((source, target), token, page=rooms.read_events(source, room_id="room", include_disbanded=True))
    else:
        with rooms._transaction(source, immediate=True) as conn:
            conn.execute("INSERT INTO hosted_room_disband_fences(room_id,authority_gateway_id,authority_epoch,started_at) VALUES(?,?,1,100)",
                         ("room", HOME))
        record = records.capture(source, room_id="room", local_gateway_id=HOME)
        records.ingest(target, record=record, token=token, secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    current = prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    assert current["blockers"]
    with pytest.raises(promotion.ManualRecoveryError):
        stage(saved, snapshot_id=current["snapshot_id"])
    assert rows(target, "hosted_rooms") == []


def test_transfer_does_not_temporarily_double_count_the_shared_byte_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", 2048)
    source, target = tmp_path / "home.db", tmp_path / "peer.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(source, room_id="room", event_id="large", kind="message.user", actor={"kind": "user", "id": "owner"},
        payload={"text": "x" * 1200}, authority_gateway_id=HOME, authority_epoch=1)
    saved = save_pair((source, target), monkeypatch)
    before = replicas.replica_state(target, room_id="room")["event_bytes"]
    assert rooms.MAX_GATEWAY_EVENT_BYTES / 2 < before < rooms.MAX_GATEWAY_EVENT_BYTES
    stage(saved)
    with sqlite3.connect(target) as conn:
        assert before < conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0] < rooms.MAX_GATEWAY_EVENT_BYTES
