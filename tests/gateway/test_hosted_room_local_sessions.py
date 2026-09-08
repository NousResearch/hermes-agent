"""Local context identities are private metadata, never successor authority."""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms as rooms
from gateway import hosted_room_local_sessions as bindings
from gateway.hosted_room_route_schema import initialize_route_schema


def running_turn(path, *, room_id="room", profile="default", clock=time.time):
    rooms.create_room(path, room_id=room_id, name="Local context", authority_gateway_id="origin", members=[
        {"member_id": "default-member", "profile": "default", "handle": "writer"},
        {"member_id": "ops-member", "profile": "ops", "handle": "reviewer"},
    ])
    event = rooms.append_event(path, room_id=room_id, event_id="request", kind="message.user",
        actor={"kind": "user", "id": "owner"}, authority_gateway_id="origin", authority_epoch=1,
        payload={"text": "Review the plan", "thread_id": "thread"})
    task = driver.TaskIdentity(room_id, "task", "thread", "turn")
    driver.admit_task(path, task, payload={"target_profile": profile, "target_member_id": profile + "-member",
        "prompt": "PRIVATE_PROMPT", "source_event_seq": event["seq"]}, clock=clock)
    lease = driver.acquire_lease(path, room_id=room_id, gateway_id="origin", authority_epoch=1,
        process_generation="worker", ttl_seconds=120, clock=clock)
    attempt = driver.start_task(path, task, lease, expected_cancel_generation=0, clock=clock)
    return SimpleNamespace(path=path, task=task, lease=lease, attempt=attempt, profile=profile)


@pytest.fixture
def turn(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "origin")
    return running_turn(tmp_path / "state.db")


def record(turn, **overrides):
    kwargs = dict(task=turn.task, execution_generation=turn.attempt.execution_generation,
                  profile=turn.profile, session_id="private-session", session_started_at=100.0)
    return bindings.record_binding(turn.path, **{**kwargs, **overrides})


def count(path):
    with sqlite3.connect(path) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='hosted_room_local_sessions'").fetchone() is None:
            return 0
        return db.execute("SELECT COUNT(*) FROM hosted_room_local_sessions").fetchone()[0]


def test_exact_binding_is_idempotent_and_never_copies_prompt_or_changes_history(turn):
    before = rooms.read_events(turn.path, room_id="room")
    first = record(turn)
    assert record(turn) == first
    assert bindings.lookup_binding(turn.path, room_id="room", profile="default") == first
    assert first["member_id"] == "default-member"
    assert first["gateway_id"] == "origin"
    assert "PRIVATE_PROMPT" not in str(first)
    assert count(turn.path) == 1
    assert rooms.read_events(turn.path, room_id="room") == before


@pytest.mark.parametrize("change", [{"session_id": "another-session"}, {"session_started_at": 101.0}])
def test_binding_cannot_be_reassigned_to_a_new_or_recreated_context(turn, change):
    first = record(turn)
    with pytest.raises(bindings.LocalSessionBindingError):
        record(turn, **change)
    assert bindings.lookup_binding(turn.path, room_id="room", profile="default") == first


@pytest.mark.parametrize("change", [
    {"execution_generation": 0}, {"execution_generation": True}, {"execution_generation": 2},
    {"profile": "ops"}, {"session_started_at": float("nan")},
])
def test_invalid_or_wrong_profile_turn_cannot_record_identity(turn, change):
    with pytest.raises((bindings.LocalSessionBindingError, driver.DriverStateError)):
        record(turn, **change)
    assert count(turn.path) == 0


@pytest.mark.parametrize("state", ["stopping", "released", "expired", "reclaimed", "closing", "disbanded", "demoted"])
def test_stale_or_closed_work_never_creates_a_binding(turn, state):
    if state == "stopping":
        driver.begin_task_cancel(turn.path, turn.task, cancel_id="stop", expected_cancel_generation=0, clock=time.time)
    elif state == "released":
        # A mismatched persisted snapshot must not substitute for a live lease.
        with sqlite3.connect(turn.path) as db:
            db.execute("UPDATE hosted_room_driver_leases SET released_at=1")
    elif state in {"expired", "reclaimed"}:
        with sqlite3.connect(turn.path) as db:
            if state == "expired":
                db.execute("UPDATE hosted_room_driver_leases SET expires_at=0")
            else:
                db.execute("UPDATE hosted_room_driver_leases SET lease_generation=lease_generation+1")
    elif state == "closing":
        with rooms._transaction(turn.path, immediate=True) as db:
            initialize_route_schema(db)
            db.execute("INSERT INTO hosted_room_disband_fences(room_id,authority_gateway_id,authority_epoch,started_at) VALUES('room','origin',1,1)")
    elif state == "disbanded":
        rooms.disband_room(turn.path, room_id="room", expected_gateway_id="origin", expected_epoch=1)
    else:
        rooms.claim_authority(turn.path, room_id="room", expected_gateway_id="origin", expected_epoch=1,
                              new_gateway_id="successor", event_id="claim")
    with pytest.raises((bindings.LocalSessionBindingError, driver.DriverStateError)):
        record(turn)
    assert count(turn.path) == 0


def test_one_private_session_cannot_be_bound_to_two_rooms(turn):
    record(turn)
    other = running_turn(turn.path, room_id="second")
    with pytest.raises(bindings.LocalSessionBindingError, match="another Group Chat"):
        record(other)
    assert count(turn.path) == 1


def test_original_identity_remains_readable_for_cleanup_after_demotion(turn, monkeypatch):
    first = record(turn)
    rooms.claim_authority(turn.path, room_id="room", expected_gateway_id="origin", expected_epoch=1,
                          new_gateway_id="successor", event_id="claim")
    assert bindings.lookup_binding(turn.path, room_id="room", profile="default") == first
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "successor")
    with pytest.raises(bindings.LocalSessionBindingError, match="another Group Chat host"):
        bindings.lookup_binding(turn.path, room_id="room", profile="default")


def test_real_room_retirement_removes_binding_without_changing_the_task_identity(turn):
    record(turn)
    rooms.disband_room(turn.path, room_id="room", expected_gateway_id="origin", expected_epoch=1, now=100)
    assert bindings.lookup_binding(turn.path, room_id="room", profile="default")["first_task_id"] == turn.task.task_id
    assert rooms.prune_disbanded_rooms(turn.path, now=101 + rooms.DISBANDED_ROOM_RETENTION_SECONDS) == 1
    assert count(turn.path) == 0


def test_registry_with_weakened_uniqueness_fails_closed(turn):
    with driver._transaction(turn.path) as db:
        db.execute(bindings._DDL.replace("UNIQUE(room_id, profile), UNIQUE(profile, session_id),", ""))
    with pytest.raises(bindings.LocalSessionBindingError, match="constraints changed"):
        record(turn)
    assert count(turn.path) == 0


def test_lease_expiring_while_waiting_for_the_writer_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "origin")
    now = [100.0]
    turn = running_turn(tmp_path / "state.db", clock=lambda: now[0])
    entered = threading.Event()
    original_connect = driver._connect

    def observed_connect(path):
        entered.set()
        return original_connect(path)

    monkeypatch.setattr(driver, "_connect", observed_connect)
    with ThreadPoolExecutor(max_workers=1) as pool, sqlite3.connect(turn.path) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        future = pool.submit(record, turn, clock=lambda: now[0])
        assert entered.wait(timeout=3)
        now[0] = 221.0
        blocker.rollback()
        with pytest.raises(driver.StaleLeaseError):
            future.result(timeout=3)
    assert count(turn.path) == 0
