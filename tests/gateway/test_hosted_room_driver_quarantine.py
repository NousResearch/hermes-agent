"""A read-only room cannot retain a live execution path behind its log fence."""

import sqlite3

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms as rooms
from tests.gateway.test_hosted_room_driver import FakeClock, _admit, _identity, _lease, _payload, db


def quarantine(path):
    # Persist the same private fence used for unsafe historical room lineage.
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO hosted_room_quarantine VALUES('room-1','unsafe_authority_demotion',101)")


@pytest.mark.parametrize("operation", ["admit", "lease", "renew", "start", "recover", "settle"])
def test_quarantine_refuses_new_execution_and_new_settlement(db, operation):
    clock, task = FakeClock(), _identity()
    _admit(db, task, clock)
    lease = _lease(db, clock)
    attempt = driver.start_task(db, task, lease, expected_cancel_generation=0, clock=clock) if operation == "settle" else None
    before = driver.get_task(db, task)
    quarantine(db)
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(db, room_id="room-1")
    operations = {
        "admit": lambda: driver.admit_task(db, _identity("new", turn_id="new"), payload=_payload(), clock=clock),
        "lease": lambda: _lease(db, clock),
        "renew": lambda: driver.renew_lease(db, lease, ttl_seconds=30, clock=clock),
        "start": lambda: driver.start_task(db, task, lease, expected_cancel_generation=0, clock=clock),
        "recover": lambda: driver.recover_room(db, lease, clock=clock),
        "settle": lambda: driver.settle_task(db, attempt, settlement_id="late", status="settled", result={"text": "late"}, clock=clock),
    }
    with pytest.raises(driver.RoomUnavailableError, match="quarantined"):
        operations[operation]()
    assert driver.get_task(db, task) == before
    assert len(driver.list_tasks(db, room_id="room-1")) == 1


def test_quarantine_keeps_exact_cancellation_and_reads_available(db):
    clock, task = FakeClock(), _identity()
    _admit(db, task, clock)
    lease = _lease(db, clock)
    driver.start_task(db, task, lease, expected_cancel_generation=0, clock=clock)
    quarantine(db)
    stopping = driver.begin_task_cancel(db, task, cancel_id="owner-stop", expected_cancel_generation=0, clock=clock)
    assert stopping["status"] == "stopping"
    with pytest.raises(driver.StaleTaskError):
        driver.complete_task_cancel(db, task, cancel_id="unrelated", expected_cancel_generation=1, clock=clock)
    cancelled = driver.complete_task_cancel(db, task, cancel_id="owner-stop", expected_cancel_generation=1, clock=clock)
    assert cancelled["status"] == "cancelled"
    assert driver.get_task(db, task)["status"] == "cancelled"
    assert driver.complete_task_cancel(db, task, cancel_id="owner-stop", expected_cancel_generation=1, clock=clock)["idempotent"]


def test_already_committed_result_replays_without_new_mutation(db):
    clock, task = FakeClock(), _identity()
    _admit(db, task, clock)
    lease = _lease(db, clock)
    attempt = driver.start_task(db, task, lease, expected_cancel_generation=0, clock=clock)
    result = {"text": "already committed"}
    driver.settle_task(db, attempt, settlement_id="known", status="settled", result=result, clock=clock)
    quarantine(db)
    assert driver.settle_task(db, attempt, settlement_id="known", status="settled", result=result, clock=clock)["idempotent"]
