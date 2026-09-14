"""Transactional source closing fences, replay, and accepted-work cleanup."""

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_link_records as links
from gateway import hosted_rooms as rooms


IDENTITY = driver.TaskIdentity("room", "task", "thread", "turn")
PAYLOAD = {"target_profile": "ops", "prompt": "Inspect", "source_event_seq": 1}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "state.db"
    rooms.create_room(path, room_id="room", name="Workshop", members=[{"profile": "ops"}], authority_gateway_id="home")
    message(path, "seed")
    return path


def message(db, event_id):
    return rooms.append_event(db, room_id="room", event_id=event_id, kind="message.user",
                              actor={"kind": "user", "id": "owner"}, payload={"text": event_id},
                              authority_gateway_id="home", authority_epoch=1)


def close(db):
    return links.begin_room_link_retirement(db, room_id="room", authority_gateway_id="home", authority_epoch=1)


def lease(db, *, now=100, process="one"):
    return driver.acquire_lease(db, room_id="room", gateway_id="home", authority_epoch=1,
                                process_generation=process, ttl_seconds=50, clock=lambda: now)


def prepared_operation(db, operation):
    driver.admit_task(db, IDENTITY, payload=PAYLOAD, clock=lambda: 100)
    held = lease(db)
    if operation == "message":
        return partial(message, db, "late")
    if operation == "admit":
        new = driver.TaskIdentity("room", "new-task", "thread", "new-turn")
        return partial(driver.admit_task, db, new, payload=PAYLOAD, clock=lambda: 100)
    if operation == "start":
        return partial(driver.start_task, db, IDENTITY, held, expected_cancel_generation=0, clock=lambda: 100)
    attempt = driver.start_task(db, IDENTITY, held, expected_cancel_generation=0, clock=lambda: 100)
    if operation == "not_admitted":
        return partial(driver.requeue_not_admitted_task, db, attempt, clock=lambda: 100)
    current = lease(db, now=200, process="recovery")
    driver.recover_room(db, current, clock=lambda: 200)
    kwargs = dict(expected_execution_generation=1, expected_cancel_generation=0, clock=lambda: 200)
    if operation == "deferred":
        driver.defer_indeterminate_task(db, IDENTITY, current, reason="unavailable", **kwargs)
        return partial(driver.requeue_deferred_task, db, IDENTITY, current, **kwargs)
    return partial(driver.requeue_indeterminate_task, db, IDENTITY, current, **kwargs)


@pytest.mark.parametrize("operation", ["message", "admit", "start", "not_admitted", "indeterminate", "deferred"])
def test_close_rejects_new_writes_and_preserves_task_state(db, operation):
    act = prepared_operation(db, operation)
    before = driver.list_tasks(db, room_id="room")
    close(db)
    with pytest.raises((rooms.HostedRoomError, driver.RoomUnavailableError), match="being disbanded"):
        act()
    assert driver.list_tasks(db, room_id="room") == before
    assert rooms.room_state(db, room_id="room")["latest_seq"] == 1


def test_existing_event_admission_and_not_admitted_replays_survive_close(db):
    driver.admit_task(db, IDENTITY, payload=PAYLOAD, clock=lambda: 100)
    held = lease(db)
    attempt = driver.start_task(db, IDENTITY, held, expected_cancel_generation=0, clock=lambda: 100)
    driver.requeue_not_admitted_task(db, attempt, clock=lambda: 100)
    close(db)
    assert message(db, "seed")["idempotent"]
    assert driver.admit_task(db, IDENTITY, payload=PAYLOAD, clock=lambda: 100)["idempotent"]
    assert driver.requeue_not_admitted_task(db, attempt, clock=lambda: 100)["idempotent"]
    with pytest.raises(driver.TaskConflictError):
        driver.admit_task(db, IDENTITY, payload={**PAYLOAD, "prompt": "different"}, clock=lambda: 100)


@pytest.mark.parametrize("finish", ["settle", "cancel", "recover", "settle_stopping"])
def test_closing_preserves_leases_status_cancellation_and_receipts(db, finish):
    driver.admit_task(db, IDENTITY, payload=PAYLOAD, clock=lambda: 100)
    held = lease(db)
    attempt = driver.start_task(db, IDENTITY, held, expected_cancel_generation=0, clock=lambda: 100)
    close(db)
    driver.require_active_lease(db, held, clock=lambda: 100)
    driver.renew_lease(db, held, ttl_seconds=50, clock=lambda: 100)
    result = dict(settlement_id="receipt", status="settled", result={"text": "Accepted result"})
    if finish == "settle":
        settled = driver.settle_task(db, attempt, clock=lambda: 100, **result)
        assert driver.settle_task(db, attempt, clock=lambda: 100, **result)["idempotent"]
    elif finish == "recover":
        held = lease(db, now=200, process="recovery")
        driver.recover_room(db, held, clock=lambda: 200)
        settled = driver.resolve_indeterminate_task(
            db, IDENTITY, held, expected_execution_generation=1, expected_cancel_generation=0,
            clock=lambda: 200, **result)
    else:
        stopping = driver.begin_task_cancel(db, IDENTITY, cancel_id="stop", expected_cancel_generation=0, clock=lambda: 100)
        assert stopping["status"] == "stopping"
        if finish == "cancel":
            settled = driver.complete_task_cancel(db, IDENTITY, cancel_id="stop", expected_cancel_generation=1, clock=lambda: 100)
        else:
            settled = driver.settle_stopping_task(
                db, IDENTITY, held, expected_execution_generation=1, expected_cancel_generation=1,
                clock=lambda: 100, **result)
    assert settled["status"] in {"settled", "cancelled"}
    assert driver.get_task(db, IDENTITY)["status"] == settled["status"]
    driver.release_lease(db, held, clock=lambda: 200 if finish == "recover" else 100)
    rooms.append_event(db, room_id="room", event_id="terminal", kind="message.member",
                       actor={"kind": "member", "id": "ops"}, payload={"text": "Accepted output"},
                       authority_gateway_id="home", authority_epoch=1)
    assert rooms.read_events(db, room_id="room")["events"][-1]["kind"] == "message.member"


def _hold_close_transaction(db, ready, release):
    original = links._transaction
    @contextmanager
    def held(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            yield conn
            ready.set()
            assert release.wait(15)
    links._transaction = held
    close(db)


@pytest.mark.parametrize("operation", ["message", "admit", "start", "not_admitted", "indeterminate", "deferred"])
def test_other_process_close_wins_before_waiting_source_write(db, operation, monkeypatch):
    act = prepared_operation(db, operation)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    child = context.Process(target=_hold_close_transaction, args=(db, ready, release))
    reached_writer = threading.Event()
    owner = rooms if operation == "message" else driver
    original = owner._transaction
    @contextmanager
    def waiting(*args, **kwargs):
        reached_writer.set()
        with original(*args, **kwargs) as conn:
            yield conn
    monkeypatch.setattr(owner, "_transaction", waiting)
    child.start()
    try:
        assert ready.wait(10)
        # The other process has inserted the fence but still owns the writer.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(act)
            assert reached_writer.wait(5)
            release.set()
            with pytest.raises((rooms.HostedRoomError, driver.RoomUnavailableError), match="being disbanded"):
                future.result(timeout=10)
        child.join(10)
        assert child.exitcode == 0
    finally:
        release.set()
        if child.is_alive():
            child.terminate()
            child.join(5)
