"""Closed deferrals must not revive unreserved terminal work in real storage."""

import sqlite3

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms


PROFILES = ("ops", "review")


def _room(db, room_id="room"):
    return rooms.create_room(
        db, room_id=room_id, name="Deferred recovery",
        members=[{"member_id": profile, "profile": profile, "handle": profile} for profile in PROFILES],
        authority_gateway_id="gateway-a", now=1)


def _events(db):
    return rooms.read_events(db, room_id="room", limit=rooms.MAX_LOG_LIMIT)["events"]


def _admit(db, index):
    rooms.append_event(
        db, room_id="room", event_id=f"source-{index}", kind="message.user",
        actor={"kind": "user", "id": "user"},
        payload={"text": "@ops inspect", "thread_id": "thread"},
        authority_gateway_id="gateway-a", authority_epoch=1, require_open_admissions=True)
    plan = discussion.plan_next_task(
        rooms.room_state(db, room_id="room"), _events(db), local_profiles=PROFILES).task
    assert plan is not None
    driver.admit_task(db, plan.identity, payload=dict(plan.payload), clock=lambda: 10 * (index + 1))
    return plan


def _publish(db, plan, status, *, generation=None):
    publication = discussion.plan_publication(
        rooms.room_state(db, room_id="room"), _events(db), plan,
        status=status, execution_generation=generation, local_profiles=PROFILES)
    return rooms.append_events(
        db, events=[event.append_kwargs("room") for event in publication.events], allow_terminal_recovery=True)


def _close(db, plan):
    rooms.append_event(
        db, room_id="room", event_id=f"close:{plan.discussion_event_id}", kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={"status": "settled", "reason_code": "silent_round", "thread_id": "thread",
                 "discussion_event_id": plan.discussion_event_id},
        authority_gateway_id="gateway-a", authority_epoch=1)


def _defer(db, index, *, closed=True):
    plan = _admit(db, index)
    started_at = 10 * (index + 1)
    lease = driver.acquire_lease(
        db, room_id="room", gateway_id="gateway-a", authority_epoch=1,
        process_generation=f"start-{index}", ttl_seconds=1, clock=lambda: started_at)
    attempt = driver.start_task(db, plan.identity, lease, expected_cancel_generation=0, clock=lambda: started_at)
    recovery = driver.acquire_lease(
        db, room_id="room", gateway_id="gateway-a", authority_epoch=1,
        process_generation=f"recover-{index}", ttl_seconds=1, clock=lambda: started_at + 2)
    assert driver.recover_room(db, recovery, clock=lambda: started_at + 2)["indeterminate"] == [plan.identity]
    driver.defer_indeterminate_task(
        db, plan.identity, recovery, expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation, reason="unavailable", clock=lambda: started_at + 2)
    _publish(db, plan, "deferred", generation=attempt.execution_generation)
    if closed:
        _close(db, plan)
    return plan


def test_stop_preserves_closed_deferrals_and_drains_only_live_work(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _room(db)
    count = rooms.TERMINAL_RECOVERY_COUNT_RESERVE // rooms.MAX_TERMINAL_PUBLICATION_EVENTS + 2
    closed = [_defer(db, index) for index in range(count)]
    retained = [driver.get_task(db, plan.identity) for plan in closed]
    deferred = _defer(db, count, closed=False)
    queued = _admit(db, count + 1)
    rooms.request_room_stop(
        db, room_id="room", cancel_id="stop-all", expected_gateway_id="gateway-a", expected_epoch=1)

    cancelled = driver.reconcile_stop_fenced_inactive_tasks(db, room_id="room", clock=lambda: 1000)
    assert cancelled == [deferred.identity, queued.identity]
    assert [driver.get_task(db, plan.identity) for plan in closed] == retained
    assert driver.reconcile_stop_fenced_inactive_tasks(db, room_id="room", clock=lambda: 1001) == []

    # Exercise the startup audit with a fresh cache, without launching a process.
    monkeypatch.setattr(driver, "_STARTUP_AUDITED_SCHEMAS", set())
    assert driver.get_task(db, queued.identity)["status"] == "cancelled"
    for plan in (deferred, queued):
        assert _publish(db, plan, "cancelled")[-1]["kind"] == "turn.cancelled"
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == set()
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: "gateway-a")
    assert replicas.demote_room(
        db, room_id="room", observed_gateway_id="gateway-b", observed_epoch=2)["authority_epoch"] == 2
    assert [driver.get_task(db, plan.identity) for plan in closed] == retained


@pytest.mark.parametrize("operation", ["cancel", "requeue"])
@pytest.mark.parametrize("occupied_room", ["room", "other-room"])
def test_closed_deferred_transition_reserves_capacity_or_rolls_back(tmp_path, monkeypatch, operation, occupied_room):
    db = tmp_path / "state.db"
    _room(db)
    plan = _defer(db, 0)
    if occupied_room != "room":
        _room(db, occupied_room)
    # One real pending discussion consumes the entire room-count / host-byte
    # allowance. A closed deferral currently consumes neither allowance.
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", rooms.MAX_TERMINAL_PUBLICATION_EVENTS)
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_BYTE_RESERVE", rooms.MAX_TERMINAL_PUBLICATION_BYTES)
    source = rooms.append_event(
        db, room_id=occupied_room, event_id="occupied", kind="message.user",
        actor={"kind": "user", "id": "user"},
        payload={"text": "@review inspect", "thread_id": "other-thread"},
        authority_gateway_id="gateway-a", authority_epoch=1, require_open_admissions=True)
    lease = driver.acquire_lease(
        db, room_id="room", gateway_id="gateway-a", authority_epoch=1,
        process_generation="retry", ttl_seconds=100, clock=lambda: 100)
    before = driver.get_task(db, plan.identity)
    with sqlite3.connect(db) as conn:
        durable_before = conn.execute(
            "SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?",
            ("room", plan.identity.task_id)).fetchone()

    def transition():
        if operation == "cancel":
            return driver.cancel_task(
                db, plan.identity, cancel_id="cancel", expected_cancel_generation=0, clock=lambda: 100)
        return driver.requeue_deferred_task(
            db, plan.identity, lease, expected_execution_generation=1, expected_cancel_generation=0,
            clock=lambda: 100)

    with pytest.raises(driver.TaskAdmissionBlockedError, match="terminal recovery headroom"):
        transition()
    assert driver.get_task(db, plan.identity) == before
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT * FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?",
            ("room", plan.identity.task_id)).fetchone() == durable_before

    rooms.append_event(
        db, room_id=occupied_room, event_id="release-occupied", kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={"status": "settled", "reason_code": "complete", "thread_id": "other-thread",
                 "discussion_event_id": source["event_id"]},
        authority_gateway_id="gateway-a", authority_epoch=1)
    resumed = transition()
    assert resumed["status"] == ("cancelled" if operation == "cancel" else "queued")
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == {("room", plan.identity.task_id)}
    monkeypatch.setattr(driver, "_STARTUP_AUDITED_SCHEMAS", set())
    assert driver.get_task(db, plan.identity)["status"] == resumed["status"]
    if operation == "requeue":
        driver.cancel_task(db, plan.identity, cancel_id="cancel-retry", expected_cancel_generation=0, clock=lambda: 101)
    else:
        assert transition()["idempotent"] is True
    assert _publish(db, plan, "cancelled")[-1]["kind"] == "turn.cancelled"
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == set()
