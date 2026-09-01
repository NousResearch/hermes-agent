"""Behavior tests for the hosted-room driver state machine."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from gateway import hosted_rooms as rooms
from gateway import hosted_room_driver as driver


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _identity(task_id: str = "task-1", *, turn_id: str = "turn-1"):
    return driver.TaskIdentity(
        room_id="room-1",
        task_id=task_id,
        thread_id="thread-1",
        turn_id=turn_id,
    )


def _payload(
    *,
    target_profile: str = "ops",
    prompt: str = "Inspect the release candidate.",
    source_event_seq: int = 1,
):
    return {
        "target_profile": target_profile,
        "prompt": prompt,
        "source_event_seq": source_event_seq,
    }


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "state.db"
    rooms.create_room(
        path,
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "member-ops", "profile": "ops", "handle": "ops"}
        ],
        authority_gateway_id="gateway-a",
        now=90,
    )
    return path


def _lease(
    db,
    clock,
    *,
    gateway="gateway-a",
    authority_epoch=1,
    process="process-a",
    process_pid=1001,
    process_start_time=5001,
    ttl=30,
):
    return driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=gateway,
        authority_epoch=authority_epoch,
        process_generation=process,
        process_pid=process_pid,
        process_start_time=process_start_time,
        ttl_seconds=ttl,
        clock=clock,
    )


def _admit(db, identity, clock, *, payload=None):
    return driver.admit_task(
        db,
        identity,
        payload=_payload() if payload is None else payload,
        clock=clock,
    )


def _open_driver_schema(path: str) -> int:
    return len(driver.list_tasks(path, room_id="room-1"))


def _begin_demotion_from_process(path: str, cancel_id: str) -> dict[str, object]:
    return driver.begin_room_demotion(
        path,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id=cancel_id,
        clock=FakeClock(),
    )


def test_driver_lease_preserves_legacy_positional_reclaimed_argument():
    lease = driver.DriverLease(
        "room-1",
        "gateway-a",
        1,
        "process-a",
        2,
        130.0,
        True,
    )

    assert lease.reclaimed is True
    assert lease.process_pid is None
    assert lease.process_start_time is None


def test_read_does_not_require_sqlite_writer_lock(db, monkeypatch):
    identity = _identity()
    _admit(db, identity, FakeClock())
    writer = sqlite3.connect(db)
    real_connect = sqlite3.connect

    def connect_with_short_timeout(*args, **kwargs):
        kwargs["timeout"] = 0.05
        return real_connect(*args, **kwargs)

    try:
        writer.execute("BEGIN IMMEDIATE")
        monkeypatch.setattr(driver.sqlite3, "connect", connect_with_short_timeout)
        assert driver.get_task(db, identity)["status"] == "queued"
    finally:
        writer.rollback()
        writer.close()


def test_two_contenders_have_one_winner(db):
    clock = FakeClock()

    def contend(process):
        try:
            return _lease(db, clock, process=process)
        except driver.LeaseHeldError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contend, ["process-a", "process-b"]))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0].lease_generation == 1


def test_terminal_receipt_is_durable_idempotent_and_generation_fenced(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    first = driver.record_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        settlement_id="reply-task-1-1",
        status="settled",
        result={"message_id": "reply-task-1-1", "text": "done"},
        clock=clock,
    )
    repeated = driver.record_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        settlement_id="reply-task-1-1",
        status="settled",
        result={"message_id": "reply-task-1-1", "text": "done"},
        clock=clock,
    )

    assert first["idempotent"] is False
    assert repeated["idempotent"] is True
    assert driver.get_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
    )["result"]["text"] == "done"
    with pytest.raises(driver.TaskConflictError):
        driver.record_terminal_receipt(
            db,
            identity,
            execution_generation=attempt.execution_generation,
            settlement_id="reply-task-1-1",
            status="failed",
            result={"message_id": "reply-task-1-1", "error": "different"},
            clock=clock,
        )
    with pytest.raises(driver.StaleTaskError):
        driver.record_terminal_receipt(
            db,
            identity,
            execution_generation=attempt.execution_generation + 1,
            settlement_id="reply-task-1-2",
            status="settled",
            result={"message_id": "reply-task-1-2", "text": "stale"},
            clock=clock,
        )


def test_expiry_allows_reclaim_and_fences_stale_renew_and_release(db):
    clock = FakeClock()
    first = _lease(db, clock, ttl=5)

    clock.advance(5)
    second = _lease(db, clock, process="process-b")

    assert second.reclaimed is True
    assert second.lease_generation == first.lease_generation + 1
    with pytest.raises(driver.StaleLeaseError):
        driver.renew_lease(db, first, ttl_seconds=30, clock=clock)
    with pytest.raises(driver.StaleLeaseError):
        driver.release_lease(db, first, clock=clock)


def test_nonexistent_and_disbanded_rooms_cannot_lease_or_admit(db):
    clock = FakeClock()
    missing = driver.TaskIdentity("missing-room", "task", "thread", "turn")

    with pytest.raises(driver.RoomUnavailableError, match="does not exist"):
        driver.acquire_lease(
            db,
            room_id="missing-room",
            gateway_id="gateway-a",
            authority_epoch=1,
            process_generation="process-a",
            ttl_seconds=30,
            clock=clock,
        )
    with pytest.raises(driver.RoomUnavailableError, match="does not exist"):
        _admit(db, missing, clock)

    rooms.disband_room(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        now=clock(),
    )
    with pytest.raises(driver.RoomUnavailableError, match="disbanded"):
        _lease(db, clock)
    with pytest.raises(driver.RoomUnavailableError, match="disbanded"):
        _admit(db, _identity(), clock)


@pytest.mark.parametrize(
    ("gateway", "authority_epoch"),
    [("gateway-b", 1), ("gateway-a", 2)],
)
def test_acquire_requires_current_room_authority(db, gateway, authority_epoch):
    with pytest.raises(driver.StaleLeaseError, match="authority changed"):
        _lease(
            db,
            FakeClock(),
            gateway=gateway,
            authority_epoch=authority_epoch,
        )


def test_same_process_acquire_and_release_are_idempotent(db):
    clock = FakeClock()
    first = _lease(db, clock)
    repeated = _lease(db, clock)

    assert repeated.lease_generation == first.lease_generation
    released = driver.release_lease(db, repeated, clock=clock)
    released_again = driver.release_lease(db, repeated, clock=clock)

    assert released["idempotent"] is False
    assert released_again["idempotent"] is True


def test_renew_extends_only_the_current_lease_generation(db):
    clock = FakeClock()
    lease = _lease(db, clock, ttl=5)

    clock.advance(2)
    renewed = driver.renew_lease(db, lease, ttl_seconds=20, clock=clock)

    assert renewed.lease_generation == lease.lease_generation
    assert renewed.expires_at == 122


def test_authority_transfer_fences_lease_and_late_settlement(db):
    clock = FakeClock()
    identity = _identity()
    queued = _identity("task-2", turn_id="turn-2")
    old_lease = _lease(db, clock)
    _admit(db, identity, clock)
    _admit(db, queued, clock)
    old_attempt = driver.start_task(
        db,
        identity,
        old_lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-gateway-b",
        now=clock(),
    )

    with pytest.raises(driver.StaleLeaseError, match="authority changed"):
        driver.renew_lease(db, old_lease, ttl_seconds=30, clock=clock)
    with pytest.raises(driver.StaleLeaseError, match="authority changed"):
        driver.start_task(
            db,
            queued,
            old_lease,
            expected_cancel_generation=0,
            clock=clock,
        )
    with pytest.raises(driver.StaleLeaseError, match="authority changed"):
        driver.recover_room(db, old_lease, clock=clock)
    with pytest.raises(driver.StaleLeaseError, match="authority changed"):
        driver.settle_task(
            db,
            old_attempt,
            settlement_id="late-settlement",
            status="settled",
            result={"text": "late"},
            clock=clock,
        )

    new_lease = _lease(
        db,
        clock,
        gateway="gateway-b",
        authority_epoch=2,
        process="process-b",
    )
    recovery = driver.recover_room(db, new_lease, clock=clock)
    assert new_lease.lease_generation == old_lease.lease_generation + 1
    assert recovery["indeterminate"] == [identity]
    assert recovery["queued"] == [queued]


def test_room_disband_fences_active_lease_operations(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    rooms.disband_room(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        now=clock(),
    )

    with pytest.raises(driver.RoomUnavailableError, match="disbanded"):
        driver.renew_lease(db, lease, ttl_seconds=30, clock=clock)
    with pytest.raises(driver.RoomUnavailableError, match="disbanded"):
        driver.start_task(
            db,
            identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        )
    with pytest.raises(driver.RoomUnavailableError, match="disbanded"):
        driver.recover_room(db, lease, clock=clock)
    with pytest.raises(driver.RoomUnavailableError, match="disbanded"):
        driver.release_lease(db, lease, clock=clock)


def test_task_admission_is_idempotent_and_identity_conflicts_fail(db):
    clock = FakeClock()
    identity = _identity()

    first = _admit(db, identity, clock)
    repeated = _admit(db, identity, clock)

    assert first["status"] == "queued"
    assert repeated["idempotent"] is True

    with pytest.raises(driver.TaskConflictError):
        driver.admit_task(
            db,
            driver.TaskIdentity(
                room_id="room-1",
                task_id="task-1",
                thread_id="thread-other",
                turn_id="turn-other",
            ),
            payload=_payload(),
            clock=clock,
        )

    with pytest.raises(driver.TaskConflictError, match="different payload"):
        _admit(
            db,
            identity,
            clock,
            payload=_payload(prompt="A different immutable prompt."),
        )
    with pytest.raises(driver.TaskConflictError):
        driver.admit_task(
            db,
            driver.TaskIdentity(
                room_id="room-1",
                task_id="task-other",
                thread_id="thread-1",
                turn_id="turn-1",
            ),
            payload=_payload(),
            clock=clock,
        )


def test_task_admission_is_atomically_fenced_by_latest_stop(db):
    clock = FakeClock()
    room = rooms.room_state(db, room_id="room-1")
    first = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    stop = rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-1",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        _admit(
            db,
            _identity(),
            clock,
            payload=_payload(source_event_seq=first["seq"]),
        )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        _admit(
            db,
            _identity("task-at-stop", turn_id="turn-at-stop"),
            clock,
            payload=_payload(source_event_seq=stop["seq"]),
        )

    newer = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-after-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect again"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    admitted = _admit(
        db,
        _identity("task-2", turn_id="turn-2"),
        clock,
        payload=_payload(source_event_seq=newer["seq"]),
    )
    assert admitted["status"] == "queued"


def test_durable_admission_barrier_blocks_new_tasks_idempotently(db):
    clock = FakeClock()
    existing_identity = _identity()
    _admit(db, existing_identity, clock)
    lease = _lease(db, clock)

    first = driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=clock,
    )
    repeated = driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=clock,
    )

    assert first["idempotent"] is False
    assert repeated["idempotent"] is True
    with pytest.raises(driver.TaskAdmissionBlockedError, match="admission barrier"):
        _admit(db, _identity("task-2", turn_id="turn-2"), clock)
    with pytest.raises(driver.TaskAdmissionBlockedError, match="admission barrier"):
        driver.start_task(
            db,
            existing_identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        )


def test_room_demotion_intent_is_atomic_idempotent_and_conflict_checked(db):
    clock = FakeClock()
    first = driver.begin_room_demotion(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id="demotion-1",
        clock=clock,
    )
    repeated = driver.begin_room_demotion(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id="demotion-racing-process",
        clock=clock,
    )

    assert first["idempotent"] is False
    assert repeated["idempotent"] is True
    assert repeated["cancel_id"] == "demotion-1"
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        stop = conn.execute(
            """SELECT kind, actor_json, authority_epoch, payload_json
                 FROM hosted_room_events
                WHERE room_id='room-1' AND event_id=?""",
            (rooms._stop_event_id("demotion-1"),),
        ).fetchone()
    assert stop is not None
    assert stop["kind"] == "room.stop_requested"
    assert stop["authority_epoch"] == 1
    assert stop["actor_json"] == '{"id":"gateway-a","kind":"gateway"}'
    assert stop["payload_json"] == '{"cancel_id":"demotion-1"}'
    assert driver.pending_room_demotion(db, room_id="room-1") == {
        key: first[key] for key in first if key != "idempotent"
    }
    with pytest.raises(driver.TaskConflictError, match="different demotion intent"):
        driver.begin_room_demotion(
            db,
            room_id="room-1",
            expected_gateway_id="gateway-a",
            expected_epoch=1,
            observed_gateway_id="gateway-c",
            observed_epoch=3,
            cancel_id="demotion-2",
            clock=clock,
        )


def test_demotion_intent_and_barrier_roll_back_when_atomic_stop_fails(
    db,
    monkeypatch,
):
    def fail_stop(*args, **kwargs):
        raise rooms.HostedRoomError("no reserved Stop capacity")

    monkeypatch.setattr(rooms, "_request_room_stop_locked", fail_stop)

    with pytest.raises(rooms.HostedRoomError, match="reserved Stop capacity"):
        driver.begin_room_demotion(
            db,
            room_id="room-1",
            expected_gateway_id="gateway-a",
            expected_epoch=1,
            observed_gateway_id="gateway-b",
            observed_epoch=2,
            cancel_id="demotion-without-capacity",
            clock=FakeClock(),
        )

    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_admission_barriers"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_demotion_intents"
        ).fetchone()[0] == 0
        assert conn.execute(
            """SELECT COUNT(*) FROM hosted_room_events
               WHERE kind='room.stop_requested'"""
        ).fetchone()[0] == 0


def test_identical_demotion_race_reuses_the_winning_process_cancel_id(db):
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                _begin_demotion_from_process,
                [str(db), str(db)],
                ["demotion-process-a", "demotion-process-b"],
            )
        )

    assert sorted(result["idempotent"] for result in results) == [False, True]
    assert len({result["cancel_id"] for result in results}) == 1
    assert results[0]["observed_gateway_id"] == "gateway-b"
    assert results[1]["observed_gateway_id"] == "gateway-b"


def test_queued_task_cannot_start_after_stop_fence(db):
    clock = FakeClock()
    room = rooms.room_state(db, room_id="room-1")
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    identity = _identity()
    _admit(
        db,
        identity,
        clock,
        payload=_payload(source_event_seq=source["seq"]),
    )
    lease = _lease(db, clock)
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-1",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        driver.start_task(
            db,
            identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        )
    assert driver.get_task(db, identity)["status"] == "queued"


def test_reconcile_stop_fenced_inactive_tasks_cancels_only_superseded_work(db):
    clock = FakeClock()
    room = rooms.room_state(db, room_id="room-1")
    stale_source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-crash-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    stale_identity = _identity()
    _admit(
        db,
        stale_identity,
        clock,
        payload=_payload(source_event_seq=stale_source["seq"]),
    )
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-before-crash",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )
    current_source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-after-crash-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect again"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    current_identity = _identity("task-2", turn_id="turn-2")
    _admit(
        db,
        current_identity,
        clock,
        payload=_payload(source_event_seq=current_source["seq"]),
    )

    reconciled = driver.reconcile_stop_fenced_inactive_tasks(
        db,
        room_id="room-1",
        clock=clock,
    )
    repeated = driver.reconcile_stop_fenced_inactive_tasks(
        db,
        room_id="room-1",
        clock=clock,
    )

    assert reconciled == [stale_identity]
    assert repeated == []
    stale = driver.get_task(db, stale_identity)
    assert stale["status"] == "cancelled"
    assert stale["cancel_id"] == "stop-before-crash"
    assert stale["cancel_generation"] == 1
    assert driver.get_task(db, current_identity)["status"] == "queued"


def test_reconcile_stop_fenced_inactive_tasks_cancels_deferred_work(db):
    clock = FakeClock()
    room = rooms.room_state(db, room_id="room-1")
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-deferred-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    identity = _identity()
    first_lease = _lease(db, clock, ttl=5)
    _admit(
        db,
        identity,
        clock,
        payload=_payload(source_event_seq=source["seq"]),
    )
    attempt = driver.start_task(
        db,
        identity,
        first_lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(5)
    recovered_lease = _lease(db, clock, process="process-b")
    driver.recover_room(db, recovered_lease, clock=clock)
    driver.defer_indeterminate_task(
        db,
        identity,
        recovered_lease,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        reason="member_unavailable",
        clock=clock,
    )
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-deferred",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )

    assert driver.reconcile_stop_fenced_inactive_tasks(
        db,
        room_id="room-1",
        clock=clock,
    ) == [identity]
    cancelled = driver.get_task(db, identity)
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_id"] == "stop-deferred"


def test_admission_barrier_is_scoped_to_superseded_authority_epoch(db):
    clock = FakeClock()
    driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=clock,
    )
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-gateway-b",
        now=clock(),
    )
    current = rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-b",
        expected_epoch=2,
        new_gateway_id="gateway-a",
        event_id="claim-gateway-a",
        now=clock(),
    )
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-after-repromotion",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect after repromotion", "thread_id": "thread-1"},
        authority_gateway_id=current["authority_gateway_id"],
        authority_epoch=current["authority_epoch"],
        require_open_admissions=True,
    )
    identity = _identity()
    admitted = _admit(
        db,
        identity,
        clock,
        payload=_payload(source_event_seq=source["seq"]),
    )
    lease = _lease(db, clock, authority_epoch=3, process="process-epoch-3")

    assert admitted["status"] == "queued"
    assert (
        driver.start_task(
            db,
            identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        ).identity
        == identity
    )
    next_barrier = driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=3,
        clock=clock,
    )
    assert next_barrier["idempotent"] is False
    assert next_barrier["authority_epoch"] == 3


def test_concurrent_task_start_has_one_winner(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)

    def start(_):
        try:
            return driver.start_task(
                db,
                identity,
                lease,
                expected_cancel_generation=0,
                clock=clock,
            )
        except driver.InvalidTaskTransitionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, range(2)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0].execution_generation == 1


def test_stale_lease_cannot_start_or_commit_task(db):
    clock = FakeClock()
    identity = _identity()
    first = _lease(db, clock, ttl=5)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )

    clock.advance(5)
    second = _lease(db, clock, process="process-b")
    driver.recover_room(db, second, clock=clock)

    with pytest.raises(driver.StaleLeaseError):
        driver.settle_task(
            db,
            attempt,
            settlement_id="settlement-old",
            status="settled",
            result={"text": "late"},
            clock=clock,
        )
    assert driver.get_task(db, identity)["status"] == "indeterminate"


@pytest.mark.parametrize("status", ["settled", "failed"])
def test_terminal_settlement_is_idempotent(db, status):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    first = driver.settle_task(
        db,
        attempt,
        settlement_id="settlement-1",
        status=status,
        result={"text": "done"},
        clock=clock,
    )
    repeated = driver.settle_task(
        db,
        attempt,
        settlement_id="settlement-1",
        status=status,
        result={"text": "done"},
        clock=clock,
    )

    assert first["status"] == status
    assert repeated["idempotent"] is True
    with pytest.raises(driver.TaskConflictError):
        driver.settle_task(
            db,
            attempt,
            settlement_id="settlement-2",
            status=status,
            result={"text": "changed"},
            clock=clock,
        )


def test_cancellation_fences_late_success(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-1",
        expected_cancel_generation=0,
        clock=clock,
    )
    cancelled = driver.complete_task_cancel(
        db,
        identity,
        lease,
        cancel_id="cancel-1",
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=1,
        clock=clock,
    )
    repeated = driver.complete_task_cancel(
        db,
        identity,
        lease,
        cancel_id="cancel-1",
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=1,
        clock=clock,
    )

    assert stopping["status"] == "stopping"
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_generation"] == 1
    assert repeated["idempotent"] is True
    with pytest.raises(driver.StaleTaskError):
        driver.settle_task(
            db,
            attempt,
            settlement_id="late-success",
            status="settled",
            result={"text": "too late"},
            clock=clock,
        )


def test_stop_ack_refuses_exact_generation_terminal_receipt(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-after-receipt",
        expected_cancel_generation=0,
        clock=clock,
    )
    driver.record_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        settlement_id="reply-before-ack",
        status="settled",
        result={"text": "done"},
        clock=clock,
    )

    with pytest.raises(driver.TaskConflictError, match="terminal receipt"):
        driver.complete_task_cancel(
            db,
            identity,
            lease,
            cancel_id="cancel-after-receipt",
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=1,
            clock=clock,
        )

    assert driver.get_task(db, identity)["status"] == "stopping"


@pytest.mark.parametrize("owner_state", ["alive", "unknown"])
def test_successor_cannot_claim_stop_without_proven_owner_exit(db, owner_state):
    clock = FakeClock()
    identity = _identity()
    old = _lease(
        db,
        clock,
        process="old-process",
        process_pid=1111,
        process_start_time=7001,
        ttl=1,
    )
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db, identity, old, expected_cancel_generation=0, clock=clock
    )
    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-owner-live",
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(2)
    successor = _lease(
        db,
        clock,
        process="successor",
        process_pid=2222,
        process_start_time=8001,
    )

    with pytest.raises(driver.LeaseHeldError, match=owner_state):
        driver.claim_stopping_task(
            db,
            identity,
            successor,
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=stopping["cancel_generation"],
            owner_liveness=lambda _pid, _started: owner_state,
            clock=clock,
        )

    current = driver.get_task(db, identity)
    assert current["run_process_generation"] == old.process_generation
    assert current["status"] == "stopping"


def test_successor_claim_and_cancel_are_process_and_lease_fenced(db):
    clock = FakeClock()
    identity = _identity()
    old = _lease(
        db,
        clock,
        process="old-process",
        process_pid=1111,
        process_start_time=7001,
        ttl=1,
    )
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db, identity, old, expected_cancel_generation=0, clock=clock
    )
    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-dead-owner",
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(2)
    successor = _lease(
        db,
        clock,
        process="successor",
        process_pid=2222,
        process_start_time=8001,
    )
    seen = []

    claimed = driver.claim_stopping_task(
        db,
        identity,
        successor,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=stopping["cancel_generation"],
        owner_liveness=lambda pid, started: seen.append((pid, started)) or "dead",
        clock=clock,
    )

    assert seen == [(1111, 7001)]
    assert claimed["run_process_generation"] == successor.process_generation
    assert claimed["run_process_pid"] == successor.process_pid
    with pytest.raises(driver.StaleLeaseError):
        driver.complete_task_cancel(
            db,
            identity,
            old,
            cancel_id="cancel-dead-owner",
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=stopping["cancel_generation"],
            clock=clock,
        )
    cancelled = driver.complete_task_cancel(
        db,
        identity,
        successor,
        cancel_id="cancel-dead-owner",
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=stopping["cancel_generation"],
        clock=clock,
    )
    assert cancelled["status"] == "cancelled"


def test_same_process_successor_claim_does_not_probe_liveness(db):
    clock = FakeClock()
    identity = _identity()
    old = _lease(db, clock, process="old-runtime", ttl=1)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db, identity, old, expected_cancel_generation=0, clock=clock
    )
    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-same-process",
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(2)
    successor = _lease(db, clock, process="new-runtime")

    claimed = driver.claim_stopping_task(
        db,
        identity,
        successor,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=stopping["cancel_generation"],
        owner_liveness=lambda _pid, _started: pytest.fail(
            "same process must not probe external liveness"
        ),
        clock=clock,
    )

    assert claimed["run_process_generation"] == successor.process_generation


def test_terminal_receipt_wins_before_stop_owner_reclaim(db):
    clock = FakeClock()
    identity = _identity()
    old = _lease(db, clock, process="old-process", ttl=1)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db, identity, old, expected_cancel_generation=0, clock=clock
    )
    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-after-terminal",
        expected_cancel_generation=0,
        clock=clock,
    )
    driver.record_terminal_receipt(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        settlement_id="reply-before-reclaim",
        status="settled",
        result={"text": "done"},
        clock=clock,
    )
    clock.advance(2)
    successor = _lease(db, clock, process="successor", process_pid=2222)

    with pytest.raises(driver.TaskConflictError, match="terminal receipt"):
        driver.claim_stopping_task(
            db,
            identity,
            successor,
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=stopping["cancel_generation"],
            owner_liveness=lambda _pid, _started: "dead",
            clock=clock,
        )


def test_stale_successor_cannot_claim_after_lease_replacement(db):
    clock = FakeClock()
    identity = _identity()
    old = _lease(db, clock, process="old-process", ttl=1)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db, identity, old, expected_cancel_generation=0, clock=clock
    )
    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-stale-successor",
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(2)
    stale = _lease(db, clock, process="stale-successor", ttl=1)
    clock.advance(2)
    _lease(db, clock, process="current-successor")

    with pytest.raises(driver.StaleLeaseError):
        driver.claim_stopping_task(
            db,
            identity,
            stale,
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=stopping["cancel_generation"],
            owner_liveness=lambda _pid, _started: "dead",
            clock=clock,
        )


def test_approval_requests_are_stale_once_task_is_stopping(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    driver.publish_approval_request(
        db,
        identity,
        execution_generation=attempt.execution_generation,
        member_id="member-ops",
        request_id="approval-1",
        session_id="session-1",
        action={"tool": "shell", "command": "inspect"},
        clock=clock,
    )

    driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-before-approval",
        expected_cancel_generation=0,
        clock=clock,
    )

    assert driver.list_pending_approval_requests(db, room_id="room-1") == []
    with pytest.raises(driver.StaleTaskError, match="no longer running"):
        driver.decide_approval_request(
            db,
            identity,
            execution_generation=attempt.execution_generation,
            member_id="member-ops",
            request_id="approval-1",
            choice="once",
            clock=clock,
        )
    with pytest.raises(driver.InvalidTaskTransitionError, match="running task"):
        driver.publish_approval_request(
            db,
            identity,
            execution_generation=attempt.execution_generation,
            member_id="member-ops",
            request_id="approval-2",
            session_id="session-1",
            action={"tool": "shell", "command": "inspect again"},
            clock=clock,
        )
    with sqlite3.connect(db) as conn:
        choice = conn.execute(
            """SELECT choice FROM hosted_room_approval_requests
               WHERE room_id=? AND task_id=? AND execution_generation=?
                 AND member_id=? AND request_id=?""",
            (
                identity.room_id,
                identity.task_id,
                attempt.execution_generation,
                "member-ops",
                "approval-1",
            ),
        ).fetchone()[0]
    assert choice is None


def test_release_fails_closed_while_its_task_is_running(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    with pytest.raises(
        driver.InvalidTaskTransitionError,
        match="tasks are running",
    ):
        driver.release_lease(db, lease, clock=clock)
    assert driver.get_task(db, identity)["status"] == "running"

    driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-before-release",
        expected_cancel_generation=0,
        clock=clock,
    )
    driver.complete_task_cancel(
        db,
        identity,
        lease,
        cancel_id="cancel-before-release",
        expected_execution_generation=1,
        expected_cancel_generation=1,
        clock=clock,
    )
    assert driver.release_lease(db, lease, clock=clock)["idempotent"] is False


def test_restart_recovery_never_requeues_indeterminate_work(db):
    clock = FakeClock()
    running = _identity()
    queued = _identity("task-2", turn_id="turn-2")
    first = _lease(db, clock, ttl=5)
    _admit(db, running, clock)
    _admit(db, queued, clock)
    driver.start_task(
        db,
        running,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )

    with pytest.raises(driver.LeaseHeldError):
        _lease(db, clock, gateway="gateway-a", process="new-process")

    clock.advance(5)
    recovered_lease = _lease(
        db,
        clock,
        gateway="gateway-a",
        process="new-process",
    )
    recovery = driver.recover_room(db, recovered_lease, clock=clock)
    repeated = driver.recover_room(db, recovered_lease, clock=clock)

    assert recovery == {"queued": [queued], "indeterminate": [running]}
    assert repeated == recovery
    with pytest.raises(driver.InvalidTaskTransitionError):
        driver.start_task(
            db,
            running,
            recovered_lease,
            expected_cancel_generation=0,
            clock=clock,
        )
    assert [task["status"] for task in driver.list_tasks(db, room_id="room-1")] == [
        "indeterminate",
        "queued",
    ]


def test_recovery_is_required_before_starting_later_work(db):
    clock = FakeClock()
    running = _identity()
    queued = _identity("task-2", turn_id="turn-2")
    first = _lease(db, clock, ttl=5)
    _admit(db, running, clock, payload=_payload(source_event_seq=1))
    _admit(db, queued, clock, payload=_payload(source_event_seq=2))
    driver.start_task(
        db,
        running,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(5)
    recovered = _lease(db, clock, process="new-process")

    with pytest.raises(
        driver.InvalidTaskTransitionError,
        match="recovery must resolve",
    ):
        driver.start_task(
            db,
            queued,
            recovered,
            expected_cancel_generation=0,
            clock=clock,
        )


def test_current_lease_can_commit_verified_indeterminate_receipt(db):
    clock = FakeClock()
    running = _identity()
    queued = _identity("task-2", turn_id="turn-2")
    first = _lease(db, clock, ttl=5)
    _admit(db, running, clock, payload=_payload(source_event_seq=1))
    _admit(db, queued, clock, payload=_payload(source_event_seq=2))
    attempt = driver.start_task(
        db,
        running,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(5)
    recovered = _lease(db, clock, process="new-process")
    driver.recover_room(db, recovered, clock=clock)

    settled = driver.resolve_indeterminate_task(
        db,
        running,
        recovered,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        settlement_id="recovered-receipt",
        status="settled",
        result={"text": "recovered"},
        clock=clock,
    )
    next_attempt = driver.start_task(
        db,
        queued,
        recovered,
        expected_cancel_generation=0,
        clock=clock,
    )

    assert settled["status"] == "settled"
    assert next_attempt.execution_generation == 1


def test_indeterminate_retry_is_explicit_and_advances_execution_generation(db):
    clock = FakeClock()
    identity = _identity()
    first = _lease(db, clock, ttl=5)
    _admit(db, identity, clock)
    original = driver.start_task(
        db,
        identity,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(5)
    recovered = _lease(db, clock, process="new-process")
    driver.recover_room(db, recovered, clock=clock)

    requeued = driver.requeue_indeterminate_task(
        db,
        identity,
        recovered,
        expected_execution_generation=original.execution_generation,
        expected_cancel_generation=original.cancel_generation,
        clock=clock,
    )
    retried = driver.start_task(
        db,
        identity,
        recovered,
        expected_cancel_generation=0,
        clock=clock,
    )

    assert requeued["status"] == "queued"
    assert retried.execution_generation == original.execution_generation + 1


def test_running_task_profile_deferral_preserves_attempt_fences(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )

    deferred = driver.defer_running_task(
        db,
        attempt,
        reason="member_unavailable",
        clock=clock,
    )
    repeated = driver.defer_running_task(
        db,
        attempt,
        reason="member_unavailable",
        clock=clock,
    )
    requeued = driver.requeue_deferred_task(
        db,
        identity,
        lease,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        clock=clock,
    )

    assert deferred["status"] == "deferred"
    assert deferred["result"] == {
        "reason": "member_unavailable",
        "retryable": True,
    }
    assert repeated["idempotent"] is True
    assert requeued["status"] == "queued"
    with pytest.raises(driver.StaleTaskError):
        driver.defer_running_task(
            db,
            attempt,
            reason="member_unavailable",
            clock=clock,
        )


def test_indeterminate_task_can_be_deferred_retried_and_cancelled(db):
    clock = FakeClock()
    identity = _identity()
    first = _lease(db, clock, ttl=5)
    _admit(db, identity, clock)
    original = driver.start_task(
        db,
        identity,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(5)
    recovered = _lease(db, clock, process="new-process", ttl=5)
    driver.recover_room(db, recovered, clock=clock)

    deferred = driver.defer_indeterminate_task(
        db,
        identity,
        recovered,
        expected_execution_generation=original.execution_generation,
        expected_cancel_generation=original.cancel_generation,
        reason="member_unavailable",
        clock=clock,
    )
    repeated = driver.defer_indeterminate_task(
        db,
        identity,
        recovered,
        expected_execution_generation=original.execution_generation,
        expected_cancel_generation=original.cancel_generation,
        reason="member_unavailable",
        clock=clock,
    )
    requeued = driver.requeue_deferred_task(
        db,
        identity,
        recovered,
        expected_execution_generation=original.execution_generation,
        expected_cancel_generation=original.cancel_generation,
        clock=clock,
    )
    retried = driver.start_task(
        db,
        identity,
        recovered,
        expected_cancel_generation=0,
        clock=clock,
    )

    assert deferred["status"] == "deferred"
    assert deferred["result"] == {
        "reason": "member_unavailable",
        "retryable": True,
    }
    assert repeated["idempotent"] is True
    assert requeued["status"] == "queued"
    assert retried.execution_generation == original.execution_generation + 1

    clock.advance(5)
    next_lease = _lease(db, clock, process="third-process")
    driver.recover_room(db, next_lease, clock=clock)
    driver.defer_indeterminate_task(
        db,
        identity,
        next_lease,
        expected_execution_generation=retried.execution_generation,
        expected_cancel_generation=retried.cancel_generation,
        reason="member_unavailable",
        clock=clock,
    )
    cancelled = driver.cancel_task(
        db,
        identity,
        cancel_id="cancel-deferred",
        expected_cancel_generation=0,
        clock=clock,
    )
    assert cancelled["status"] == "cancelled"


def test_stop_fence_blocks_requeue_of_an_indeterminate_task(db):
    clock = FakeClock()
    identity = _identity()
    first = _lease(db, clock, ttl=5)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )
    clock.advance(5)
    recovered = _lease(db, clock, process="recovered-process", ttl=30)
    driver.recover_room(db, recovered, clock=clock)
    room = rooms.room_state(db, room_id="room-1")
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-before-indeterminate-requeue",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        driver.requeue_indeterminate_task(
            db,
            identity,
            recovered,
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=attempt.cancel_generation,
            clock=clock,
        )

    assert driver.get_task(db, identity)["status"] == "indeterminate"


def test_stop_fence_blocks_requeue_of_a_deferred_task(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    driver.defer_running_task(
        db,
        attempt,
        reason="member_unavailable",
        clock=clock,
    )
    room = rooms.room_state(db, room_id="room-1")
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-before-deferred-requeue",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        driver.requeue_deferred_task(
            db,
            identity,
            lease,
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=attempt.cancel_generation,
            clock=clock,
        )

    assert driver.get_task(db, identity)["status"] == "deferred"


def test_admission_is_bounded_by_terminal_recovery_liability(db, monkeypatch):
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 2)
    clock = FakeClock()
    _admit(db, _identity("task-1", turn_id="turn-1"), clock)

    with pytest.raises(driver.TaskAdmissionBlockedError, match="recovery headroom"):
        _admit(db, _identity("task-2", turn_id="turn-2"), clock)

    assert len(driver.list_tasks(db, room_id="room-1")) == 1


def test_terminal_publication_transfers_reservation_until_room_activity(
    db,
    monkeypatch,
):
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 2)
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="discussion-1",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect", "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        require_open_admissions=True,
    )

    admitted = _admit(
        db,
        _identity(),
        FakeClock(),
        payload=_payload(source_event_seq=source["seq"]),
    )

    assert admitted["status"] == "queued"
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == {
            ("room-1", "task-1")
        }

    driver.cancel_task(
        db,
        _identity(),
        cancel_id="cancel-discussion-1",
        expected_cancel_generation=0,
        clock=FakeClock(),
    )
    rooms.append_events(
        db,
        events=[
            {
                "room_id": "room-1",
                "event_id": "terminal-discussion-1",
                "kind": "turn.cancelled",
                "actor": {"kind": "gateway", "id": "gateway-a"},
                "payload": {
                    "task_id": "task-1",
                    "member_id": "member-ops",
                    "member_index": 0,
                    "round_index": 0,
                    "seen_through_seq": source["seq"],
                    "thread_id": "thread-1",
                    "turn_id": "turn-1",
                    "discussion_event_id": "discussion-1",
                },
                "authority_gateway_id": "gateway-a",
                "authority_epoch": 1,
            }
        ],
        allow_terminal_recovery=True,
    )
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == {
            rooms._discussion_liability_key("room-1", "thread-1")
        }

    latest = rooms.room_state(db, room_id="room-1")["latest_seq"]
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", latest)
    closed = rooms.append_events(
        db,
        events=[
            {
                "room_id": "room-1",
                "event_id": "activity-discussion-1",
                "kind": "room.activity",
                "actor": {"kind": "gateway", "id": "gateway-a"},
                "payload": {
                    "status": "settled",
                    "reason_code": "complete",
                    "thread_id": "thread-1",
                    "discussion_event_id": "discussion-1",
                },
                "authority_gateway_id": "gateway-a",
                "authority_epoch": 1,
            }
        ],
    )

    assert closed[0]["kind"] == "room.activity"
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == set()


def test_closed_published_deferred_task_releases_liability_and_is_pruned(db):
    clock = FakeClock()
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="discussion-deferred",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect", "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        require_open_admissions=True,
    )
    identity = _identity()
    _admit(
        db,
        identity,
        clock,
        payload=_payload(source_event_seq=source["seq"]),
    )
    lease = _lease(db, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    deferred = driver.defer_running_task(
        db,
        attempt,
        reason="member_unavailable",
        clock=clock,
    )
    for event_id, kind, payload in (
        (
            "deferred-task-1",
            "turn.deferred",
            {
                "task_id": "task-1",
                "execution_generation": deferred["execution_generation"],
                "member_id": "member-ops",
                "member_index": 0,
                "round_index": 0,
                "seen_through_seq": source["seq"],
                "thread_id": "thread-1",
                "turn_id": "turn-1",
                "discussion_event_id": "discussion-deferred",
            },
        ),
        (
            "activity-deferred",
            "room.activity",
            {
                "status": "bounded",
                "reason_code": "deferred",
                "thread_id": "thread-1",
                "discussion_event_id": "discussion-deferred",
            },
        ),
    ):
        rooms.append_event(
            db,
            room_id="room-1",
            event_id=event_id,
            kind=kind,
            actor={"kind": "gateway", "id": "gateway-a"},
            payload=payload,
            authority_gateway_id="gateway-a",
            authority_epoch=1,
        )

    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == set()
    assert (
        driver.prune_closed_published_deferred_tasks(db, room_id="room-1")
        == 1
    )
    assert driver.list_tasks(db, room_id="room-1") == []


def test_uncorrelated_terminal_event_cannot_consume_existing_liability_headroom(
    db,
    monkeypatch,
):
    _admit(db, _identity(), FakeClock())
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 0)
    monkeypatch.setattr(rooms, "STOP_EVENT_COUNT_RESERVE", 1)
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 2)
    monkeypatch.setattr(rooms, "CONTROL_EVENT_COUNT_RESERVE", 4)

    with pytest.raises(rooms.HostedRoomError, match="preserve terminal recovery"):
        rooms.append_events(
            db,
            events=[
                {
                    "room_id": "room-1",
                    "event_id": "terminal-for-unrelated-task",
                    "kind": "turn.cancelled",
                    "actor": {"kind": "gateway", "id": "gateway-a"},
                    "payload": {"task_id": "task-1"},
                    "authority_gateway_id": "gateway-a",
                    "authority_epoch": 1,
                }
            ],
            allow_terminal_recovery=True,
        )


def test_room_activity_thread_must_match_before_releasing_discussion_liability(
    db,
):
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="discussion-thread-fence",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect", "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        require_open_admissions=True,
    )
    rooms.append_event(
        db,
        room_id="room-1",
        event_id="activity-wrong-thread",
        kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={
            "status": "bounded",
            "thread_id": "thread-other",
            "discussion_event_id": source["event_id"],
        },
        authority_gateway_id="gateway-a",
        authority_epoch=1,
    )

    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == {
            rooms._discussion_liability_key("room-1", "thread-1")
        }

    rooms.append_event(
        db,
        room_id="room-1",
        event_id="activity-exact-thread",
        kind="room.activity",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={
            "status": "bounded",
            "thread_id": "thread-1",
            "discussion_event_id": source["event_id"],
        },
        authority_gateway_id="gateway-a",
        authority_epoch=1,
    )

    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == set()


def test_durable_terminal_publication_releases_liability_at_exact_boundary(
    db,
    monkeypatch,
):
    clock = FakeClock()
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="discussion-at-boundary",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect", "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        require_open_admissions=True,
    )
    identity = _identity()
    _admit(
        db,
        identity,
        clock,
        payload=_payload(source_event_seq=source["seq"]),
    )
    driver.cancel_task(
        db,
        identity,
        cancel_id="cancel-at-boundary",
        expected_cancel_generation=0,
        clock=clock,
    )
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", source["seq"])
    monkeypatch.setattr(rooms, "STOP_EVENT_COUNT_RESERVE", 1)
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 2)
    monkeypatch.setattr(rooms, "CONTROL_EVENT_COUNT_RESERVE", 5)

    appended = rooms.append_events(
        db,
        events=[
            {
                "room_id": "room-1",
                "event_id": "terminal-for-task-1",
                "kind": "turn.cancelled",
                "actor": {"kind": "gateway", "id": "gateway-a"},
                "payload": {
                    "task_id": "task-1",
                    "member_id": "member-ops",
                    "member_index": 0,
                    "round_index": 0,
                    "seen_through_seq": source["seq"],
                    "thread_id": "thread-1",
                    "turn_id": "turn-1",
                    "discussion_event_id": "discussion-at-boundary",
                    "reason": "member turn cancelled",
                },
                "authority_gateway_id": "gateway-a",
                "authority_epoch": 1,
            }
        ],
        allow_terminal_recovery=True,
    )

    assert appended[0]["kind"] == "turn.cancelled"
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == {
            rooms._discussion_liability_key("room-1", "thread-1")
        }


def test_existing_liabilities_over_recovery_reserve_fail_loudly(db, monkeypatch):
    identity = _identity()
    _admit(db, identity, FakeClock())
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 0)
    current_pid = driver.os.getpid()
    monkeypatch.setattr(driver.os, "getpid", lambda: current_pid + 1)

    with pytest.raises(driver.DriverStateError, match="exceed durable terminal"):
        driver.get_task(db, identity)


def test_terminal_recovery_audit_runs_once_per_process_schema(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=90,
    )
    original = driver._raise_if_terminal_recovery_headroom_is_unrecoverable
    calls = 0

    def counted(conn):
        nonlocal calls
        calls += 1
        return original(conn)

    monkeypatch.setattr(
        driver,
        "_raise_if_terminal_recovery_headroom_is_unrecoverable",
        counted,
    )
    identity = _identity()
    _admit(db, identity, FakeClock())

    for _ in range(5):
        assert driver.get_task(db, identity)["status"] == "queued"
        assert len(driver.list_tasks(db, room_id="room-1")) == 1

    assert calls == 1


def test_state_survives_sqlite_reopen_and_concurrent_duplicate_admission(db):
    clock = FakeClock()
    identity = _identity()

    def admit(_):
        return _admit(db, identity, clock)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(admit, range(8)))

    assert sum(not result["idempotent"] for result in results) == 1
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_tasks"
        ).fetchone()[0]
    assert count == 1
    reopened = driver.get_task(db, identity)
    listed = driver.list_tasks(db, room_id="room-1")
    assert reopened["identity"] == identity
    assert reopened["payload"] == _payload()
    assert listed[0]["payload"] == _payload()


def test_pre_admission_barrier_schema_is_migrated_without_losing_work(db):
    identity = _identity()
    _admit(db, identity, FakeClock())
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_admission_barriers")

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        table = conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_admission_barriers'"""
        ).fetchone()
        task_count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_tasks"
        ).fetchone()[0]

    assert table == ("hosted_room_driver_admission_barriers",)
    assert task_count == 1


def test_pre_demotion_intent_schema_is_migrated_without_losing_work(db):
    identity = _identity()
    _admit(db, identity, FakeClock())
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_demotion_intents")

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        table = conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_demotion_intents'"""
        ).fetchone()
        task_count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_tasks"
        ).fetchone()[0]

    assert table == ("hosted_room_driver_demotion_intents",)
    assert task_count == 1


def test_barrier_only_demotion_schema_fails_loudly_instead_of_wedging(db):
    driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=FakeClock(),
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_demotion_intents")

    with pytest.raises(driver.DriverStateError, match="lacks resumable target"):
        driver.pending_room_demotion(db, room_id="room-1")


def test_completed_legacy_demotion_barrier_does_not_block_schema_migration(db):
    driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=FakeClock(),
    )
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="authority-claimed-by-gateway-b",
        now=101.0,
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_demotion_intents")

    assert driver.pending_room_demotion(db, room_id="room-1") is None
    with sqlite3.connect(db) as conn:
        table = conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_demotion_intents'"""
        ).fetchone()
    assert table == ("hosted_room_driver_demotion_intents",)


def test_current_demotion_intent_without_atomic_stop_fails_loudly(db):
    driver.begin_room_demotion(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id="demotion-lost-stop",
        clock=FakeClock(),
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "DELETE FROM hosted_room_events WHERE room_id='room-1' AND event_id=?",
            (rooms._stop_event_id("demotion-lost-stop"),),
        )

    with pytest.raises(driver.DriverStateError, match="lacks its atomic Stop"):
        driver.pending_room_demotion(db, room_id="room-1")


def test_prune_removes_only_old_published_terminal_tasks(db):
    clock = FakeClock()
    lease = _lease(db, clock)
    published = _identity("task-published", turn_id="turn-published")
    unpublished = _identity("task-unpublished", turn_id="turn-unpublished")
    for identity in (published, unpublished):
        _admit(db, identity, clock)
        attempt = driver.start_task(
            db,
            identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        )
        driver.settle_task(
            db,
            attempt,
            settlement_id=f"result:{identity.task_id}",
            status="settled",
            result={"text": "done"},
            clock=clock,
        )

    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_policy_publications (
                room_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                execution_generation INTEGER NOT NULL DEFAULT 0,
                seq INTEGER NOT NULL,
                PRIMARY KEY(room_id, task_id, kind, execution_generation)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_publications
               VALUES ('room-1', 'task-published', 'turn.settled', 0, 3)"""
        )

    clock.advance(driver.TERMINAL_TASK_RETENTION_SECONDS + 1)
    assert (
        driver.prune_published_terminal_tasks(
            db,
            room_id="room-1",
            clock=clock,
        )
        == 1
    )
    assert [
        task["identity"].task_id for task in driver.list_tasks(db, room_id="room-1")
    ] == ["task-unpublished"]


def test_unpublished_legacy_driver_schema_fails_closed(db):
    conn = sqlite3.connect(db)
    try:
        conn.execute("DROP TABLE IF EXISTS hosted_room_driver_tasks")
        conn.execute("DROP TABLE IF EXISTS hosted_room_driver_leases")
        conn.execute(
            """CREATE TABLE hosted_room_driver_leases (
                room_id TEXT PRIMARY KEY,
                gateway_id TEXT NOT NULL,
                process_generation TEXT NOT NULL,
                lease_generation INTEGER NOT NULL,
                expires_at REAL NOT NULL,
                acquired_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                released_at REAL
            )"""
        )
        conn.execute(
            """CREATE TABLE hosted_room_driver_tasks (
                room_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                status TEXT NOT NULL,
                execution_generation INTEGER NOT NULL,
                cancel_generation INTEGER NOT NULL,
                run_gateway_id TEXT,
                run_process_generation TEXT,
                run_lease_generation INTEGER,
                cancel_id TEXT,
                settlement_id TEXT,
                settlement_status TEXT,
                result_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_at REAL,
                terminal_at REAL,
                indeterminate_at REAL,
                PRIMARY KEY (room_id, task_id),
                UNIQUE (room_id, thread_id, turn_id)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_driver_leases
               VALUES ('room-1', 'gateway-a', 'old-process', 1,
                       200, 100, 100, NULL)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_driver_tasks
               (room_id, task_id, thread_id, turn_id, status,
                execution_generation, cancel_generation,
                run_gateway_id, run_process_generation, run_lease_generation,
                created_at, updated_at, started_at)
               VALUES ('room-1', 'task-1', 'thread-1', 'turn-1', 'running',
                       1, 0, 'gateway-a', 'old-process', 1, 100, 100, 100)"""
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(driver.DriverStateError, match="unsupported unpublished"):
        driver.get_task(db, _identity())


def test_pre_owner_identity_schema_is_migrated_without_losing_work(db):
    clock = FakeClock()
    identity = _identity()
    _admit(db, identity, clock)
    _lease(db, clock)

    with sqlite3.connect(db) as conn:
        conn.execute(
            "ALTER TABLE hosted_room_driver_leases DROP COLUMN process_pid"
        )
        conn.execute(
            "ALTER TABLE hosted_room_driver_leases DROP COLUMN process_start_time"
        )
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks DROP COLUMN run_process_pid"
        )
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks DROP COLUMN run_process_start_time"
        )

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        lease_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_driver_leases)")
        }
        task_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_driver_tasks)")
        }
        legacy_lease = conn.execute(
            """SELECT process_generation, process_pid, process_start_time,
                      lease_generation
               FROM hosted_room_driver_leases WHERE room_id='room-1'"""
        ).fetchone()

    assert {"process_pid", "process_start_time"} <= lease_columns
    assert {"run_process_pid", "run_process_start_time"} <= task_columns
    assert legacy_lease == ("process-a", None, None, 1)
    with pytest.raises(driver.LeaseHeldError, match="another generation"):
        _lease(db, clock)

    clock.advance(31)
    successor = _lease(
        db,
        clock,
        process="process-b",
        process_pid=2002,
        process_start_time=6002,
    )
    assert successor.lease_generation == 2
    assert successor.reclaimed is True
    driver.start_task(
        db,
        identity,
        successor,
        expected_cancel_generation=0,
        clock=clock,
    )
    running = driver.get_task(db, identity)
    assert running["run_process_generation"] == "process-b"
    assert running["run_process_pid"] == 2002
    assert running["run_process_start_time"] == 6002


def test_pre_stopping_schema_is_migrated_without_losing_tasks(db):
    clock = FakeClock()
    identity = _identity()
    _admit(db, identity, clock)

    with sqlite3.connect(db) as conn:
        current_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='hosted_room_driver_tasks'"
        ).fetchone()[0]
        old_sql = current_sql.replace(", 'stopping'", "")
        conn.execute("DROP INDEX idx_hosted_room_driver_tasks_status")
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks RENAME TO hosted_room_driver_tasks_current"
        )
        conn.execute(old_sql)
        columns = ", ".join(driver._TASK_COLUMN_ORDER)
        conn.execute(
            f"""INSERT INTO hosted_room_driver_tasks ({columns})
                 SELECT {columns} FROM hosted_room_driver_tasks_current"""
        )
        conn.execute("DROP TABLE hosted_room_driver_tasks_current")
        conn.execute(
            """CREATE INDEX idx_hosted_room_driver_tasks_status
               ON hosted_room_driver_tasks(
                   room_id, status, source_event_seq, created_at, task_id
               )"""
        )

    assert driver.get_task(db, identity)["status"] == "queued"
    lease = _lease(db, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    stopping = driver.begin_task_cancel(
        db,
        identity,
        cancel_id="cancel-after-upgrade",
        expected_cancel_generation=attempt.cancel_generation,
        clock=clock,
    )

    assert stopping["status"] == "stopping"
    with sqlite3.connect(db) as conn:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='hosted_room_driver_tasks'"
        ).fetchone()[0]
    assert "'stopping'" in table_sql


def test_pre_deferred_schema_is_migrated_without_losing_tasks(db):
    clock = FakeClock()
    identity = _identity()
    _admit(db, identity, clock)

    with sqlite3.connect(db) as conn:
        current_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='hosted_room_driver_tasks'"
        ).fetchone()[0]
        old_sql = current_sql.replace(", 'deferred'", "")
        conn.execute("DROP INDEX idx_hosted_room_driver_tasks_status")
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks RENAME TO hosted_room_driver_tasks_current"
        )
        conn.execute(old_sql)
        columns = ", ".join(driver._TASK_COLUMN_ORDER)
        conn.execute(
            f"""INSERT INTO hosted_room_driver_tasks ({columns})
                 SELECT {columns} FROM hosted_room_driver_tasks_current"""
        )
        conn.execute("DROP TABLE hosted_room_driver_tasks_current")
        conn.execute(
            """CREATE INDEX idx_hosted_room_driver_tasks_status
               ON hosted_room_driver_tasks(
                   room_id, status, source_event_seq, created_at, task_id
               )"""
        )

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='hosted_room_driver_tasks'"
        ).fetchone()[0]
    assert "'deferred'" in table_sql


def test_first_schema_creation_is_safe_across_processes(db):
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE IF EXISTS hosted_room_driver_tasks")
        conn.execute("DROP TABLE IF EXISTS hosted_room_driver_leases")
        conn.commit()

    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_open_driver_schema, [str(db)] * 4))

    assert results == [0, 0, 0, 0]


def test_tasks_follow_source_event_order_not_admission_time(db):
    clock = FakeClock()
    later = _identity("task-2", turn_id="turn-2")
    earlier = _identity("task-1", turn_id="turn-1")
    _admit(db, later, clock, payload=_payload(source_event_seq=2))
    _admit(db, earlier, clock, payload=_payload(source_event_seq=1))
    lease = _lease(db, clock)

    assert [task["identity"] for task in driver.list_tasks(db, room_id="room-1")] == [
        earlier,
        later,
    ]
    with pytest.raises(driver.InvalidTaskTransitionError, match="event order"):
        driver.start_task(
            db,
            later,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        )
    assert (
        driver.start_task(
            db,
            earlier,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        ).identity
        == earlier
    )


def test_payload_digest_is_verified_on_read(db):
    identity = _identity()
    _admit(db, identity, FakeClock())
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_driver_tasks
               SET payload_json=REPLACE(payload_json, 'Inspect', 'Replace')
               WHERE room_id=? AND task_id=?""",
            (identity.room_id, identity.task_id),
        )
        conn.commit()

    with pytest.raises(driver.TaskConflictError, match="integrity"):
        driver.get_task(db, identity)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"target_profile": "ops", "prompt": "hello"}, "missing payload fields"),
        (
            {**_payload(), "unexpected": True},
            "unknown payload fields",
        ),
        (_payload(target_profile="bad profile"), "invalid target_profile"),
        (_payload(prompt="   "), "prompt must not be empty"),
        (_payload(prompt="x" * (driver.MAX_PROMPT_BYTES + 1)), "prompt is too large"),
        (_payload(source_event_seq=0), "source_event_seq"),
        (_payload(source_event_seq=True), "source_event_seq"),
    ],
)
def test_invalid_task_payload_is_rejected(db, payload, match):
    with pytest.raises(driver.DriverValidationError, match=match):
        _admit(db, _identity(), FakeClock(), payload=payload)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: driver.TaskIdentity("bad room", "task", "thread", "turn"),
            "invalid room_id",
        ),
        (
            lambda: driver.TaskIdentity("room", "", "thread", "turn"),
            "invalid task_id",
        ),
    ],
)
def test_invalid_task_identity_is_rejected(factory, match):
    with pytest.raises(driver.DriverValidationError, match=match):
        factory()


def test_invalid_lease_clock_ttl_and_settlement_schema_are_rejected(db):
    clock = FakeClock()

    with pytest.raises(driver.DriverValidationError, match="ttl_seconds"):
        _lease(db, clock, ttl=0)
    with pytest.raises(driver.DriverValidationError, match="clock"):
        _lease(db, lambda: float("nan"))
    with pytest.raises(driver.DriverValidationError, match="expiry"):
        _lease(db, FakeClock(1e308), ttl=1e308)

    identity = _identity()
    lease = _lease(db, clock)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    with pytest.raises(driver.DriverValidationError, match="JSON-serializable"):
        driver.settle_task(
            db,
            attempt,
            settlement_id="settlement-1",
            status="settled",
            result={"bad": object()},
            clock=clock,
        )


def test_renewal_never_shortens_an_active_lease(db):
    clock = FakeClock()
    lease = _lease(db, clock, ttl=30)
    clock.advance(1)

    renewed = driver.renew_lease(db, lease, ttl_seconds=2, clock=clock)

    assert renewed.expires_at == lease.expires_at
