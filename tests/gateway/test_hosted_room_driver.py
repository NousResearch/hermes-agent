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
    target_member_id: str | None = None,
):
    payload = {
        "target_profile": target_profile,
        "prompt": prompt,
        "source_event_seq": source_event_seq,
    }
    if target_member_id is not None:
        payload["target_member_id"] = target_member_id
    return payload


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "state.db"
    rooms.create_room(
        path,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "ops", "handle": "ops"}],
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
    ttl=30,
):
    return driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=gateway,
        authority_epoch=authority_epoch,
        process_generation=process,
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
        cancel_id="cancel-1",
        expected_cancel_generation=1,
        clock=clock,
    )
    repeated = driver.complete_task_cancel(
        db,
        identity,
        cancel_id="cancel-1",
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
        cancel_id="cancel-before-release",
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
    repeated = driver.resolve_indeterminate_task(
        db,
        running,
        recovered,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        settlement_id="recovered-receipt",
        status="settled",
        result={"text": "recovered"},
        clock=clock,
        retry_id="retry-terminal",
    )
    next_attempt = driver.start_task(
        db,
        queued,
        recovered,
        expected_cancel_generation=0,
        clock=clock,
    )

    assert settled["status"] == "settled"
    assert repeated["idempotent"] is True
    assert driver.retry_receipt_exists(
        db,
        room_id=running.room_id,
        task_id=running.task_id,
        retry_id="retry-terminal",
    )
    assert next_attempt.execution_generation == 1


def test_current_lease_can_commit_verified_indeterminate_cancellation(db):
    clock = FakeClock()
    running = _identity()
    first = _lease(db, clock, ttl=5)
    _admit(db, running, clock, payload=_payload(source_event_seq=1))
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

    cancelled = driver.resolve_indeterminate_cancellation(
        db,
        running,
        recovered,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        cancel_id="remote-cancel:1",
        clock=clock,
    )
    repeated = driver.resolve_indeterminate_cancellation(
        db,
        running,
        recovered,
        expected_execution_generation=attempt.execution_generation,
        expected_cancel_generation=attempt.cancel_generation,
        cancel_id="remote-cancel:1",
        clock=clock,
        retry_id="retry-cancelled",
    )

    assert cancelled["status"] == "cancelled"
    assert cancelled["execution_generation"] == attempt.execution_generation
    assert repeated["idempotent"] is True
    assert driver.retry_receipt_exists(
        db,
        room_id=running.room_id,
        task_id=running.task_id,
        retry_id="retry-cancelled",
    )


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


def test_proven_not_admitted_attempt_returns_to_queue_under_exact_fence(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    admitted = _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=admitted["cancel_generation"],
        clock=clock,
    )

    queued = driver.requeue_not_admitted_task(db, attempt, clock=clock)
    repeated = driver.requeue_not_admitted_task(db, attempt, clock=clock)

    assert queued["status"] == "queued"
    assert queued["execution_generation"] == attempt.execution_generation
    assert queued["payload"] == admitted["payload"]
    assert queued["run_gateway_id"] is None
    assert queued["run_process_generation"] is None
    assert queued["run_lease_generation"] is None
    assert repeated["idempotent"] is True


def test_not_admitted_requeue_rejects_stale_lease_and_task_generation(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock, ttl=5)
    _admit(db, identity, clock)
    attempt = driver.start_task(
        db,
        identity,
        lease,
        expected_cancel_generation=0,
        clock=clock,
    )
    stale_attempt = driver.TaskAttempt(
        identity=identity,
        lease=lease,
        execution_generation=attempt.execution_generation + 1,
        cancel_generation=attempt.cancel_generation,
    )

    with pytest.raises(driver.StaleTaskError, match="lost its fence"):
        driver.requeue_not_admitted_task(db, stale_attempt, clock=clock)

    clock.advance(5)
    with pytest.raises(driver.StaleLeaseError):
        driver.requeue_not_admitted_task(db, attempt, clock=clock)

    assert driver.get_task(db, identity)["status"] == "running"


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


def test_prune_retains_recoverable_artifact_retry_but_releases_expired_block(db):
    clock = FakeClock()
    lease = _lease(db, clock)
    identities = (
        _identity("task-expired-block", turn_id="turn-expired-block"),
        _identity("task-recoverable", turn_id="turn-recoverable"),
    )
    for identity in identities:
        _admit(db, identity, clock)
        attempt = driver.start_task(
            db, identity, lease, expected_cancel_generation=0, clock=clock
        )
        driver.settle_task(
            db,
            attempt,
            settlement_id=f"result:{identity.task_id}",
            status="settled",
            result={"artifacts": {"version": 1}},
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
        conn.executemany(
            "INSERT INTO hosted_room_policy_publications VALUES (?, ?, ?, 0, ?)",
            (
                ("room-1", identity.task_id, "turn.settled", index)
                for index, identity in enumerate(identities, start=1)
            ),
        )
        conn.execute(
            """CREATE TABLE hosted_room_artifact_retries (
                room_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                execution_generation INTEGER NOT NULL,
                member_id TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                next_attempt_at REAL NOT NULL,
                blocked INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY(room_id, task_id, execution_generation)
            )"""
        )
        conn.executemany(
            """INSERT INTO hosted_room_artifact_retries
               VALUES ('room-1', ?, 1, 'ops', 1, 0, 1, ?)""",
            ((identities[0].task_id, clock()), (identities[1].task_id, clock())),
        )

    clock.advance(driver.ARTIFACT_RETRY_RETENTION_SECONDS + 1)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """UPDATE hosted_room_artifact_retries SET updated_at=?
               WHERE task_id=?""",
            (clock(), identities[1].task_id),
        )

    assert driver.prune_published_terminal_tasks(
        db, room_id="room-1", clock=clock
    ) == 1
    assert [
        task["identity"].task_id for task in driver.list_tasks(db, room_id="room-1")
    ] == [identities[1].task_id]


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


def test_optional_target_member_id_is_durable_and_digest_bound(db):
    identity = _identity()
    _admit(
        db,
        identity,
        FakeClock(),
        payload=_payload(target_member_id="member-remote"),
    )

    assert driver.get_task(db, identity)["payload"]["target_member_id"] == (
        "member-remote"
    )


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


def test_active_lease_revalidation_does_not_extend_expiry(db):
    clock = FakeClock()
    lease = _lease(db, clock, ttl=5)

    validated = driver.require_active_lease(db, lease, clock=clock)

    assert validated == lease
    clock.advance(5)
    with pytest.raises(driver.StaleLeaseError):
        driver.require_active_lease(db, lease, clock=clock)


@pytest.mark.parametrize("cooldown", [None, 10.0])
def test_start_task_atomically_honors_room_stop_fence(db, cooldown):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-start",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "Start", "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
    )
    _admit(db, identity, clock)
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-before-start",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
    )

    assert (
        driver.start_task(
            db,
            identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
            not_before=clock() + cooldown if cooldown is not None else None,
        )
        is None
    )
    assert driver.get_task(db, identity)["status"] == "cancelled"


def test_start_task_preserves_attempt_until_cooldown_finishes(db):
    clock = FakeClock()
    identity = _identity()
    lease = _lease(db, clock)
    before = _admit(db, identity, clock)
    ready_at = clock() + 10
    assert driver.start_task(db, identity, lease, expected_cancel_generation=0,
                             clock=clock, not_before=ready_at) is None
    assert driver.get_task(db, identity) == before
    clock.advance(10)
    attempt = driver.start_task(db, identity, lease, expected_cancel_generation=0,
                                clock=clock, not_before=ready_at)
    assert attempt.execution_generation == 1
    assert driver.get_task(db, identity)["status"] == "running"


@pytest.mark.parametrize("retry_state", ["deferred", "indeterminate"])
def test_retry_atomically_honors_room_stop_fence(db, retry_state):
    clock = FakeClock()
    identity = _identity()
    first = _lease(db, clock, ttl=5)
    rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-retry",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "Retry", "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
    )
    _admit(db, identity, clock)
    original = driver.start_task(
        db,
        identity,
        first,
        expected_cancel_generation=0,
        clock=clock,
    )
    assert original is not None
    clock.advance(5)
    recovered = _lease(db, clock, process="new-process", ttl=30)
    driver.recover_room(db, recovered, clock=clock)
    if retry_state == "deferred":
        driver.defer_indeterminate_task(
            db,
            identity,
            recovered,
            expected_execution_generation=original.execution_generation,
            expected_cancel_generation=original.cancel_generation,
            reason="member_unavailable",
            clock=clock,
        )
    rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-before-retry",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
    )

    if retry_state == "deferred":
        retried = driver.requeue_deferred_task(
            db,
            identity,
            recovered,
            expected_execution_generation=original.execution_generation,
            expected_cancel_generation=original.cancel_generation,
            clock=clock,
        )
    else:
        retried = driver.requeue_indeterminate_task(
            db,
            identity,
            recovered,
            expected_execution_generation=original.execution_generation,
            expected_cancel_generation=original.cancel_generation,
            clock=clock,
        )

    assert retried["status"] == "cancelled"
    assert retried["cancel_id"].startswith("stop-fence:")


def test_draft_retry_id_column_is_migrated_to_receipt_ledger(db):
    clock = FakeClock()
    identity = _identity()
    _admit(db, identity, clock)

    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE hosted_room_driver_tasks ADD COLUMN retry_id TEXT")
        conn.execute(
            """INSERT INTO hosted_room_retry_receipts(
                   retry_id, room_id, task_id, source_execution_generation, created_at
               ) VALUES ('existing-retry', ?, ?, 0, 100)""",
            (identity.room_id, identity.task_id),
        )
        conn.execute(
            "UPDATE hosted_room_driver_tasks SET retry_id='draft-retry'"
        )
        conn.commit()

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_driver_tasks)")
        }
    assert "retry_id" not in columns
    assert driver.retry_receipt_exists(
        db,
        room_id=identity.room_id,
        task_id=identity.task_id,
        retry_id="draft-retry",
    )
    assert driver.retry_receipt_exists(
        db,
        room_id=identity.room_id,
        task_id=identity.task_id,
        retry_id="existing-retry",
    )


def test_retry_schema_failure_closes_the_open_connection(db, monkeypatch):
    _lease(db, FakeClock())
    opened = []
    connect = driver.connect

    def tracked(*args, **kwargs):
        conn = connect(*args, **kwargs)
        opened.append(conn)
        return conn

    def fail(_conn):
        raise sqlite3.OperationalError("retry schema unavailable")

    monkeypatch.setattr(driver, "connect", tracked)
    monkeypatch.setattr(driver, "_initialize_retry_receipt_table", fail)
    with pytest.raises(sqlite3.OperationalError, match="retry schema unavailable"):
        driver._connect(db)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_prune_keeps_task_that_owns_pending_artifact_retry(db):
    clock = FakeClock()
    lease = _lease(db, clock)
    identity = _identity("task-retry", turn_id="turn-retry")
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
        settlement_id="result:task-retry",
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
               VALUES ('room-1', 'task-retry', 'turn.settled', 0, 3)"""
        )
        conn.execute(
            """CREATE TABLE hosted_room_artifact_retries (
                room_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                execution_generation INTEGER NOT NULL,
                PRIMARY KEY(room_id, task_id, execution_generation)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_artifact_retries
               VALUES ('room-1', 'task-retry', 1)"""
        )

    clock.advance(driver.TERMINAL_TASK_RETENTION_SECONDS + 1)
    assert driver.prune_published_terminal_tasks(
        db,
        room_id="room-1",
        clock=clock,
        retain=0,
    ) == 0
    assert driver.get_task(db, identity)["status"] == "settled"

    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM hosted_room_artifact_retries")
    assert driver.prune_published_terminal_tasks(
        db,
        room_id="room-1",
        clock=clock,
        retain=0,
    ) == 1
