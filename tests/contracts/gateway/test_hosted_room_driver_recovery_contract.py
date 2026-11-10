"""Contract cases for the recovery, cancellation and pruning verbs of ``persistence.gateway.hosted_rooms.HostedRoomDriverRepository``.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin these rules: a new lease fences the previous lease's running work as indeterminate, the lease-first recovery transitions are fenced on status and both generations, cancellation is direct for queued and deferred work and two-phase for running work, and pruning removes only terminal tasks whose outcome the policy checkpoint recorded as published.
"""

from __future__ import annotations

import pytest

from persistence.gateway.hosted_rooms import (
    DriverLease,
    HostedTaskIdentity,
    InvalidTaskTransitionError,
    RoomRecovery,
    StaleLeaseError,
    StaleTaskError,
    TaskAttempt,
    TaskConflictError,
)

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
ROOM = "room"


def _identity(n: int = 1) -> HostedTaskIdentity:
    return HostedTaskIdentity(ROOM, f"task-{n}", f"thread-{n}", f"turn-{n}")


def _lease(store, process: str = "proc-a", *, now: float = 100.0) -> DriverLease:
    return store.hosted_room_driver.acquire_lease(placement="profile", room_id=ROOM, gateway_id="gw", authority_epoch=1, process_generation=process, ttl_seconds=60.0, now=now)


def _admit(store, n: int = 1, *, now: float = 100.0):
    payload = {"target_profile": "p1", "prompt": "hello", "source_event_seq": n}
    return store.hosted_room_driver.admit_task(_identity(n), placement="profile", payload=payload, now=now)


def _start(store, lease: DriverLease, n: int = 1, *, now: float = 110.0) -> TaskAttempt:
    return store.hosted_room_driver.start_task(_identity(n), lease, placement="profile", expected_cancel_generation=0, now=now)


def _running(store) -> tuple[DriverLease, TaskAttempt]:
    store.hosted_rooms.create_room(placement="profile", room_id=ROOM, name="Room", members=MEMBERS, authority_gateway_id="gw", now=100.0)
    lease = _lease(store)
    _admit(store)
    return lease, _start(store, lease)


def _indeterminate(store) -> DriverLease:
    """Task 1 started under lease A (expired at 160) and fenced by lease B's recovery at 201."""
    _running(store)
    successor = _lease(store, "proc-b", now=200.0)
    store.hosted_room_driver.recover_room(successor, placement="profile", now=201.0)
    return successor


def _fields(task, *names: str) -> dict:
    return {name: task[name] for name in names}


# ── recover_room ──────────────────────────────────────────────────────────────


def test_recover_room_fences_running_work_of_other_leases_as_indeterminate(backend, scope_a):
    store = backend.open(scope_a)
    lease, _ = _running(store)
    _admit(store, 2)
    driver = store.hosted_room_driver
    # CONTRACT: the lease's own running task is left alone.
    assert driver.recover_room(lease, placement="profile", now=120.0) == RoomRecovery(queued=(_identity(2),), indeterminate=())
    successor = _lease(store, "proc-b", now=200.0)
    assert successor.reclaimed and successor.lease_generation == 2
    assert driver.recover_room(successor, placement="profile", now=201.0) == RoomRecovery(queued=(_identity(2),), indeterminate=(_identity(1),))
    task = driver.get_task(_identity(1), placement="profile")
    assert _fields(task, "status", "indeterminate_at", "updated_at", "execution_generation", "run_process_generation") == {
        "status": "indeterminate", "indeterminate_at": 201.0, "updated_at": 201.0, "execution_generation": 1, "run_process_generation": "proc-a"}
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        driver.recover_room(lease, placement="profile", now=202.0)
    with pytest.raises(InvalidTaskTransitionError, match="^room recovery must resolve the prior task before starting new work$"):
        _start(store, successor, 2, now=202.0)


# ── the lease-first recovery transitions ──────────────────────────────────────


def test_resolve_indeterminate_task_settles_under_the_current_lease_and_replays(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    driver = store.hosted_room_driver

    def resolve(*, status: str = "settled", generation: int = 1, now: float = 210.0, current: DriverLease = lease, n: int = 1):
        return driver.resolve_indeterminate_task(_identity(n), current, placement="profile", expected_execution_generation=generation, expected_cancel_generation=0, settlement_id="settle-1", status=status, result={"text": "recovered"}, now=now)

    # CONTRACT: lease first: a stale lease refuses before the task row is read.
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        resolve(current=DriverLease(ROOM, "gw", 1, "proc-a", 1, 160.0), n=9)
    with pytest.raises(StaleTaskError, match="^indeterminate task generation changed$"):
        resolve(generation=2)
    settled = resolve()
    assert _fields(settled, "status", "settlement_id", "settlement_status", "result", "terminal_at", "idempotent") == {
        "status": "settled", "settlement_id": "settle-1", "settlement_status": "settled", "result": {"text": "recovered"}, "terminal_at": 210.0, "idempotent": False}
    assert resolve(now=220.0) == {**settled, "idempotent": True}
    with pytest.raises(TaskConflictError, match="^task already has a different terminal settlement$"):
        resolve(status="failed")


def test_resolve_indeterminate_cancellation_cancels_with_the_next_cancel_generation(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    driver = store.hosted_room_driver

    def cancel(cancel_generation: int = 0, now: float = 210.0):
        return driver.resolve_indeterminate_cancellation(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=cancel_generation, cancel_id="cancel-1", now=now)

    with pytest.raises(StaleTaskError, match="^indeterminate cancellation proof is stale$"):
        cancel(1)
    cancelled = cancel()
    assert _fields(cancelled, "status", "cancel_generation", "cancel_id", "terminal_at", "idempotent") == {
        "status": "cancelled", "cancel_generation": 1, "cancel_id": "cancel-1", "terminal_at": 210.0, "idempotent": False}
    assert cancel(now=220.0) == {**cancelled, "idempotent": True}


def test_requeue_indeterminate_task_clears_the_attempt_without_a_replay(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    driver = store.hosted_room_driver
    queued = driver.requeue_indeterminate_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=210.0)
    assert _fields(queued, "status", "run_gateway_id", "run_process_generation", "run_lease_generation", "started_at", "indeterminate_at", "updated_at", "execution_generation") == {
        "status": "queued", "run_gateway_id": None, "run_process_generation": None, "run_lease_generation": None, "started_at": None,
        "indeterminate_at": None, "updated_at": 210.0, "execution_generation": 1}
    with pytest.raises(StaleTaskError, match="^indeterminate task generation changed$"):
        driver.requeue_indeterminate_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=211.0)
    assert _start(store, lease, now=212.0).execution_generation == 2


def test_defer_then_requeue_deferred_releases_later_work_and_retries_the_turn(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    _admit(store, 2, now=205.0)
    driver = store.hosted_room_driver

    def defer(reason: str = "peer-unreachable", now: float = 210.0):
        return driver.defer_indeterminate_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, reason=reason, now=now)

    deferred = defer()
    assert _fields(deferred, "status", "result", "terminal_at", "idempotent") == {
        "status": "deferred", "result": {"reason": "peer-unreachable", "retryable": True}, "terminal_at": 210.0, "idempotent": False}
    assert defer(now=215.0) == {**deferred, "idempotent": True}
    # CONTRACT: another reason is no replay, and the row is no longer indeterminate.
    with pytest.raises(StaleTaskError, match="^indeterminate task generation changed$"):
        defer("other-reason")
    # CONTRACT: a deferred task does not block the room's next task.
    assert _start(store, lease, 2, now=216.0).execution_generation == 1
    store.hosted_room_driver.settle_task(TaskAttempt(_identity(2), lease, 1, 0), placement="profile", settlement_id="settle-2", status="failed", result=None, now=217.0)
    requeued = driver.requeue_deferred_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=220.0)
    assert _fields(requeued, "status", "result", "started_at", "terminal_at", "indeterminate_at", "run_gateway_id", "updated_at") == {
        "status": "queued", "result": None, "started_at": None, "terminal_at": None, "indeterminate_at": None, "run_gateway_id": None, "updated_at": 220.0}
    with pytest.raises(StaleTaskError, match="^deferred task generation changed$"):
        driver.requeue_deferred_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=221.0)
    assert _start(store, lease, now=222.0).execution_generation == 2


def test_defer_unconfirmed_stop_defers_a_stopping_task_and_replays(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    driver = store.hosted_room_driver

    def defer(reason: str = "stop-unconfirmed", cancel_generation: int = 1, now: float = 220.0):
        return driver.defer_unconfirmed_stop(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=cancel_generation, reason=reason, now=now)

    # CONTRACT: only a stopping task defers this way.
    with pytest.raises(StaleTaskError, match="^stopping task generation changed$"):
        defer(cancel_generation=0)
    driver.begin_task_cancel(_identity(), placement="profile", cancel_id="stop-1", expected_cancel_generation=0, now=210.0)
    deferred = defer()
    assert _fields(deferred, "status", "result", "terminal_at", "cancel_generation", "idempotent") == {
        "status": "deferred", "result": {"reason": "stop-unconfirmed", "retryable": True}, "terminal_at": 220.0, "cancel_generation": 1, "idempotent": False}
    assert defer(now=225.0) == {**deferred, "idempotent": True}
    with pytest.raises(StaleTaskError, match="^stopping task generation changed$"):
        defer("other-reason")
    # CONTRACT: deferred, never cancelled: a later room task can start.
    _admit(store, 2, now=226.0)
    assert _start(store, lease, 2, now=227.0).execution_generation == 1


def test_reopen_deferred_task_returns_it_to_indeterminate_at_its_generation(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    driver = store.hosted_room_driver
    driver.defer_indeterminate_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, reason="peer-unreachable", now=210.0)
    reopened = driver.reopen_deferred_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=230.0)
    assert _fields(reopened, "status", "result", "terminal_at", "started_at", "indeterminate_at", "updated_at", "execution_generation", "cancel_generation") == {
        "status": "indeterminate", "result": None, "terminal_at": None, "started_at": 230.0, "indeterminate_at": 230.0, "updated_at": 230.0,
        "execution_generation": 1, "cancel_generation": 0}
    with pytest.raises(StaleTaskError, match="^deferred task generation changed$"):
        driver.reopen_deferred_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=231.0)
    assert driver.requeue_indeterminate_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=232.0)["status"] == "queued"
    with pytest.raises(StaleLeaseError):
        driver.reopen_deferred_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, now=900.0)


def test_mark_attempt_indeterminate_retires_exactly_that_attempt_under_any_active_lease(backend, scope_a):
    store = backend.open(scope_a)
    lease, attempt = _running(store)
    driver = store.hosted_room_driver
    renewed = driver.renew_lease(lease, placement="profile", ttl_seconds=60.0, now=120.0)
    assert driver.mark_attempt_indeterminate(attempt, renewed, placement="profile", now=130.0) is None
    task = driver.get_task(_identity(), placement="profile")
    assert _fields(task, "status", "indeterminate_at", "updated_at", "execution_generation", "run_gateway_id") == {
        "status": "indeterminate", "indeterminate_at": 130.0, "updated_at": 130.0, "execution_generation": 1, "run_gateway_id": "gw"}
    # CONTRACT: an attempt no longer running is left as it is.
    driver.mark_attempt_indeterminate(attempt, renewed, placement="profile", now=140.0)
    assert driver.get_task(_identity(), placement="profile")["updated_at"] == 130.0
    with pytest.raises(StaleLeaseError):
        driver.mark_attempt_indeterminate(attempt, renewed, placement="profile", now=900.0)


def test_settle_stopping_task_settles_a_completion_that_won_the_stop_race(backend, scope_a):
    store = backend.open(scope_a)
    lease, attempt = _running(store)
    driver = store.hosted_room_driver

    def settle(cancel_generation: int = 1, now: float = 130.0):
        return driver.settle_stopping_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=cancel_generation, settlement_id="settle-1", status="settled", result={"text": "done"}, now=now)

    # CONTRACT: no row guard: a task that is not stopping refuses at the fence.
    with pytest.raises(StaleTaskError, match="^task completion lost the stop race$"):
        settle(0)
    driver.begin_task_cancel(_identity(), placement="profile", cancel_id="stop-1", expected_cancel_generation=0, now=120.0)
    with pytest.raises(StaleTaskError, match="^task attempt is stale or cancelled$"):
        driver.settle_task(attempt, placement="profile", settlement_id="settle-1", status="settled", result={"text": "done"}, now=125.0)
    settled = settle()
    assert _fields(settled, "status", "cancel_generation", "cancel_id", "settlement_id", "terminal_at", "idempotent") == {
        "status": "settled", "cancel_generation": 1, "cancel_id": "stop-1", "settlement_id": "settle-1", "terminal_at": 130.0, "idempotent": False}
    # CONTRACT: the settlement replays before the lease check.
    assert settle(now=500.0) == {**settled, "idempotent": True}


# ── cancellation ──────────────────────────────────────────────────────────────


def test_cancel_task_cancels_queued_and_deferred_work_directly(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    _admit(store, 2, now=205.0)
    driver = store.hosted_room_driver

    def cancel(n: int, cancel_id: str = "cancel-1", generation: int = 0, now: float = 210.0):
        return driver.cancel_task(_identity(n), placement="profile", cancel_id=cancel_id, expected_cancel_generation=generation, now=now)

    with pytest.raises(InvalidTaskTransitionError, match="^running work requires acknowledged two-phase cancellation$"):
        cancel(1)
    with pytest.raises(StaleTaskError, match="^task cancellation generation changed$"):
        cancel(2, generation=1)
    cancelled = cancel(2)
    assert _fields(cancelled, "status", "cancel_generation", "cancel_id", "terminal_at", "idempotent") == {
        "status": "cancelled", "cancel_generation": 1, "cancel_id": "cancel-1", "terminal_at": 210.0, "idempotent": False}
    assert cancel(2, now=220.0) == {**cancelled, "idempotent": True}
    with pytest.raises(InvalidTaskTransitionError, match="^cannot cancel task in state 'cancelled'$"):
        cancel(2, cancel_id="cancel-2", generation=1)
    driver.defer_indeterminate_task(_identity(), lease, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, reason="peer-unreachable", now=230.0)
    store.hosted_rooms.disband_room(placement="profile", room_id=ROOM, expected_gateway_id="gw", expected_epoch=1, now=240.0)
    # CONTRACT: no room check: a deferred task of a disbanded room still cancels.
    assert cancel(1, now=250.0)["status"] == "cancelled"


def test_begin_task_cancel_records_a_stop_intent_for_running_and_indeterminate_work(backend, scope_a):
    store = backend.open(scope_a)
    lease = _indeterminate(store)
    _admit(store, 2, now=205.0)
    driver = store.hosted_room_driver

    def begin(n: int, cancel_id: str = "stop-1", generation: int = 0, now: float = 210.0):
        return driver.begin_task_cancel(_identity(n), placement="profile", cancel_id=cancel_id, expected_cancel_generation=generation, now=now)

    with pytest.raises(InvalidTaskTransitionError, match="^cannot request remote stop in state 'queued'$"):
        begin(2)
    with pytest.raises(StaleTaskError, match="^task cancellation generation changed$"):
        begin(1, generation=1)
    stopping = begin(1)
    assert _fields(stopping, "status", "cancel_generation", "cancel_id", "terminal_at", "updated_at", "idempotent") == {
        "status": "stopping", "cancel_generation": 1, "cancel_id": "stop-1", "terminal_at": None, "updated_at": 210.0, "idempotent": False}
    assert begin(1, now=220.0) == {**stopping, "idempotent": True}
    # CONTRACT: the guard lets a stopping or deferred row through and its fence (running or indeterminate) refuses it.
    with pytest.raises(StaleTaskError, match="^task changed during stop request$"):
        begin(1, cancel_id="stop-2", generation=1)
    driver.complete_task_cancel(_identity(1), placement="profile", cancel_id="stop-1", expected_cancel_generation=1, now=225.0)
    _start(store, lease, 2, now=230.0)
    successor = _lease(store, "proc-c", now=300.0)
    driver.recover_room(successor, placement="profile", now=301.0)
    driver.defer_indeterminate_task(_identity(2), successor, placement="profile", expected_execution_generation=1, expected_cancel_generation=0, reason="peer-unreachable", now=302.0)
    with pytest.raises(StaleTaskError, match="^task changed during stop request$"):
        begin(2, now=303.0)


def test_complete_task_cancel_acknowledges_the_exact_stop(backend, scope_a):
    store = backend.open(scope_a)
    _running(store)
    driver = store.hosted_room_driver

    def complete(cancel_id: str = "stop-1", generation: int = 1, now: float = 130.0):
        return driver.complete_task_cancel(_identity(), placement="profile", cancel_id=cancel_id, expected_cancel_generation=generation, now=now)

    with pytest.raises(StaleTaskError, match="^task stop acknowledgement is stale$"):
        complete()
    driver.begin_task_cancel(_identity(), placement="profile", cancel_id="stop-1", expected_cancel_generation=0, now=120.0)
    for cancel_id, generation in (("stop-2", 1), ("stop-1", 0)):
        with pytest.raises(StaleTaskError, match="^task stop acknowledgement is stale$"):
            complete(cancel_id, generation)
    cancelled = complete()
    assert _fields(cancelled, "status", "cancel_generation", "cancel_id", "terminal_at", "idempotent") == {
        "status": "cancelled", "cancel_generation": 1, "cancel_id": "stop-1", "terminal_at": 130.0, "idempotent": False}
    assert complete(now=140.0) == {**cancelled, "idempotent": True}
    with pytest.raises(TaskConflictError, match="^task does not exist$"):
        driver.complete_task_cancel(_identity(2), placement="profile", cancel_id="stop-1", expected_cancel_generation=1, now=140.0)


# ── prune_published_terminal_tasks ────────────────────────────────────────────


def _publish(store, n: int, seq: int, kind: str = "turn.settled") -> None:
    store.hosted_rooms.append_event(
        placement="profile", room_id=ROOM, event_id=f"outcome-{n}", kind=kind, actor={"kind": "system", "id": "driver"},
        payload={"task_id": f"task-{n}", "thread_id": f"thread-{n}", "discussion_event_id": f"d-{n}"}, authority_gateway_id="gw", authority_epoch=1, now=135.0)
    store.hosted_room_policy.snapshot(placement="profile", room_id=ROOM, latest_seq=seq)


def test_prune_published_terminal_tasks_drops_only_published_outcomes_past_the_bounds(backend, scope_a):
    store = backend.open(scope_a)
    lease, attempt = _running(store)
    driver = store.hosted_room_driver
    for n in (2, 3, 4):
        _admit(store, n)
    driver.settle_task(attempt, placement="profile", settlement_id="settle-1", status="settled", result=None, now=120.0)
    driver.settle_task(_start(store, lease, 2, now=125.0), placement="profile", settlement_id="settle-2", status="failed", result=None, now=130.0)
    driver.cancel_task(_identity(3), placement="profile", cancel_id="cancel-3", expected_cancel_generation=0, now=131.0)
    _publish(store, 1, 1)
    _publish(store, 2, 2, "turn.failed")
    # CONTRACT: a deferral receipt is not a terminal publication.
    _publish(store, 3, 3, "turn.deferred")

    def prune(**bounds):
        return driver.prune_published_terminal_tasks(placement="profile", room_id=ROOM, now=140.0, **bounds)

    assert prune() == 0
    # CONTRACT: terminal_at DESC, task_id order: the count bound keeps the newest published outcome.
    assert prune(retain=1) == 1
    assert [task["identity"].task_id for task in driver.list_tasks(placement="profile", room_id=ROOM)] == ["task-2", "task-3", "task-4"]
    assert prune(retention_seconds=15.0) == 0
    assert prune(retention_seconds=10.0) == 1
    assert [task["identity"].task_id for task in driver.list_tasks(placement="profile", room_id=ROOM)] == ["task-3", "task-4"]
    assert prune(retain=0) == 0
