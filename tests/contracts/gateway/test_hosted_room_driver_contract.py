"""Contract cases for the leases, admission, start, settlement and reads of ``persistence.gateway.hosted_rooms.HostedRoomDriverRepository``.

Every case runs on each registered backend through the ``backend`` fixture and seeds through public verbs. The cases pin monotonic fenced leases, idempotent admission bound to one turn and one payload, start in room event order under the lease's run fence, settlement replays before the lease check, whose expiry term alone the live settle may drop. The caller-side validators stay with the caller; the cases pass what the callers pass.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from persistence.gateway.hosted_rooms import (
    DriverLease,
    DriverValidationError,
    HostedMemberGuard,
    HostedOwnerClaim,
    HostedTaskIdentity,
    InvalidTaskTransitionError,
    LeaseHeldError,
    LeaseRelease,
    RoomUnavailableError,
    StaleLeaseError,
    StaleTaskError,
    TaskAttempt,
    TaskConflictError,
)

MEMBERS = [{"member_id": "m1", "profile": "p1"}]
ROOM = "room"


def _room(store, *, placement: str = "profile", room_id: str = ROOM, owner: HostedOwnerClaim | None = None, now: float = 100.0):
    return store.hosted_rooms.create_room(placement=placement, room_id=room_id, name="Room", members=MEMBERS, authority_gateway_id="gw", owner=owner, now=now)


def _identity(n: int = 1, room_id: str = ROOM) -> HostedTaskIdentity:
    return HostedTaskIdentity(room_id, f"task-{n}", f"thread-{n}", f"turn-{n}")


def _payload(seq: int = 1, **extra) -> dict:
    return {"target_profile": "p1", "prompt": "hello", "source_event_seq": seq, **extra}


def _lease(store, *, placement: str = "profile", process: str = "proc-a", epoch: int = 1, ttl: float = 60.0, now: float = 100.0) -> DriverLease:
    return store.hosted_room_driver.acquire_lease(placement=placement, room_id=ROOM, gateway_id="gw", authority_epoch=epoch, process_generation=process, ttl_seconds=ttl, now=now)


def _admit(store, n: int = 1, *, seq: int | None = None, placement: str = "profile", now: float = 100.0):
    return store.hosted_room_driver.admit_task(_identity(n), placement=placement, payload=_payload(n if seq is None else seq), now=now)


def _start(store, lease: DriverLease, n: int = 1, *, placement: str = "profile", now: float = 110.0) -> TaskAttempt:
    return store.hosted_room_driver.start_task(_identity(n), lease, placement=placement, expected_cancel_generation=0, now=now)


def _settle(store, attempt: TaskAttempt, *, settlement_id: str = "settle-1", status: str = "settled", result=None, now: float = 120.0, **flags):
    return store.hosted_room_driver.settle_task(attempt, placement="profile", settlement_id=settlement_id, status=status, result={"text": "done"} if result is None else result, now=now, **flags)


# ── leases ────────────────────────────────────────────────────────────────────


def test_acquire_lease_creates_extends_and_refuses_another_live_generation(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    assert _lease(store) == DriverLease(ROOM, "gw", 1, "proc-a", 1, 160.0, reclaimed=False)
    # CONTRACT: the same live generation extends to the later of the stored and requested expiry, never shortening it.
    assert _lease(store, ttl=10.0, now=120.0).expires_at == 160.0
    assert _lease(store, ttl=60.0, now=130.0) == DriverLease(ROOM, "gw", 1, "proc-a", 1, 190.0)
    with pytest.raises(LeaseHeldError, match="^room driver lease is held by another generation$"):
        _lease(store, process="proc-b", now=150.0)


def test_acquire_lease_takes_over_an_expired_or_released_lease_with_the_next_generation(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    first = _lease(store)
    # CONTRACT: an expired lease (expires_at <= now) is taken over by any generation of the room's authority.
    taken = _lease(store, process="proc-b", now=160.0)
    assert taken == DriverLease(ROOM, "gw", 1, "proc-b", 2, 220.0, reclaimed=True)
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        store.hosted_room_driver.renew_lease(first, placement="profile", ttl_seconds=60.0, now=161.0)
    store.hosted_room_driver.release_lease(taken, placement="profile", now=170.0)
    # CONTRACT: a released lease is taken over even by the generation that released it.
    assert _lease(store, process="proc-b", now=171.0) == DriverLease(ROOM, "gw", 1, "proc-b", 3, 231.0, reclaimed=True)


def test_acquire_lease_refuses_missing_disbanded_and_foreign_authority_rooms(backend, scope_a):
    store = backend.open(scope_a)
    with pytest.raises(RoomUnavailableError, match="^hosted room does not exist$"):
        _lease(store)
    _room(store)
    with pytest.raises(StaleLeaseError, match="^hosted room authority changed$"):
        _lease(store, epoch=2)
    with pytest.raises(StaleLeaseError, match="^hosted room authority changed$"):
        store.hosted_room_driver.acquire_lease(placement="profile", room_id=ROOM, gateway_id="other", authority_epoch=1, process_generation="proc-a", ttl_seconds=60.0, now=100.0)
    store.hosted_rooms.disband_room(placement="profile", room_id=ROOM, expected_gateway_id="gw", expected_epoch=1, now=110.0)
    with pytest.raises(RoomUnavailableError, match="^hosted room is disbanded$"):
        _lease(store, now=120.0)


def test_renew_lease_keeps_the_later_expiry_and_refuses_stale_or_expired_leases(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    renewed = store.hosted_room_driver.renew_lease(lease, placement="profile", ttl_seconds=60.0, now=150.0)
    assert renewed == DriverLease(ROOM, "gw", 1, "proc-a", 1, 210.0, reclaimed=False)
    assert store.hosted_room_driver.renew_lease(renewed, placement="profile", ttl_seconds=5.0, now=151.0).expires_at == 210.0
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        store.hosted_room_driver.renew_lease(renewed, placement="profile", ttl_seconds=60.0, now=210.0)
    # CONTRACT: the room authority is checked before the lease row.
    with pytest.raises(StaleLeaseError, match="^hosted room authority changed$"):
        store.hosted_room_driver.renew_lease(DriverLease(ROOM, "gw", 2, "proc-a", 1, 210.0), placement="profile", ttl_seconds=60.0, now=150.0)


def test_release_lease_replays_and_refuses_running_work_stale_and_expired_leases(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    with pytest.raises(InvalidTaskTransitionError, match="^cannot release a room lease while tasks are running$"):
        store.hosted_room_driver.release_lease(lease, placement="profile", now=130.0)
    _settle(store, attempt)
    with pytest.raises(StaleLeaseError, match="^driver lease is stale$"):
        store.hosted_room_driver.release_lease(DriverLease(ROOM, "gw", 1, "proc-b", 1, 160.0), placement="profile", now=130.0)
    released = store.hosted_room_driver.release_lease(lease, placement="profile", now=130.0)
    assert released == LeaseRelease(lease=DriverLease(ROOM, "gw", 1, "proc-a", 1, 130.0), idempotent=False)
    assert store.hosted_room_driver.release_lease(lease, placement="profile", now=140.0) == LeaseRelease(lease=DriverLease(ROOM, "gw", 1, "proc-a", 1, 130.0), idempotent=True)
    expired = _lease(store, now=150.0)
    with pytest.raises(StaleLeaseError, match="^driver lease expired before release$"):
        store.hosted_room_driver.release_lease(expired, placement="profile", now=210.0)


# ── admission ─────────────────────────────────────────────────────────────────


def test_admit_task_returns_the_full_task_view_and_replays_identically(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    encoded = json.dumps(_payload(), sort_keys=True, separators=(",", ":"))
    task = _admit(store)
    assert task == {
        "identity": _identity(), "payload": _payload(), "payload_digest": hashlib.sha256(encoded.encode()).hexdigest(), "status": "queued",
        "execution_generation": 0, "cancel_generation": 0, "run_gateway_id": None, "run_process_generation": None, "run_lease_generation": None,
        "cancel_id": None, "settlement_id": None, "settlement_status": None, "result": None, "created_at": 100.0, "updated_at": 100.0,
        "started_at": None, "terminal_at": None, "indeterminate_at": None, "idempotent": False}
    assert _admit(store, now=150.0) == {**task, "idempotent": True}


def test_admit_task_refuses_conflicting_bindings_and_unavailable_rooms(backend, scope_a):
    store = backend.open(scope_a)
    with pytest.raises(RoomUnavailableError, match="^hosted room does not exist$"):
        _admit(store)
    _room(store)
    _admit(store)
    driver = store.hosted_room_driver
    with pytest.raises(TaskConflictError, match="^task_id is already bound to a different payload$"):
        driver.admit_task(_identity(), placement="profile", payload=_payload(prompt="other"), now=110.0)
    with pytest.raises(TaskConflictError, match="^task_id is already bound to a different turn$"):
        driver.admit_task(HostedTaskIdentity(ROOM, "task-1", "thread-1", "turn-9"), placement="profile", payload=_payload(), now=110.0)
    with pytest.raises(TaskConflictError, match="^thread_id and turn_id are already bound to a task$"):
        driver.admit_task(HostedTaskIdentity(ROOM, "task-9", "thread-1", "turn-1"), placement="profile", payload=_payload(), now=110.0)
    store.hosted_rooms.disband_room(placement="profile", room_id=ROOM, expected_gateway_id="gw", expected_epoch=1, now=120.0)
    with pytest.raises(RoomUnavailableError, match="^hosted room is disbanded$"):
        _admit(store, 2, now=130.0)


def test_admit_task_stores_the_normalized_payload(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    attachment = {"attachment_id": "att_" + "0" * 32, "kind": "image", "name": " a.png ", "size": 3, "mime": "IMAGE/PNG", "event_id": " e1 "}
    raw = _payload(target_profile=" p1 ", target_member_id="m1", input_context={"watermark": 0, "event_seqs": [1]}, attachments=[attachment])
    task = store.hosted_room_driver.admit_task(_identity(), placement="profile", payload=raw, now=100.0)
    # CONTRACT: the stored payload is normalized (identifiers and names stripped, MIME lowered), from which the digest derives.
    assert task["payload"] == {
        "target_profile": "p1", "prompt": "hello", "source_event_seq": 1, "target_member_id": "m1", "input_context": {"watermark": 0, "event_seqs": [1]},
        "attachments": [{"attachment_id": "att_" + "0" * 32, "kind": "image", "name": "a.png", "size": 3, "mime": "image/png", "event_id": "e1"}]}
    assert task["payload_digest"] == hashlib.sha256(json.dumps(task["payload"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(DriverValidationError, match="^missing payload fields: prompt$"):
        store.hosted_room_driver.admit_task(_identity(2), placement="profile", payload={"target_profile": "p1", "source_event_seq": 2}, now=100.0)
    assert store.hosted_room_driver.list_tasks(placement="profile", room_id=ROOM) == (task,)


def test_admit_task_takes_lead_routing_only_as_exactly_true(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    task = store.hosted_room_driver.admit_task(_identity(), placement="profile", payload=_payload(lead_routing=True), now=100.0)
    # CONTRACT: a lead's routing turn carries lead_routing, stored only as exactly true; any other value refuses.
    assert task["payload"]["lead_routing"] is True
    for value in (False, 1, "true"):
        with pytest.raises(DriverValidationError, match="^lead_routing must be true when present$"):
            store.hosted_room_driver.admit_task(_identity(2), placement="profile", payload=_payload(2, lead_routing=value), now=100.0)


# ── start and settlement ──────────────────────────────────────────────────────


def test_start_task_runs_the_next_queued_task_under_the_lease_run_fence(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store, 2, seq=2)
    _admit(store, 1, seq=1, now=105.0)
    # CONTRACT: room event order is (source_event_seq, created_at, task_id), not admission order.
    with pytest.raises(InvalidTaskTransitionError, match="^task is not next in the hosted room event order$"):
        _start(store, lease, 2)
    attempt = _start(store, lease, 1)
    assert attempt == TaskAttempt(identity=_identity(1), lease=lease, execution_generation=1, cancel_generation=0)
    task = store.hosted_room_driver.get_task(_identity(1), placement="profile")
    assert (task["status"], task["run_gateway_id"], task["run_process_generation"], task["run_lease_generation"], task["started_at"]) == ("running", "gw", "proc-a", 1, 110.0)
    with pytest.raises(InvalidTaskTransitionError, match="^room recovery must resolve the prior task before starting new work$"):
        _start(store, lease, 2)


def test_start_task_refusals_check_the_task_then_the_lease_generation_and_state(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    driver = store.hosted_room_driver
    with pytest.raises(TaskConflictError, match="^task does not exist$"):
        _start(store, lease)
    _admit(store)
    # CONTRACT: the lease is checked before the task row.
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        _start(store, lease, now=160.0)
    with pytest.raises(StaleTaskError, match="^task cancellation generation changed$"):
        driver.start_task(_identity(), lease, placement="profile", expected_cancel_generation=1, now=110.0)
    _settle(store, _start(store, lease))
    with pytest.raises(InvalidTaskTransitionError, match="^cannot start task in state 'settled'$"):
        _start(store, lease, now=130.0)


def test_settle_task_settles_and_replays_before_the_lease_check(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    settled = _settle(store, attempt, result={"text": "d\u00e9j\u00e0"})
    assert {key: settled[key] for key in ("status", "settlement_id", "settlement_status", "result", "terminal_at", "updated_at", "idempotent")} == {
        "status": "settled", "settlement_id": "settle-1", "settlement_status": "settled", "result": {"text": "d\u00e9j\u00e0"}, "terminal_at": 120.0, "updated_at": 120.0, "idempotent": False}
    # CONTRACT: an identical settlement replays even after the lease expired; a different one conflicts.
    assert _settle(store, attempt, result={"text": "d\u00e9j\u00e0"}, now=500.0) == {**settled, "idempotent": True}
    with pytest.raises(TaskConflictError, match="^task already has a different terminal settlement$"):
        _settle(store, attempt, status="failed", result={"text": "d\u00e9j\u00e0"}, now=130.0)


def test_settle_task_refuses_a_stale_attempt_after_its_lease_check(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        _settle(store, attempt, now=160.0)
    for stale in (TaskAttempt(attempt.identity, lease, 2, 0), TaskAttempt(attempt.identity, lease, 1, 1)):
        with pytest.raises(StaleTaskError, match="^task attempt is stale or cancelled$"):
            _settle(store, stale)
    assert store.hosted_room_driver.get_task(_identity(), placement="profile")["status"] == "running"


def test_settle_task_with_allow_expired_lease_commits_past_plain_expiry_of_its_own_lease(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        _settle(store, attempt, settlement_id="settle-strict", now=160.0)
    # CONTRACT: the flag drops only the expiry term of the post-replay lease check; the lease is still this generation's, unreleased.
    settled = _settle(store, attempt, settlement_id="settle-live", now=160.0, allow_expired_lease=True)
    assert (settled["status"], settled["settlement_id"], settled["terminal_at"]) == ("settled", "settle-live", 160.0)
    assert store.hosted_room_driver.get_task(_identity(), placement="profile")["status"] == "settled"


def test_settle_task_with_allow_expired_lease_still_refuses_a_reclaimed_lease(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    successor = _lease(store, process="proc-b", now=200.0)
    store.hosted_room_driver.recover_room(successor, placement="profile", now=201.0)
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        _settle(store, attempt, now=202.0, allow_expired_lease=True)
    assert store.hosted_room_driver.get_task(_identity(), placement="profile")["status"] == "indeterminate"


def test_settle_task_with_allow_expired_lease_still_refuses_a_released_lease(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    store.hosted_room_driver.begin_task_cancel(_identity(), placement="profile", cancel_id="stop-1", expected_cancel_generation=0, now=115.0)
    store.hosted_room_driver.release_lease(lease, placement="profile", now=120.0)
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        _settle(store, attempt, now=170.0, allow_expired_lease=True)
    assert store.hosted_room_driver.get_task(_identity(), placement="profile")["status"] == "stopping"


def test_settle_task_with_allow_expired_lease_still_refuses_after_an_authority_change(backend, scope_a):
    store = backend.open(scope_a)
    store.hosted_rooms.create_room(placement="profile", room_id=ROOM, name="Room", members=MEMBERS, authority_gateway_id="legacy", now=100.0)
    lease = store.hosted_room_driver.acquire_lease(placement="profile", room_id=ROOM, gateway_id="legacy", authority_epoch=1, process_generation="proc-a", ttl_seconds=60.0, now=100.0)
    _admit(store)
    attempt = _start(store, lease)
    assert _room(store, now=170.0)["authority_epoch"] == 2
    with pytest.raises(StaleLeaseError, match="^hosted room authority changed$"):
        _settle(store, attempt, now=171.0, allow_expired_lease=True)
    assert store.hosted_room_driver.get_task(_identity(), placement="profile")["status"] == "running"


def test_requeue_not_admitted_task_returns_the_attempt_to_the_queue_and_replays(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store)
    attempt = _start(store, lease)
    driver = store.hosted_room_driver
    queued = driver.requeue_not_admitted_task(attempt, placement="profile", now=120.0)
    assert {key: queued[key] for key in ("status", "execution_generation", "run_gateway_id", "run_process_generation", "run_lease_generation", "started_at", "updated_at", "idempotent")} == {
        "status": "queued", "execution_generation": 1, "run_gateway_id": None, "run_process_generation": None, "run_lease_generation": None,
        "started_at": None, "updated_at": 120.0, "idempotent": False}
    assert driver.requeue_not_admitted_task(attempt, placement="profile", now=130.0) == {**queued, "idempotent": True}
    restarted = _start(store, lease, now=140.0)
    assert restarted.execution_generation == 2
    with pytest.raises(StaleTaskError, match="^not-admitted task attempt lost its fence$"):
        driver.requeue_not_admitted_task(attempt, placement="profile", now=150.0)
    # CONTRACT: the lease is checked first on this path.
    with pytest.raises(StaleLeaseError, match="^driver lease is stale or expired$"):
        driver.requeue_not_admitted_task(restarted, placement="profile", now=160.0)


# ── reads ─────────────────────────────────────────────────────────────────────


def test_get_task_and_get_task_for_turn_read_one_task(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    driver = store.hosted_room_driver
    with pytest.raises(TaskConflictError, match="^task does not exist$"):
        driver.get_task(_identity(), placement="profile")
    assert driver.get_task_for_turn(_identity(), placement="profile") is None
    task = _admit(store)
    assert driver.get_task(_identity(), placement="profile") == task
    with pytest.raises(TaskConflictError, match="^task_id is already bound to a different turn$"):
        driver.get_task(HostedTaskIdentity(ROOM, "task-1", "thread-1", "turn-9"), placement="profile")
    # CONTRACT: the turn lookup ignores the identity's task_id.
    assert driver.get_task_for_turn(HostedTaskIdentity(ROOM, "other-task", "thread-1", "turn-1"), placement="profile") == task


def test_list_tasks_reads_the_room_in_event_order_with_an_optional_status(backend, scope_a):
    store = backend.open(scope_a)
    _room(store)
    lease = _lease(store)
    _admit(store, 3, seq=2)
    _admit(store, 2, seq=2, now=90.0)
    _admit(store, 1, seq=5)
    driver = store.hosted_room_driver
    assert [task["identity"].task_id for task in driver.list_tasks(placement="profile", room_id=ROOM)] == ["task-2", "task-3", "task-1"]
    _start(store, lease, 2)
    assert [task["identity"].task_id for task in driver.list_tasks(placement="profile", room_id=ROOM, status="queued")] == ["task-3", "task-1"]
    assert [task["status"] for task in driver.list_tasks(placement="profile", room_id=ROOM, status="running")] == ["running"]
    assert driver.list_tasks(placement="profile", room_id="other") == ()


def test_placements_keep_driver_tables_apart(backend, scope_a, scope_b):
    store, other = backend.open(scope_a), backend.open(scope_b)
    _room(store, placement="shared")
    lease = _lease(store, placement="shared")
    # CONTRACT: the shared file is one per install: a second profile of the install sees the same lease and tasks.
    with pytest.raises(LeaseHeldError):
        _lease(other, placement="shared", process="proc-b")
    _admit(other, placement="shared")
    assert _start(store, lease, placement="shared").execution_generation == 1
    with pytest.raises(RoomUnavailableError, match="^hosted room does not exist$"):
        _lease(store)
    _room(store)
    assert _lease(store, process="proc-b") == DriverLease(ROOM, "gw", 1, "proc-b", 1, 160.0)
    assert store.hosted_room_driver.list_tasks(placement="profile", room_id=ROOM) == ()
    _admit(store, 2, seq=1)
    assert [task["identity"].task_id for task in other.hosted_room_driver.list_tasks(placement="shared", room_id=ROOM)] == ["task-1"]
    assert [task["identity"].task_id for task in store.hosted_room_driver.list_tasks(placement="profile", room_id=ROOM)] == ["task-2"]
    assert other.hosted_room_driver.list_tasks(placement="profile", room_id=ROOM) == ()


def test_the_member_check_reads_the_driver_task_rows(backend, scope_a):
    store = backend.open(scope_a)
    epoch = backend.begin_epoch(scope_a, "instance-contract")
    _room(store, owner=HostedOwnerClaim(epoch, "alice", rule=None))
    lease = _lease(store)
    store.hosted_room_driver.admit_task(_identity(), placement="profile", payload=_payload(target_member_id="m1"), now=100.0)
    guard = HostedMemberGuard(room_id=ROOM, owner="alice", gateway_id="gw", authority_epoch=1, member_id="m1", profile="p1", home_current=True, task=_identity(), generation=1)
    assert store.hosted_rooms.member_authorized(guard, epoch=epoch, operation="submit") is False
    attempt = _start(store, lease)
    assert store.hosted_rooms.member_authorized(guard, epoch=epoch, operation="submit") is True
    store.hosted_room_driver.begin_task_cancel(_identity(), placement="profile", cancel_id="stop-1", expected_cancel_generation=0, now=115.0)
    # CONTRACT: a stopping task still authorizes operations other than submit and execute.
    assert store.hosted_rooms.member_authorized(guard, epoch=epoch, operation="submit") is False
    assert store.hosted_rooms.member_authorized(guard, epoch=epoch, operation="cancel") is True
    assert attempt.execution_generation == guard.generation


def test_reference_stored_payload_integrity_is_checked_on_every_read(backend, backend_name, scope_a):
    if backend_name != "in_memory":
        pytest.skip("rewrites a stored row through the reference's tables; a backend pins the same check with its own storage")
    from tests.contracts._reference.gateway.hosted_room_tables import write_tables

    store = backend.open(scope_a)
    _room(store)
    _admit(store)
    with write_tables(store, "profile") as tables:
        tables["hosted_room_driver_tasks"][ROOM]["task-1"]["payload_digest"] = "0" * 64
    with pytest.raises(TaskConflictError, match="^stored task payload failed its integrity check$"):
        store.hosted_room_driver.get_task(_identity(), placement="profile")
    with write_tables(store, "profile") as tables:
        tables["hosted_room_driver_tasks"][ROOM]["task-1"]["payload_json"] = '{"prompt":"x"}'
    with pytest.raises(TaskConflictError, match="^stored task payload is invalid$"):
        store.hosted_room_driver.list_tasks(placement="profile", room_id=ROOM)
    with pytest.raises(TaskConflictError, match="^stored task payload is invalid$"):
        store.hosted_room_driver.get_task_for_turn(_identity(), placement="profile")
