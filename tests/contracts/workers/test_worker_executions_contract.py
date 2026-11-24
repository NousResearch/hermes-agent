"""Contract suite for ``WorkerExecutionRepository``: registration, adoption, the claim check and the execution reads.

The receipt transaction every worker write shares is pinned in ``test_worker_receipt_contract.py``; the usage, finish, child and turn-lease operations in ``test_worker_operations_contract.py``. Construction goes through public verbs only (``tests/contracts/workers/_support.py``).
"""

from __future__ import annotations

import pytest

from persistence import AuthorityEpoch, ExecutionGeneration, FenceStale, SessionId, StaleFenceKind
from persistence.conversation.sessions import NewSession
from persistence.workers.executions import WorkerExecutionId, WorkerExecutionRecord, WorkerKind, WorkerStatus
from tests.contracts.workers._support import (
    EXECUTION,
    SECRET,
    SESSION,
    begin_epoch,
    ident,
    refused,
    register,
    require_workers,
    seed,
    worker,
)

PRINCIPAL = "principal-a"


def _admit(store, epoch, request_id: str, session_id: SessionId = SESSION):
    record, _created = store.admissions.admit(epoch=epoch, principal_id=PRINCIPAL, session_id=session_id, request_id=request_id, payload={"text": request_id})
    return record


def test_worker_execution_record_refuses_negative_sequence():
    with pytest.raises(ValueError):
        WorkerExecutionRecord(execution_id=EXECUTION, session_id=SESSION, kind=WorkerKind.CRON, owner_epoch=AuthorityEpoch(1),
                              generation=ExecutionGeneration(0), status=WorkerStatus.REGISTERED, last_sequence=-1)


def test_register_creates_a_registered_worker_with_zero_sequence(backend, scope_a):
    epoch = seed(backend, scope_a)
    record = register(backend, scope_a, epoch, kind=WorkerKind.CRON)
    assert record == WorkerExecutionRecord(execution_id=EXECUTION, session_id=SESSION, kind=WorkerKind.CRON, owner_epoch=epoch,
                                           generation=ExecutionGeneration(0), status=WorkerStatus.REGISTERED, last_sequence=0)


def test_register_refuses_invalid_inputs_before_the_store(backend, scope_a):
    epoch = seed(backend, scope_a)
    for field, value in (("execution_id", ""), ("session_id", ""), ("adoption_secret", ""), ("execution_id", "x" * 1025), ("kind", "remote")):
        with refused("invalid_params"):
            register(backend, scope_a, epoch, **{field: value})
    # Input-shape refusals stay ValueError, and even a stale epoch answers them first.
    with pytest.raises(ValueError):
        register(backend, scope_a, AuthorityEpoch(99), adoption_secret="")


def test_register_fences_stale_and_never_issued_epochs(backend, scope_a):
    epoch = seed(backend, scope_a)
    with refused("stale_epoch") as caught:
        register(backend, scope_a, AuthorityEpoch(int(epoch) + 1))
    assert isinstance(caught.value, FenceStale) and caught.value.kind is StaleFenceKind.STALE_EPOCH
    with refused("stale_epoch"):
        register(backend, scope_a, AuthorityEpoch(0))


def test_register_requires_the_session_and_its_current_generation(backend, scope_a):
    epoch = seed(backend, scope_a)
    with refused("not_found"):
        register(backend, scope_a, epoch, session_id=SessionId("sess-absent"))
    with refused("stale_generation"):
        register(backend, scope_a, epoch, generation=7)


def test_register_replays_exact_duplicates_and_conflicts_on_any_mismatch(backend, scope_a):
    epoch = seed(backend, scope_a)
    first = register(backend, scope_a, epoch)
    assert register(backend, scope_a, epoch) == first
    with refused("admission_conflict"):
        register(backend, scope_a, epoch, kind=WorkerKind.KANBAN)
    with refused("admission_conflict"):
        register(backend, scope_a, epoch, adoption_secret="a-different-secret-0123456789")
    # The same registration under a NEWER epoch is a mismatch, not a replay: re-owning is adopt's job.
    with refused("admission_conflict"):
        register(backend, scope_a, begin_epoch(backend, scope_a))


def test_register_admits_one_live_worker_per_runtime_lineage(backend, scope_a):
    epoch = seed(backend, scope_a)
    register(backend, scope_a, epoch)
    with refused("stale_generation"):
        register(backend, scope_a, epoch, execution_id=WorkerExecutionId("exec-2"))
    # The exclusion covers the compression lineage: a worker on the parent bars its continuation.
    store = backend.open(scope_a)
    store.sessions.end(SESSION, "compression")
    store.sessions.create(NewSession(session_id=SessionId("sess-cont"), source="contract", parent_session_id=SESSION))
    with refused("stale_generation"):
        register(backend, scope_a, epoch, execution_id=WorkerExecutionId("exec-cont"), session_id=SessionId("sess-cont"))


def test_register_require_idle_refuses_a_session_with_pending_admissions(backend, scope_a):
    epoch = seed(backend, scope_a)
    _admit(backend.open(scope_a), epoch, "req-queued")
    with refused("stale_generation"):
        register(backend, scope_a, epoch, require_idle=True)
    # Without the idle requirement a queued admission does not block.
    assert register(backend, scope_a, epoch).status is WorkerStatus.REGISTERED


def test_register_fails_closed_on_an_unknown_admission_barrier(backend, scope_a):
    epoch = seed(backend, scope_a)
    store = backend.open(scope_a)
    _admit(store, epoch, "req-unknown")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    rotated = begin_epoch(backend, scope_a)
    store.admissions.recover(epoch=rotated)
    with refused("unknown_execution"):
        register(backend, scope_a, rotated, generation=claimed.generation)


def test_fence_values_must_be_exact_ints(backend, scope_a):
    # The fences test type(x) is not int: a bool or float that compares equal is still stale.
    epoch = seed(backend, scope_a)
    assert int(epoch) == 1
    for bad_epoch in (True, 1.0):
        with refused("stale_epoch"):
            register(backend, scope_a, bad_epoch)
    for bad_generation in (False, 0.0):
        with refused("stale_generation"):
            register(backend, scope_a, epoch, generation=bad_generation)


def _adopt(backend, scope, epoch, **overrides):
    values = {"epoch": epoch, "execution_id": EXECUTION, "session_id": SESSION, "generation": ExecutionGeneration(0), "adoption_secret": SECRET} | overrides
    return backend.open(scope).worker_executions.adopt(**values)


def test_adopt_reowns_a_worker_under_the_current_epoch_on_proof(backend, scope_a):
    epoch = seed(backend, scope_a)
    register(backend, scope_a, epoch)
    rotated = begin_epoch(backend, scope_a)
    record = _adopt(backend, scope_a, rotated)
    assert (record.status, record.owner_epoch) == (WorkerStatus.RUNNING, rotated)


def test_adopt_resolution_order_missing_wrong_session_stale_generation(backend, scope_a):
    epoch = seed(backend, scope_a)
    backend.open(scope_a).sessions.create(NewSession(session_id=SessionId("sess-other"), source="contract"))
    register(backend, scope_a, epoch)
    with refused("not_found"):
        _adopt(backend, scope_a, epoch, execution_id=WorkerExecutionId("exec-absent"))
    with refused("permission_denied"):
        _adopt(backend, scope_a, epoch, session_id=SessionId("sess-other"))
    with refused("stale_generation"):
        _adopt(backend, scope_a, epoch, generation=ExecutionGeneration(3))
    with refused("invalid_params"):
        _adopt(backend, scope_a, AuthorityEpoch(99), adoption_secret="")


def test_adopt_rejects_a_wrong_secret_without_side_effects(backend, scope_a):
    epoch = seed(backend, scope_a)
    original = register(backend, scope_a, epoch)
    with refused("permission_denied"):
        _adopt(backend, scope_a, epoch, adoption_secret="wrong-secret-0123456789abcd")
    # No state change: the exact-duplicate registration still replays REGISTERED.
    assert register(backend, scope_a, epoch) == original


def test_adopt_refuses_terminal_workers(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.worker_executions.finish(**ident(epoch))
    with refused("stale_generation"):
        _adopt(backend, scope_a, epoch)


def test_adopt_reowns_the_linked_started_admission(backend, scope_a):
    epoch = seed(backend, scope_a)
    store = backend.open(scope_a)
    admitted = _admit(store, epoch, "req-started")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    register(backend, scope_a, epoch, generation=claimed.generation)
    rotated = begin_epoch(backend, scope_a)
    record = _adopt(backend, scope_a, rotated, generation=ExecutionGeneration(claimed.generation))
    assert record.owner_epoch == rotated
    admission = store.admissions.get(admission_id=admitted.admission_id)
    assert (admission.status, admission.owner_epoch) == ("started", rotated)


def test_adopt_refuses_an_admission_worker_of_another_epoch(backend, scope_a):
    # An admission-reserved worker reports to the owner that reserved it; a late handshake reaching a restarted owner is refused.
    epoch = seed(backend, scope_a)
    store = backend.open(scope_a)
    admitted = _admit(store, epoch, "req-reserved")
    store.admissions.claim(epoch=epoch, session_id=SESSION)
    scope = store.worker_executions.reserve_for_admission(epoch=epoch, admission_id=admitted.admission_id, principal_id=PRINCIPAL,
                                                          profile_id="profile-a", pid=4242, birth=1.5, secret="s" * 32, owner=None)
    rotated = begin_epoch(backend, scope_a)
    claim = store.worker_executions.verify_claim(execution_id=scope.execution_id, session_id=scope.session_id, generation=scope.generation,
                                                 adoption_secret=_reservation_claim(scope, PRINCIPAL))
    with refused("stale_epoch"):
        _adopt(backend, scope_a, rotated, execution_id=claim.execution_id, generation=scope.generation, adoption_secret=_reservation_claim(scope, PRINCIPAL))


def _reservation_claim(scope, principal: str) -> str:
    """The claim a reserved worker presents: ``admission_fingerprint`` over its scope and principal."""
    from persistence.authority.vocabulary import admission_fingerprint

    fields = {"profile_id": scope.profile_id, "session_id": scope.session_id, "execution_id": scope.execution_id, "generation": scope.generation,
              "pid": scope.pid, "birth": scope.birth, "secret": scope.secret}
    return admission_fingerprint(canonical_target=scope.session_id, payload=fields | {"principal": principal})


def test_verify_claim_returns_the_current_record_after_the_id_session_generation_and_secret_checks(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    check = store.worker_executions.verify_claim
    assert check(execution_id=EXECUTION, session_id=SESSION, generation=ExecutionGeneration(0), adoption_secret=SECRET).status is WorkerStatus.REGISTERED
    with refused("not_found"):
        check(execution_id=WorkerExecutionId("ghost"), session_id=SESSION, generation=ExecutionGeneration(0), adoption_secret=SECRET)
    with refused("permission_denied"):
        check(execution_id=EXECUTION, session_id=SessionId("elsewhere"), generation=ExecutionGeneration(0), adoption_secret=SECRET)
    with refused("stale_generation"):
        check(execution_id=EXECUTION, session_id=SESSION, generation=ExecutionGeneration(1), adoption_secret=SECRET)
    with refused("permission_denied"):
        check(execution_id=EXECUTION, session_id=SESSION, generation=ExecutionGeneration(0), adoption_secret="not-the-secret-0123456789")
    store.worker_executions.finish(**ident(epoch))
    assert check(execution_id=EXECUTION, session_id=SESSION, generation=ExecutionGeneration(0), adoption_secret=SECRET).status is WorkerStatus.TERMINAL


def test_execution_session_and_has_live_worker(backend, scope_a):
    store = backend.open(scope_a)
    require_workers(backend)
    assert store.worker_executions.has_live_worker() is False
    assert store.worker_executions.execution_session(execution_id=EXECUTION) is None
    _store, epoch = worker(backend, scope_a)
    assert store.worker_executions.execution_session(execution_id=EXECUTION) == SESSION
    assert store.worker_executions.has_live_worker() is True
    store.worker_executions.finish(**ident(epoch))
    assert store.worker_executions.has_live_worker() is False
    assert store.worker_executions.execution_session(execution_id=EXECUTION) == SESSION


def test_worker_state_is_scope_isolated(backend, scope_a, scope_b):
    epoch_a = seed(backend, scope_a)
    register(backend, scope_a, epoch_a)
    # Scope B never issued an epoch: A's value is a stale fence there, and nothing leaks across.
    with refused("stale_epoch"):
        register(backend, scope_b, epoch_a)
    epoch_b = seed(backend, scope_b)
    assert register(backend, scope_b, epoch_b).last_sequence == 0
    backend.open(scope_a).worker_executions.append_transcript(**ident(epoch_a), messages=[{"role": "assistant", "content": "scope-a-only"}])
    backend.open(scope_b).worker_executions.append_transcript(**ident(epoch_b), messages=[{"role": "assistant", "content": "scope-b-only"}])
    assert [r.content for r in backend.open(scope_a).messages.read_transcript(SESSION)] == ["scope-a-only"]
    assert [r.content for r in backend.open(scope_b).messages.read_transcript(SESSION)] == ["scope-b-only"]
    assert backend.open(scope_b).worker_executions.has_live_worker() is True
