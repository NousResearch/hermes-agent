"""Contract suite for ``AdmissionRepository``.

Runs against every registered backend: the reference proves honest fail-closed semantics and injected crash windows through the shared authority state's failpoints; SQLite proves the same contract through its real single-transaction lifecycle verbs, with the process-crash instrumentation exercising the settle transaction THROUGH the adapter.

Test-construction seams: the epoch comes from ``SessionRuntimeRepository.begin_epoch``, retirement from ``mutate(Delete())``, worker rows from the worker-executions capability's own ``register``.
"""

from __future__ import annotations

import dataclasses


import pytest

from persistence import (
    AuthorityEpoch,
    AuthorityRefusal,
    ExecutionGeneration,
    FenceStale,
    SessionId,
    StaleFenceKind,
    ProfileScope,
    UnknownExecution,
)
from persistence.authority.admissions import (
    AdmissionId,
    AdmissionIntent,
    AdmissionOutcome,
    AdmissionRecord,
    AdmissionStatus,
)
from persistence.authority.runtime import Delete
from persistence.authority.vocabulary import ADMITTED_REQUEST_CONFLICT
from persistence.conversation.sessions import NewSession
from persistence.workers.executions import WorkerExecutionId, WorkerKind
from tests.contracts._reference.authority.admissions import (
    CLAIM_FAILPOINT,
    SETTLE_FAILPOINT,
)
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts.conftest import require_capability, require_failpoints

SESSION = SessionId("sess-admit")
PRINCIPAL = "principal-a"
SECRET = "adoption-secret-0123456789abcdef"


def _require(backend) -> None:
    require_capability(backend, "durable_admissions")


def _begin_epoch(backend, scope: ProfileScope) -> AuthorityEpoch:
    return AuthorityEpoch(backend.open(scope).runtime.begin_epoch(instance_id="instance-contract"))


def _seed(backend, scope: ProfileScope, session_id: SessionId = SESSION) -> AuthorityEpoch:
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=session_id, source="contract"))
    return _begin_epoch(backend, scope)


def _admit(backend, scope, epoch, *, request_id: str = "req-1", session_id: SessionId = SESSION, payload=None, intent=AdmissionIntent.QUEUE, principal_id: str = PRINCIPAL) -> AdmissionRecord:
    store = backend.open(scope)
    record, _created = store.admissions.admit(
        epoch=epoch, principal_id=principal_id, session_id=session_id,
        request_id=request_id, payload=payload if payload is not None else {"input": "hello"}, intent=intent,
    )
    return record


# ── record validation (contract boundary, no backend) ─────────────────────────


def test_admission_record_is_the_row_without_digest_or_lineage():
    # CONTRACT: the record carries the row's columns without digest and lineage, as plain strings.
    record = AdmissionRecord(seq=1, admission_id="adm-1", request_id="req-1", principal_id=PRINCIPAL, target_session_id=SESSION,
                             payload={}, intent="queue", status="queued", outcome=None, owner_epoch=1, generation=None)
    assert [f.name for f in dataclasses.fields(record)] == [
        "seq", "admission_id", "request_id", "principal_id", "target_session_id", "payload", "intent", "status", "outcome", "owner_epoch", "generation"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.status = "started"  # type: ignore[misc]
    assert record.status == AdmissionStatus.QUEUED and record.intent == AdmissionIntent.QUEUE


# ── admit ─────────────────────────────────────────────────────────────────────


def test_admit_creates_a_queued_admission_with_required_identity(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    assert record.status == AdmissionStatus.QUEUED
    assert record.outcome is None
    assert record.generation is None
    assert (record.principal_id, record.request_id, record.target_session_id) == (PRINCIPAL, "req-1", SESSION)
    assert record.payload == {"input": "hello"}
    assert record.owner_epoch == epoch
    assert store_a.admissions.get(admission_id=record.admission_id) == record


def test_admit_validates_inputs_before_touching_the_store(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    for kwargs in (
        {"principal_id": ""},
        {"principal_id": "p" * 1025},
        {"request_id": ""},
        {"payload": {"bad": float("nan")}},
    ):
        with pytest.raises(ValueError):
            _admit(backend, scope_a, epoch, **kwargs)
    assert store_a.admissions.list(session_id=SESSION) == []


def test_admit_fences_epoch_and_requires_the_session(backend, scope_a):
    _require(backend)
    epoch = _seed(backend, scope_a)
    for bad in (AuthorityEpoch(int(epoch) + 1), True, 1.0, AuthorityEpoch(0)):
        with pytest.raises(FenceStale) as refused:
            _admit(backend, scope_a, bad)
        assert refused.value.kind is StaleFenceKind.STALE_EPOCH
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        _admit(backend, scope_a, epoch, session_id=SessionId("sess-absent"))


def test_admit_dedups_on_identity_replaying_exact_and_conflicting_on_digest(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    first = _admit(backend, scope_a, epoch)
    assert _admit(backend, scope_a, epoch) == first
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _admit(backend, scope_a, epoch, payload={"input": "changed"})
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _admit(backend, scope_a, epoch, intent=AdmissionIntent.STEER)
    # A different identity component is a NEW admission, not a conflict.
    other = _admit(backend, scope_a, epoch, request_id="req-2")
    assert other.admission_id != first.admission_id
    assert len(store_a.admissions.list(session_id=SESSION)) == 2


def test_admit_replays_the_dedup_row_after_it_went_terminal(backend, scope_a):
    # The dedup identity outlives the lifecycle: a duplicate of settled work
    # replays the terminal record instead of re-queueing it.
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id,
        generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED,
    )
    replayed = _admit(backend, scope_a, epoch)
    assert replayed.admission_id == record.admission_id
    assert replayed.status == AdmissionStatus.TERMINAL
    assert replayed.outcome == AdmissionOutcome.COMPLETED


def _retire_session(backend, scope: ProfileScope, epoch: AuthorityEpoch, terminal: AdmissionRecord) -> None:
    """Retire ``SESSION`` through the runtime delete (test construction; the retirement contract is ``tests/contracts/conversation/test_sessions_contract.py``): the row goes, the marker is set, and each terminal admission becomes a tombstone that keeps its digest while its payload is scrubbed."""
    store = backend.open(scope)
    row = store.sessions.get(SESSION)
    store.runtime.mutate(epoch, principal_id=PRINCIPAL, session_id=SESSION, request_id="req-delete", expected_revision=row.runtime_revision,
                         expected_generation=row.runtime_generation, command=Delete(), owner=None)


def test_an_admitted_request_conflict_names_the_request_id(backend, scope_a):
    # CONTRACT: another digest for an admitted (principal, target, request), live or as a retired target's tombstone, and in retired_replay, refuses admission_conflict carrying ADMITTED_REQUEST_CONFLICT's field and detail.
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    expected = ("admission_conflict", ADMITTED_REQUEST_CONFLICT["field"], ADMITTED_REQUEST_CONFLICT["detail"])
    with pytest.raises(AuthorityRefusal) as live:
        _admit(backend, scope_a, epoch, payload={"input": "changed"})
    assert (live.value.reason, live.value.field, live.value.detail) == expected
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    settled = store_a.admissions.settle(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED)
    _retire_session(backend, scope_a, epoch, settled)
    with pytest.raises(AuthorityRefusal) as tombstone:
        _admit(backend, scope_a, epoch, payload={"input": "changed"})
    assert (tombstone.value.reason, tombstone.value.field, tombstone.value.detail) == expected
    with pytest.raises(AuthorityRefusal) as replay:
        store_a.admissions.retired_replay(epoch=epoch, principal_id=PRINCIPAL, session_id=SESSION, request_id="req-1", payload={"input": "changed"})
    assert (replay.value.reason, replay.value.field, replay.value.detail) == expected


def test_admit_on_a_retired_session_permits_only_the_exact_original_digest(backend, scope_a):
    # The retired branch of admit: exact digest
    # replays the terminal evidence, a different payload conflicts, an unknown
    # identity is NotFound; nothing ever re-queues on a retired id.
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    settled = store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id,
        generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED,
    )
    _retire_session(backend, scope_a, epoch, settled)
    replayed = _admit(backend, scope_a, epoch)
    assert replayed.admission_id == record.admission_id
    assert replayed.status == AdmissionStatus.TERMINAL
    # The tombstone keeps the digest but scrubs the payload (user data must
    # not outlive the delete); the exact-digest gate still discriminates.
    assert replayed.payload == {}
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _admit(backend, scope_a, epoch, payload={"input": "changed"})
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        _admit(backend, scope_a, epoch, request_id="req-fresh")


# ── get / list ────────────────────────────────────────────────────────────────


def test_get_returns_typed_none_for_unknown_ids(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    _seed(backend, scope_a)
    assert store_a.admissions.get(admission_id=AdmissionId("adm-absent")) is None


def test_list_is_fifo_and_pending_only_excludes_terminal(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    first = _admit(backend, scope_a, epoch, request_id="req-1")
    second = _admit(backend, scope_a, epoch, request_id="req-2")
    third = _admit(backend, scope_a, epoch, request_id="req-3")
    assert [r.admission_id for r in store_a.admissions.list(session_id=SESSION)] == [
        first.admission_id, second.admission_id, third.admission_id,
    ]
    store_a.admissions.cancel(epoch=epoch, admission_id=second.admission_id)
    pending = store_a.admissions.list(session_id=SESSION)
    assert [r.admission_id for r in pending] == [first.admission_id, third.admission_id]
    everything = store_a.admissions.list(session_id=SESSION, pending_only=False)
    assert [r.admission_id for r in everything] == [first.admission_id, second.admission_id, third.admission_id]
    assert store_a.admissions.list(session_id=SessionId("sess-unknown")) == []


def test_find_is_the_exact_identity_and_match_filters_across_sessions(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    other = SessionId("sess-other")
    store_a.sessions.create(NewSession(session_id=other, source="contract"))
    first = _admit(backend, scope_a, epoch, request_id="run-1")
    second = _admit(backend, scope_a, epoch, request_id="run-1", session_id=other)
    third = _admit(backend, scope_a, epoch, request_id="run-2", principal_id="principal-b")
    store_a.admissions.cancel(epoch=epoch, admission_id=second.admission_id)
    ids = lambda rows: [r.admission_id for r in rows]
    # CONTRACT: find is the UNIQUE (principal, target, request) key.
    assert store_a.admissions.find(principal_id=PRINCIPAL, session_id=SESSION, request_id="run-1") == first
    assert store_a.admissions.find(principal_id=PRINCIPAL, session_id=other, request_id="run-1").status == AdmissionStatus.TERMINAL
    assert store_a.admissions.find(principal_id="nobody", session_id=SESSION, request_id="run-1") is None
    # CONTRACT: the (principal, request) pair is not unique across sessions; every match comes back in seq order.
    assert ids(store_a.admissions.match(principal_id=PRINCIPAL, request_id="run-1")) == [first.admission_id, second.admission_id]
    # CONTRACT: filters combine with AND; an unknown value is an empty list, never not_found.
    assert ids(store_a.admissions.match(session_id=SESSION, request_id="run-1")) == [first.admission_id]
    assert ids(store_a.admissions.match(admission_id=third.admission_id, principal_id="principal-b")) == [third.admission_id]
    assert store_a.admissions.match(principal_id="nobody") == []
    (found,) = store_a.admissions.match(request_id="run-2")
    assert found == third


def test_sequence_orders_admissions_and_survives_retirement(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    other = SessionId("sess-seq-other")
    store_a.sessions.create(NewSession(session_id=other, source="contract"))
    first = _admit(backend, scope_a, epoch, request_id="seq-1")
    second = _admit(backend, scope_a, epoch, request_id="seq-2", session_id=other)
    third = _admit(backend, scope_a, epoch, request_id="seq-3")
    # CONTRACT: sequence is strictly increasing in admission order, across sessions.
    assert first.seq < second.seq < third.seq
    # CONTRACT: every read verb reports the same sequence the admit returned; an exact replay keeps it.
    assert (store_a.admissions.get(admission_id=second.admission_id)).seq == second.seq
    assert [r.seq for r in store_a.admissions.list(session_id=SESSION)] == [first.seq, third.seq]
    assert (_admit(backend, scope_a, epoch, request_id="seq-1")).seq == first.seq
    store_a.admissions.cancel(epoch=epoch, admission_id=third.admission_id)
    terminal = store_a.admissions.cancel(epoch=epoch, admission_id=first.admission_id)
    _retire_session(backend, scope_a, epoch, terminal)
    # CONTRACT: the tombstone keeps the retired row's sequence; a later admission never reuses it.
    assert (store_a.admissions.get(admission_id=first.admission_id)).seq == first.seq
    later = _admit(backend, scope_a, epoch, request_id="seq-4", session_id=other)
    assert later.seq > third.seq


def test_list_unknown_reports_recovered_unknown_work(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    admitted = _admit(backend, scope_a, epoch, request_id="req-unknown")
    store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    queued = _admit(backend, scope_a, epoch, request_id="req-queued")
    successor = _begin_epoch(backend, scope_a)
    store_a.admissions.recover(epoch=successor)
    # CONTRACT: work a superseded epoch left started reads as unknown.
    assert [(r.admission_id, r.status) for r in store_a.admissions.list_unknown()] == [(admitted.admission_id, AdmissionStatus.UNKNOWN)]
    assert store_a.admissions.pending_targets(statuses=frozenset({"unknown"})) == frozenset({SESSION})
    assert [r.admission_id for r in store_a.admissions.list(session_id=SESSION)] == [admitted.admission_id, queued.admission_id]


def test_live_reads_do_not_see_retired_rows_that_get_still_answers(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    admitted = _admit(backend, scope_a, epoch, request_id="req-retire")
    terminal = store_a.admissions.cancel(epoch=epoch, admission_id=admitted.admission_id)
    _retire_session(backend, scope_a, epoch, terminal)
    # CONTRACT: retirement removes the row from the live ledger; get still answers from the tombstone.
    assert store_a.admissions.match(request_id="req-retire") == []
    assert store_a.admissions.find(principal_id=PRINCIPAL, session_id=SESSION, request_id="req-retire") is None
    tombstone = store_a.admissions.get(admission_id=admitted.admission_id)
    assert (tombstone.seq, tombstone.status, tombstone.outcome, tombstone.payload) == (terminal.seq, AdmissionStatus.TERMINAL, AdmissionOutcome.CANCELLED, {})


# ── claim ─────────────────────────────────────────────────────────────────────


def test_claim_starts_the_fifo_head_under_a_fresh_generation(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    first = _admit(backend, scope_a, epoch, request_id="req-1")
    _admit(backend, scope_a, epoch, request_id="req-2")
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    assert claimed.admission_id == first.admission_id
    assert claimed.status == AdmissionStatus.STARTED
    assert claimed.generation is not None and int(claimed.generation) >= 1
    assert claimed.owner_epoch == epoch
    # The second claim observes the busy loser, typed None.
    assert store_a.admissions.claim(epoch=epoch, session_id=SESSION) is None


def test_claim_types_the_empty_queue_and_fences_epoch_and_session(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    assert store_a.admissions.claim(epoch=epoch, session_id=SESSION) is None
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.admissions.claim(epoch=epoch, session_id=SessionId("sess-absent"))
    with pytest.raises(FenceStale) as refused:
        store_a.admissions.claim(epoch=AuthorityEpoch(int(epoch) + 1), session_id=SESSION)
    assert refused.value.kind is StaleFenceKind.STALE_EPOCH


def test_claim_is_excluded_by_a_live_worker(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    require_capability(backend, "receipted_writes")
    epoch = _seed(backend, scope_a)
    _admit(backend, scope_a, epoch)
    store_a.worker_executions.register(
        epoch=epoch, execution_id=WorkerExecutionId("exec-live"), session_id=SESSION,
        generation=ExecutionGeneration(0), kind=WorkerKind.COMPUTE, adoption_secret=SECRET,
    )
    assert store_a.admissions.claim(epoch=epoch, session_id=SESSION) is None


def test_claim_fails_closed_behind_unknown_work(backend, scope_a):
    # Uncertainty BLOCKS the queue; it never requeues or skips.
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    rotated = _begin_epoch(backend, scope_a)
    marked = store_a.admissions.recover(epoch=rotated)
    assert marked == 1
    _admit(backend, scope_a, rotated, request_id="req-2")
    with pytest.raises(UnknownExecution):
        store_a.admissions.claim(epoch=rotated, session_id=SESSION)


def test_claim_fails_closed_behind_an_unknown_worker_alone(backend, scope_a):
    # The second barrier population: no unknown ADMISSION exists, only a
    # worker whose owning epoch died; recover marks it (returning the
    # admission-only count, zero here) and claim still refuses.
    store_a = backend.open(scope_a)
    _require(backend)
    require_capability(backend, "receipted_writes")
    epoch = _seed(backend, scope_a)
    store_a.worker_executions.register(
        epoch=epoch, execution_id=WorkerExecutionId("exec-orphan"), session_id=SESSION,
        generation=ExecutionGeneration(0), kind=WorkerKind.COMPUTE, adoption_secret=SECRET,
    )
    rotated = _begin_epoch(backend, scope_a)
    assert store_a.admissions.recover(epoch=rotated) == 0
    _admit(backend, scope_a, rotated)
    with pytest.raises(UnknownExecution):
        store_a.admissions.claim(epoch=rotated, session_id=SESSION)


# ── settle ────────────────────────────────────────────────────────────────────


def test_settle_commits_terminal_outcome_and_result_evidence_once(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    settled = store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id,
        generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED,
        result={"answer": 42},
    )
    assert settled.status == AdmissionStatus.TERMINAL
    assert settled.outcome == AdmissionOutcome.COMPLETED
    # Settle never replays: the terminal row refuses a second settlement.
    with pytest.raises(FenceStale) as refused:
        store_a.admissions.settle(
            epoch=epoch, admission_id=record.admission_id,
            generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED,
        )
    assert refused.value.kind is StaleFenceKind.STALE_GENERATION
    # The queue reopens: the next admission claims a HIGHER generation.
    _admit(backend, scope_a, epoch, request_id="req-2")
    reclaimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert reclaimed is not None
    assert int(reclaimed.generation) == int(claimed.generation) + 1


def test_settle_validates_outcome_and_fences_the_composite(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    with pytest.raises(ValueError):
        store_a.admissions.settle(
            epoch=epoch, admission_id=record.admission_id,
            generation=ExecutionGeneration(1), outcome=AdmissionOutcome.CANCELLED,
        )
    # Queued (never claimed) work cannot settle.
    with pytest.raises(FenceStale) as queued:
        store_a.admissions.settle(
            epoch=epoch, admission_id=record.admission_id,
            generation=ExecutionGeneration(1), outcome=AdmissionOutcome.COMPLETED,
        )
    assert queued.value.kind is StaleFenceKind.STALE_GENERATION
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.admissions.settle(
            epoch=epoch, admission_id=AdmissionId("adm-absent"),
            generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED,
        )
    for bad_generation in (int(claimed.generation) + 1, True, 0.5):
        with pytest.raises(FenceStale) as refused:
            store_a.admissions.settle(
                epoch=epoch, admission_id=record.admission_id,
                generation=bad_generation, outcome=AdmissionOutcome.COMPLETED,
            )
        assert refused.value.kind is StaleFenceKind.STALE_GENERATION
    # The started row survived every refusal.
    row = store_a.admissions.get(admission_id=record.admission_id)
    assert row is not None and row.status == AdmissionStatus.STARTED


def test_settle_retires_the_linked_worker_atomically(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    require_capability(backend, "receipted_writes")
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    store_a.worker_executions.register(
        epoch=epoch, execution_id=WorkerExecutionId("exec-linked"), session_id=SESSION,
        generation=claimed.generation, kind=WorkerKind.COMPUTE, adoption_secret=SECRET,
    )
    store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id,
        generation=claimed.generation, outcome=AdmissionOutcome.FAILED,
    )
    # The worker retired with the settlement: a fresh claim is not excluded.
    _admit(backend, scope_a, epoch, request_id="req-2")
    assert store_a.admissions.claim(epoch=epoch, session_id=SESSION) is not None


# ── cancel ────────────────────────────────────────────────────────────────────


def test_settled_result_pairs_the_record_with_its_committed_result(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    # CONTRACT: an unknown id is not_found; a non-terminal admission answers None.
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.admissions.settled_result(admission_id=AdmissionId("adm-missing"))
    assert store_a.admissions.settled_result(admission_id=record.admission_id) is None
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    # CONTRACT: fenced, a started row answers with no result.
    started = store_a.admissions.settled_result(admission_id=record.admission_id, epoch=epoch, generation=claimed.generation)
    assert started.record == store_a.admissions.get(admission_id=record.admission_id) and started.result is None
    settled = store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id, generation=claimed.generation,
        outcome=AdmissionOutcome.COMPLETED, result={"result": {"final_response": "done"}},
    )
    found = store_a.admissions.settled_result(admission_id=record.admission_id)
    assert found.record == settled
    assert found.result == {"result": {"final_response": "done"}}
    # CONTRACT: the result is detached; mutating it leaves the store unchanged.
    found.result["result"]["final_response"] = "mutated"
    assert store_a.admissions.settled_result(admission_id=record.admission_id).result["result"]["final_response"] == "done"
    # CONTRACT: a terminal admission without a stored result answers None unfenced; fenced it refuses storage_unavailable.
    other = _admit(backend, scope_a, epoch, request_id="req-2")
    store_a.admissions.cancel(epoch=epoch, admission_id=other.admission_id)
    assert store_a.admissions.settled_result(admission_id=other.admission_id) is None
    third = _admit(backend, scope_a, epoch, request_id="req-3")
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    store_a.admissions.settle(epoch=epoch, admission_id=third.admission_id, generation=claimed.generation, outcome="failed")
    with pytest.raises(AuthorityRefusal, match="^storage_unavailable$"):
        store_a.admissions.settled_result(admission_id=third.admission_id, epoch=epoch, generation=claimed.generation)


def test_settled_result_fence_is_epoch_then_lookup_then_owner(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    generation = claimed.generation
    # CONTRACT: the fence pair is both-or-neither.
    for kwargs in ({"epoch": epoch}, {"generation": generation}):
        with pytest.raises(ValueError):
            store_a.admissions.settled_result(admission_id=record.admission_id, **kwargs)
    # CONTRACT: a wrong epoch refuses before the lookup, even for an unknown id.
    for admission_id in (record.admission_id, AdmissionId("adm-missing")):
        with pytest.raises(FenceStale) as stale:
            store_a.admissions.settled_result(
                admission_id=admission_id, epoch=AuthorityEpoch(int(epoch) + 7), generation=generation)
        assert stale.value.kind is StaleFenceKind.STALE_EPOCH
    # CONTRACT: under the current epoch an unknown id is not_found, and a foreign generation is the generation-kind fence.
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.admissions.settled_result(admission_id=AdmissionId("adm-missing"), epoch=epoch, generation=generation)
    with pytest.raises(FenceStale) as stale:
        store_a.admissions.settled_result(
            admission_id=record.admission_id, epoch=epoch, generation=ExecutionGeneration(int(generation) + 1))
    assert stale.value.kind is StaleFenceKind.STALE_GENERATION
    # CONTRACT: an admission owned by a superseded epoch refuses under the new one (the owner, not the current epoch, must match).
    newer = _begin_epoch(backend, scope_a)
    with pytest.raises(FenceStale) as stale:
        store_a.admissions.settled_result(
            admission_id=record.admission_id, epoch=newer, generation=generation)
    assert stale.value.kind is StaleFenceKind.STALE_GENERATION


def test_settled_result_survives_retirement(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    settled = store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id, generation=claimed.generation,
        outcome=AdmissionOutcome.COMPLETED, result={"answer": 42},
    )
    _retire_session(backend, scope_a, epoch, settled)
    # CONTRACT: the retired admission answers from its tombstone and keeps its committed result, fenced or not.
    found = store_a.admissions.settled_result(admission_id=record.admission_id)
    assert found.record.status == AdmissionStatus.TERMINAL and found.record.seq == settled.seq
    assert found.result == {"answer": 42}
    fenced = store_a.admissions.settled_result(
        admission_id=record.admission_id, epoch=epoch, generation=claimed.generation)
    assert fenced.result == {"answer": 42}


def test_cancel_terminates_queued_replays_terminal_refuses_started_and_unknown(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    cancelled = store_a.admissions.cancel(epoch=epoch, admission_id=record.admission_id)
    assert cancelled.status == AdmissionStatus.TERMINAL
    assert cancelled.outcome == AdmissionOutcome.CANCELLED
    # Idempotent on terminal.
    assert store_a.admissions.cancel(epoch=epoch, admission_id=record.admission_id) == cancelled
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store_a.admissions.cancel(epoch=epoch, admission_id=AdmissionId("adm-absent"))
    started = _admit(backend, scope_a, epoch, request_id="req-2")
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    with pytest.raises(FenceStale) as refused:
        store_a.admissions.cancel(epoch=epoch, admission_id=started.admission_id)
    assert refused.value.kind is StaleFenceKind.STALE_GENERATION
    rotated = _begin_epoch(backend, scope_a)
    store_a.admissions.recover(epoch=rotated)
    with pytest.raises(UnknownExecution):
        store_a.admissions.cancel(epoch=rotated, admission_id=started.admission_id)


# ── recover / resolve_unknown ─────────────────────────────────────────────────


def test_recover_marks_only_foreign_epoch_work_and_never_requeues(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    # Same-epoch recovery touches nothing.
    assert store_a.admissions.recover(epoch=epoch) == 0
    row = store_a.admissions.get(admission_id=record.admission_id)
    assert row is not None and row.status == AdmissionStatus.STARTED
    rotated = _begin_epoch(backend, scope_a)
    assert store_a.admissions.recover(epoch=rotated) == 1
    row = store_a.admissions.get(admission_id=record.admission_id)
    assert row is not None and row.status == AdmissionStatus.UNKNOWN
    # Recovery is idempotent: unknown rows stay unknown, count zero.
    assert store_a.admissions.recover(epoch=rotated) == 0


def test_resolve_unknown_is_the_only_exit_and_requires_the_exact_generation(backend, scope_a):
    store_a = backend.open(scope_a)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    rotated = _begin_epoch(backend, scope_a)
    store_a.admissions.recover(epoch=rotated)
    # A queued or terminal row refuses resolution; only unknown resolves.
    fresh = _admit(backend, scope_a, rotated, request_id="req-2")
    with pytest.raises(FenceStale):
        store_a.admissions.resolve_unknown(
            epoch=rotated, admission_id=fresh.admission_id, generation=claimed.generation)
    with pytest.raises(FenceStale) as wrong_generation:
        store_a.admissions.resolve_unknown(
            epoch=rotated, admission_id=record.admission_id,
            generation=ExecutionGeneration(int(claimed.generation) + 1))
    assert wrong_generation.value.kind is StaleFenceKind.STALE_GENERATION
    resolved = store_a.admissions.resolve_unknown(
        epoch=rotated, admission_id=record.admission_id, generation=claimed.generation)
    assert resolved.status == AdmissionStatus.TERMINAL
    assert resolved.outcome == AdmissionOutcome.INTERRUPTED
    # The barrier cleared: the queue claims again.
    reclaimed = store_a.admissions.claim(epoch=rotated, session_id=SESSION)
    assert reclaimed is not None and reclaimed.admission_id == fresh.admission_id


# ── crash consistency ───────────────────────────────────────


def test_r30a_claim_crash_window_publishes_nothing_then_retries(backend, scope_a):
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    _require(backend)
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    state = authority_state(backend, scope_a)
    generation_before = state.snapshot["sessions"].get(SESSION, {}).get("generation", 0)
    state.failpoints.arm(CLAIM_FAILPOINT)
    with pytest.raises(InjectedFailure):
        store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    # Neither-or-both: no generation bump, the head still queued.
    assert state.snapshot["sessions"].get(SESSION, {}).get("generation", 0) == generation_before
    row = store_a.admissions.get(admission_id=record.admission_id)
    assert row is not None and row.status == AdmissionStatus.QUEUED
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    assert int(claimed.generation) == generation_before + 1


def test_r30a_settle_crash_window_publishes_nothing_then_retries(backend, scope_a):
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    _require(backend)
    require_capability(backend, "receipted_writes")
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    store_a.worker_executions.register(
        epoch=epoch, execution_id=WorkerExecutionId("exec-crash"), session_id=SESSION,
        generation=claimed.generation, kind=WorkerKind.COMPUTE, adoption_secret=SECRET,
    )
    state = authority_state(backend, scope_a)
    revision_before = state.snapshot["sessions"][SESSION]["revision"]
    state.failpoints.arm(SETTLE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        store_a.admissions.settle(
            epoch=epoch, admission_id=record.admission_id,
            generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED, result={"answer": 42},
        )
    # Neither-or-both: admission still started, worker still live, no revision
    # bump, no result evidence.
    row = store_a.admissions.get(admission_id=record.admission_id)
    assert row is not None and row.status == AdmissionStatus.STARTED
    assert state.snapshot["workers"]["exec-crash"]["status"] == "registered"
    assert state.snapshot["sessions"][SESSION]["revision"] == revision_before
    assert ("admission_result", str(record.admission_id)) not in state.snapshot["meta"]
    # Retry lands the whole transition.
    settled = store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id,
        generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED, result={"answer": 42},
    )
    assert settled.status == AdmissionStatus.TERMINAL
    assert state.snapshot["workers"]["exec-crash"]["status"] == "terminal"
    assert state.snapshot["sessions"][SESSION]["revision"] == revision_before + 1


def test_r30b_admission_lifecycle_survives_reopen(backend, scope_a):
    # The settled evidence, dedup identity, and FIFO state read
    # back identically through independent public reads after reopen.
    store_a = backend.open(scope_a)
    _require(backend)
    require_capability(backend, "reopen_durability")
    epoch = _seed(backend, scope_a)
    record = _admit(backend, scope_a, epoch)
    queued = _admit(backend, scope_a, epoch, request_id="req-2")
    claimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed is not None
    settled = store_a.admissions.settle(
        epoch=epoch, admission_id=record.admission_id,
        generation=claimed.generation, outcome=AdmissionOutcome.COMPLETED, result={"answer": 42},
    )
    backend.reopen()
    assert store_a.admissions.get(admission_id=record.admission_id) == settled
    assert _admit(backend, scope_a, epoch) == settled
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _admit(backend, scope_a, epoch, payload={"input": "changed"})
    pending = store_a.admissions.list(session_id=SESSION)
    assert [r.admission_id for r in pending] == [queued.admission_id]
    reclaimed = store_a.admissions.claim(epoch=epoch, session_id=SESSION)
    assert reclaimed is not None
    assert reclaimed.admission_id == queued.admission_id
    assert int(reclaimed.generation) == int(claimed.generation) + 1


# ── scope isolation ───────────────────────────────────────────────────────────


def test_admission_state_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _require(backend)
    epoch_a = _seed(backend, scope_a)
    record_a = _admit(backend, scope_a, epoch_a)
    # Scope B never issued an epoch: A's value is a stale fence there.
    with pytest.raises(FenceStale) as refused:
        _admit(backend, scope_b, epoch_a)
    assert refused.value.kind is StaleFenceKind.STALE_EPOCH
    epoch_b = _seed(backend, scope_b)
    # The same identity admits independently per scope; neither sees the other.
    record_b = _admit(backend, scope_b, epoch_b)
    assert record_b.status == AdmissionStatus.QUEUED
    assert store_b.admissions.get(admission_id=record_a.admission_id) in (None, record_b)
    claimed_b = store_b.admissions.claim(epoch=epoch_b, session_id=SESSION)
    assert claimed_b is not None
    row_a = store_a.admissions.get(admission_id=record_a.admission_id)
    assert row_a is not None and row_a.status == AdmissionStatus.QUEUED
