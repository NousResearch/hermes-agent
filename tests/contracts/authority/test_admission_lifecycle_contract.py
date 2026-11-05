"""Contract cases for the admission lifecycle verbs beyond admit and settle: claim's gate and head, return_claim, steering transfers, mark_unknown, rehoming, recovery and resolution."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, ExecutionGeneration, FenceStale, StaleFenceKind
from persistence.authority.admissions import CapturedResult, GateRefusal, TurnBoundary
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.workers.executions import WorkerExecutionId, WorkerKind
from tests.contracts.conftest import require_capability

SESSION = "sess-life"
PRINCIPAL = "principal-life"


def _seed(store, *session_ids: str) -> int:
    for sid in session_ids or (SESSION,):
        store.sessions.create(NewSession(session_id=sid, source="contract"))
    return store.runtime.begin_epoch(instance_id="instance-life")


def _admit(store, epoch: int, request_id: str, *, session_id: str = SESSION, principal_id: str = PRINCIPAL, payload=None):
    record, created = store.admissions.admit(epoch=epoch, principal_id=principal_id, session_id=session_id, request_id=request_id,
                                             payload=payload if payload is not None else {"text": request_id})
    assert created
    return record


def test_claim_raises_the_gate_after_the_epoch_and_before_the_session(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    _admit(store, epoch, "r1")
    gate = GateRefusal("runtime_draining", field="runtime", detail="draining")
    with pytest.raises(FenceStale) as stale:
        store.admissions.claim(epoch=epoch + 1, session_id=SESSION, gate=gate)
    assert stale.value.kind is StaleFenceKind.STALE_EPOCH
    for session_id in (SESSION, "sess-absent"):
        with pytest.raises(AuthorityRefusal) as refused:
            store.admissions.claim(epoch=epoch, session_id=session_id, gate=gate)
        assert (refused.value.reason, refused.value.field, refused.value.detail) == ("runtime_draining", "runtime", "draining")
    assert store.admissions.list(session_id=SESSION)[0].status == "queued"


def test_a_gate_refusal_claims_nothing(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    first = _admit(store, epoch, "r1")
    session = store.sessions.get(SESSION)
    # CONTRACT: a gate refusal raises with its reason, field and detail and writes nothing: the row stays queued and the generation does not move.
    with pytest.raises(AuthorityRefusal) as refused:
        store.admissions.claim(epoch=epoch, session_id=SESSION, gate=GateRefusal("runtime_draining", field="runtime", detail="draining"))
    assert (refused.value.reason, refused.value.field, refused.value.detail) == ("runtime_draining", "runtime", "draining")
    assert store.admissions.list(session_id=SESSION) == [first]
    assert store.sessions.get(SESSION) == session and session.runtime_generation == 0
    # CONTRACT: no gate claims.
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION, gate=None)
    assert claimed.admission_id == first.admission_id and claimed.status == "started"


def test_claim_head_names_the_row_the_caller_validated(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    # CONTRACT: head=None means the caller saw nothing queued; a queued head is then False, as is a different head id.
    assert store.admissions.claim(epoch=epoch, session_id=SESSION, head=None) is None
    first = _admit(store, epoch, "r1")
    assert store.admissions.claim(epoch=epoch, session_id=SESSION, head=None) is False
    assert store.admissions.claim(epoch=epoch, session_id=SESSION, head="someone-else") is False
    store.messages.append(SESSION, NewMessage(role="user", content="before the claim"))
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION, head=first.admission_id)
    assert claimed.admission_id == first.admission_id and claimed.status == "started" and claimed.generation == 1
    # CONTRACT: the claim carries the transcript high-water mark and the owner epoch the row held while queued.
    assert claimed.transcript_mark == max(row.id for row in store.messages.read_transcript(SESSION))
    assert claimed.queued_owner_epoch == epoch
    assert store.sessions.get(SESSION).runtime_generation == 1


def test_return_claim_restores_the_queued_row_and_the_generation(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    first = _admit(store, epoch, "r1")
    second = _admit(store, epoch, "r2")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    revision = store.sessions.get(SESSION).runtime_revision
    with pytest.raises(FenceStale) as stale:
        store.admissions.return_claim(epoch=epoch + 1, admission_id=first.admission_id, generation=claimed.generation, queued_owner_epoch=epoch)
    assert stale.value.kind is StaleFenceKind.STALE_EPOCH
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store.admissions.return_claim(epoch=epoch, admission_id="absent", generation=1, queued_owner_epoch=epoch)
    with pytest.raises(FenceStale):
        store.admissions.return_claim(epoch=epoch, admission_id=first.admission_id, generation=claimed.generation + 1, queued_owner_epoch=epoch)
    returned = store.admissions.return_claim(epoch=epoch, admission_id=first.admission_id, generation=claimed.generation, queued_owner_epoch=claimed.queued_owner_epoch)
    assert (returned.status, returned.generation, returned.owner_epoch, returned.seq) == ("queued", None, epoch, first.seq)
    assert store.sessions.get(SESSION).runtime_generation == claimed.generation - 1
    assert store.sessions.get(SESSION).runtime_revision == revision
    # CONTRACT: a repeat is an idempotent replay; the row keeps its FIFO place, so it is claimed first again.
    assert store.admissions.return_claim(epoch=epoch, admission_id=first.admission_id, generation=claimed.generation, queued_owner_epoch=epoch) == returned
    assert store.admissions.claim(epoch=epoch, session_id=SESSION).admission_id == first.admission_id
    assert second.status == "queued"


def test_return_claim_refuses_once_a_worker_or_a_steer_is_bound(backend, scope_a):
    store = backend.open(scope_a)
    require_capability(backend, "receipted_writes")
    epoch = _seed(store)
    first = _admit(store, epoch, "r1")
    steered = _admit(store, epoch, "r2")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.admissions.steer(epoch=epoch, admission_id=steered.admission_id, generation=claimed.generation)
    with pytest.raises(FenceStale) as stale:
        store.admissions.return_claim(epoch=epoch, admission_id=first.admission_id, generation=claimed.generation, queued_owner_epoch=epoch)
    assert stale.value.kind is StaleFenceKind.STALE_GENERATION
    store.admissions.revert_steer(epoch=epoch, admission_id=steered.admission_id, generation=claimed.generation)
    store.worker_executions.register(epoch=epoch, execution_id=WorkerExecutionId("exec-bound"), session_id=SESSION,
                                     generation=ExecutionGeneration(claimed.generation), kind=WorkerKind.COMPUTE, adoption_secret="adoption-secret-0123456789abcdef")
    with pytest.raises(FenceStale):
        store.admissions.return_claim(epoch=epoch, admission_id=first.admission_id, generation=claimed.generation, queued_owner_epoch=epoch)
    assert store.admissions.get(admission_id=first.admission_id).status == "started"


def test_steer_transfers_a_queued_row_and_settle_decides_it(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    head = _admit(store, epoch, "r1")
    delivered = _admit(store, epoch, "r2")
    undelivered = _admit(store, epoch, "r3")
    with pytest.raises(AuthorityRefusal, match="^session_idle$"):
        store.admissions.steer(epoch=epoch, admission_id=delivered.admission_id, generation=1)
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    with pytest.raises(AuthorityRefusal, match="^admission_started$"):
        store.admissions.steer(epoch=epoch, admission_id=head.admission_id, generation=claimed.generation)
    row, committed = store.admissions.steer(epoch=epoch, admission_id=delivered.admission_id, generation=claimed.generation)
    assert committed and (row.status, row.outcome, row.generation) == ("terminal", "steering", claimed.generation)
    # CONTRACT: an exact retry replays the committed transfer; an in-flight steer is not cancellable.
    assert store.admissions.steer(epoch=epoch, admission_id=delivered.admission_id, generation=claimed.generation) == (row, False)
    with pytest.raises(FenceStale):
        store.admissions.cancel(epoch=epoch, admission_id=delivered.admission_id)
    store.admissions.steer(epoch=epoch, admission_id=undelivered.admission_id, generation=claimed.generation)
    store.admissions.settle(epoch=epoch, admission_id=head.admission_id, generation=claimed.generation, outcome="completed",
                            transfers={delivered.admission_id: True, undelivered.admission_id: False})
    assert store.admissions.get(admission_id=delivered.admission_id).outcome == "steered"
    back = store.admissions.get(admission_id=undelivered.admission_id)
    assert (back.status, back.outcome, back.generation, back.seq) == ("queued", None, None, undelivered.seq)
    # CONTRACT: revert_steer answers False once the settlement decided the transfer.
    assert store.admissions.revert_steer(epoch=epoch, admission_id=delivered.admission_id, generation=claimed.generation) is False


def test_mark_unknown_fences_the_owner_and_settles_transfers_as_unknown(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    head = _admit(store, epoch, "r1")
    steered = _admit(store, epoch, "r2")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.admissions.steer(epoch=epoch, admission_id=steered.admission_id, generation=claimed.generation)
    revision = store.sessions.get(SESSION).runtime_revision
    with pytest.raises(FenceStale):
        store.admissions.mark_unknown(epoch=epoch, admission_id=head.admission_id, generation=claimed.generation + 1)
    assert store.admissions.mark_unknown(epoch=epoch, admission_id=head.admission_id, generation=claimed.generation,
                                         transfers={steered.admission_id: True}) == "unknown"
    assert store.admissions.get(admission_id=steered.admission_id).outcome == "interrupted"
    assert store.sessions.get(SESSION).runtime_revision == revision + 1
    # CONTRACT: a row already unknown or terminal answers its status without writing.
    assert store.admissions.mark_unknown(epoch=epoch, admission_id=head.admission_id, generation=claimed.generation) == "unknown"
    assert store.sessions.get(SESSION).runtime_revision == revision + 1


def test_rehome_queued_moves_prefixed_rows_keeping_identity_and_order(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, SESSION, "sess-new")
    first = _admit(store, epoch, "r1", principal_id="local:alice")
    _admit(store, epoch, "r2", principal_id="api")
    third = _admit(store, epoch, "r3", principal_id="local:bob")
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        store.admissions.rehome_queued(epoch=epoch, old_session_id=SESSION, new_session_id="sess-absent", principal_prefix="local:")
    moved = store.admissions.rehome_queued(epoch=epoch, old_session_id=SESSION, new_session_id="sess-new", principal_prefix="local:")
    assert moved == [first.admission_id, third.admission_id]
    rows = store.admissions.list(session_id="sess-new")
    assert [(r.admission_id, r.request_id, r.seq) for r in rows] == [(first.admission_id, "r1", first.seq), (third.admission_id, "r3", third.seq)]
    # CONTRACT: the digest follows the new target, so an exact retry on the new target replays.
    replay, created = store.admissions.admit(epoch=epoch, principal_id="local:alice", session_id="sess-new", request_id="r1", payload={"text": "r1"})
    assert (replay.admission_id, created) == (first.admission_id, False)
    assert store.sessions.get("sess-new").runtime_revision == 1 and store.sessions.get(SESSION).runtime_revision == 1


def test_retarget_local_automation_points_stale_targets_at_the_new_one(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, SESSION, "sess-tip")
    stale = _admit(store, epoch, "auto-1", payload={"local_automation_v1": {"target": "sess-old", "kind": "notice"}})
    _admit(store, epoch, "auto-2", payload={"local_automation_v1": {"target": "sess-other"}})
    _admit(store, epoch, "plain", payload={"text": "x"})
    moved = store.admissions.retarget_local_automation(epoch=epoch, session_id=SESSION, principal_id=PRINCIPAL, old_targets=["sess-old"], new_target="sess-tip")
    assert moved == [stale.admission_id]
    assert store.admissions.get(admission_id=stale.admission_id).payload == {"local_automation_v1": {"target": "sess-tip", "kind": "notice"}}
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        store.admissions.retarget_local_automation(epoch=epoch, session_id=SESSION, principal_id=PRINCIPAL, old_targets=[""], new_target="sess-tip")


def test_nothing_moves_behind_an_unknown_row(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store, SESSION, "sess-new")
    _admit(store, epoch, "r1", principal_id="local:a")
    store.admissions.claim(epoch=epoch, session_id=SESSION)
    _admit(store, epoch, "r2", principal_id="local:a", payload={"local_automation_v1": {"target": "x"}})
    rotated = store.runtime.begin_epoch(instance_id="next")
    assert store.admissions.recover(epoch=rotated) == 1
    assert store.admissions.rehome_queued(epoch=rotated, old_session_id=SESSION, new_session_id="sess-new", principal_prefix="local:") == []
    assert store.admissions.retarget_local_automation(epoch=rotated, session_id=SESSION, principal_id="local:a", old_targets=["x"], new_target="sess-new") == []


def test_recover_interrupts_foreign_in_flight_steers(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    _admit(store, epoch, "r1")
    steered = _admit(store, epoch, "r2")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.admissions.steer(epoch=epoch, admission_id=steered.admission_id, generation=claimed.generation)
    rotated = store.runtime.begin_epoch(instance_id="next")
    assert store.admissions.recover(epoch=rotated) == 1
    assert store.admissions.get(admission_id=steered.admission_id).outcome == "interrupted"


def test_settle_boundary_closes_the_turn_on_the_transcript(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    record = _admit(store, epoch, "r1")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.messages.append(SESSION, NewMessage(role="user", content="please"))
    store.admissions.settle(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation, outcome="interrupted",
                            boundary=TurnBoundary("The gateway restarted.", replace_after=claimed.transcript_mark, display_metadata={"error": "restart"}))
    tail = store.messages.read_transcript(SESSION)[-1]
    assert (tail.role, tail.content, tail.display_kind, tail.display_metadata) == ("assistant", "The gateway restarted.", "failed_turn", {"error": "restart"})


def test_resolve_unknown_commits_a_captured_result_of_this_owner_only(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    record = _admit(store, epoch, "r1")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    store.messages.append(SESSION, NewMessage(role="user", content="lost"))
    assert store.admissions.mark_unknown(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation) == "unknown"
    captured = CapturedResult("completed", {"result": {"final_response": "kept"}})
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        store.admissions.resolve_unknown(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation, captured=CapturedResult("rejected", {}))  # type: ignore[arg-type]
    resolved = store.admissions.resolve_unknown(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation, notice="ignored", captured=captured)
    assert (resolved.status, resolved.outcome) == ("terminal", "completed")
    assert store.admissions.settled_result(admission_id=record.admission_id).result == {"result": {"final_response": "kept"}}
    # CONTRACT: the closer ran without the notice, closing the open user tail with the partial copy.
    tail = store.messages.read_transcript(SESSION)[-1]
    assert (tail.role, tail.display_kind) == ("assistant", "failed_turn") and tail.content != "ignored"
    # CONTRACT: a terminal row with a stored result replays.
    assert store.admissions.resolve_unknown(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation) == resolved


def test_resolve_unknown_of_another_owner_writes_the_notice(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    record = _admit(store, epoch, "r1")
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    rotated = store.runtime.begin_epoch(instance_id="next")
    store.admissions.recover(epoch=rotated)
    resolved = store.admissions.resolve_unknown(epoch=rotated, admission_id=record.admission_id, generation=claimed.generation,
                                                notice="An automated delivery was interrupted.", captured=CapturedResult("completed", {"result": {}}))
    assert resolved.outcome == "interrupted"
    assert store.admissions.settled_result(admission_id=record.admission_id) is None
    tail = store.messages.read_transcript(SESSION)[-1]
    assert (tail.content, tail.display_kind) == ("An automated delivery was interrupted.", "failed_turn")
