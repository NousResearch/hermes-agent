"""Contract suite for the worker authority verbs of the gateway owner.

``WorkerExecutionRepository.verify_claim`` is the owner's per-request claim check; ``reserve_for_admission`` and ``lose_reservation`` are the owner's private admission-worker reservation and its loss; ``SessionBindingRepository.validate_lineage`` is the local-lineage read the reservation relies on; ``SessionBindingRepository.authorize_history`` is the history-control policy of the owner's mutation path. Seeding goes through public verbs (``tests/contracts/workers/_support.py``, the admissions, bindings and sessions verbs).
"""

from __future__ import annotations

from persistence import AuthorityEpoch, ExecutionGeneration, ProfileScope, SessionId
from persistence.authority.admissions import AdmissionId, AdmissionStatus
from persistence.authority.bindings import local_session_id
from persistence.authority.vocabulary import admission_fingerprint
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from persistence.workers.executions import AdmissionWorkerScope, WorkerExecutionId, WorkerKind, WorkerStatus
from tests.contracts.authority.test_bindings_contract import make_binding
from tests.contracts.authority.test_owner_rule_contract import OWNER, RULE, STRANGER
from tests.contracts.workers._support import EXECUTION, SECRET, SESSION, begin_epoch, ident, refused, register, require_workers, seed

PRINCIPAL = "principal-a"
WORKER_SECRET = "w" * 43


def _verify(backend, scope, **overrides):
    args = {"execution_id": EXECUTION, "session_id": SESSION, "generation": ExecutionGeneration(0), "adoption_secret": SECRET}
    return backend.open(scope).worker_executions.verify_claim(**(args | overrides))


def test_verify_claim_returns_the_current_record_without_writing(backend, scope_a):
    epoch = seed(backend, scope_a)
    registered = register(backend, scope_a, epoch)
    assert _verify(backend, scope_a) == registered
    assert register(backend, scope_a, epoch) == registered


def test_verify_claim_resolution_order(backend, scope_a):
    epoch = seed(backend, scope_a)
    backend.open(scope_a).sessions.create(NewSession(session_id=SessionId("sess-other"), source="contract"))
    register(backend, scope_a, epoch)
    wrong = "wrong-secret-0123456789abcd"
    with refused("not_found"):
        _verify(backend, scope_a, execution_id=WorkerExecutionId("exec-absent"), session_id=SessionId("sess-other"))
    # A wrong session outranks a wrong generation and a wrong secret.
    with refused("permission_denied"):
        _verify(backend, scope_a, session_id=SessionId("sess-other"), generation=ExecutionGeneration(9), adoption_secret=wrong)
    # A wrong generation outranks a wrong secret; a non-int generation is the same refusal.
    for generation in (ExecutionGeneration(9), "0", True, 0.0):
        with refused("stale_generation"):
            _verify(backend, scope_a, generation=generation, adoption_secret=wrong)
    with refused("permission_denied"):
        _verify(backend, scope_a, adoption_secret=wrong)


def test_verify_claim_answers_for_terminal_workers(backend, scope_a):
    epoch = seed(backend, scope_a)
    register(backend, scope_a, epoch)
    backend.open(scope_a).worker_executions.finish(**ident(epoch))
    record = _verify(backend, scope_a)
    assert record.status is WorkerStatus.TERMINAL and record.generation == 0


def test_verify_claim_malformed_inputs(backend, scope_a):
    epoch = seed(backend, scope_a)
    register(backend, scope_a, epoch)
    # The secret is checked last.
    for secret in ("", "x" * 1025, 7):
        with refused("invalid_params"):
            _verify(backend, scope_a, adoption_secret=secret)
    for execution_id in ("", 7, None):
        with refused("not_found"):
            _verify(backend, scope_a, execution_id=execution_id, adoption_secret="")


# ── reserve_for_admission / lose_reservation ─────────────────────────────────


def _started(backend, scope: ProfileScope, epoch: AuthorityEpoch, session_id: SessionId = SESSION, request: str = "req-1"):
    store = backend.open(scope)
    admitted = store.admissions.admit(epoch=epoch, principal_id=PRINCIPAL, session_id=session_id, request_id=request, payload={"input": request})[0]
    claimed = store.admissions.claim(epoch=epoch, session_id=session_id)
    assert claimed and claimed.admission_id == admitted.admission_id
    return claimed


def _reserve(backend, scope, epoch, admission_id, *, principal_id=PRINCIPAL, owner=None, worker_subject=None) -> AdmissionWorkerScope:
    return backend.open(scope).worker_executions.reserve_for_admission(
        epoch=epoch, admission_id=admission_id, principal_id=principal_id, profile_id="profile-a", pid=4242, birth=1700000000.5, secret=WORKER_SECRET,
        owner=owner, worker_subject=worker_subject)


def _claim(scope: AdmissionWorkerScope, principal: str = PRINCIPAL) -> str:
    """The claim the reserved worker presents."""
    fields = {"profile_id": scope.profile_id, "session_id": scope.session_id, "execution_id": scope.execution_id, "generation": scope.generation,
              "pid": scope.pid, "birth": scope.birth, "secret": scope.secret}
    return admission_fingerprint(canonical_target=scope.session_id, payload=fields | {"principal": principal})


def test_reserve_inserts_a_registered_compute_worker_on_the_owner(backend, scope_a):
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    scope = _reserve(backend, scope_a, epoch, admission.admission_id)
    assert scope == AdmissionWorkerScope(profile_id="profile-a", session_id=SESSION, execution_id=WorkerExecutionId("admission-worker:" + admission.admission_id),
                                         generation=admission.generation, pid=4242, birth=1700000000.5, secret=WORKER_SECRET, epoch=epoch)
    # The adoption evidence is the digest of the claim over the scope and the principal.
    record = _verify(backend, scope_a, execution_id=scope.execution_id, generation=scope.generation, adoption_secret=_claim(scope))
    assert (record.kind, record.status, record.owner_epoch, record.last_sequence) == (WorkerKind.COMPUTE, WorkerStatus.REGISTERED, epoch, 0)
    with refused("permission_denied"):
        _verify(backend, scope_a, execution_id=scope.execution_id, generation=scope.generation, adoption_secret=_claim(scope, "someone-else"))
    assert backend.open(scope_a).admissions.get(admission_id=admission.admission_id).status == AdmissionStatus.STARTED


def test_reserve_refusal_order_writes_nothing(backend, scope_a):
    store = backend.open(scope_a)
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    queued = store.admissions.admit(epoch=epoch, principal_id=PRINCIPAL, session_id=SESSION, request_id="req-queued", payload={"input": "q"})[0]
    with refused("stale_epoch"):
        _reserve(backend, scope_a, AuthorityEpoch(int(epoch) + 7), AdmissionId("absent"))
    with refused("not_found"):
        _reserve(backend, scope_a, epoch, AdmissionId("absent"), principal_id="someone-else")
    with refused("producer_not_started"):
        _reserve(backend, scope_a, epoch, queued.admission_id, principal_id="someone-else")
    with refused("permission_denied"):
        _reserve(backend, scope_a, epoch, admission.admission_id, principal_id="someone-else")
    assert store.worker_executions.execution_session(execution_id=WorkerExecutionId("admission-worker:" + admission.admission_id)) is None
    _reserve(backend, scope_a, epoch, admission.admission_id)
    # An existing execution is a conflict, never a replay.
    with refused("admission_conflict"):
        _reserve(backend, scope_a, epoch, admission.admission_id)


def test_reserve_refuses_a_live_worker_on_the_physical_target(backend, scope_a):
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    register(backend, scope_a, epoch, generation=admission.generation)
    with refused("stale_generation"):
        _reserve(backend, scope_a, epoch, admission.admission_id)


def test_reserve_reads_the_stored_principal_through_the_owner_rule(backend, scope_a):
    store = backend.open(scope_a)
    epoch = seed(backend, scope_a)
    store.admissions.admit(epoch=epoch, principal_id="basic:admin", session_id=SESSION, request_id="req-login", payload={"input": "x"})
    admission = store.admissions.claim(epoch=epoch, session_id=SESSION)
    # CONTRACT: the admission's stored principal belongs to the caller under the owner rule (a sign-in stamp is the host owner's); without a rule only the equal stamp does.
    for principal, rule in ((STRANGER, RULE), (OWNER, None)):
        with refused("permission_denied"):
            _reserve(backend, scope_a, epoch, admission.admission_id, principal_id=principal, owner=rule)
    assert _reserve(backend, scope_a, epoch, admission.admission_id, principal_id=OWNER, owner=RULE).session_id == SESSION


def test_reserve_binds_the_claim_to_the_worker_subject_when_given(backend, scope_a):
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    scope = _reserve(backend, scope_a, epoch, admission.admission_id, worker_subject=OWNER)
    # CONTRACT: the adoption claim names the identity the worker authenticates as (``worker_subject``) when given, not the principal.
    _verify(backend, scope_a, execution_id=scope.execution_id, generation=scope.generation, adoption_secret=_claim(scope, OWNER))
    with refused("permission_denied"):
        _verify(backend, scope_a, execution_id=scope.execution_id, generation=scope.generation, adoption_secret=_claim(scope))


def test_reserve_refuses_an_admission_started_by_another_epoch(backend, scope_a):
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    with refused("producer_not_started"):
        _reserve(backend, scope_a, begin_epoch(backend, scope_a), admission.admission_id)


def _bound_session(backend, scope: ProfileScope, epoch: AuthorityEpoch) -> SessionId:
    store = backend.open(scope)
    sid = local_session_id(scope.name, PRINCIPAL, "req-local")
    store.bindings.reserve(epoch=epoch, binding=make_binding(str(sid), route="route-local", principal_id=PRINCIPAL, request_id="req-local", profile_id=scope.name), owner=None)
    return sid


def test_physical_target_is_the_session_itself_when_unbound_or_freshly_bound(backend, scope_a):
    store = backend.open(scope_a)
    epoch = seed(backend, scope_a)
    assert store.bindings.binding_marks(SESSION).policy is False
    assert store.bindings.binding_marks(SessionId("never-created")).policy is False
    sid = _bound_session(backend, scope_a, epoch)
    assert store.bindings.validate_lineage(store.bindings.receipt(sid)) == sid


def test_physical_target_refuses_a_receipt_whose_id_is_not_the_local_identity(backend, scope_a):
    store = backend.open(scope_a)
    require_workers(backend)
    store.bindings.reserve(epoch=begin_epoch(backend, scope_a), binding=make_binding("s-not-derived"), owner=None)
    with refused("storage_unavailable"):
        store.bindings.validate_lineage(store.bindings.receipt(SessionId("s-not-derived")))


def test_physical_target_refuses_a_lineage_tail_that_was_compressed_past(backend, scope_a):
    """The receipt still names the root after the root was compressed into a continuation: the recorded tail is stale."""
    store = backend.open(scope_a)
    require_workers(backend)
    sid = _bound_session(backend, scope_a, begin_epoch(backend, scope_a))
    store.sessions.end(sid, END_REASON_COMPRESSION)
    store.sessions.create(NewSession(session_id=SessionId("sess-cont"), source="contract", parent_session_id=sid))
    with refused("storage_unavailable"):
        store.bindings.validate_lineage(store.bindings.receipt(sid))


def test_reserve_resolves_the_physical_target_inside_the_transition(backend, scope_a):
    store = backend.open(scope_a)
    require_workers(backend)
    epoch = begin_epoch(backend, scope_a)
    sid = _bound_session(backend, scope_a, epoch)
    admission = _started(backend, scope_a, epoch, session_id=sid)
    assert _reserve(backend, scope_a, epoch, admission.admission_id).session_id == sid
    # On a stale lineage the physical-target refusal outranks the execution-id conflict.
    store.sessions.end(sid, END_REASON_COMPRESSION)
    store.sessions.create(NewSession(session_id=SessionId("sess-cont"), source="contract", parent_session_id=sid))
    with refused("storage_unavailable"):
        _reserve(backend, scope_a, epoch, admission.admission_id)


def test_lose_reservation_retires_the_worker_and_raises_the_unknown_barrier(backend, scope_a):
    store = backend.open(scope_a)
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    scope = _reserve(backend, scope_a, epoch, admission.admission_id)
    revision = store.sessions.get(SESSION).runtime_revision
    store.worker_executions.lose_reservation(epoch=epoch, admission_id=admission.admission_id, execution_id=scope.execution_id, generation=admission.generation)
    assert store.admissions.get(admission_id=admission.admission_id).status == AdmissionStatus.UNKNOWN
    assert store.sessions.get(SESSION).runtime_revision == revision + 1
    assert _verify(backend, scope_a, execution_id=scope.execution_id, generation=scope.generation, adoption_secret=_claim(scope)).status is WorkerStatus.TERMINAL
    store.admissions.admit(epoch=epoch, principal_id=PRINCIPAL, session_id=SESSION, request_id="req-next", payload={"input": "n"})
    with refused("unknown_execution"):
        store.admissions.claim(epoch=epoch, session_id=SESSION)
    # The admission is no longer started: a repeated loss is fenced, never re-applied.
    with refused("stale_generation"):
        store.worker_executions.lose_reservation(epoch=epoch, admission_id=admission.admission_id, execution_id=scope.execution_id, generation=admission.generation)


def test_lose_reservation_refusal_order_writes_nothing(backend, scope_a):
    store = backend.open(scope_a)
    epoch = seed(backend, scope_a)
    admission = _started(backend, scope_a, epoch)
    scope = _reserve(backend, scope_a, epoch, admission.admission_id)
    lose, generation = store.worker_executions.lose_reservation, admission.generation
    with refused("stale_epoch"):
        lose(epoch=AuthorityEpoch(int(epoch) + 7), admission_id=AdmissionId("absent"), execution_id=scope.execution_id, generation=generation)
    with refused("not_found"):
        lose(epoch=epoch, admission_id=AdmissionId("absent"), execution_id=scope.execution_id, generation=generation)
    for execution_id, gen in ((scope.execution_id, ExecutionGeneration(int(generation) + 1)), (WorkerExecutionId("absent"), generation)):
        with refused("stale_generation"):
            lose(epoch=epoch, admission_id=admission.admission_id, execution_id=execution_id, generation=gen)
    assert store.admissions.get(admission_id=admission.admission_id).status == AdmissionStatus.STARTED
    assert _verify(backend, scope_a, execution_id=scope.execution_id, generation=generation, adoption_secret=_claim(scope)).status is WorkerStatus.REGISTERED


# ── authorize_history ─────────────────────────────────────────────────────────


def _history(backend, scope, session_id="hist-1", **spec):
    backend.open(scope).sessions.create(NewSession(session_id=SessionId(session_id), **({"source": "cli"} | spec)))
    return SessionId(session_id)


def _authorize(backend, scope, session_id, *, profile_id="p", subject="alice", may_create=True, native_claim=True):
    return backend.open(scope).bindings.authorize_history(session_id=session_id, profile_id=profile_id, subject=subject, may_create=may_create, native_transport=native_claim, owner=None)


def test_authorize_history_is_false_for_missing_routed_and_non_history_sessions(backend, scope_a):
    require_workers(backend)
    assert _authorize(backend, scope_a, SessionId("absent"), native_claim=False) is False
    bound = _bound_session(backend, scope_a, begin_epoch(backend, scope_a))
    # A routed session is never history, even for a caller without the native proof.
    assert _authorize(backend, scope_a, bound, native_claim=False, may_create=False) is False
    assert _authorize(backend, scope_a, _history(backend, scope_a, "hist-tg", source="telegram")) is False


def test_authorize_history_order_for_an_unowned_session(backend, scope_a):
    require_workers(backend)
    sid = _history(backend, scope_a)
    owned = _history(backend, scope_a, "hist-bob", user_id="bob")
    tg = _history(backend, scope_a, "hist-tg", source="telegram")
    # The native-transport proof is checked before the source, the owner and the create capability.
    for target in (sid, owned, tg):
        with refused("permission_denied"):
            _authorize(backend, scope_a, target, native_claim=False, may_create=False)
    with refused("permission_denied"):
        _authorize(backend, scope_a, owned, may_create=False)
    with refused("permission_denied"):
        _authorize(backend, scope_a, sid, may_create=False)
    assert _authorize(backend, scope_a, sid) is True
    assert _authorize(backend, scope_a, owned, subject="bob") is True
