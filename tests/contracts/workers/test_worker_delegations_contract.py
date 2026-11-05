"""Contract suite for the worker delegation ledger (``persistence.workers.delegations.WorkerDelegationLedger``).

Each receipted verb is one operation of the worker receipt transaction with its handler on the ledger :class:`persistence.authority.coordination.CoordinationRepository` serves, so every case reads the row back through ``coordination.get_delegation`` as well as through the worker's own ``read_delegation``. The process probe is the caller's: ``dispatch_delegation`` takes its outcome as ``owner_started_at`` and ``probe_refused``.
"""

from __future__ import annotations

import pytest

from persistence.authority.coordination import DELEGATION_MAX_DELIVERY_ATTEMPTS, DELEGATION_ROUTE_KEYS, DelegationState, DeliveryState
from persistence.conversation.sessions import NewSession
from persistence.workers.delegations import WorkerDelegationChild, WorkerDelegationClaim, WorkerDelegationCompletion, WorkerDelegationDispatch, WorkerDelegationView
from persistence.workers.executions import WorkerExecutionId
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts.conftest import require_failpoints
from tests.contracts.workers._support import EXECUTION, SESSION, begin_epoch, ident, refused, register, require_workers

PID = 4242
BIRTH = 1_700_000_000
DELEGATION = "deleg-1"


def seed_delegations(backend, scope, *, source: str = "cli", session_key=None):
    require_workers(backend)
    backend.open(scope).sessions.create(NewSession(session_id=SESSION, source=source, session_key=session_key))
    epoch = begin_epoch(backend, scope)
    register(backend, scope, epoch)
    return epoch


def dispatch_record(delegation_id: str = DELEGATION, **overrides) -> WorkerDelegationDispatch:
    fields = {"delegation_id": delegation_id, "task": {"goal": "inert"}, "dispatched_at": 1700000000.5, "parent_session_id": str(SESSION),
              "session_key": "", "origin_ui_session_id": "", "origin_session_id": ""}
    return WorkerDelegationDispatch(**(fields | overrides))


def event(delegation_id: str = DELEGATION, **overrides) -> dict:
    return {"type": "async_delegation", "delegation_id": delegation_id, "parent_session_id": str(SESSION), "session_key": "", "origin_ui_session_id": "",
            "origin_session_id": "", "status": "completed", "completed_at": 1700000100.25, "summary": "durable"} | overrides


class LedgerWorker:
    """One registered worker's strict sequence over the ledger verbs: each call takes the next free sequence unless one is given, and only a call that returns advances it; a ledger verb carries the worker's verified ``PID`` and ``BIRTH`` unless the call names its own."""

    def __init__(self, backend, scope, epoch, execution_id=EXECUTION):
        self.store, self.epoch, self.execution_id, self.sequence = backend.open(scope), epoch, execution_id, 0

    def call(self, verb, sequence=None, **kwargs):
        sequence = self.sequence + 1 if sequence is None else sequence
        if "delegation" in verb:
            kwargs = {"worker_pid": PID, "worker_birth": BIRTH} | kwargs
        result = getattr(self.store.worker_executions, verb)(**ident(self.epoch, sequence=sequence, execution_id=self.execution_id), **kwargs)
        self.sequence = max(self.sequence, sequence)
        return result

    def dispatch(self, record=None, sequence=None, pid=PID, started=BIRTH, probe=None):
        return self.call("dispatch_delegation", sequence, dispatch=record or dispatch_record(), worker_pid=pid, owner_started_at=started, probe_refused=probe)

    def complete(self, notification=None, result=None, sequence=None):
        return self.call("complete_delegation", sequence, completion=WorkerDelegationCompletion(notification or event(), {"summary": "durable"} if result is None else result))

    def read(self, delegation_id=DELEGATION, sequence=None):
        return self.call("read_delegation", sequence, delegation_id=delegation_id)

    def deliver(self, action, token, delegation_id=DELEGATION, sequence=None):
        return self.call(f"{action}_delegation_delivery", sequence, claim=WorkerDelegationClaim(delegation_id, token))

    def pending_event(self, delegation_id=DELEGATION, session_id=SESSION, execution_id=None):
        return self.store.worker_executions.pending_delegation_event(execution_id=execution_id or self.execution_id, session_id=session_id, delegation_id=delegation_id)

    def row(self, delegation_id=DELEGATION):
        return self.store.coordination.get_delegation(delegation_id=delegation_id)

    def owner(self, delegation_id=DELEGATION):
        """The stored owner stamp of a running delegation, through the coordinator's in-flight read."""
        row = next(r for r in self.store.coordination.list_inflight_delegations() if r.delegation_id == delegation_id)
        return row.owner_pid, row.owner_started_at


# ── lifecycle ─────────────────────────────────────────────────────────────────


def test_dispatch_complete_claim_ack_lifecycle(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    assert worker.dispatch() is None
    assert worker.read() == WorkerDelegationView(delegation_id=DELEGATION, origin_session="", state="running", dispatched_at=1700000000.5, completed_at=None,
                                                 delivery_state="pending", delivery_attempts=0, origin_session_id="", result=None)
    assert worker.row().state == DelegationState.RUNNING
    assert worker.deliver("claim", "early") is False  # still running
    assert worker.pending_event() is None  # no event before completion
    worker.complete()
    assert worker.pending_event() == event()
    view = worker.read()
    assert (view.state, view.completed_at, view.result) == ("completed", 1700000100.25, {"summary": "durable"})
    assert (worker.row().state, worker.row().result) == (DelegationState.COMPLETED, {"summary": "durable"})
    assert worker.deliver("claim", "first") is True
    assert worker.deliver("ack", "foreign") is False
    assert worker.deliver("ack", "first") is True
    assert worker.pending_event() is None
    view = worker.read()
    assert (view.delivery_state, view.delivery_attempts, worker.row().delivery_state) == ("delivered", 1, DeliveryState.DELIVERED)
    assert worker.deliver("claim", "late") is False


def test_complete_keeps_the_first_result(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch()
    worker.complete()
    with refused("admission_conflict"):
        worker.complete(event(summary="overwrite"), {"summary": "overwrite"})
    assert worker.read().result == {"summary": "durable"}


# ── receipts ──────────────────────────────────────────────────────────────────


def test_exact_duplicates_replay_and_a_changed_record_conflicts(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch(sequence=1)
    worker.dispatch(sequence=1)  # a network retry replays; the existing row raises no conflict
    with refused("admission_conflict"):
        worker.dispatch(dispatch_record(dispatched_at=1.0), sequence=1)
    first = worker.read(sequence=2)
    worker.complete(sequence=3)
    assert worker.read(sequence=2) == first  # the replay answers what sequence 2 saw
    assert worker.read(sequence=4).state == "completed"


def test_fences_and_receipts_answer_before_the_record_is_judged(backend, scope_a):
    epoch = seed_delegations(backend, scope_a)
    worker = LedgerWorker(backend, scope_a, epoch)
    worker.dispatch(sequence=1)
    bad = WorkerDelegationDispatch(None, None, None, None, None, None, None)
    with refused("stale_epoch"):
        LedgerWorker(backend, scope_a, epoch + 1).dispatch(bad, sequence=2)
    with refused("admission_conflict"):
        worker.dispatch(bad, sequence=1)
    with refused("invalid_params"):
        worker.dispatch(bad, sequence=3)  # not the next sequence
    with refused("invalid_params"):
        worker.dispatch(bad, sequence=2)  # the record itself, after the fences
    worker.read(sequence=2)  # the refused record left sequence 2 free


def test_boundary_arguments_refuse_before_the_store(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    for pid in (0, True, -1):
        with refused("invalid_params"):
            worker.dispatch(sequence=1, pid=pid)
    with refused("invalid_params"):
        worker.dispatch(dispatch_record(task={"goal": {1, 2}}), sequence=1)  # not JSON-encodable
    worker.dispatch(sequence=1)
    # Ownership comes first, then the token.
    with refused("not_found"):
        worker.deliver("claim", "", "deleg-missing", sequence=2)
    with refused("invalid_params"):
        worker.deliver("claim", "", sequence=2)
    with refused("invalid_params"):
        worker.deliver("claim", "t", "x" * 513, sequence=2)
    assert worker.deliver("claim", "t", sequence=2) is False


# ── dispatch ──────────────────────────────────────────────────────────────────


def test_dispatch_requires_a_local_origin(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a, source="telegram"))
    with refused("unsupported_producer"):
        worker.dispatch(sequence=1)
    assert worker.row() is None


def test_dispatch_refuses_foreign_routes_and_existing_ids(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a, session_key="key-1"))
    for bad in ({"session_key": ""}, {"parent_session_id": "other"}, {"origin_ui_session_id": "other"}, {"origin_session_id": "other"}):
        with refused("permission_denied"):
            worker.dispatch(dispatch_record(**{"session_key": "key-1"} | bad), sequence=1)
    for bad in ({"delegation_id": ""}, {"delegation_id": "x" * 513}, {"dispatched_at": -1}, {"dispatched_at": True}, {"task": ["goal"]}, {"task": {"unknown": 1}}):
        with refused("invalid_params"):
            worker.dispatch(dispatch_record(**{"session_key": "key-1"} | bad), sequence=1)
    worker.dispatch(dispatch_record(session_key="key-1", origin_session_id=str(SESSION)), sequence=1)
    with refused("admission_conflict"):
        worker.dispatch(dispatch_record(session_key="key-1", dispatched_at=2.0), sequence=2)
    view = worker.read(sequence=2)
    assert (view.origin_session, view.origin_session_id) == ("key-1", str(SESSION))


def test_dispatch_stores_the_callers_process_identity(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch(dispatch_record("deleg-a"), pid=PID, started=BIRTH)
    worker.dispatch(dispatch_record("deleg-b"), pid=PID, started=None)
    assert (worker.owner("deleg-a"), worker.owner("deleg-b")) == ((PID, BIRTH), (PID, None))


@pytest.mark.parametrize("probe", ["permission_denied", "worker_not_live"])
def test_a_refused_probe_raises_after_the_duplicate_check_and_before_the_insert(backend, scope_a, probe):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    with refused(probe):
        worker.dispatch(sequence=1, probe=probe)
    assert worker.row() is None
    worker.dispatch(sequence=1)
    # An existing id answers as an existing id whatever the probe found.
    with refused("admission_conflict"):
        worker.dispatch(sequence=2, probe=probe)
    # An exact replay ignores both the probe outcome and the identity.
    assert worker.dispatch(sequence=1, probe=probe, started=None, pid=1) is None
    assert worker.owner() == (PID, BIRTH)


def test_delegations_belong_to_their_execution(backend, scope_a):
    epoch = seed_delegations(backend, scope_a)
    worker = LedgerWorker(backend, scope_a, epoch)
    worker.dispatch()
    worker.call("finish")
    other = WorkerExecutionId("exec-2")
    register(backend, scope_a, epoch, execution_id=other)
    successor = LedgerWorker(backend, scope_a, epoch, execution_id=other)
    with refused("permission_denied"):
        successor.read()
    with refused("permission_denied"):
        successor.dispatch(sequence=1)  # the id exists and is not this execution's
    with refused("not_found"):
        successor.read("deleg-missing", sequence=1)
    with refused("permission_denied"):
        successor.pending_event()
    with refused("invalid_params"):
        successor.pending_event(delegation_id="")


# ── completion, children, withdrawal ──────────────────────────────────────────


def test_complete_validates_then_checks_route_and_state(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch(sequence=1)
    for completion in (WorkerDelegationCompletion(event(status="done"), {}), WorkerDelegationCompletion(event(), ["x"]),
                       WorkerDelegationCompletion("event", {}), WorkerDelegationCompletion(event(type="other"), {}),
                       WorkerDelegationCompletion(event(unknown=1), {})):
        with refused("invalid_params"):
            worker.call("complete_delegation", 2, completion=completion)
    with refused("not_found"):
        worker.complete(event("deleg-missing"), sequence=2)
    with refused("invalid_params"):
        worker.complete(event(delegation_id=""), sequence=2)
    for key in DELEGATION_ROUTE_KEYS:
        with refused("permission_denied"):
            worker.complete(event(**{key: "foreign"}), sequence=2)
    with refused("permission_denied"):
        worker.complete(event(completed_at="soon", origin_session_id="x"), sequence=2)  # the route before the timestamp
    with refused("invalid_params"):
        worker.complete(event(completed_at="soon"), sequence=2)
    notification = event() | {"results": ("a", "b")}
    worker.complete(notification, result={"items": ("a", "b")}, sequence=2)
    # Documents are stored as their JSON form.
    assert worker.pending_event() == event() | {"results": ["a", "b"]}
    assert worker.read(sequence=3).result == {"items": ["a", "b"]}


def test_children_keep_one_entry_per_index_until_completion(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch()

    def child(entry):
        worker.call("record_delegation_child", child=WorkerDelegationChild(DELEGATION, entry))

    child({"task_index": 1, "summary": "a"})
    child({"task_index": 0, "summary": "b"})
    child({"task_index": 1, "summary": "c"})
    assert worker.read().result == {"results": [{"task_index": 0, "summary": "b"}, {"task_index": 1, "summary": "c"}], "partial": True}
    for entry in ({"task_index": -1}, {"task_index": True}, {}, ["task_index"]):
        with refused("invalid_params"):
            child(entry)
    with refused("admission_conflict"):
        worker.call("withdraw_delegation", delegation_id=DELEGATION)  # it holds a result
    worker.complete()
    child({"task_index": 2, "summary": "late"})  # a no-op on a finished delegation; the receipt still commits
    assert worker.read().result == {"summary": "durable"}


def test_withdraw_deletes_only_an_untouched_running_delegation(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch()
    worker.call("withdraw_delegation", delegation_id=DELEGATION)
    assert worker.row() is None
    with refused("not_found"):
        worker.read()
    worker.dispatch(dispatch_record("deleg-2"))
    worker.complete(event("deleg-2"))
    with refused("admission_conflict"):
        worker.call("withdraw_delegation", delegation_id="deleg-2")


# ── the delivery claim ────────────────────────────────────────────────────────


def _completed(backend, scope):
    worker = LedgerWorker(backend, scope, seed_delegations(backend, scope))
    worker.dispatch()
    worker.complete()
    return worker


def test_a_live_claim_blocks_another_claimant(backend, scope_a):
    worker = _completed(backend, scope_a)
    assert worker.deliver("claim", "a") is True
    assert worker.deliver("claim", "b") is False
    assert worker.deliver("ack", "b") is False
    assert worker.read().delivery_attempts == 1


def test_defer_refunds_and_release_drops_at_the_cap(backend, scope_a):
    worker = _completed(backend, scope_a)
    for index in range(3):
        assert worker.deliver("claim", f"c{index}") is True
        assert worker.deliver("defer", f"c{index}") is True
    assert worker.read().delivery_attempts == 0
    for index in range(DELEGATION_MAX_DELIVERY_ATTEMPTS - 1):
        assert worker.deliver("claim", f"r{index}") is True
        assert worker.deliver("release", f"r{index}") is True
        assert worker.read().delivery_state == "pending"
    assert worker.deliver("claim", "last") is True
    assert worker.deliver("release", "wrong") is False
    assert worker.deliver("release", "last") is True
    view = worker.read()
    assert (view.delivery_state, view.delivery_attempts) == ("dropped", DELEGATION_MAX_DELIVERY_ATTEMPTS)
    assert worker.deliver("claim", "after") is False


def test_drop_ends_delivery(backend, scope_a):
    worker = _completed(backend, scope_a)
    assert worker.deliver("claim", "t") is True
    assert worker.deliver("drop", "t") is True
    assert worker.pending_event() is None
    assert (worker.read().delivery_state, worker.row().delivery_state) == ("dropped", DeliveryState.DROPPED)


# ── crash window and isolation ────────────────────────────────────────────────


def test_dispatch_crash_window_publishes_nothing(backend, scope_a):
    require_failpoints(backend)
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    authority_state(backend, scope_a).failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        worker.dispatch(sequence=1)
    with refused("not_found"):
        worker.pending_event()
    worker.dispatch(sequence=1)
    assert worker.read(sequence=2).state == "running"


def test_delegations_are_scope_isolated(backend, scope_a, scope_b):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch()
    other = LedgerWorker(backend, scope_b, seed_delegations(backend, scope_b))
    with refused("not_found"):
        other.read()
    other.dispatch()
    assert other.read().state == "running"
