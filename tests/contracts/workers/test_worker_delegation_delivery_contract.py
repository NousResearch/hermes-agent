"""Contract cases for a completion's delivery state: ``delegation.complete``'s ``delivery`` and ``record_delegation_completion(delivery=)``.

``delivered`` records a completion that is never offered (every child of the unit waits on a promised automatic resume); ``pending`` is the default and the worker omits the key for it, so a record with the default maps to the two-key payload the producer sends.
"""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, DelegationId
from persistence.authority.coordination import DelegationRecovery, DeliveryState, OwnerStamp
from persistence.workers.delegations import WorkerDelegationCompletion, WorkerDelegationPayload, delegation_payload
from tests.contracts.workers.test_worker_delegations_contract import LedgerWorker, event, seed_delegations


def _complete(worker, payload, sequence=None):
    return worker.call("complete_delegation", sequence, completion=payload)


def test_a_default_completion_maps_to_the_two_key_payload():
    notification, result = event(), {"summary": "durable"}
    # CONTRACT: the default delivery is not on the wire; ``delivered`` is.
    assert delegation_payload(WorkerDelegationCompletion(notification, result)) == {"event": notification, "result": result}
    assert delegation_payload(WorkerDelegationCompletion(notification, result, delivery="delivered")) == {"event": notification, "result": result, "delivery": "delivered"}


def test_complete_delivered_records_a_completion_that_is_never_offered(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch()
    _complete(worker, WorkerDelegationCompletion(event(), {"summary": "waiting"}, delivery="delivered"))
    # CONTRACT: the row is terminal with its event and result, its delivery ``delivered``, and nothing is pending.
    view = worker.read()
    assert (view.state, view.delivery_state, view.result) == ("completed", "delivered", {"summary": "waiting"})
    assert worker.row().delivery_state is DeliveryState.DELIVERED
    assert worker.pending_event() is None
    assert worker.deliver("claim", "late") is False


@pytest.mark.parametrize(("payload", "expected"), [
    pytest.param({"delivery": "delivered"}, DeliveryState.DELIVERED, id="delivered"),
    pytest.param({"delivery": "pending"}, DeliveryState.PENDING, id="explicit-pending"),
    pytest.param({}, DeliveryState.PENDING, id="absent"),
])
def test_complete_payload_carries_the_delivery_key(backend, scope_a, payload, expected):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch()
    _complete(worker, WorkerDelegationPayload({"event": event(), "result": {"summary": "durable"}} | payload))
    # CONTRACT: the optional ``delivery`` key sets the delivery state; absent is ``pending``.
    assert worker.row().delivery_state is expected


@pytest.mark.parametrize("delivery", ["dropped", "Delivered", "", None, 1])
def test_complete_refuses_an_unknown_delivery_and_writes_nothing(backend, scope_a, delivery):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch(sequence=1)
    # CONTRACT: a delivery other than pending or delivered is invalid_params, as main's payload validator answers.
    with pytest.raises(AuthorityRefusal) as caught:
        _complete(worker, WorkerDelegationPayload({"event": event(), "result": {}, "delivery": delivery}), sequence=2)
    assert caught.value.reason == "invalid_params"
    assert worker.read(sequence=2).state == "running"


def test_complete_replays_the_producer_payload_for_a_default_record(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    worker.dispatch(sequence=1)
    _complete(worker, WorkerDelegationCompletion(event(), {"summary": "durable"}), sequence=2)
    # CONTRACT: the record and the two-key wire payload are one operation, so the retry replays instead of conflicting.
    _complete(worker, WorkerDelegationPayload({"event": event(), "result": {"summary": "durable"}}), sequence=2)
    with pytest.raises(AuthorityRefusal) as caught:
        _complete(worker, WorkerDelegationCompletion(event(), {"summary": "durable"}, delivery="delivered"), sequence=2)
    assert caught.value.reason == "admission_conflict"


# ── the coordinator's own completion write ──────────────────────────────────


def _dispatched(store, delegation_id: str) -> DelegationId:
    did = DelegationId(delegation_id)
    store.coordination.record_delegation_dispatch(delegation_id=did, origin_session="", origin_ui_session_id="", parent_session_id=None,
                                                  dispatched_at=1700000000.0, owner=OwnerStamp(pid=1, started_at=1), task={"goal": "g"})
    return did


@pytest.mark.parametrize(("kwargs", "expected"), [
    pytest.param({}, DeliveryState.PENDING, id="default"),
    pytest.param({"delivery": "pending"}, DeliveryState.PENDING, id="pending"),
    pytest.param({"delivery": "delivered"}, DeliveryState.DELIVERED, id="delivered"),
])
def test_record_delegation_completion_stores_the_delivery(backend, scope_a, kwargs, expected):
    store = backend.open(scope_a)
    did = _dispatched(store, "d-own")
    store.coordination.record_delegation_completion(delegation_id=did, state="completed", completed_at=1700000100.0,
                                                    event={"delegation_id": did}, result={"summary": "s"}, **kwargs)
    # CONTRACT: the completion writes the delivery it is given, ``pending`` by default.
    row = store.coordination.get_delegation(delegation_id=did)
    assert (row.state, row.delivery_state) == ("completed", expected)
    assert (store.coordination.has_replayable_completions()) is (expected is DeliveryState.PENDING)


@pytest.mark.parametrize("delivery", ["dropped", "bogus"])
def test_record_delegation_completion_refuses_an_unknown_delivery(backend, scope_a, delivery):
    store = backend.open(scope_a)
    did = _dispatched(store, "d-bad")
    # CONTRACT: only pending or delivered; the refusal writes nothing.
    with pytest.raises(AuthorityRefusal) as caught:
        store.coordination.record_delegation_completion(delegation_id=did, state="completed", completed_at=1700000100.0,
                                                        event={"delegation_id": did}, result={}, delivery=delivery)
    assert caught.value.reason == "invalid_params"
    assert store.coordination.get_delegation(delegation_id=did).state == "running"


def test_recovery_stores_the_delivery_it_carries(backend, scope_a):
    store = backend.open(scope_a)
    silent, loud = _dispatched(store, "d-silent"), _dispatched(store, "d-loud")
    inflight = {row.delegation_id: row for row in store.coordination.list_inflight_delegations()}
    recoveries = [DelegationRecovery(delegation_id=did, expected_state="running", expected_updated_at=inflight[did].updated_at, completed_at=1700000200.0,
                                     event={"delegation_id": did}, result={"status": "unknown"}, **extra)
                  for did, extra in ((silent, {"delivery": "delivered"}), (loud, {}))]
    assert store.coordination.recover_abandoned_delegations(recoveries=recoveries) == 2
    # CONTRACT: a recovery whose every child waits on a promised resume is recorded delivered; the default is pending.
    assert store.coordination.get_delegation(delegation_id=silent).delivery_state is DeliveryState.DELIVERED
    assert store.coordination.get_delegation(delegation_id=loud).delivery_state is DeliveryState.PENDING


def test_delegation_recovery_refuses_an_unknown_delivery():
    # CONTRACT: the record holds only pending or delivered.
    with pytest.raises(ValueError):
        DelegationRecovery(delegation_id=DelegationId("d"), expected_state="running", expected_updated_at=0.0, completed_at=0.0, event={}, result={}, delivery="dropped")
