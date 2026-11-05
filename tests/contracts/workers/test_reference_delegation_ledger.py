"""Worker and coordinator operations share one delegation ledger."""

from datetime import UTC, datetime

import pytest

from persistence.authority.coordination import DelegationState, DeliveryState, OwnerStamp
from tests.contracts.workers._support import refused
from tests.contracts.workers.test_worker_delegations_contract import DELEGATION, LedgerWorker, dispatch_record, event, seed_delegations

STAMP = datetime(2023, 11, 14, 22, 13, 20, 250000, tzinfo=UTC)


def test_worker_delegation_is_visible_to_the_coordinator(backend, scope_a):
    coord = backend.open(scope_a).coordination
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a, session_key="route"))
    worker.dispatch(dispatch_record(session_key="route"))
    assert coord.get_delegation(delegation_id=DELEGATION).state == DelegationState.RUNNING
    assert worker.deliver("claim", "early") is False
    worker.complete(event(session_key="route"))
    record = coord.get_delegation(delegation_id=DELEGATION)
    assert (record.state, record.result) == (DelegationState.COMPLETED, {"summary": "durable"})
    assert worker.deliver("claim", "worker") is True
    assert coord.release_delivery(delegation_id=DELEGATION, claim_id="worker") is True
    assert coord.claim_delivery(delegation_id=DELEGATION, claim_id="coordinator") is True
    assert worker.deliver("ack", "coordinator") is True
    assert coord.get_delegation(delegation_id=DELEGATION).delivery_state is DeliveryState.DELIVERED


@pytest.mark.parametrize("completed", [False, True])
def test_coordinator_delegations_are_not_owned_by_a_worker(backend, scope_a, completed):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    coord = backend.open(scope_a).coordination
    coord.record_delegation_dispatch(delegation_id=DELEGATION, origin_session="route", origin_ui_session_id="", parent_session_id=None,
                                     dispatched_at=STAMP.timestamp(), owner=OwnerStamp(pid=1, started_at=None), task={})
    if completed:
        coord.record_delegation_completion(delegation_id=DELEGATION, state="completed", completed_at=STAMP.timestamp(), event={}, result={"coordinator": True})
    with refused("permission_denied"):
        worker.read()
    with refused("permission_denied"):
        worker.dispatch()
    record = coord.get_delegation(delegation_id=DELEGATION)
    assert record.origin_session == "route"
    assert record.state == (DelegationState.COMPLETED if completed else DelegationState.RUNNING)
