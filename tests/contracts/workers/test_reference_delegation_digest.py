"""Delegation retries compare the original request, not decoded JSON text."""

from tests.contracts.workers._support import refused
from tests.contracts.workers.test_worker_delegations_contract import LedgerWorker, dispatch_record, seed_delegations


def test_surrogate_pair_and_astral_scalar_are_distinct_requests(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    dispatch = dispatch_record(task={"context": "\ud83d\ude42"})
    worker.dispatch(dispatch, sequence=1)
    worker.dispatch(dispatch, sequence=1)
    with refused("admission_conflict"):
        worker.dispatch(dispatch_record(task={"context": "\U0001f642"}), sequence=1)
    assert worker.read().state == "running"
