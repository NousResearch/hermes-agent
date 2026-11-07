"""Reference only: ledger mutations update their clock once, reads and retries do not, and a claim goes stale after ``DELEGATION_CLAIM_STALE_SECONDS``."""

import time

from persistence.authority.coordination import DELEGATION_CLAIM_STALE_SECONDS
from tests.contracts._reference import InMemoryBackend
from tests.contracts._scopes import profile_scope
from tests.contracts.workers.test_worker_delegations_contract import DELEGATION, LedgerWorker, seed_delegations


def _updated(backend, scope):
    return backend.scope_store(scope)["async_delegations"][DELEGATION]["updated_at"]


def test_delegation_mutations_preserve_the_update_clock(monkeypatch):
    backend = InMemoryBackend()
    scope = profile_scope("clock")
    try:
        worker = LedgerWorker(backend, scope, seed_delegations(backend, scope))
        monkeypatch.setattr(time, "time", lambda: 1700000001.25)
        worker.dispatch(sequence=1)
        assert _updated(backend, scope) == 1700000001.25
        monkeypatch.setattr(time, "time", lambda: 1700000002.25)
        worker.dispatch(sequence=1)
        worker.read()
        assert _updated(backend, scope) == 1700000001.25
        worker.complete()
        assert _updated(backend, scope) == 1700000002.25
        monkeypatch.setattr(time, "time", lambda: 1700000003.25)
        assert worker.deliver("claim", "owner") is True
        assert _updated(backend, scope) == 1700000003.25
        monkeypatch.setattr(time, "time", lambda: 1700000004.25)
        assert worker.deliver("release", "wrong") is False
        assert _updated(backend, scope) == 1700000003.25
        assert worker.deliver("release", "owner") is True
        assert _updated(backend, scope) == 1700000004.25
    finally:
        backend.close()


def test_a_stale_claim_no_longer_blocks_another_claimant(monkeypatch):
    backend = InMemoryBackend()
    scope = profile_scope("stale")
    try:
        worker = LedgerWorker(backend, scope, seed_delegations(backend, scope))
        worker.dispatch()
        worker.complete()
        now = 1700000000.0
        monkeypatch.setattr(time, "time", lambda: now)
        assert worker.deliver("claim", "a") is True
        now += DELEGATION_CLAIM_STALE_SECONDS
        assert worker.deliver("claim", "b") is False
        now += 1
        assert worker.deliver("claim", "b") is True
        assert worker.deliver("ack", "a") is False
        assert worker.read().delivery_attempts == 2
    finally:
        backend.close()
