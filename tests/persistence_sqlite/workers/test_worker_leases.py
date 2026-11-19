"""The worker lease verbs, which run the coordination family's lease bodies (the lease claim, the conversation lease
key, the dead-holder probe): each one transaction that a failure injected after its first statement leaves without a
trace, the reclaim of a live-looking lease whose holder process is provably gone included. The cleanup verbs also read
the target's lineage through the conversation family's read bodies."""

from __future__ import annotations

import os
from typing import Any

import pytest

from persistence.workers.executions import WorkerTurnLease
from tests.persistence_sqlite.workers._support import (
    SESSION, frozen, ident, one_transaction, rolls_back, seed_store, store_path,
)
from tests.persistence_sqlite.workers.test_worker_verbs import Case, _w

__all__ = ["frozen"]


def _dead_pid() -> int:
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _store_turn_lease(holder: str) -> Any:
    def seed(store: Any, epoch: int) -> None:
        store.coordination.try_acquire_turn_lease(session_id=SESSION, holder=holder, ttl_seconds=600)
    return seed


def _store_compression_lock(holder: str) -> Any:
    def seed(store: Any, epoch: int) -> None:
        store.coordination.try_acquire_compression_lock(session_id=SESSION, holder=holder, ttl_seconds=600)
    return seed


def _holder(pid: int, name: str) -> str:
    from process_identity import holder_namespace_token

    return f"pid={pid}{holder_namespace_token()}:{name}"


DEAD = _holder(_dead_pid(), "gone")
LIVE = _holder(os.getppid(), "live")
LEASE = {"holder": "pid=1:turn", "ttl_seconds": 30.0}

CASES = [
    Case("acquire_turn_lease", _w("acquire_turn_lease", lease=WorkerTurnLease(**LEASE))),
    Case("acquire_turn_lease_reclaims_a_dead_holder", _w("acquire_turn_lease", lease=WorkerTurnLease(**LEASE)),
         store_seed=_store_turn_lease(DEAD)),
    Case("acquire_turn_lease_loses_to_a_live_holder", _w("acquire_turn_lease", lease=WorkerTurnLease(**LEASE)),
         store_seed=_store_turn_lease(LIVE)),
    Case("renew_turn_lease", _w("renew_turn_lease", lease=WorkerTurnLease(**LEASE)), store_seed=_store_turn_lease("pid=1:turn")),
    Case("cleanup_turn_lease", _w("cleanup_turn_lease", target_session_id=SESSION, holder="pid=1:turn"),
         store_seed=_store_turn_lease("pid=1:turn")),
    Case("acquire_compression_lock", _w("acquire_compression_lock", lease=WorkerTurnLease(**LEASE))),
    Case("acquire_compression_lock_reclaims_a_dead_holder", _w("acquire_compression_lock", lease=WorkerTurnLease(**LEASE)),
         store_seed=_store_compression_lock(DEAD)),
    Case("cleanup_compression_lock", _w("cleanup_compression_lock", target_session_id=SESSION, holder="pid=1:lock"),
         store_seed=_store_compression_lock("pid=1:lock")),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_verb_is_one_transaction(backend, scope_a, frozen, case):
    path = store_path(scope_a)
    store = backend.open(scope_a)
    epoch = seed_store(store, extra=case.store_seed)
    idn = ident(epoch, case.sequence)
    rolls_back(store, path, lambda: case.call(store, idn), *case.fail_at)
    one_transaction(store, lambda: case.call(store, idn))
