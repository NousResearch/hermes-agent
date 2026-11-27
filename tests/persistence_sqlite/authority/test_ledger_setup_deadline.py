"""A ledger's table setup waits for the store's locks no longer than the ledgers' lock wait.

The setup takes the process-wide setup lock and the holder's writer lock against one deadline, as every write takes the writer lock under its patience (``writer_mutex_until``): a writer holding the holder's lock (another thread's long write) refuses the setup with the store's contention error instead of holding it, and every ledger verb behind it, without a bound.
"""

from __future__ import annotations

import threading
import time

import pytest

from persistence import StoreContention
from persistence.sqlite.authority import coordination as coordination_module


def test_a_held_writer_lock_refuses_the_setup_within_the_ledger_wait(backend, scope_a, monkeypatch):
    store = backend.open(scope_a)
    holder = store.holder()
    monkeypatch.setattr(coordination_module, "LEDGER_PATIENCE_S", 0.3)
    coordination = store.coordination
    ran = []
    release = threading.Event()
    held = threading.Event()

    def hold():
        with holder._lock:
            held.set()
            release.wait(30)

    threading.Thread(target=hold, daemon=True).start()
    assert held.wait(5)
    outcome: dict = {}

    def setup():
        started = time.monotonic()
        try:
            coordination._ledger_setup(lambda conn: ran.append(conn))
        except BaseException as exc:
            outcome["error"] = exc
        outcome["took"] = time.monotonic() - started

    worker = threading.Thread(target=setup, daemon=True)
    worker.start()
    worker.join(10)
    hung = worker.is_alive()
    release.set()
    worker.join(10)
    # CONTRACT: the setup gives up on a held writer lock with the store's contention error, inside a small multiple of the wait.
    assert not hung, "_ledger_setup waited on the holder's lock without a bound"
    assert isinstance(outcome.get("error"), StoreContention), outcome
    assert outcome["took"] < 5.0 and ran == []


def test_the_setup_runs_once_the_lock_is_free(backend, scope_a):
    store = backend.open(scope_a)
    ran = []
    step = lambda conn: ran.append(conn)
    store.coordination._ledger_setup(step)
    store.coordination._ledger_setup(step)
    assert len(ran) == 1


@pytest.fixture(autouse=True)
def _fresh_setup_registry(monkeypatch):
    monkeypatch.setattr(coordination_module, "_setup_done", type(coordination_module._setup_done)())
