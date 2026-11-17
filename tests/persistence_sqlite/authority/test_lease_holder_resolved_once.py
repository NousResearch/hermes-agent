"""A lease verb resolves its store holder once, outside the write: a profile sweep during the write cannot deadlock it.

The lease writes ran ``self._store.holder()`` inside their write transaction. After ``close_all_under`` cleared the
holder's registry ownership, that call reacquired from the registry, which waits on the sweep's teardown, which waits on
the write lock the transaction holds. Main's lease writes resolve nothing inside the transaction (they work on the
connection they were given).
"""

from __future__ import annotations

import threading
import time

import pytest

from persistence import SessionId
from persistence.conversation.sessions import NewSession
from persistence.sqlite import _registry

DEADLINE = 20.0
VERBS = {
    "try_acquire_turn_lease": {"ttl_seconds": 300.0},
    "refresh_turn_lease": {"ttl_seconds": 300.0},
    "release_turn_lease": {},
}


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_a_profile_sweep_during_a_lease_write_never_deadlocks_it(backend, scope_a, monkeypatch, verb):
    store = backend.open(scope_a)
    sid = SessionId("lease-sweep")
    store.sessions.create(NewSession(session_id=sid, source="cli"))
    assert store.coordination.try_acquire_turn_lease(session_id=sid, holder="h", ttl_seconds=300.0)
    holder = store.holder()
    home = backend._stores[scope_a.hermes_home_key].parent
    swept = threading.Event()
    real_write = holder._execute_write

    def write_with_a_sweep_in_flight(fn, *args, **kwargs):
        def body(conn):
            # The write transaction is open and its lock held: start a profile sweep and let it clear the holder's
            # registry ownership (it then blocks closing the holder behind this write) before the verb's body runs.
            threading.Thread(target=lambda: (_registry.close_all_under(home), swept.set()), daemon=True).start()
            until = time.monotonic() + 5.0
            while holder._shared_registry_owned and time.monotonic() < until:
                time.sleep(0.01)
            return fn(conn)
        monkeypatch.setattr(holder, "_execute_write", real_write)
        return real_write(body, *args, **kwargs)

    monkeypatch.setattr(holder, "_execute_write", write_with_a_sweep_in_flight)
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = getattr(store.coordination, verb)(session_id=sid, holder="h", **VERBS[verb])
        except BaseException as exc:  # a typed refusal is a valid answer; only a hang fails
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(DEADLINE)
    hung = worker.is_alive()
    if hung:
        # Free the deadlock so the process can exit: the sweep waits on the write lock the hung verb holds.
        holder._shared_registry_owned = True
    # CONTRACT: the verb returns (or raises its typed error) and the sweep completes; neither waits on the other.
    assert not hung, f"{verb} deadlocked with a profile sweep"
    assert swept.wait(DEADLINE), "the profile sweep never completed"
