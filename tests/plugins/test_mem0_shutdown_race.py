"""Regression: mem0 teardown must not close the backend under a live sync worker.

Observed 2026-08-24 in ~/.hermes-primary/logs/errors.log (6 occurrences/24h):

    ERROR mem0.memory.main: Failed to insert memory <id>: Cannot send a request,
        as the client has been closed.
    ERROR mem0.memory.main: Failed to add history for <id>: 'NoneType' object has
        no attribute 'add_history'
    WARNING plugins.memory.mem0: Mem0 sync failed: 'NoneType' object has no
        attribute 'save_messages'

Cause: `sync_turn()` starts a daemon thread that captures `backend = self._backend`
and calls `backend.add(...)`. `_shutdown_backend()` (reached via the atexit hook
registered in __init__, and via any teardown path that does not go through
`shutdown()`) closed that backend WITHOUT joining the worker, so the httpx client
and mem0 history DB died mid-write and the turn's memory was lost silently.

Fix: `_shutdown_backend()` joins in-flight workers first.

Run:
    .venv/bin/python -m pytest tests/plugins/test_mem0_shutdown_race.py -q
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_provider_class():
    from plugins.memory.mem0 import Mem0MemoryProvider

    return Mem0MemoryProvider


class _SlowBackend:
    """Backend whose add() takes long enough for a racing close() to land."""

    def __init__(self, work_s=0.35):
        self.closed = False
        self.completed = []
        self.torn_down_mid_write = False
        self._work_s = work_s

    def add(self, messages, **kwargs):
        deadline = time.time() + self._work_s
        while time.time() < deadline:
            if self.closed:
                # This is the production bug: the client vanished mid-write.
                self.torn_down_mid_write = True
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            time.sleep(0.01)
        self.completed.append(messages)
        return {"results": []}

    def close(self):
        self.closed = True


def _make_provider(backend):
    cls = _load_provider_class()
    p = object.__new__(cls)
    p._config = {}
    p._backend = backend
    p._mode = "oss"
    p._user_id = "sam"
    p._agent_id = "hermes"
    p._channel = "test"
    p._rerank_default = False
    p._sync_thread = None
    p._prefetch_thread = None
    p._prefetch_query = ""
    p._prefetch_result = ""
    p._prefetch_done = False
    p._sync_lock = threading.Lock()
    p._atexit_registered = True
    # Circuit breaker: force closed so sync_turn actually dispatches.
    for attr, val in (
        ("_failure_count", 0),
        ("_breaker_open_until", 0.0),
        ("_consecutive_failures", 0),
    ):
        setattr(p, attr, val)
    return p


def test_shutdown_backend_joins_inflight_sync_worker():
    """_shutdown_backend() must not close the backend under a running worker."""
    backend = _SlowBackend()
    p = _make_provider(backend)

    started = threading.Event()

    def _work():
        started.set()
        try:
            backend.add(
                [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
                user_id="sam",
                agent_id="hermes",
                infer=True,
            )
        except RuntimeError:
            pass

    t = threading.Thread(target=_work, daemon=True, name="mem0-sync")
    p._sync_thread = t
    t.start()
    assert started.wait(2.0), "worker never started"

    # Teardown while the write is in flight — the exact production race.
    p._shutdown_backend()

    t.join(timeout=5.0)
    assert not t.is_alive(), "worker still running after teardown"
    assert not backend.torn_down_mid_write, (
        "backend was closed mid-write: the sync worker lost the turn's memory "
        "(this is the bug _shutdown_backend()'s join fixes)"
    )
    assert backend.completed, "the in-flight memory write did not land"
    assert backend.closed, "backend should still end up closed"


def test_shutdown_backend_is_reentrant_from_worker_thread():
    """A join must never deadlock when called from the worker itself."""
    backend = _SlowBackend(work_s=0.01)
    p = _make_provider(backend)
    done = threading.Event()

    def _work():
        p._sync_thread = threading.current_thread()
        p._shutdown_backend()  # self-join would hang without the guard
        done.set()

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    assert done.wait(5.0), "_shutdown_backend deadlocked on its own thread"


def test_shutdown_leaves_backend_open_when_worker_outlives_join():
    """The 2026-08-24 cron case: worker still running after the join expires.

    Cron's `_run_cron_cleanup_with_timeout` ABANDONS teardown after its own
    bound and lets the thread keep running, so an orphaned close can fire while
    the write is mid-flight (observed 23s after teardown was abandoned, killing
    12 writes). Closing a backend a worker still holds is strictly worse than
    leaking it: teardown must decline to close.
    """
    backend = _SlowBackend(work_s=3.0)
    p = _make_provider(backend)
    mem0mod = sys.modules[type(p).__module__]

    started = threading.Event()

    def _work():
        started.set()
        try:
            backend.add([{"role": "user", "content": "hi"}], user_id="sam", agent_id="hermes")
        except RuntimeError:
            pass

    t = threading.Thread(target=_work, daemon=True, name="mem0-sync")
    p._sync_thread = t
    t.start()
    assert started.wait(2.0)

    # Force the join to expire well before the worker finishes.
    orig = getattr(mem0mod, "_SYNC_JOIN_TIMEOUT_S")
    setattr(mem0mod, "_SYNC_JOIN_TIMEOUT_S", 0.2)
    try:
        p._shutdown_backend()
    finally:
        setattr(mem0mod, "_SYNC_JOIN_TIMEOUT_S", orig)

    assert not backend.closed, (
        "teardown closed the backend while the sync worker was still writing; "
        "it must leave it open instead of corrupting the write"
    )
    t.join(timeout=10.0)
    assert not backend.torn_down_mid_write
    assert backend.completed, "the write should have been allowed to finish"


def test_shutdown_still_closes_when_no_worker():
    backend = _SlowBackend()
    p = _make_provider(backend)
    p.shutdown()
    assert backend.closed
    assert p._backend is None
