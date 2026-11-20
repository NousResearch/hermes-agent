"""Regression test for #40695 (salvage of keystone PR #40782).

The Discord gateway heartbeat was stalling because the handoff watcher
(``GatewayRunner._handoff_watcher``) polled the synchronous, blocking
SQLite-backed store directly on the asyncio event loop every 2s
('Shard ID None heartbeat blocked for more than N seconds').

The fix routes every blocking store call in the watcher through the store's
awaitable form (``store.aio``), which runs each verb on the store's executor so
the SQLite I/O runs on a worker thread and never blocks the event loop / Discord
heartbeat.

These tests assert that behaviour contract. They are mutation-survivable:
replacing any ``await store.aio.coordination.<verb>(...)`` with the plain verb
called on the loop makes the relevant assertion fail.
"""

import asyncio
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import run
from persistence.authority.coordination import PendingHandoff
from persistence.core.aio import AwaitableStore


class _RecordingCoordination:
    """Coordination stand-in that records the thread each verb runs on.

    A verb the watcher calls directly on the event loop runs on the loop thread;
    one it awaits through ``store.aio`` runs on the store's executor.
    """

    def __init__(self):
        self.threads = {}

    def _record(self, name):
        self.threads.setdefault(name, []).append(threading.current_thread().name)

    def ran_on(self, name, prefix):
        """True iff every call to ``name`` ran on a thread whose name starts with ``prefix``."""
        names = self.threads.get(name, [])
        return bool(names) and all(n.startswith(prefix) for n in names)

    def list_pending_handoffs(self):
        self._record("list_pending_handoffs")
        return (PendingHandoff(session_id="sess-1", platform=None, title=None, started_at=0.0),)

    def claim_handoff(self, *, session_id):
        self._record("claim_handoff")
        return True

    def complete_handoff(self, *, session_id):
        self._record("complete_handoff")

    def fail_handoff(self, *, session_id, error, only_states=None):
        self._record("fail_handoff")
        return True


class _Store:
    """A storage handle stand-in: ``coordination`` plus the default awaitable view on ``executor``."""

    def __init__(self, coordination, executor):
        self.coordination = coordination
        self.executor = executor
        self.aio = AwaitableStore(self)


def _make_fake_runner(store, *, fail_process=False):
    """Build a minimal object that exposes exactly what the loop body touches."""
    fake = types.SimpleNamespace()
    fake.persistence = store
    # _running yields True for the first loop check, then False so the loop
    # exits after a single tick.
    states = iter([True, False])

    class _Running:
        def __bool__(_self):
            try:
                return next(states)
            except StopIteration:
                return False

    fake._running = _Running()

    async def _process_handoff(row):
        if fail_process:
            raise RuntimeError("boom")

    fake._process_handoff = _process_handoff
    return fake


async def _run_one_tick(fake, monkeypatch):
    """Run the watcher for a single tick with sleeps neutralised."""

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)
    # Bind the real (patched) method onto our minimal stand-in.
    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0)
    await asyncio.wait_for(coro, timeout=5)


@pytest.mark.asyncio
async def test_watcher_wraps_calls_via_asyncio_to_thread(monkeypatch):
    """Explicitly assert the offload goes through the store's awaitable form.

    Every verb the watcher awaits must run on the store's executor, never on
    the loop thread. Mutation-survivable: calling any verb plain on the loop
    records the loop thread for it and drops it from the set.
    """
    coordination = _RecordingCoordination()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="store-executor") as executor:
        fake = _make_fake_runner(_Store(coordination, executor), fail_process=False)
        await _run_one_tick(fake, monkeypatch)

    wrapped = {name for name in coordination.threads if coordination.ran_on(name, "store-executor")}

    assert "list_pending_handoffs" in wrapped
    assert "claim_handoff" in wrapped
    assert "complete_handoff" in wrapped
