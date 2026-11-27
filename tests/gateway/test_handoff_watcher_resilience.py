"""Handoff watcher resilience: no head-of-line blocking, no stranded rows.

Two failure modes raised by adversarial review of the multi-profile handoff
work, both fixed here and pinned by these tests.

1. HEAD-OF-LINE BLOCKING. ``_process_handoff`` runs a full agent turn plus
   platform delivery. Awaiting it inline meant one slow handoff in profile A
   stopped the watcher from even POLLING B, C and D. Since the CLI gives up
   after 60s, a perfectly good handoff could time out purely because another
   profile's was ahead of it. Dispatch is now fire-and-forget.

2. STRANDED ``running`` ROWS. Only the watcher sets ``running``, for the span
   of one in-process dispatch. A gateway that dies mid-dispatch leaves the row
   there forever — and ``request_handoff`` refuses a NEW request unless the
   state is NULL/completed/failed, so that session can never hand off again,
   silently. Startup now reclaims those rows to ``failed``.
"""

import asyncio
import types
from pathlib import Path

import pytest

from gateway import run
from persistence.authority.coordination import PendingHandoff


def _store(coordination):
    """A storage handle stand-in exposing ``coordination`` in its awaitable form."""
    return types.SimpleNamespace(aio=types.SimpleNamespace(coordination=coordination))


def _running_flag(ticks):
    """A ``_running`` stand-in that is True for ``ticks`` reads, then False."""
    states = iter([True] * ticks + [False])

    class _Running:
        def __bool__(_self):
            try:
                return next(states)
            except StopIteration:
                return False

    return _Running()


class _SlowDB:
    """One pending row; ``_process_handoff`` for it never finishes on its own."""

    def __init__(self):
        self.polls = 0
        self.claimed = []

    async def list_pending_handoffs(self):
        self.polls += 1
        return (PendingHandoff(session_id="slow-row", platform=None, title=None, started_at=0.0),)

    async def claim_handoff(self, *, session_id):
        # Real claim is atomic pending→running: it succeeds exactly once.
        if session_id in self.claimed:
            return False
        self.claimed.append(session_id)
        return True

    async def complete_handoff(self, *, session_id):
        return None

    async def fail_handoff(self, *, session_id, error, only_states=None):
        return True


@pytest.mark.asyncio
async def test_slow_handoff_does_not_block_later_polls(monkeypatch):
    """A handoff that never returns must not stop the poll loop.

    Mutation-survivable by construction: ``_process_handoff`` blocks on an
    Event that is only set AFTER the watcher has been given room to keep
    polling. Restoring the inline ``await self._process_handoff`` deadlocks
    tick 1 — the watcher never reaches tick 2, ``release`` is never set, and
    ``wait_for`` raises TimeoutError.

    Note the sleep stub must yield control (``asyncio.sleep(0)``); a stub that
    returns immediately without yielding lets an inline dispatch monopolise
    the loop and masks the bug.
    """
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])

    # Capture the real sleep BEFORE patching: the stub runs as
    # ``run.asyncio.sleep``, which is the same module object the test imports,
    # so calling ``asyncio.sleep`` inside it would recurse into itself.
    _real_sleep = asyncio.sleep

    async def _yield_sleep(_seconds):
        await _real_sleep(0)

    monkeypatch.setattr(run.asyncio, "sleep", _yield_sleep)

    db = _SlowDB()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _process_handoff(row, profile_name=None):
        started.set()
        await release.wait()

    fake = types.SimpleNamespace()
    fake.persistence = _store(db)
    fake._running = _running_flag(3)
    fake._process_handoff = _process_handoff

    async def _watch():
        await run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)

    task = asyncio.ensure_future(_watch())
    await asyncio.wait_for(started.wait(), timeout=5)

    # Give the loop real turns to poll while the handoff is stuck.
    for _ in range(20):
        await _real_sleep(0)

    polls_while_stuck = db.polls
    release.set()
    # The watcher may already have exited its loop and be draining; either way
    # it must finish once the stuck handoff is released.
    try:
        await asyncio.wait_for(task, timeout=5)
    except TimeoutError:
        task.cancel()
        raise AssertionError("watcher did not finish after the handoff was released")

    assert polls_while_stuck >= 2, (
        "poll loop must keep polling while a handoff is in flight; "
        f"polls={polls_while_stuck}"
    )


@pytest.mark.asyncio
async def test_inflight_row_is_not_claimed_twice(monkeypatch):
    """A row already dispatched must be skipped by later ticks."""
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    db = _SlowDB()
    calls = []

    async def _process_handoff(row, profile_name=None):
        calls.append(row["id"])
        await asyncio.sleep(3600)

    fake = types.SimpleNamespace()
    fake.persistence = _store(db)
    fake._running = _running_flag(4)
    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)
    await asyncio.wait_for(coro, timeout=5)

    assert calls == ["slow-row"], f"dispatched more than once: {calls}"


class _ReclaimDB:
    """Records the reclaim call and reports nothing pending."""

    def __init__(self, stale_ids=("dead-row",)):
        self.stale_ids = list(stale_ids)
        self.reclaim_calls = []

    async def reclaim_stale_running_handoffs(self, *, error):
        self.reclaim_calls.append(error)
        return self.stale_ids

    async def list_pending_handoffs(self):
        return []


@pytest.mark.asyncio
async def test_startup_reclaims_rows_stranded_in_running(monkeypatch):
    """Rows left 'running' by a dead gateway are failed at startup.

    Without this, ``request_handoff`` keeps rejecting new requests for that
    session forever and nothing tells the user why.
    """
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    db = _ReclaimDB()
    fake = types.SimpleNamespace()
    fake.persistence = _store(db)
    fake._running = _running_flag(1)

    async def _process_handoff(row, profile_name=None):
        return None

    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)
    await asyncio.wait_for(coro, timeout=5)

    assert len(db.reclaim_calls) == 1, "reclaim must run exactly once per store"
    assert "/handoff" in db.reclaim_calls[0], (
        "the recorded error should tell the user how to retry"
    )


@pytest.mark.asyncio
async def test_reclaim_runs_per_profile_store(monkeypatch):
    """Every served profile's store gets reclaimed, not just the root's."""
    scopes = [
        (None, None),
        ("bala", Path("/h/profiles/bala")),
        ("medicina", Path("/h/profiles/medicina")),
    ]
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: scopes)

    class _Scope:
        def __init__(self, home):
            self.home = home

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(run, "_async_profile_runtime_scope", _Scope)

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    db = _ReclaimDB(stale_ids=[])
    fake = types.SimpleNamespace()
    fake.persistence = _store(db)
    fake._running = _running_flag(1)

    async def _process_handoff(row, profile_name=None):
        return None

    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)
    await asyncio.wait_for(coro, timeout=5)

    assert len(db.reclaim_calls) == 3, (
        f"expected one reclaim per scope (root + 2 profiles), got {len(db.reclaim_calls)}"
    )




def test_reclaim_stale_running_handoffs_flips_only_running_rows(tmp_path):
    """DB-level: only 'running' rows are touched, and their ids are returned."""
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    with open_persistence(profile_scope(home=tmp_path)) as store:
        handoffs = store.coordination
        for sid in ("dead", "queued", "done"):
            seed_session(store, sid, "cli")
            handoffs.request_handoff(session_id=sid, platform="telegram")  # pending
        for sid in ("dead", "done"):
            handoffs.claim_handoff(session_id=sid)  # running
        handoffs.complete_handoff(session_id="done")  # completed

        reclaimed = handoffs.reclaim_stale_running_handoffs(error="gateway died")

        assert reclaimed == ("dead",)
        assert handoffs.get_handoff(session_id="dead").state == "failed"
        assert handoffs.get_handoff(session_id="dead").error == "gateway died"
        assert handoffs.get_handoff(session_id="queued").state == "pending"
        assert handoffs.get_handoff(session_id="done").state == "completed"


def test_reclaimed_session_can_request_handoff_again(tmp_path):
    """The point of the reclaim: the session is usable again.

    ``request_handoff`` only accepts NULL/completed/failed, so a stranded
    'running' row is what permanently locks the session out.
    """
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    with open_persistence(profile_scope(home=tmp_path)) as store:
        handoffs = store.coordination
        seed_session(store, "stuck", "cli")
        handoffs.request_handoff(session_id="stuck", platform="telegram")
        handoffs.claim_handoff(session_id="stuck")  # running

        assert handoffs.request_handoff(session_id="stuck", platform="telegram") is False, (
            "precondition: a stranded 'running' row blocks new handoff requests"
        )

        handoffs.reclaim_stale_running_handoffs(error="gateway died")

        assert handoffs.request_handoff(session_id="stuck", platform="telegram") is True, (
            "after reclaim the session must be able to hand off again"
        )
