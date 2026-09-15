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
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import run


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
        return [{"id": "slow-row"}]

    async def claim_handoff(self, sid):
        # Real claim is atomic pending→running: it succeeds exactly once.
        if sid in self.claimed:
            return False
        self.claimed.append(sid)
        return True

    async def complete_handoff(self, sid):
        return None

    async def fail_handoff(self, sid, err):
        return None


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
    fake._session_db = db
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
    except asyncio.TimeoutError:
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
    fake._session_db = db
    fake._running = _running_flag(4)
    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)
    await asyncio.wait_for(coro, timeout=5)

    assert calls == ["slow-row"], f"dispatched more than once: {calls}"


@pytest.mark.asyncio
async def test_programmatic_handoff_waits_for_active_turn(monkeypatch):
    """A handoff queued by the current turn must be claimed only after that turn is idle."""
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])
    real_sleep = asyncio.sleep

    async def _yield_sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(run.asyncio, "sleep", _yield_sleep)
    db = _SlowDB()
    calls = []
    busy = iter([True, False, False])

    async def _process_handoff(row, profile_name=None):
        calls.append(row["id"])

    fake = types.SimpleNamespace()
    fake._session_db = db
    fake._running = _running_flag(3)
    fake._process_handoff = _process_handoff
    fake.session_store = types.SimpleNamespace(
        lookup_by_session_id=lambda _sid: types.SimpleNamespace(session_key="active-key")
    )
    fake._is_session_running = lambda _key: next(busy, False)

    await asyncio.wait_for(
        run.GatewayRunner._handoff_watcher(
            fake, interval=0.0, drain_timeout=1.0  # type: ignore[arg-type]
        ),
        timeout=5,
    )

    assert db.polls >= 2
    assert db.claimed == ["slow-row"]
    assert calls == ["slow-row"]


@pytest.mark.asyncio
async def test_handoff_reserves_source_before_blocking_db_claim(monkeypatch):
    """A successor turn cannot enter while the watcher awaits its durable claim."""
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])
    real_sleep = asyncio.sleep

    async def _yield_sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(run.asyncio, "sleep", _yield_sleep)

    class GateDB(_SlowDB):
        def __init__(self):
            super().__init__()
            self.claim_started = asyncio.Event()
            self.allow_claim = asyncio.Event()

        async def claim_handoff(self, sid):
            self.claim_started.set()
            await self.allow_claim.wait()
            return await super().claim_handoff(sid)

    db = GateDB()
    reserved = False
    released = []
    processed = []

    def reserve(_sid):
        nonlocal reserved
        if reserved:
            return ("source-key", 0)
        reserved = True
        return ("source-key", 7)

    async def process(row, profile_name=None):
        processed.append(row["id"])

    fake = types.SimpleNamespace(
        _session_db=db,
        _running=_running_flag(1),
        _process_handoff=process,
        _reserve_handoff_source=reserve,
        _release_handoff_source_reservation=lambda value: released.append(value),
    )

    watcher = asyncio.create_task(
        run.GatewayRunner._handoff_watcher(
            fake, interval=0.0, drain_timeout=1.0  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(db.claim_started.wait(), timeout=5)
    assert reserved is True
    assert reserve("slow-row") == ("source-key", 0)
    assert processed == []
    db.allow_claim.set()
    await asyncio.wait_for(watcher, timeout=5)

    assert processed == ["slow-row"]
    assert ("source-key", 7) in released


@pytest.mark.asyncio
async def test_claim_error_releases_source_reservation(monkeypatch):
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])
    monkeypatch.setattr(run.asyncio, "sleep", AsyncMock())

    class RaisingDB(_ReclaimDB):
        async def list_pending_handoffs(self):
            return [{"id": "row"}]

        async def claim_handoff(self, _sid):
            raise RuntimeError("locked")

    released = []
    fake = types.SimpleNamespace(
        _session_db=RaisingDB(stale_ids=[]),
        _running=_running_flag(1),
        _process_handoff=AsyncMock(),
        _reserve_handoff_source=lambda _sid: ("source-key", 8),
        _release_handoff_source_reservation=lambda value: released.append(value),
    )

    await run.GatewayRunner._handoff_watcher(
        cast(Any, fake), interval=0.0, drain_timeout=0.01
    )

    assert released == [("source-key", 8)]


@pytest.mark.asyncio
async def test_watcher_waits_for_durable_local_turn_lease(monkeypatch):
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])
    monkeypatch.setattr(run.asyncio, "sleep", AsyncMock())

    class LeaseDB(_ReclaimDB):
        def __init__(self):
            super().__init__(stale_ids=[])
            self.acquire_results = iter([False, True])
            self.claimed = []
            self.released = []

        async def list_pending_handoffs(self):
            return [{"id": "local-session"}]

        async def try_acquire_session_turn_lease(self, *_args, **_kwargs):
            return next(self.acquire_results)

        async def release_session_turn_lease(self, sid, holder):
            self.released.append((sid, holder))

        async def claim_handoff(self, sid):
            self.claimed.append(sid)
            return True

        async def complete_handoff(self, _sid):
            return None

    db = LeaseDB()
    processed = []

    async def process(row):
        processed.append(row["id"])

    fake = types.SimpleNamespace(
        _session_db=db,
        _running=_running_flag(2),
        _process_handoff=process,
        _reserve_handoff_source=lambda _sid: None,
        _release_handoff_source_reservation=lambda _value: None,
    )

    await run.GatewayRunner._handoff_watcher(
        cast(Any, fake), interval=0.0, drain_timeout=1.0
    )

    assert db.claimed == ["local-session"]
    assert processed == ["local-session"]
    assert len(db.released) == 1


@pytest.mark.asyncio
async def test_task_spawn_error_fails_row_and_releases_reservation(monkeypatch):
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])
    monkeypatch.setattr(run.asyncio, "sleep", AsyncMock())

    class SpawnDB(_ReclaimDB):
        def __init__(self):
            super().__init__(stale_ids=[])
            self.failed = []

        async def list_pending_handoffs(self):
            return [{"id": "row"}]

        async def claim_handoff(self, _sid):
            return True

        async def fail_handoff(self, sid, error):
            self.failed.append((sid, error))

    db = SpawnDB()
    released = []
    fake = types.SimpleNamespace(
        _session_db=db,
        _running=_running_flag(1),
        _process_handoff=AsyncMock(),
        _reserve_handoff_source=lambda _sid: ("source-key", 9),
        _release_handoff_source_reservation=lambda value: released.append(value),
    )
    monkeypatch.setattr(
        run.asyncio, "ensure_future", MagicMock(side_effect=RuntimeError("spawn"))
    )

    await run.GatewayRunner._handoff_watcher(
        cast(Any, fake), interval=0.0, drain_timeout=0.01
    )

    assert released == [("source-key", 9)]
    assert db.failed and db.failed[0][0] == "row"


class _ReclaimDB:
    """Records the reclaim call and reports nothing pending."""

    def __init__(self, stale_ids=("dead-row",)):
        self.stale_ids = list(stale_ids)
        self.reclaim_calls = []

    async def reclaim_stale_running_handoffs(self, error):
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
    fake._session_db = db
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
    fake._session_db = db
    fake._running = _running_flag(1)

    async def _process_handoff(row, profile_name=None):
        return None

    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)
    await asyncio.wait_for(coro, timeout=5)

    assert len(db.reclaim_calls) == 3, (
        f"expected one reclaim per scope (root + 2 profiles), got {len(db.reclaim_calls)}"
    )


@pytest.mark.asyncio
async def test_reclaim_tolerates_store_without_the_method(monkeypatch):
    """An older/duck-typed store must not abort watcher startup."""
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: [(None, None)])

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    class _OldDB:
        def __init__(self):
            self.polls = 0

        async def list_pending_handoffs(self):
            self.polls += 1
            return []

    db = _OldDB()
    fake = types.SimpleNamespace()
    fake._session_db = db
    fake._running = _running_flag(1)

    async def _process_handoff(row, profile_name=None):
        return None

    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0, drain_timeout=0.01)
    await asyncio.wait_for(coro, timeout=5)

    assert db.polls == 1, "the watcher must still poll when reclaim is unavailable"


def test_reclaim_stale_running_handoffs_flips_only_running_rows(tmp_path):
    """DB-level: only 'running' rows are touched, and their ids are returned."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    for sid, state in (
        ("dead", "running"),
        ("queued", "pending"),
        ("done", "completed"),
    ):
        db.create_session(sid, "cli")
        db._execute_write(
            lambda conn, s=sid, st=state: conn.execute(
                "UPDATE sessions SET handoff_state = ? WHERE id = ?", (st, s)
            )
        )

    reclaimed = db.reclaim_stale_running_handoffs("gateway died")

    assert reclaimed == ["dead"]
    assert db.get_handoff_state("dead")["state"] == "failed"
    assert db.get_handoff_state("dead")["error"] == "gateway died"
    assert db.get_handoff_state("queued")["state"] == "pending"
    assert db.get_handoff_state("done")["state"] == "completed"


def test_reclaimed_session_can_request_handoff_again(tmp_path):
    """The point of the reclaim: the session is usable again.

    ``request_handoff`` only accepts NULL/completed/failed, so a stranded
    'running' row is what permanently locks the session out.
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stuck", "cli")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET handoff_state = 'running' WHERE id = 'stuck'"
        )
    )

    assert db.request_handoff("stuck", "telegram") is False, (
        "precondition: a stranded 'running' row blocks new handoff requests"
    )

    db.reclaim_stale_running_handoffs("gateway died")

    assert db.request_handoff("stuck", "telegram") is True, (
        "after reclaim the session must be able to hand off again"
    )
