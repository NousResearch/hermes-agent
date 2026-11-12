"""Async ``TurnLease`` wrapper over the merged cross-process turn lease.

The storage substrate (``session_turn_leases`` table, lineage-root key,
dead-PID reclaim, ``SessionTurnLeaseLostError`` fencing) is covered by
``tests/state/test_session_turn_lease.py``, and the sync ``AIAgent`` consumer
by ``tests/run_agent/test_cross_process_turn_lease.py``.  This file covers
only the async adapter in ``gateway/turn_lease.py``: non-blocking polling with
backoff, fail-open timeout, guaranteed release on every exit path (including
exception and cancellation), and the background refresh loop — unit-tested
against a fake state, plus one real-DB integration test proving it stacks on
the merged substrate (#67442).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from gateway.turn_lease import TurnLease


def _run(coro):
    return asyncio.run(coro)


class _FakeState:
    """Minimal sync surface the wrapper needs (merged API names)."""

    def __init__(self, acquire_results=None):
        # The LAST element is repeated forever, so [True] acquires
        # immediately and [False] never does.
        self.acquire_results = list(acquire_results or [True])
        self.acquire_calls = []
        self.refresh_calls = []
        self.release_calls = []

    def try_acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.acquire_calls.append((session_id, holder, kwargs))
        return (
            self.acquire_results[-1]
            if len(self.acquire_results) == 1
            else self.acquire_results.pop(0)
        )

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        self.refresh_calls.append((session_id, holder, kwargs))
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.release_calls.append((session_id, holder))


# ---------------------------------------------------------------------------
# Acquisition / release
# ---------------------------------------------------------------------------


def test_acquire_immediate_success_and_release():
    """A free lease is taken on the first attempt and released on exit."""
    state = _FakeState([True])

    async def scenario():
        lease = TurnLease(state, "s1", "pid=1:turn=a")
        async with lease as acquired:
            assert acquired is True
            assert lease.degraded is False

    _run(scenario())
    assert [c[0] for c in state.acquire_calls] == ["s1"]
    assert state.acquire_calls[0][1] == "pid=1:turn=a"
    assert state.acquire_calls[0][2] == {"ttl_seconds": 300.0}
    assert state.release_calls == [("s1", "pid=1:turn=a")]


def test_degraded_flag_reflects_exclusivity():
    """``degraded`` is True only when the lease was NOT proven exclusive."""
    state_ok = _FakeState([True])
    lease_ok = TurnLease(state_ok, "s1", "pid=1:turn=a")
    assert _run(lease_ok.__aenter__()) is True
    assert lease_ok.degraded is False
    _run(lease_ok.__aexit__(None, None, None))

    state_busy = _FakeState([False])
    lease_busy = TurnLease(
        state_busy, "s1", "pid=1:turn=b",
        wait_timeout=0.05, poll_interval=0.01,
    )
    assert _run(lease_busy.__aenter__()) is False
    assert lease_busy.degraded is True
    _run(lease_busy.__aexit__(None, None, None))


def test_polls_with_backoff_until_lease_is_free():
    """A contended lease is polled (non-blocking) until the holder releases."""
    state = _FakeState([False, False, True])

    async def scenario():
        async with TurnLease(
            state, "s1", "pid=1:turn=a",
            wait_timeout=2.0, poll_interval=0.01,
        ) as acquired:
            return acquired

    assert _run(scenario()) is True
    assert len(state.acquire_calls) == 3
    # No release before acquisition completes.
    assert state.release_calls == [("s1", "pid=1:turn=a")]


def test_timeout_fails_open_without_release():
    """A lease held past the wait budget returns False and releases nothing."""
    state = _FakeState([False])

    async def scenario():
        async with TurnLease(
            state, "s1", "pid=1:turn=a",
            wait_timeout=0.05, poll_interval=0.01,
        ) as acquired:
            return acquired

    assert _run(scenario()) is False
    assert len(state.acquire_calls) >= 2  # polled, then gave up
    assert state.release_calls == []  # never held → nothing to release


def test_empty_session_id_never_acquires():
    """A falsy session_id short-circuits to fail-open with no storage calls."""
    state = _FakeState([True])

    async def scenario():
        async with TurnLease(state, "", "pid=1:turn=a") as acquired:
            return acquired

    assert _run(scenario()) is False
    assert state.acquire_calls == []
    assert state.release_calls == []


# ---------------------------------------------------------------------------
# Guaranteed release
# ---------------------------------------------------------------------------


def test_release_on_exception():
    """An exception inside the body still releases the lease."""
    state = _FakeState([True])

    async def scenario():
        try:
            async with TurnLease(state, "s1", "pid=1:turn=a"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    _run(scenario())
    assert state.release_calls == [("s1", "pid=1:turn=a")]


def test_release_on_cancellation():
    """Cancelling the task mid-body releases the lease (aexit runs)."""
    state = _FakeState([True])

    async def scenario():
        async with TurnLease(state, "s1", "pid=1:turn=a"):
            await asyncio.sleep(30)

    async def driver():
        task = asyncio.create_task(scenario())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    _run(driver())
    assert state.release_calls == [("s1", "pid=1:turn=a")]


# ---------------------------------------------------------------------------
# Background refresh
# ---------------------------------------------------------------------------


def test_background_refresh_extends_and_stops_on_exit():
    """refresh_interval spawns a loop that bumps expiry; exit cancels it."""
    state = _FakeState([True])

    async def scenario():
        async with TurnLease(
            state, "s1", "pid=1:turn=a",
            ttl_seconds=1.0, refresh_interval=0.02,
        ) as acquired:
            assert acquired is True
            await asyncio.sleep(0.09)
        return len(state.refresh_calls)

    refreshes = _run(scenario())
    assert refreshes >= 2  # loop ran at ~0.02/0.04/0.06/0.08
    assert all(c[0] == "s1" and c[1] == "pid=1:turn=a" for c in state.refresh_calls)
    assert state.release_calls == [("s1", "pid=1:turn=a")]


def test_no_refresh_without_refresh_interval():
    state = _FakeState([True])

    async def scenario():
        async with TurnLease(state, "s1", "pid=1:turn=a"):
            await asyncio.sleep(0.03)

    _run(scenario())
    assert state.refresh_calls == []


# ---------------------------------------------------------------------------
# Integration on the merged substrate
# ---------------------------------------------------------------------------


def test_real_db_serializes_two_handles(tmp_path):
    """The async wrapper acquires the same row the merged storage exposes."""
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    db1 = SessionDB(path)
    db2 = SessionDB(path)
    db1.create_session("shared", source="test")

    h1 = f"pid={os.getpid()}:turn=async1"
    h2 = f"pid={os.getpid()}:turn=async2"

    async def scenario():
        async with TurnLease(db1, "shared", h1, ttl_seconds=5) as acquired:
            assert acquired is True
            # Second handle sees the lease held (cross-process shape).
            assert not db2.try_acquire_session_turn_lease("shared", h2, ttl_seconds=5)
        # Released on exit → the second handle can now take it.
        assert db2.try_acquire_session_turn_lease("shared", h2, ttl_seconds=5)
        db2.release_session_turn_lease("shared", h2)

    _run(scenario())


def test_real_db_async_waiter_polls_until_contention_clears(tmp_path):
    """A second async wrapper waits (non-blocking) and wins after release."""
    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    db1 = SessionDB(path)
    db2 = SessionDB(path)
    db1.create_session("shared", source="test")

    h1 = f"pid={os.getpid()}:turn=holder"
    h2 = f"pid={os.getpid()}:turn=waiter"
    results = {}

    async def holder():
        async with TurnLease(db1, "shared", h1, ttl_seconds=5):
            await asyncio.sleep(0.15)  # simulate a turn

    async def waiter():
        async with TurnLease(
            db2, "shared", h2,
            ttl_seconds=5, wait_timeout=3.0, poll_interval=0.01,
        ) as acquired:
            results["waiter"] = acquired

    async def main():
        await asyncio.gather(holder(), waiter())

    _run(main())
    assert results["waiter"] is True
