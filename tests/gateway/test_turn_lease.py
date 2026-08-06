"""Behavior tests for the per-session turn lease (#64934).

The lease serializes the [load history → run → flush] region per RESOLVED
session_id, closing the alias-key overlap route: two routing keys mapped to
one session_id via switch_session() run turns on two different agent objects,
invisible to every routing-key guard.

Covers:
- alias-key turn waits until the first turn's flush, and flush order is
  preserved (the second turn loads history AFTER the first turn's release)
- distinct sessions do not contend
- generation-scoped, idempotent release: a stale unwind can never free a
  newer turn's lease; double-release is a no-op
- timeout fail-open: a stuck holder degrades to unserialized with a degraded
  token, never a wedged session, and the degraded token releases nothing
- registry stays bounded; live leases are never evicted
- GatewayRunner._release_turn_lease wiring (bare-runner safe, token-scoped)
"""

import asyncio
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from gateway.turn_lease import (
    DurableTurnLease,
    SessionTurnLeaseRegistry,
    TurnLeaseCancelledError,
    TurnLeaseTimeoutError,
)


def _desktop_turn_process(db_path, session_id, acquired, release):
    """Hold the Desktop side of a cross-process turn, then flush and release."""
    from hermes_state import SessionDB

    db = SessionDB(Path(db_path))
    lease = DurableTurnLease.acquire(
        db,
        session_id,
        owner_key="tui:desktop",
        generation=1,
        timeout=5,
    )
    acquired.set()
    try:
        if not release.wait(10):
            raise TimeoutError("test parent did not release Desktop turn")
        db.append_message(session_id, "user", "desktop question")
        db.append_message(session_id, "assistant", "desktop answer")
    finally:
        if lease is not None:
            lease.release()
        db.close()


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Serialization behavior
# ---------------------------------------------------------------------------


def test_alias_key_turn_waits_and_order_is_preserved():
    """Second routing key on the same session_id waits for the first turn's
    release; events interleave in strict [load1, flush1, load2, flush2] order."""

    async def scenario():
        registry = SessionTurnLeaseRegistry()
        events = []

        async def turn(owner_key, generation, hold):
            token = await registry.acquire(
                "sess-1", owner_key=owner_key, generation=generation, timeout=5
            )
            assert token is not None and not token.degraded
            events.append(f"load:{owner_key}")
            await asyncio.sleep(hold)  # simulate run + flush
            events.append(f"flush:{owner_key}")
            registry.release(token)

        t1 = asyncio.create_task(turn("key-a", 1, hold=0.05))
        await asyncio.sleep(0.01)  # let turn 1 take the lease
        t2 = asyncio.create_task(turn("key-b", 1, hold=0))
        await asyncio.gather(t1, t2)
        return events

    events = _run(scenario())
    assert events == ["load:key-a", "flush:key-a", "load:key-b", "flush:key-b"]


def test_distinct_sessions_do_not_contend():
    async def scenario():
        registry = SessionTurnLeaseRegistry()
        order = []

        async def turn(session_id, owner_key):
            token = await registry.acquire(
                session_id, owner_key=owner_key, generation=1, timeout=5
            )
            order.append(f"start:{session_id}")
            await asyncio.sleep(0.05)
            order.append(f"end:{session_id}")
            registry.release(token)

        await asyncio.gather(turn("sess-a", "key-a"), turn("sess-b", "key-b"))
        return order

    order = _run(scenario())
    # Both started before either finished — no serialization across sessions.
    assert order[:2] == ["start:sess-a", "start:sess-b"]


def test_durable_lease_serializes_two_processes_and_waiter_reads_flushed_history(
    tmp_path,
):
    """The API-side DB handle waits for Desktop, then sees its committed turn."""
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    setup_db = SessionDB(db_path)
    session_id = setup_db.create_session("shared-session", "tui")
    setup_db.close()

    ctx = multiprocessing.get_context("spawn")
    desktop_acquired = ctx.Event()
    release_desktop = ctx.Event()
    desktop = ctx.Process(
        target=_desktop_turn_process,
        args=(str(db_path), session_id, desktop_acquired, release_desktop),
    )
    desktop.start()
    assert desktop_acquired.wait(10)

    api_db = SessionDB(db_path)
    waiter_acquired = threading.Event()
    observed = {}

    def api_wake():
        lease = DurableTurnLease.acquire(
            api_db,
            session_id,
            owner_key="api:wake",
            generation=1,
            timeout=5,
        )
        try:
            observed["history"] = api_db.get_messages_as_conversation(session_id)
            waiter_acquired.set()
        finally:
            if lease is not None:
                lease.release()

    waiter = threading.Thread(target=api_wake)
    waiter.start()
    time.sleep(0.15)
    assert not waiter_acquired.is_set()

    release_desktop.set()
    assert waiter_acquired.wait(5)
    waiter.join(timeout=5)
    desktop.join(timeout=10)
    api_db.close()

    assert desktop.exitcode == 0
    assert [
        {"role": message["role"], "content": message["content"]}
        for message in observed["history"]
    ] == [
        {"role": "user", "content": "desktop question"},
        {"role": "assistant", "content": "desktop answer"},
    ]


def test_durable_lease_wait_is_bounded_and_cancellable_across_db_handles(tmp_path):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    holder_db = SessionDB(db_path)
    waiter_db = SessionDB(db_path)
    holder = DurableTurnLease.acquire(
        holder_db, "shared", owner_key="tui", generation=1, timeout=1
    )
    try:
        with pytest.raises(TurnLeaseTimeoutError):
            DurableTurnLease.acquire(
                waiter_db, "shared", owner_key="api", generation=1, timeout=0.05
            )

        with pytest.raises(TurnLeaseCancelledError):
            DurableTurnLease.acquire(
                waiter_db,
                "shared",
                owner_key="api",
                generation=2,
                timeout=1,
                cancelled=lambda: True,
            )

        assert holder is not None
        assert waiter_db.release_session_turn_lease("shared", "wrong-holder") is False
        assert waiter_db.get_session_turn_lease_holder("shared") == holder.holder
    finally:
        if holder is not None:
            holder.release()
        holder_db.close()
        waiter_db.close()


def test_durable_lease_rebinds_rotation_alias_and_renews_both_rows(tmp_path):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    owner_db = SessionDB(db_path)
    contender_db = SessionDB(db_path)
    lease = DurableTurnLease.acquire(
        owner_db,
        "parent",
        owner_key="tui",
        generation=1,
        timeout=1,
        ttl_seconds=1,
    )
    try:
        assert lease is not None and lease.rebind("child")
        time.sleep(1.2)
        assert contender_db.get_session_turn_lease_holder("parent") == lease.holder
        assert contender_db.get_session_turn_lease_holder("child") == lease.holder
        assert not contender_db.try_acquire_session_turn_lease("parent", "other", 1)
        assert not contender_db.try_acquire_session_turn_lease("child", "other", 1)
    finally:
        if lease is not None:
            lease.release()
        owner_db.close()
        contender_db.close()


# ---------------------------------------------------------------------------
# Release semantics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fail-open on timeout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bounded registry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mid-turn rotation rebind
# ---------------------------------------------------------------------------


def test_rebind_moves_serialization_to_new_session_id():
    """After a mid-turn compression rotation, an alias key acquiring the NEW
    id must wait behind the holder; release under the new id frees it."""

    async def scenario():
        registry = SessionTurnLeaseRegistry()
        token = await registry.acquire("parent", owner_key="key-a", generation=1, timeout=5)
        assert registry.rebind(token, "child") is True
        assert token is not None and token.session_id == "child"

        # Alias key resolving the fresh child id serializes behind the holder.
        waiter = asyncio.create_task(
            registry.acquire("child", owner_key="key-b", generation=1, timeout=5)
        )
        await asyncio.sleep(0.02)
        assert not waiter.done()

        assert registry.release(token) is True
        t2 = await waiter
        assert t2 is not None and not t2.degraded
        registry.release(t2)

    _run(scenario())


# ---------------------------------------------------------------------------
# GatewayRunner wiring
# ---------------------------------------------------------------------------


def test_runner_release_turn_lease_is_token_scoped_and_bare_safe():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    # Bare runner without __init__: must be a safe no-op (pitfall #17).
    assert runner._release_turn_lease("key-a", 1) is False

    async def scenario():
        runner._turn_leases = SessionTurnLeaseRegistry()
        runner._turn_lease_tokens = {}
        token = await runner._turn_leases.acquire(
            "sess-r", owner_key="key-a", generation=1, timeout=5
        )
        runner._turn_lease_tokens[("key-a", 1)] = token
        # Wrong generation: pops nothing, releases nothing.
        assert runner._release_turn_lease("key-a", 2) is False
        assert runner._turn_leases._leases["sess-r"].holder is token
        # Right (key, generation): releases.
        assert runner._release_turn_lease("key-a", 1) is True
        # Idempotent.
        assert runner._release_turn_lease("key-a", 1) is False
        # Empty key guard.
        assert runner._release_turn_lease("", 1) is False

    _run(scenario())
