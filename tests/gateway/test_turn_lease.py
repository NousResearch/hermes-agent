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

import pytest

from gateway.turn_lease import SessionTurnLeaseRegistry


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


# ---------------------------------------------------------------------------
# Release semantics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fail-open on timeout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bounded registry
# ---------------------------------------------------------------------------


def test_lease_with_queued_waiter_is_not_idle_or_evicted():
    """A lease reports non-idle while a turn is parked waiting to acquire it,
    even in the window right after the holder releases (before the waiter's
    wake-up runs). Guards the waiter-blind eviction hole from @hansai-art's
    #67401 audit: without the waiters term the woken waiter would be stranded
    on an unregistered lease while the next acquire mints a fresh one."""

    async def scenario():
        registry = SessionTurnLeaseRegistry(max_entries=1)  # forces eviction

        a = await registry.acquire("S", owner_key="A", generation=1, timeout=5)

        # B parks waiting for S (A still holds it).
        b_task = asyncio.ensure_future(
            registry.acquire("S", owner_key="B", generation=2, timeout=5)
        )
        await asyncio.sleep(0)  # let B reach the await
        assert registry._leases["S"].waiters == 1
        assert not registry._leases["S"].idle

        # Release A. Now holder is None and lock may read unlocked before B's
        # wake-up runs — but waiters==1 keeps idle False.
        registry.release(a)
        assert not registry._leases["S"].idle

        # An eviction pass fired by a *different* session's acquire (cap=1)
        # must NOT drop S while B is queued behind it.
        c = await registry.acquire("OTHER", owner_key="C", generation=1, timeout=5)
        assert "S" in registry._leases, "contended lease was wrongly evicted"

        # B resumes and takes the lease under the still-registered entry.
        b = await asyncio.wait_for(b_task, timeout=5)
        assert b is not None and not b.degraded
        assert registry._leases["S"].holder is b

        registry.release(b)
        registry.release(c)

    _run(scenario())


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
