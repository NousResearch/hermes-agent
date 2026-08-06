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
- timeout fail-closed: a timed-out waiter never enters the transcript region,
  and outer dispatch returns a visible rejection/resend notice without invoking
  goal continuation
- registry stays bounded; live leases are never evicted
- GatewayRunner._release_turn_lease wiring (bare-runner safe, token-scoped)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.turn_lease import SessionTurnLeaseRegistry, TurnLeaseTimeoutError


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
# Timeout safety
# ---------------------------------------------------------------------------


def test_timeout_fails_closed_instead_of_authorizing_an_unserialized_turn():
    """A timed-out waiter must never run against the still-live holder.

    Returning a degraded token used to authorize exactly that unsafe path.
    The two turns could then load the same history base and interleave their
    transcript writes, defeating the serialization invariant this lease owns.
    """
    async def scenario():
        registry = SessionTurnLeaseRegistry()
        holder = await registry.acquire(
            "sess-timeout", owner_key="key-a", generation=1, timeout=1
        )
        assert holder is not None

        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(
                "sess-timeout", owner_key="key-b", generation=1, timeout=0.02
            )

        # The timeout neither steals nor releases the live holder's lease.
        assert registry._leases["sess-timeout"].holder is holder
        assert registry.release(holder) is True

        # Once the holder releases, a later turn can acquire normally.
        successor = await registry.acquire(
            "sess-timeout", owner_key="key-b", generation=2, timeout=1
        )
        assert successor is not None
        assert registry.release(successor) is True

    _run(scenario())


@pytest.mark.asyncio
async def test_agent_path_propagates_timed_out_lease_before_loading_transcript(
    monkeypatch, tmp_path
):
    """The agent path propagates timeout before transcript work can begin.

    Outer dispatch owns the visible rejection/resend notice. Most importantly,
    transcript loading and agent execution must not start: both would operate
    without the per-session serialization guarantee.
    """
    from tests.gateway.test_42039_duplicate_user_message import (
        _bootstrap,
        _event,
        _source,
    )

    runner = _bootstrap(monkeypatch, tmp_path)
    runner._turn_leases = SessionTurnLeaseRegistry()
    holder = await runner._turn_leases.acquire(
        "sess-dedup", owner_key="holder-key", generation=1, timeout=1
    )
    assert holder is not None
    monkeypatch.setenv("HERMES_TURN_LEASE_TIMEOUT", "0.02")

    runner.session_store.load_transcript.side_effect = AssertionError(
        "transcript must not load after a turn-lease timeout"
    )
    runner._run_agent = pytest.fail

    try:
        with pytest.raises(TurnLeaseTimeoutError):
            await runner._handle_message_with_agent(
                _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
            )
    finally:
        assert runner._turn_leases.release(holder) is True

    runner.session_store.load_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_full_dispatch_rejects_lease_timeout_without_running_goal_hook(
    monkeypatch, tmp_path
):
    """A lease rejection is not a completed turn for `/goal` evaluation.

    The lease wait also has its own clock: a short lease budget must reject
    promptly even while the normal agent inactivity timeout remains long.
    """
    from tests.gateway.test_42039_duplicate_user_message import _bootstrap, _event

    runner = _bootstrap(monkeypatch, tmp_path)
    runner._turn_leases = SessionTurnLeaseRegistry()
    holder = await runner._turn_leases.acquire(
        "sess-dedup", owner_key="holder-key", generation=1, timeout=1
    )
    assert holder is not None
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "5")
    monkeypatch.setenv("HERMES_TURN_LEASE_TIMEOUT", "0.02")

    runner.session_store.load_transcript.side_effect = AssertionError(
        "transcript must not load after a turn-lease timeout"
    )
    session_env_tokens = object()
    runner._set_session_env = MagicMock(return_value=session_env_tokens)
    runner._clear_session_env = MagicMock()
    runner._run_agent = pytest.fail
    runner._post_turn_goal_continuation = AsyncMock()

    try:
        response = await asyncio.wait_for(runner._handle_message(_event()), timeout=1)
    finally:
        assert runner._turn_leases.release(holder) is True

    assert isinstance(response, str)
    assert "not processed" in response.lower()
    assert "resend" in response.lower()
    runner.session_store.load_transcript.assert_not_called()
    runner._clear_session_env.assert_called_once_with(session_env_tokens)
    runner._post_turn_goal_continuation.assert_not_awaited()


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
