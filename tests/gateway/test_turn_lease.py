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
- registry stays bounded; live and pending leases are never evicted
- timed-out and cancelled acquire attempts do not pin idle registry entries
- GatewayRunner._release_turn_lease wiring (bare-runner safe, token-scoped)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.turn_lease import (
    DEFAULT_LEASE_WAIT,
    SessionTurnLeaseRegistry,
    TurnLeaseTimeoutError,
)


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
            assert token is not None
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


def test_default_wait_cannot_head_of_line_block_platform_updates_for_minutes():
    """A contended topic must fail/queue promptly, not pin Telegram's updater.

    Telegram dispatches updates sequentially. Awaiting a held session lease for
    1,800 seconds blocks unrelated topics behind the waiter even though their
    sessions do not share a transcript.
    """
    assert DEFAULT_LEASE_WAIT == 5.0


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


class _ObservedLock:
    """Forwarding lock that signals when an acquire has to wait."""

    def __init__(self, lock):
        self._lock = lock
        self.blocked = asyncio.Queue()

    async def acquire(self):
        if self._lock.locked():
            self.blocked.put_nowait(None)
        return await self._lock.acquire()

    def locked(self):
        return self._lock.locked()

    def release(self):
        self._lock.release()


class _GatedLock:
    """Forwarding lock that pauses an otherwise-uncontended acquire."""

    def __init__(self, lock):
        self._lock = lock
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()

    async def acquire(self):
        self.started.set()
        await self.proceed.wait()
        return await self._lock.acquire()

    def locked(self):
        return self._lock.locked()

    def release(self):
        self._lock.release()


def test_registry_does_not_evict_lease_during_waiter_handoff():
    """A woken waiter must stay in the original serialization domain.

    ``asyncio.Lock.release()`` unlocks before the selected waiter resumes.
    Capacity eviction in that handoff window must not orphan the old lock and
    let a later acquire for the same session take a second lock concurrently.
    """

    async def scenario():
        registry = SessionTurnLeaseRegistry(max_entries=1)
        first = await registry.acquire(
            "shared", owner_key="first", generation=1, timeout=1
        )
        lease = registry._leases["shared"]
        observed = _ObservedLock(lease.lock)
        lease.lock = observed

        waking_task = asyncio.create_task(
            registry.acquire("shared", owner_key="waking", generation=1, timeout=1)
        )
        await asyncio.wait_for(observed.blocked.get(), timeout=1)

        assert registry.release(first) is True
        # _get_or_create("other") runs before this acquire first yields, in
        # the unlocked handoff window before waking_task resumes.
        other = await registry.acquire(
            "other", owner_key="other", generation=1, timeout=1
        )
        waking = await waking_task

        assert registry._leases.get("shared") is lease

        successor_task = asyncio.create_task(
            registry.acquire("shared", owner_key="successor", generation=2, timeout=1)
        )
        await asyncio.wait_for(observed.blocked.get(), timeout=1)
        assert not successor_task.done()

        assert registry.release(waking) is True
        successor = await successor_task
        assert registry.release(successor) is True
        assert registry.release(other) is True

    _run(scenario())


def test_registry_does_not_evict_an_uncontended_acquire_before_it_locks():
    """Every pending acquire is protected, even if the lock looked free."""

    async def scenario():
        registry = SessionTurnLeaseRegistry(max_entries=1)
        seed = await registry.acquire(
            "shared", owner_key="seed", generation=1, timeout=1
        )
        assert registry.release(seed) is True

        lease = registry._leases["shared"]
        gated = _GatedLock(lease.lock)
        lease.lock = gated
        pending_task = asyncio.create_task(
            registry.acquire("shared", owner_key="pending", generation=2, timeout=1)
        )
        await asyncio.wait_for(gated.started.wait(), timeout=1)

        other = await registry.acquire(
            "other", owner_key="other", generation=1, timeout=1
        )
        assert registry._leases.get("shared") is lease

        gated.proceed.set()
        pending = await pending_task
        assert registry.release(pending) is True
        assert registry.release(other) is True

    _run(scenario())


def test_timed_out_acquire_does_not_pin_idle_registry_entry():
    async def scenario():
        registry = SessionTurnLeaseRegistry(max_entries=1)
        holder = await registry.acquire(
            "shared", owner_key="holder", generation=1, timeout=1
        )

        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(
                "shared", owner_key="timeout", generation=2, timeout=0.02
            )

        assert registry.release(holder) is True
        other = await registry.acquire(
            "other", owner_key="other", generation=1, timeout=1
        )
        assert set(registry._leases) == {"other"}
        assert registry.release(other) is True

    _run(scenario())


def test_cancelled_acquire_does_not_pin_idle_registry_entry():
    async def scenario():
        registry = SessionTurnLeaseRegistry(max_entries=1)
        holder = await registry.acquire(
            "shared", owner_key="holder", generation=1, timeout=1
        )
        lease = registry._leases["shared"]
        observed = _ObservedLock(lease.lock)
        lease.lock = observed

        cancelled_task = asyncio.create_task(
            registry.acquire("shared", owner_key="cancelled", generation=2, timeout=1)
        )
        await asyncio.wait_for(observed.blocked.get(), timeout=1)
        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task

        assert registry.release(holder) is True
        other = await registry.acquire(
            "other", owner_key="other", generation=1, timeout=1
        )
        assert set(registry._leases) == {"other"}
        assert registry.release(other) is True

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
        assert t2 is not None
        registry.release(t2)

    _run(scenario())


def test_rebind_does_not_replace_target_during_waiter_handoff():
    """A target with a waking waiter is still a live lease domain."""

    async def scenario():
        registry = SessionTurnLeaseRegistry()
        target_holder = await registry.acquire(
            "target", owner_key="target-holder", generation=1, timeout=1
        )
        target_lease = registry._leases["target"]
        observed = _ObservedLock(target_lease.lock)
        target_lease.lock = observed
        target_waiter_task = asyncio.create_task(
            registry.acquire(
                "target", owner_key="target-waiter", generation=2, timeout=1
            )
        )
        await asyncio.wait_for(observed.blocked.get(), timeout=1)

        source_holder = await registry.acquire(
            "source", owner_key="source-holder", generation=1, timeout=1
        )
        assert registry.release(target_holder) is True

        # The target lock is briefly unlocked, but its selected waiter has
        # not resumed. Rebind must not replace that serialization domain.
        assert registry.rebind(source_holder, "target") is False
        target_waiter = await target_waiter_task
        assert registry._leases["target"] is target_lease

        assert registry.release(target_waiter) is True
        assert registry.release(source_holder) is True

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
