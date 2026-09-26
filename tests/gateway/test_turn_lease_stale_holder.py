"""A STALE lease holder must not be reported as alias-key contention.

The lease's contention WARNING unconditionally says "two routing keys are mapped to one
session_id (#64934)". That is false for the common case where a turn was ``/stop``'d (or
``/new``'d) and is still unwinding a tool call: the generation was invalidated, the busy guard
was released, but the lease is only freed in the dispatch layer's ``finally``. The next message
arrives on the SAME routing key at a NEWER generation and contends with that zombie — and the
log blames aliased routing keys, sending every reader down the wrong path. There is also no
signal at all that a stopped turn is sitting on the session.

Fixed by giving the registry the runner's ``_is_session_run_current`` predicate: a holder whose
generation is no longer current is reported honestly, and one past ``STALE_HOLDER_LOG_AFTER``
emits a single structured ``PHASE=stale_lease_holder`` line. Fail-open — no predicate, or a
raising one, keeps the previous behavior.
"""

import asyncio
import logging
import time

import pytest

from gateway.turn_lease import (
    STALE_HOLDER_LOG_AFTER,
    SessionTurnLeaseRegistry,
    TurnLeaseTimeoutError,
    TurnLeaseToken,
)

SESSION = "sess-stale"
KEY = "agent:main:discord:thread:12345:12345"
ALIAS_KEY = "agent:main:discord:thread:99999:99999"
_ALIAS_TEXT = "two routing keys are mapped to one session_id"


def _run(coro):
    return asyncio.run(coro)


def _registry(current: dict) -> SessionTurnLeaseRegistry:
    """Registry wired to a generation map standing in for the runner's session state."""
    return SessionTurnLeaseRegistry(
        is_generation_current=lambda key, gen: int(current.get(key, 0)) == int(gen)
    )


def _messages(caplog) -> list:
    return [r.getMessage() for r in caplog.records]


def test_stale_holder_is_not_reported_as_alias_contention(caplog):
    async def scenario():
        current = {KEY: 2}
        registry = _registry(current)
        held = await registry.acquire(SESSION, owner_key=KEY, generation=2, timeout=1)
        current[KEY] = 4  # /stop invalidates the holder's generation
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(SESSION, owner_key=KEY, generation=4, timeout=0.05)
        registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    messages = _messages(caplog)
    assert not any(_ALIAS_TEXT in m for m in messages), (
        "a same-key stale holder must not be blamed on aliased routing keys"
    )
    assert any("STALE" in m and "still draining a tool call" in m for m in messages)


def test_live_alias_holder_keeps_the_original_wording(caplog):
    async def scenario():
        registry = _registry({KEY: 2})  # holder's generation stays current
        held = await registry.acquire(SESSION, owner_key=KEY, generation=2, timeout=1)
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(SESSION, owner_key=ALIAS_KEY, generation=1, timeout=0.05)
        registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    messages = _messages(caplog)
    assert any(_ALIAS_TEXT in m for m in messages)
    assert not any("PHASE=stale_lease_holder" in m for m in messages)


def test_long_held_stale_holder_emits_one_structured_line(caplog):
    async def scenario():
        current = {KEY: 2}
        registry = _registry(current)
        held = await registry.acquire(SESSION, owner_key=KEY, generation=2, timeout=1)
        current[KEY] = 4
        # Backdate past the detector threshold, as a real long drain would be.
        registry._leases[SESSION].acquired_at = time.time() - STALE_HOLDER_LOG_AFTER - 30
        for generation in (4, 5):
            with pytest.raises(TurnLeaseTimeoutError):
                await registry.acquire(SESSION, owner_key=KEY, generation=generation, timeout=0.05)
        registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    lines = [m for m in _messages(caplog) if "PHASE=stale_lease_holder" in m]
    assert len(lines) == 1, f"one-shot latch expected, got {lines}"
    assert f"session={SESSION}" in lines[0]
    assert f"key={KEY}" in lines[0]
    assert "gen=2" in lines[0]
    assert "held=" in lines[0]


def test_briefly_held_stale_holder_does_not_emit_the_detector(caplog):
    async def scenario():
        current = {KEY: 2}
        registry = _registry(current)
        held = await registry.acquire(SESSION, owner_key=KEY, generation=2, timeout=1)
        current[KEY] = 4
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(SESSION, owner_key=KEY, generation=4, timeout=0.05)
        registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    assert not any("PHASE=stale_lease_holder" in m for m in _messages(caplog))


def test_detector_is_rearmed_after_release(caplog):
    """A LATER zombie on the same session must log again."""

    async def scenario():
        current = {KEY: 2}
        registry = _registry(current)
        for generation, waiter_gen in ((2, 4), (6, 8)):
            current[KEY] = generation
            held = await registry.acquire(
                SESSION, owner_key=KEY, generation=generation, timeout=1
            )
            current[KEY] = waiter_gen
            registry._leases[SESSION].acquired_at = time.time() - STALE_HOLDER_LOG_AFTER - 30
            with pytest.raises(TurnLeaseTimeoutError):
                await registry.acquire(
                    SESSION, owner_key=KEY, generation=waiter_gen, timeout=0.05
                )
            registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    assert len([m for m in _messages(caplog) if "PHASE=stale_lease_holder" in m]) == 2


def test_no_predicate_keeps_previous_behavior(caplog):
    """Fail-open: a registry built without a predicate treats every holder as live."""

    async def scenario():
        registry = SessionTurnLeaseRegistry()
        held = await registry.acquire(SESSION, owner_key=KEY, generation=2, timeout=1)
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(SESSION, owner_key=KEY, generation=4, timeout=0.05)
        registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    messages = _messages(caplog)
    assert any(_ALIAS_TEXT in m for m in messages)
    assert not any("PHASE=stale_lease_holder" in m for m in messages)


def test_raising_predicate_fails_open_to_live(caplog):
    """A broken predicate must never mislabel a genuinely running turn."""

    def boom(_key, _gen):
        raise RuntimeError("session state unavailable")

    async def scenario():
        registry = SessionTurnLeaseRegistry(is_generation_current=boom)
        held = await registry.acquire(SESSION, owner_key=KEY, generation=2, timeout=1)
        with pytest.raises(TurnLeaseTimeoutError):
            await registry.acquire(SESSION, owner_key=KEY, generation=4, timeout=0.05)
        registry.release(held)

    with caplog.at_level(logging.WARNING, logger="gateway.turn_lease"):
        _run(scenario())

    assert any(_ALIAS_TEXT in m for m in _messages(caplog))


def test_runner_wires_the_predicate_into_its_registry():
    """The gateway must actually supply _is_session_run_current.

    Drives the REAL initialiser on a bare instance so deleting the
    ``is_generation_current=`` argument at ``gateway/run.py`` fails here.
    Constructing a registry in the test and handing it the bound method by hand
    exercises the registry, not the wiring: it stays green with the production
    argument deleted (verified by mutation).
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    GatewayRunner._init_lifecycle_state(runner)

    registry = runner._turn_leases
    assert registry._is_generation_current is not None
    # The predicate must be the runner's own bound method, not an arbitrary callable.
    assert getattr(registry._is_generation_current, "__self__", None) is runner

    # And it must actually CLASSIFY: a holder whose generation is no longer current
    # is stale. Assert through the runner's registry, using its real session state.
    from gateway.session_state import SessionState

    session_key = "agent:main:discord:thread:12345:12345"
    state = SessionState()
    state.persistent.run_generation = 7
    runner._sessions = {session_key: state}

    from gateway.turn_lease import _SessionLease

    lease = _SessionLease()
    holder = TurnLeaseToken(SESSION, session_key, 3, lease=lease)
    live = TurnLeaseToken(SESSION, session_key, 7, lease=lease)
    assert registry._holder_is_stale(holder) is True
    assert registry._holder_is_stale(live) is False
