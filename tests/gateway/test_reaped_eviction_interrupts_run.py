"""Regression tests for interrupting work evicted from a gateway turn slot."""

from __future__ import annotations

import pytest

from gateway.run import (
    GatewayRunner,
    _AGENT_PENDING_SENTINEL,
    _is_control_interrupt_message,
)
from gateway.run_inbound import GatewayInboundMixin


KEY = "agent:main:telegram:dm:106963"
EVICTION_REASON = "Session ended while the turn was running"


class _RecordingAgent:
    def __init__(self, events: list[tuple], slot_agent) -> None:
        self._events = events
        self._slot_agent = slot_agent
        self.interrupted = False

    def hard_interrupt(self, message: str | None = None, **_kwargs) -> None:
        self.interrupted = True
        self._events.append(("interrupt", message, self._slot_agent() is self))


class _RaisingAgent:
    def hard_interrupt(self, _message: str | None = None, **_kwargs) -> None:
        raise RuntimeError("interrupt transport failed")


class _ReapedStore:
    def peek_session_id(self, _session_key: str) -> str:
        return "session-106963"

    def _is_session_ended_in_db(self, session_id: str) -> bool:
        return session_id == "session-106963"


def _build_gateway(agent, events: list[tuple]):
    gateway = object.__new__(GatewayRunner)
    gateway._persist_active_agents = lambda: None
    gateway._agent_cache_lock = None
    gateway._agent_cache = {KEY: (agent, "signature", 0)}
    gateway._spawn_release_thread = lambda target, args, name, inline_fallback: events.append(
        ("cache_release", args[0])
    )
    state = gateway._session_state(KEY)
    state.turn.agent = agent
    return gateway, state


@pytest.mark.parametrize("entrypoint", ("direct", "reaped"))
def test_eviction_interrupts_before_release_and_drops_cached_agent(entrypoint: str) -> None:
    events: list[tuple] = []
    gateway = None

    def current_agent():
        state = gateway._peek_session_state(KEY)
        return state.turn.agent if state else None

    agent = _RecordingAgent(events, current_agent)
    gateway, _state = _build_gateway(agent, events)
    release = gateway._release_running_agent_state

    def release_with_record(session_key: str, **kwargs) -> bool:
        events.append(("release", current_agent() is agent))
        return release(session_key, **kwargs)

    gateway._release_running_agent_state = release_with_record
    if entrypoint == "reaped":
        gateway.session_store = _ReapedStore()
        gateway._hm_evict_reaped_agent(KEY)
    else:
        gateway._hm_evict_running_agent(KEY, "stale_running_agent_eviction")

    assert isinstance(gateway, GatewayInboundMixin)
    assert agent.interrupted
    assert events[0] == ("interrupt", EVICTION_REASON, True)
    release_events = [event for event in events if event[0] == "release"]
    assert release_events == [("release", True)]
    assert events.index(events[0]) < events.index(release_events[0])
    assert gateway._peek_session_state(KEY).turn.agent is None
    assert KEY not in gateway._agent_cache
    assert _is_control_interrupt_message(EVICTION_REASON)


@pytest.mark.parametrize("agent", (None, _AGENT_PENDING_SENTINEL, _RaisingAgent()))
def test_eviction_cleanup_survives_empty_pending_or_failed_interrupt(agent) -> None:
    events: list[tuple] = []
    gateway, state = _build_gateway(agent, events)

    gateway._hm_evict_running_agent(KEY, "reaped_session_eviction")

    assert state.turn.agent is None
    assert KEY not in gateway._agent_cache


def test_stale_finalizer_cannot_release_replacement_generation() -> None:
    events: list[tuple] = []
    old_agent = _RecordingAgent(events, lambda: None)
    gateway, state = _build_gateway(old_agent, events)
    state.persistent.run_generation = 2

    # Eviction releases generation 2 before the cold path claims the replacement.
    gateway._invalidate_session_run_generation(KEY, reason="reaped_session_eviction")
    gateway._release_running_agent_state(KEY)
    replacement = object()
    replacement_state = gateway._session_state(KEY)
    replacement_state.turn.agent = replacement
    replacement_state.persistent.run_generation = 4

    # Generation 2 is unwinding after generation 4 claimed the key.
    assert gateway._release_running_agent_state(KEY, run_generation=2) is False
    assert gateway._peek_session_state(KEY).turn.agent is replacement


def test_moa_one_shot_restoration_generation_owned() -> None:
    events: list[tuple] = []
    gateway, state = _build_gateway(object(), events)
    state.persistent.run_generation = 1
    state.conversation.model_override = {"provider": "custom", "model": "test-model"}

    # Simulate /moa command handling
    class _MockEvent:
        _moa_disable_after_turn = True
        _moa_restore_override = {"provider": "custom", "model": "test-model"}
        _moa_run_generation = None

    event = _MockEvent()
    # Turn claims generation 2
    claimed_gen = 2
    state.persistent.run_generation = claimed_gen
    event._moa_run_generation = claimed_gen
    state.conversation.model_override = {"provider": "moa", "model": "moa-model"}

    # 1. Stale generation finalizer (e.g. gen 1) must NOT restore or clear MoA
    gateway._restore_moa_one_shot(event, KEY, run_generation=1)
    assert event._moa_disable_after_turn is True
    assert state.conversation.model_override == {"provider": "moa", "model": "moa-model"}

    # 2. Owning generation finalizer (gen 2) restores prior override
    gateway._restore_moa_one_shot(event, KEY, run_generation=claimed_gen)
    assert event._moa_disable_after_turn is False
    assert state.conversation.model_override == {"provider": "custom", "model": "test-model"}


def test_model_once_snapshot_preserved_from_stale_finalizer() -> None:
    events: list[tuple] = []
    gateway, state = _build_gateway(object(), events)
    state.persistent.run_generation = 1
    state.conversation.model_override = {"model": "once-model", "provider": "test"}
    state.conversation.one_turn_restore = {
        "had_override": True,
        "override": {"model": "original-model", "provider": "test"},
        "run_generation": 2,
    }

    # Stale finalizer for generation 1 must NOT clear one_turn_restore or restore override
    state.persistent.run_generation = 2
    gateway._restore_pending_one_turn_model_override(KEY, run_generation=1)
    assert state.conversation.one_turn_restore is not None
    assert state.conversation.model_override["model"] == "once-model"

    # Owning generation 2 restores snapshot and clears one_turn_restore
    gateway._restore_pending_one_turn_model_override(KEY, run_generation=2)
    assert state.conversation.one_turn_restore is None
    assert state.conversation.model_override["model"] == "original-model"


@pytest.mark.asyncio
async def test_turn_lease_rebind_preserves_parent_lock_domain_and_releases() -> None:
    from gateway.turn_lease import SessionTurnLeaseRegistry

    registry = SessionTurnLeaseRegistry()
    token = await registry.acquire("parent-session", owner_key="key-1", generation=1, timeout=5)
    assert token is not None
    assert registry.rebind(token, "child-session") is True
    assert token.session_id == "child-session"

    # Both parent and child session IDs must be registered to the same lease
    assert registry._leases.get("parent-session") is registry._leases.get("child-session")
    assert registry._leases["parent-session"].holder is token

    # Parent lock domain remains busy while child is held
    import asyncio
    waiter = asyncio.create_task(
        registry.acquire("parent-session", owner_key="key-2", generation=1, timeout=5)
    )
    await asyncio.sleep(0.01)
    assert not waiter.done()

    # Release by token identity frees the lock and wakes the parent waiter
    assert registry.release(token) is True
    parent_token = await waiter
    assert parent_token is not None
    assert parent_token.owner_key == "key-2"
    assert registry.release(parent_token) is True


def test_displaced_turn_lease_release_by_owning_generation() -> None:
    from gateway.turn_lease import SessionTurnLeaseRegistry

    events: list[tuple] = []
    gateway, state = _build_gateway(object(), events)
    registry = SessionTurnLeaseRegistry()
    gateway._turn_leases = registry

    import asyncio
    token1 = asyncio.run(registry.acquire("sess-106963", owner_key=KEY, generation=1, timeout=5))
    assert token1 is not None

    state.turn.lease_tokens[1] = token1
    state.turn.lease_token = token1
    state.turn.lease_generation = 1

    # Unwind of generation 2 has no token
    assert gateway._release_turn_lease(KEY, run_generation=2) is False
    assert token1.released is False

    # Owning generation 1 releases token1
    assert gateway._release_turn_lease(KEY, run_generation=1) is True
    assert token1.released is True
    assert 1 not in state.turn.lease_tokens
