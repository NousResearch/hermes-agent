"""Regression tests for interrupting work evicted from a gateway turn slot."""

from __future__ import annotations

import pytest

from gateway.run import (
    GatewayRunner,
    _AGENT_PENDING_SENTINEL,
    _INTERRUPT_REASON_EVICTED,
    _is_control_interrupt_message,
)
from gateway.run_inbound import GatewayInboundMixin


KEY = "agent:main:telegram:dm:106963"


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

    assert agent.interrupted
    assert events[0] == ("interrupt", _INTERRUPT_REASON_EVICTED, True)
    release_events = [event for event in events if event[0] == "release"]
    assert release_events == [("release", True)]  # the interrupt was requested BEFORE the slot release
    assert gateway._peek_session_state(KEY).turn.agent is None
    assert KEY not in gateway._agent_cache
    # The reason must be a registered control message or the finalizer treats it as user text.
    assert _is_control_interrupt_message(_INTERRUPT_REASON_EVICTED)


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


def test_one_shot_override_settles_on_stop_and_stale_finalizer_is_a_noop() -> None:
    """/model --once (and /moa, which shares the snapshot) mid-turn: a /stop, /new or eviction
    settles the override BEFORE bumping the generation, and the displaced turn's finalizer then
    finds nothing to restore — so the one-shot model neither leaks nor clobbers a successor."""
    events: list[tuple] = []
    gateway, state = _build_gateway(object(), events)
    prior = {"model": "original-model", "provider": "test"}
    state.conversation.model_override = {"model": "once-model", "provider": "test"}
    state.conversation.one_turn_restore = {"had_override": True, "override": dict(prior)}
    owning_gen = gateway._begin_session_run_generation(KEY)

    gateway._invalidate_session_run_generation(KEY, reason="user_stop")  # settlement point
    assert state.conversation.model_override == prior
    assert state.conversation.one_turn_restore is None

    # The successor claims its own --once; the displaced finalizer (owning_gen) must not touch it.
    state.conversation.model_override = {"model": "successor-once", "provider": "test"}
    state.conversation.one_turn_restore = {"had_override": False, "override": None,
                                           "run_generation": state.persistent.run_generation}
    gateway._restore_pending_one_turn_model_override(KEY, run_generation=owning_gen)
    assert state.conversation.model_override == {"model": "successor-once", "provider": "test"}
    assert state.conversation.one_turn_restore is not None


@pytest.mark.asyncio
async def test_turn_lease_rebind_preserves_parent_lock_domain_and_releases() -> None:
    from gateway.turn_lease import SessionTurnLeaseRegistry

    registry = SessionTurnLeaseRegistry()
    token = await registry.acquire("parent-session", owner_key="key-1", generation=1, timeout=5)
    assert token is not None
    assert registry.rebind(token, "child-session") is True
    assert token.session_id == "child-session"


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

    # Unwind of generation 2 has no token
    assert gateway._release_turn_lease(KEY, run_generation=2) is False
    assert token1.released is False

    # Owning generation 1 releases token1
    assert gateway._release_turn_lease(KEY, run_generation=1) is True
    assert token1.released is True
    assert 1 not in state.turn.lease_tokens
