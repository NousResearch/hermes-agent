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
