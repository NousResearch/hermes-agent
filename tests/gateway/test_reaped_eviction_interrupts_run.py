"""#106963: evicting a reaped session must interrupt the in-flight run.

Bug
---
``_hm_evict_running_agent`` (durable-session reap and stale-agent eviction)
only bumps the run generation and releases the turn slot.  The in-flight
agent is never hard-interrupted, so the turn loop keeps calling the model
after its result will be dropped (51 calls / 7.9M tokens in the report).

``/stop`` already interrupts first via ``_interrupt_and_clear_session`` →
``request_hard_interrupt(agent, _INTERRUPT_REASON_STOP)``.  Reap eviction
must use the same ABI, and must interrupt BEFORE
``_release_running_agent_state`` empties the slot (otherwise a later
``/stop`` cannot reach the agent).
"""

from __future__ import annotations

from gateway.run import (
    GatewayRunner,
    _AGENT_PENDING_SENTINEL,
    _INTERRUPT_REASON_STOP,
)
from gateway.run_inbound import GatewayInboundMixin


KEY = "agent:main:telegram:dm:106963"


class _RecordingAgent:
    """Records the interrupt ABI used by ``request_hard_interrupt``."""

    def __init__(self, events, slot_holder):
        self.events = events
        self._slot_holder = slot_holder

    def hard_interrupt(self, message=None, **_kwargs):
        still_held = self._slot_holder() is self
        self.events.append(("interrupt", message, still_held))

    def interrupt(self, reason=None, **_kwargs):
        still_held = self._slot_holder() is self
        self.events.append(("interrupt", reason, still_held))


class _ReapedStore:
    def peek_session_id(self, _key):
        return "sess-106963"

    def _is_session_ended_in_db(self, session_id):
        return session_id == "sess-106963"


def _make_inbound():
    """Real ``GatewayInboundMixin`` via ``GatewayRunner`` (owns eviction)."""
    assert issubclass(GatewayRunner, GatewayInboundMixin)
    runner = object.__new__(GatewayRunner)
    runner._persist_active_agents = lambda: None
    return runner


def _occupy(runner, key, agent):
    state = runner._session_state(key)
    state.turn.agent = agent
    return state


def _current_agent(runner, key):
    state = runner._peek_session_state(key)
    return state.turn.agent if state else None


def test_reaped_eviction_interrupts_in_flight_run_before_slot_release():
    """Positive path: durable reap → ``_hm_evict_reaped_agent`` interrupts first."""
    events = []
    runner = _make_inbound()
    agent = _RecordingAgent(events, lambda: _current_agent(runner, KEY))
    _occupy(runner, KEY, agent)

    orig_release = runner._release_running_agent_state

    def _release(session_key, **kwargs):
        events.append(("release", session_key, _current_agent(runner, KEY) is agent))
        return orig_release(session_key, **kwargs)

    runner._release_running_agent_state = _release
    runner.session_store = _ReapedStore()
    runner._hm_evict_reaped_agent(KEY)

    interrupt_events = [e for e in events if e[0] == "interrupt"]
    assert interrupt_events, (
        "in-flight run never interrupted — reaped-session eviction must "
        "request_hard_interrupt before generation invalidation / slot release"
    )
    assert interrupt_events[0][1] == _INTERRUPT_REASON_STOP
    assert interrupt_events[0][2] is True, (
        "interrupt must be requested while the turn slot still holds the agent"
    )
    release_events = [e for e in events if e[0] == "release"]
    assert release_events, "eviction must still release the running-agent slot"
    assert events.index(interrupt_events[0]) < events.index(release_events[0]), (
        "interrupt MUST happen before _release_running_agent_state empties the slot"
    )
    assert _current_agent(runner, KEY) is None


def test_evict_running_agent_interrupts_live_agent_directly():
    """``_hm_evict_running_agent`` itself must interrupt a live turn agent."""
    events = []
    runner = _make_inbound()
    agent = _RecordingAgent(events, lambda: _current_agent(runner, KEY))
    _occupy(runner, KEY, agent)

    runner._hm_evict_running_agent(KEY, "stale_running_agent_eviction")

    assert events, "in-flight run never interrupted"
    assert events[0][0] == "interrupt"
    assert events[0][1] == _INTERRUPT_REASON_STOP
    assert events[0][2] is True
    assert _current_agent(runner, KEY) is None


def test_fail_open_no_session_state(monkeypatch):
    """No session state for key → no interrupt call; invalidate+release still run."""
    calls = []
    import gateway.run as gr

    monkeypatch.setattr(
        gr,
        "request_hard_interrupt",
        lambda *a, **k: calls.append((a, k)) or True,
    )
    runner = _make_inbound()
    runner._hm_evict_running_agent("missing-key", "reaped_session_eviction")
    assert calls == []
    # Invalidate creates persistent state; release must complete without crash.
    state = runner._peek_session_state("missing-key")
    assert state is not None
    assert state.turn.agent is None


def test_fail_open_none_agent(monkeypatch):
    """``state.turn.agent is None`` → no interrupt; eviction still completes."""
    calls = []
    import gateway.run as gr

    monkeypatch.setattr(
        gr,
        "request_hard_interrupt",
        lambda *a, **k: calls.append((a, k)) or True,
    )
    runner = _make_inbound()
    _occupy(runner, KEY, None)
    runner._hm_evict_running_agent(KEY, "reaped_session_eviction")
    assert calls == []
    assert _current_agent(runner, KEY) is None


def test_fail_open_pending_sentinel(monkeypatch):
    """Pending sentinel is not a real agent — do not interrupt it."""
    calls = []
    import gateway.run as gr

    monkeypatch.setattr(
        gr,
        "request_hard_interrupt",
        lambda *a, **k: calls.append((a, k)) or True,
    )
    runner = _make_inbound()
    _occupy(runner, KEY, _AGENT_PENDING_SENTINEL)
    runner._hm_evict_running_agent(KEY, "stale_running_agent_eviction")
    assert calls == []
    assert _current_agent(runner, KEY) is None
