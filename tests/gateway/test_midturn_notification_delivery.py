"""Which running agent the gateway's mid-turn notice seam reaches.

``GatewayNotificationsMixin._midturn_notice_agent`` is the gateway half of the mid-turn seam
(``AIAgent.notify_between_tool_calls``): the routing predicate in ``_deliver_notice_mid_turn``
(covered in ``tests/agent/test_midturn_notifications.py``) hands it a source and expects the agent
of the turn running for that session, or ``None`` so the notice keeps its admit-as-a-turn path.

Both ownership models are pinned here — the session-authority claim (its adopted agent while that
generation still executes) and the classic running-agent slot — because a refactor of either one
makes every notice silently queue a whole extra turn instead of reaching the live one.
"""

from types import SimpleNamespace

import pytest

from gateway.run import _AGENT_PENDING_SENTINEL
from gateway.run_notifications import GatewayNotificationsMixin


def _runner(*, entry=SimpleNamespace(session_id="sid-1"), state=None):
    runner = SimpleNamespace()
    runner._session_key_for_source = lambda source: "agent:main:telegram:dm:c1"
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda key: entry)
    runner._peek_session_state = lambda key: state
    return runner


def _bind(runner):
    return GatewayNotificationsMixin._midturn_notice_agent.__get__(runner)


@pytest.fixture()
def authority_calls(monkeypatch):
    """Route the authority path at a fake ``active_authority``/``_receiving_agent`` pair."""
    import gateway.session_authorities as authorities
    import gateway.session_queued_controls as controls

    calls = SimpleNamespace(receiving=[], authority=None, agent=None)
    monkeypatch.setattr(authorities, "active_authority", lambda runner: calls.authority, raising=False)

    def _receiving(authority, session_id, generation):
        calls.receiving.append((session_id, generation))
        return calls.agent

    monkeypatch.setattr(controls, "_receiving_agent", _receiving, raising=False)
    return calls


def test_authority_path_returns_the_adopted_agent(authority_calls):
    agent = SimpleNamespace(name="adopted")
    generation = object()
    authority_calls.authority = SimpleNamespace(executing={"sid-1": generation})
    authority_calls.agent = agent

    assert _bind(_runner())(SimpleNamespace()) is agent
    assert authority_calls.receiving == [("sid-1", generation)]


def test_authority_without_an_executing_generation_is_idle(authority_calls):
    authority_calls.authority = SimpleNamespace(executing={})

    assert _bind(_runner())(SimpleNamespace()) is None


def test_authority_without_a_stored_entry_is_idle(authority_calls):
    authority_calls.authority = SimpleNamespace(executing={"sid-1": object()})

    assert _bind(_runner(entry=None))(SimpleNamespace()) is None


def test_classic_slot_returns_the_running_agent(authority_calls):
    authority_calls.authority = None
    agent = SimpleNamespace(name="running")
    state = SimpleNamespace(turn=SimpleNamespace(agent=agent))

    assert _bind(_runner(state=state))(SimpleNamespace()) is agent


def test_classic_slot_pending_sentinel_is_not_an_agent(authority_calls):
    authority_calls.authority = None
    state = SimpleNamespace(turn=SimpleNamespace(agent=_AGENT_PENDING_SENTINEL))

    assert _bind(_runner(state=state))(SimpleNamespace()) is None
