"""Cross-session agent reuse must not carry the previous conversation's todo list.

``_lookup_cached_agent`` deliberately reuses a cached agent when the cached
``session_id`` differs from the current one (same ``session_key``) so the prompt
cache survives a session switch. ``_todo_store`` hangs off the agent object and
``agent/turn_context.py::_hydrate_from_history`` only hydrates when the store is
empty, so a non-empty store from the previous conversation would both survive the
switch and block hydration from the new session's own history.
"""

import threading
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner
from tools.todo_tool import TodoStore

SIG = "sig-1"
TODOS = [{"id": "1", "content": "previous conversation task", "status": "pending"}]


def _turn(session_id):
    ctx = SimpleNamespace(session_key="key", session_id=session_id, _interrupt_depth=0)
    runner = SimpleNamespace(_init_cached_agent_for_turn=lambda agent, depth: None)
    return TurnRunner(runner, ctx)


def _cached_agent():
    agent = SimpleNamespace(max_iterations=1, _todo_store=TodoStore())
    agent._todo_store.write(TODOS)
    return agent


def _lookup(turn, agent, cached_session_id):
    cache = {"key": (agent, SIG, 7, cached_session_id)}
    return turn._lookup_cached_agent(
        SIG, threading.Lock(), cache, max_iterations=1,
        peek_sid=cached_session_id, dead=False, msg_count=7,
    )


def test_cross_session_reuse_clears_the_todo_store():
    agent = _cached_agent()
    found = _lookup(_turn("sid-new"), agent, "sid-old")

    assert found.reused is True
    assert found.agent is agent
    assert agent._todo_store.has_items() is False


def test_same_session_reuse_keeps_the_todo_store():
    agent = _cached_agent()
    found = _lookup(_turn("sid-same"), agent, "sid-same")

    assert found.reused is True
    assert agent._todo_store.read() == [
        {"id": "1", "content": "previous conversation task", "status": "pending"}
    ]
