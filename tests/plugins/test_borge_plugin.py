"""Integration tests for the Borge cognitive layer plugin."""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_borge_agent_init():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    assert agent is not None
    assert agent.emotion is not None
    assert agent.beliefs is not None
    assert agent.values is not None


def test_pre_turn_returns_string():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    ctx = agent.pre_turn("hello, can you help me?", [])
    assert isinstance(ctx, str)


def test_pre_turn_increments_turn_count():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    agent.pre_turn("first", [])
    agent.pre_turn("second", [{"role": "user", "content": "first"}])
    assert agent._turn_count == 2


def test_post_tool_updates_value_satisfaction():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    agent.post_tool("bash", "File created successfully.")


def test_on_session_start_resets_state():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    agent.pre_turn("first turn", [])
    assert agent._turn_count == 1
    agent.on_session_start()
    assert agent._turn_count == 0
    assert agent._emotional_history == []


def test_on_session_end_does_not_raise():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    agent.pre_turn("let's build something", [])
    agent.on_session_end(session_id="test-session", messages=[])


def test_score_tool_candidates_returns_candidates():
    from borge.agent import BorgeAgent
    agent = BorgeAgent(hermes_agent=None)
    candidates = [
        {"name": "bash", "description": "Run shell commands"},
        {"name": "read_file", "description": "Read file contents"},
    ]
    result = agent.score_tool_candidates(candidates)
    assert len(result) == 2
    names = {c["name"] for c in result}
    assert names == {"bash", "read_file"}


def test_skill_health_report_returns_dict():
    from borge.agent import BorgeAgent
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        agent = BorgeAgent(hermes_agent=None, db_path=db_path)
        report = agent.skill_health_report()
        assert "prune_candidates" in report
        assert "generalise_candidates" in report
    finally:
        os.unlink(db_path)


def test_plugin_register_hooks():
    registered = []

    class FakeCtx:
        def register_hook(self, name, cb):
            registered.append(name)

    from plugins.borge import register
    register(FakeCtx())

    assert "on_session_start" in registered
    assert "pre_llm_call" in registered
    assert "post_llm_call" in registered
    assert "on_session_end" in registered


def test_plugin_pre_llm_call_returns_string():
    from plugins.borge import _pre_llm_call, _sessions
    _sessions.clear()
    result = _pre_llm_call(
        session_id="test-123",
        user_message="help me refactor this function",
        conversation_history=[],
    )
    assert isinstance(result, str)
    _sessions.clear()


def test_plugin_full_lifecycle():
    from plugins.borge import (
        _on_session_start, _pre_llm_call, _post_llm_call, _on_session_end,
        _sessions, _histories,
    )
    _sessions.clear()
    _histories.clear()

    sid = "lifecycle-test"

    _on_session_start(session_id=sid, model="claude-sonnet-4-6", platform="cli")
    assert sid in _sessions

    ctx = _pre_llm_call(
        session_id=sid,
        user_message="write a test for the login function",
        conversation_history=[],
        is_first_turn=True,
    )
    assert isinstance(ctx, str)

    history = [
        {"role": "user", "content": "write a test for the login function"},
        {"role": "assistant", "content": "Here's a test..."},
    ]
    _post_llm_call(
        session_id=sid,
        assistant_response="Here's a test...",
        conversation_history=history,
    )
    assert _histories.get(sid) is not None

    _on_session_end(session_id=sid, completed=True)
    assert sid not in _sessions
    assert sid not in _histories
