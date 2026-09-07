"""Iteration checkpoints preserve the transcript and the hard budget."""
from copy import deepcopy

import pytest


def _agent(tmp_path, monkeypatch, ratio):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"agent:\n  budget_warning_ratio: {ratio}\n", encoding="utf-8"
    )
    from run_agent import AIAgent
    from hermes_state import SessionDB
    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


@pytest.mark.parametrize("ratio", ["null", "true", "0", "1", ".nan", "junk", "0.75"])
def test_checkpoint_is_opt_in_and_does_not_rewrite_prior_turns(tmp_path, monkeypatch, ratio):
    from agent.turn_iteration_prep import prepare_iteration
    agent = _agent(tmp_path, monkeypatch, ratio)
    for _ in range(3):
        agent.iteration_budget.consume()
    messages = [{"role": "user", "content": "work"},
                {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t", "content": "result"}]
    original = deepcopy(messages)
    prepare_iteration(agent, messages=messages, api_call_count=3)
    assert messages[:-1] == original[:-1]
    assert (messages[-1]["content"] != "result") == (ratio == "0.75")
    snapshot = deepcopy(messages)
    prepare_iteration(agent, messages=messages, api_call_count=3)
    assert messages == snapshot
    assert agent.iteration_budget.used == 3


@pytest.mark.parametrize("content", ["result", [{"type": "text", "text": "result"}]])
def test_checkpoint_rearms_per_turn_with_tools_still_available(tmp_path, monkeypatch, content):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration
    agent = _agent(tmp_path, monkeypatch, "0.75")
    for _ in range(2):
        _reset_per_turn_agent_state(agent)
        for _ in range(3):
            agent.iteration_budget.consume()
        messages = [{"role": "user", "content": "work"},
                    {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "t", "content": deepcopy(content)}]
        from agent.tool_executor import _flush_session_db_after_tool_progress
        assert _flush_session_db_after_tool_progress(agent, messages, stage="checkpoint")
        persisted = agent._session_db.get_messages(agent.session_id)
        assert "3 of 4" in str(persisted[-1]["content"])
        snapshot = deepcopy(messages)
        prepare_iteration(agent, messages=messages, api_call_count=3)
        assert messages == snapshot
        assert "3 of 4" in str(messages[-1]["content"])
        assert agent.iteration_budget.consume()
        assert not agent.iteration_budget.consume()
