import pytest
from tests.agent.test_iteration_budget_warning import _agent

@pytest.mark.parametrize("owned", [True, False])
def test_kanban_checkpoint_is_only_for_dispatcher_owner(tmp_path, monkeypatch, owned):
    from agent.delegation_context import non_dispatcher_owned_context
    from contextlib import nullcontext
    from agent.turn_iteration_prep import prepare_iteration
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_checkpoint")
    with nullcontext() if owned else non_dispatcher_owned_context():
        agent = _agent(tmp_path, monkeypatch, "null")
        for _ in range(4):
            agent.iteration_budget.consume()
        messages = [{"role": "user", "content": "work"},
                    {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "t", "content": "verified artifact"}]
        prepare_iteration(agent, messages=messages, api_call_count=4)
        assert ("kanban_complete" in messages[-1]["content"]) is owned
        assert agent.iteration_budget.remaining == 0
