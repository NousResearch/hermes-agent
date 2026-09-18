"""Checkpoint eligibility follows explicit opt-in or dispatcher ownership."""
from contextlib import nullcontext
from copy import deepcopy

import pytest
from tests.agent.test_iteration_budget_warning import _agent


@pytest.mark.parametrize("ratio,scope,expected,kanban_notice", [
    *[(ratio, "ordinary", False, False) for ratio in ("null", "true", "0", "1", ".nan", "junk")],
    ("0.75", "ordinary", True, False),
    ("null", "owner", True, True),
    ("null", "non-owner", False, False),
    ("null", "child", False, False),
    ("0.75", "child", True, False),
    ("null", "no-completion-tool", False, False),
])
def test_checkpoint_requires_opt_in_or_dispatcher_completion_scope(
    tmp_path, monkeypatch, ratio, scope, expected, kanban_notice
):
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    from agent.turn_iteration_prep import prepare_iteration

    if scope != "ordinary":
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_checkpoint")
    contexts = {"non-owner": non_dispatcher_owned_context, "child": delegated_child_context}
    with contexts.get(scope, nullcontext)():
        agent = _agent(tmp_path, monkeypatch, ratio)
        try:
            if scope == "no-completion-tool":
                agent.valid_tool_names.discard("kanban_complete")
            for _ in range(3):
                agent.iteration_budget.consume()
            messages = [{"role": "user", "content": "work"},
                        {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "t", "content": "verified artifact"}]
            original = deepcopy(messages)
            prepare_iteration(agent, messages=messages, api_call_count=3)
            assert messages[:-1] == original[:-1]
            assert (messages[-1]["content"] != "verified artifact") is expected
            assert ("kanban_complete" in messages[-1]["content"]) is kanban_notice
            snapshot = deepcopy(messages)
            prepare_iteration(agent, messages=messages, api_call_count=3)
            assert messages == snapshot
            assert agent.iteration_budget.remaining == 1
        finally:
            agent._session_db.close()
