"""A delegated child returns exactly once and cannot resume after its iteration budget runs out,
so its checkpoint notice must tell it to checkpoint artifacts and finish the summary now — not the
session-shaped "then continue the task", which is meaningless for a child. Regression for #124291.
"""
from agent.delegation_context import delegated_child_context
from agent.turn_iteration_prep import prepare_iteration
from tests.agent.test_iteration_budget_warning import _agent


def _messages():
    return [
        {"role": "user", "content": "work"},
        {"role": "assistant", "tool_calls": [
            {"id": "t", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "t", "content": "verified artifact"},
    ]


def test_child_gets_child_shaped_wording_not_session_wording(tmp_path, monkeypatch):
    with delegated_child_context():
        agent = _agent(tmp_path, monkeypatch, "0.75")
        try:
            for _ in range(3):
                agent.iteration_budget.consume()
            messages = _messages()
            prepare_iteration(agent, messages=messages, api_call_count=3)
            notice = str(messages[-1]["content"])
            assert "write any durable artifacts to disk now" in notice
            assert "then continue the task" not in notice
        finally:
            agent._session_db.close()


def test_ordinary_session_keeps_session_shaped_wording(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, "0.75")
    try:
        for _ in range(3):
            agent.iteration_budget.consume()
        messages = _messages()
        prepare_iteration(agent, messages=messages, api_call_count=3)
        notice = str(messages[-1]["content"])
        assert "then continue the task" in notice
        assert "write any durable artifacts to disk now" not in notice
    finally:
        agent._session_db.close()
