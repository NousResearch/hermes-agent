"""Regression coverage for routing metadata on LLM lifecycle hooks."""

import pytest

from agent.turn_finalizer import finalize_turn
from tests.agent.test_turn_context import _FakeAgent as TurnContextAgent, _build
from tests.agent.test_turn_finalizer_final_response_persistence import (
    FakeAgent as TurnFinalizerAgent,
)


@pytest.mark.parametrize("sender_id, chat_id", [("user-123", "chat-456"), (None, None)])
def test_llm_hooks_receive_gateway_routing_metadata(monkeypatch, sender_id, chat_id):
    events = {}
    calls = []

    def capture(hook_name, **payload):
        calls.append(hook_name)
        events[hook_name] = payload
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", capture)

    pre_agent = TurnContextAgent()
    pre_agent.platform = "feishu"
    pre_agent._user_id = sender_id
    pre_agent._chat_id = chat_id
    _build(pre_agent)

    post_agent = TurnFinalizerAgent()
    post_agent.platform = "feishu"
    post_agent._user_id = sender_id
    post_agent._chat_id = chat_id
    finalize_turn(
        post_agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Done."},
        ],
        conversation_history=[],
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    expected = {
        "platform": "feishu",
        "sender_id": sender_id or "",
        "chat_id": chat_id or "",
    }
    assert calls.count("pre_llm_call") == 1
    assert calls.count("post_llm_call") == 1
    assert {key: events["pre_llm_call"][key] for key in expected} == expected
    assert {key: events["post_llm_call"][key] for key in expected} == expected
