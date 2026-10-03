"""An explicit wait for the user is a final answer, not an unexecuted tool plan."""
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import (
    promoted_reasoning_announces_action,
    trailing_continue_intent,
)
from tests.agent.test_degenerate_final_recovery import loop_agent, _final, _tool_round


@pytest.mark.parametrize("text", [
    "Next, I will wait for your decision.",
    "Completed: four tests passed. Options A and B are ready. "
    "No more action until you choose. Next, I will wait for your decision.",
    "I will now wait for your approval.",
    "Now I'll await your reply.",
    "Let me now wait for your input.",
    "Next: I await your choice.",
])
@pytest.mark.parametrize("detect", [trailing_continue_intent, promoted_reasoning_announces_action])
def test_user_handoff_is_not_an_action_plan(detect, text):
    assert not detect(text)


@pytest.mark.parametrize("text", [
    "Next, I will check the logs.",
    "Next, I will wait for the background process.",
    "I will now check the logs and wait for your decision.",
    "I will wait for your decision. Next, I will run the tests.",
])
@pytest.mark.parametrize("detect", [trailing_continue_intent, promoted_reasoning_announces_action])
def test_real_action_tail_still_detected(detect, text):
    assert detect(text)


@pytest.mark.parametrize("separator", [" — ", " – ", ": "])
@pytest.mark.parametrize("detect", [trailing_continue_intent, promoted_reasoning_announces_action])
def test_user_handoff_after_clause_separator(detect, separator):
    assert not detect("Options ready" + separator + "Next, I will wait for your decision.")
    assert detect("Options ready" + separator + "Next, I will check the logs.")
    assert detect("I will now check the logs" + separator + "I will wait for your decision.")


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("promoted", [True, False])
def test_tool_turn_ends_at_user_decision_without_synthetic_authority(loop_agent, enabled, promoted):
    agent = loop_agent
    agent._stall_guards = enabled
    agent._intent_ack_continuation = False
    agent.valid_tool_names = {"web_search"}
    text = "Checks completed. A: deploy now. B: keep it unchanged. Next, I will wait for your decision."
    final = _final("" if promoted else text)
    if promoted:
        final.choices[0].message.reasoning_content = text
    agent.client.chat.completions.create.side_effect = [_tool_round("evidence"), final, final, final]
    prompt = "Read the evidence once, offer A/B, then wait for my choice. Do not check again or deploy."
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value='{"task_complete":true}') as tool,
    ):
        result = agent.run_conversation(prompt)
    assert result["final_response"] == text
    assert agent.client.chat.completions.create.call_count == 2
    assert tool.call_count == 1
    assert [m["content"] for m in result["messages"] if m.get("role") == "user"] == [prompt]
