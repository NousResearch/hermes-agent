"""Old announced-action recovery must not inherit the new suggestion veto."""
from unittest.mock import patch

import pytest

from tests.agent.test_degenerate_final_recovery import loop_agent, _final, _tool_round, _run


@pytest.mark.parametrize("prefix", [
    "Deployment is blocked, but local inspection is authorized.",
    "The earlier unavailable fixture is now restored.",
    "Yesterday we were waiting for approval. Local inspection is authorized.",
    'The log contained "waiting for approval"; no approval is needed for reading it.',
    'Example: `status = "blocked"`. Local read access is confirmed.',
])
@pytest.mark.parametrize("promoted", [False, True])
def test_safe_tail_action_survives_unrelated_blocker(loop_agent, prefix, promoted):
    loop_agent.valid_tool_names = {"web_search"}
    text = prefix + (" I need to inspect the local trace." if promoted
                     else " Next, I will inspect the local trace.")
    with patch("agent.turn_final_response.answer_in_reasoning_capability", return_value=promoted), patch.object(
        loop_agent, "_extract_reasoning", return_value=text if promoted else None,
    ):
        result = _run(loop_agent, [_tool_round("inspect"), _final("" if promoted else text),
                                   _final("Trace inspected.")])
    assert result["final_response"] == "Trace inspected."
    assert loop_agent.client.chat.completions.create.call_count == 3


@pytest.mark.parametrize("text", [
    "Next, I will wait for your decision.",
    "Inspection complete. Next, I will wait for your approval.",
    "Next, I will await your permission.",
])
def test_permission_wait_tail_is_not_action(loop_agent, text):
    loop_agent.valid_tool_names = {"web_search"}
    result = _run(loop_agent, [_tool_round("inspect"), _final(text), _final("Unexpected continuation.")])
    assert result["final_response"] == text
    assert loop_agent.client.chat.completions.create.call_count == 2


def test_ack_helper_receives_original_response(loop_agent):
    loop_agent.valid_tool_names = {"web_search"}
    text = "<think>blocked unavailable</think>I will inspect the repository."
    with patch.object(loop_agent, "_looks_like_codex_intermediate_ack",
                      wraps=loop_agent._looks_like_codex_intermediate_ack) as acknowledgement:
        result = _run(loop_agent, [_final(text), _final("Inspection complete.")])
    assert result["final_response"] == "Inspection complete."
    assert acknowledgement.call_args_list[0].kwargs["assistant_content"] == text
