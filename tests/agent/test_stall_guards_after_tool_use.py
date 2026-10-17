"""Stall guard after tool use (#122119).

A turn that already ran tools and then stops on a visible line announcing its next action
("I can proceed with the first fix.", "Let me write the parse script to a file first, then run
it.") used to end as if it were the answer: the narrow ``let me now`` tail missed the phrasing,
and the intent-ack fallback never applies once the history holds a tool result. These tests run
the real loop with a mocked client and check that such a line gets the bounded continuation nudge,
while real answers, sign-offs and conditional offers still end the turn.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE


@pytest.mark.parametrize("text", [
    "Let me write the parse script to a file first, then run it.",
    "I found the config. Now I need to check the gateway log.",
    "The build failed on step 3. Let me look at the stack trace:",
    "I'll run the migration and verify the row counts.",
    "Next, I'll patch the router config.",
])
def test_visible_detector_catches_announced_actions(text):
    from agent.agent_runtime_helpers_stall_intent import visible_reply_announces_action
    assert visible_reply_announces_action(text)


@pytest.mark.parametrize("text", [
    "The answer is 42.",
    "Done - the file is saved at /tmp/x.txt. Let me know if you want changes.",
    "I'll wait for your confirmation before deleting anything.",
    "Fixed the bug in parser.py; all 12 tests pass.",
    "I can run the migration if you want.",
    "Should I proceed?",
    "Let me know if anything else breaks.",
    "I'll proceed once you confirm the branch name.",
    "Summary:\n- A\n- B\nI'll be here if you need more.",
    "I need to note that the backup is from yesterday.",
    "",
])
def test_visible_detector_ignores_final_answers(text):
    from agent.agent_runtime_helpers_stall_intent import visible_reply_announces_action
    assert not visible_reply_announces_action(text)


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_degenerate_final_recovery)."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent.model = "local-test-model"
        agent.valid_tool_names = {"terminal", "web_search"}
        return agent


def _run(agent, stages):
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value="ok"),
    ):
        return agent.run_conversation("parse the export")


def _tool_round(call_id):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="web_search", arguments='{"query": "x"}', call_id=call_id)],
    )


def _final(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop")


def _nudged_after(result, stall):
    contents = [m.get("content") for m in result["messages"]]
    return contents.index(stall) + 1 == contents.index(_CODEX_ACK_CONTINUATION_NUDGE)


_PLAN_STALL = "Let me write the parse script to a file first, then run it."
_PROCEED_STALL = (
    "The only pending change is the cursor write in sync.py at line 160. "
    "I can proceed with the first fix using the terminal tool."
)


def test_can_proceed_stall_after_tool_round_is_nudged(loop_agent):
    # Default scope: intent-ack continuation is off on chat_completions, so only the stall
    # guard can catch this line.
    loop_agent._intent_ack_continuation = "auto"

    result = _run(loop_agent, [_tool_round("c1"), _final(_PROCEED_STALL), _final("Fixed and tested.")])

    assert loop_agent.client.chat.completions.create.call_count == 3
    assert result["final_response"] == "Fixed and tested."
    assert _nudged_after(result, _PROCEED_STALL)


def test_conditional_can_proceed_after_tool_round_ends_the_turn(loop_agent):
    loop_agent._intent_ack_continuation = "auto"
    offer = "I found the cause. I can proceed with the fix once you confirm the branch name."

    result = _run(loop_agent, [_tool_round("c1"), _final(offer)])

    assert loop_agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == offer


def test_visible_plan_stall_after_tool_round_is_nudged_when_opted_in(loop_agent):
    # The ack fallback never fires once the history holds a tool result; the visible plan-tail
    # check covers that case for opted-in models.
    loop_agent._intent_ack_continuation = ["local-test"]

    result = _run(loop_agent, [_tool_round("c1"), _final(_PLAN_STALL), _final("Parsed 120 rows.")])

    assert loop_agent.client.chat.completions.create.call_count == 3
    assert result["final_response"] == "Parsed 120 rows."
    assert _nudged_after(result, _PLAN_STALL)


def test_visible_plan_stall_not_nudged_for_unlisted_model(loop_agent):
    loop_agent._intent_ack_continuation = ["some-other-model"]

    result = _run(loop_agent, [_tool_round("c1"), _final(_PLAN_STALL)])

    assert loop_agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == _PLAN_STALL


def test_visible_plan_stall_nudges_are_capped_at_two(loop_agent):
    loop_agent._intent_ack_continuation = True

    result = _run(loop_agent, [_tool_round("c1"), _final(_PLAN_STALL), _final(_PLAN_STALL), _final(_PLAN_STALL)])

    assert loop_agent.client.chat.completions.create.call_count == 4
    assert result["final_response"] == _PLAN_STALL


def test_genuine_answer_after_tool_round_is_not_nudged(loop_agent):
    loop_agent._intent_ack_continuation = True
    answer = "Parsed 120 rows. Let me know if you want a CSV."

    result = _run(loop_agent, [_tool_round("c1"), _final(answer)])

    assert loop_agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == answer
