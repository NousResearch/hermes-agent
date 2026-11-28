"""A contentless reasoning-only clean stop is a decode collapse, not an answer.

Observed on NVIDIA-hosted Kimi K3: right after a refused tool result the provider returned
``finish_reason=stop`` with empty content, no tool call and reasoning made of 32 copies of
token id 0 (``!``). The reasoning-only clean-stop promotion returned that run as the final
answer and the turn ended "complete" mid-task, bypassing the empty-response ladder.

Promotion exists for parsers that file a real answer as reasoning, so a fragment-sized symbol
answer (``✓``) must still be promoted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

COLLAPSE = "!" * 32
REFUSED = '{"error": "Refusing to overwrite report.json: file not read this task.", "stale_write_blocked": true}'


@pytest.fixture()
def agent():
    """Reasoning echo-back route (reasoning_content replayed), as for Kimi K3."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://api.deepseek.com/v1",
                        model="deepseek-reasoner", provider="deepseek",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"write_file", "read_file"}
    return agent


def _run(agent, stages, user_message):
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.turn_recovery.interruptible_backoff_sleep", return_value=None),
        patch("agent.turn_empty_response.interruptible_backoff_sleep", return_value=None),
        patch("model_tools.handle_function_call", return_value=REFUSED),
    ):
        return agent.run_conversation(user_message)


def _tool(name, call_id):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(content="", finish_reason="tool_calls", reasoning_content="Write it.",
                          tool_calls=[_mock_tool_call(name=name, arguments='{"path": "report.json"}', call_id=call_id)])


def _reasoning_stop(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content="", finish_reason="stop", reasoning_content=text)


def _final(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop", reasoning_content="Done.")


def test_contentless_reasoning_stop_recovers_instead_of_ending_turn(agent):
    result = _run(agent, [
        _tool("write_file", "write_file:0"), _reasoning_stop(COLLAPSE),
        _tool("read_file", "read_file:0"), _final("Report updated and verified."),
    ], "update report.json")

    assert result["final_response"].startswith("Report updated and verified.")
    assert agent.client.chat.completions.create.call_count == 4
    assert all(m.get("content") != COLLAPSE for m in result["messages"])
    # The recovery request still pairs the refused-write result with its call id.
    sent = agent.client.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert [(m["tool_call_id"], m["content"]) for m in sent if m.get("role") == "tool"] == [("write_file:0", REFUSED)]


@pytest.mark.parametrize("answer", ["✓", "!!!", "Paris."])
def test_fragment_sized_reasoning_answer_is_still_promoted(agent, answer):
    result = _run(agent, [_reasoning_stop(answer)], "did it work?")

    assert result["final_response"] == answer
    assert agent.client.chat.completions.create.call_count == 1
