"""Reasoning promoted on a reasoning-only clean stop is returned, never persisted as a reply.

A clean ``stop`` with empty content and reasoning text is promoted to ``final_response`` (the
vLLM nemotron parser files the whole answer as reasoning; re-running the empty-response ladder
re-bills the prompt). The promoted text must NOT become the assistant row's ordinary ``content``:
chain-of-thought stored as content is indistinguishable from a real reply on every history
surface (#111761). The row keeps ``content`` empty and carries the text as the ``api_content``
sidecar, so the next request still replays the answer byte-identically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

REASONING = "嗯，长度合适。Let me check the file first."


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            provider="deepseek",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _run(agent, responses, user_message="hello", conversation_history=None):
    agent.client.chat.completions.create.side_effect = list(responses)
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(user_message, conversation_history=conversation_history)


def test_promoted_reasoning_is_returned_but_persisted_row_keeps_content_empty(loop_agent):
    from tests.agent.test_run_agent import _mock_response

    result = _run(loop_agent, [_mock_response(content="", finish_reason="stop", reasoning_content=REASONING)])

    # Return contract from the parser-compat fix survives: one call, the reasoning is the answer.
    assert result["final_response"] == REASONING
    assert result["api_calls"] == 1

    row = result["messages"][-1]
    assert row["role"] == "assistant"
    assert not row.get("content")  # never chain-of-thought as an ordinary reply
    assert row["reasoning"] == REASONING
    assert row["api_content"] == REASONING

    # The next request still replays the promoted text as the assistant's own turn.
    _run(loop_agent, [_mock_response(content="Second answer.", finish_reason="stop")],
         user_message="next", conversation_history=result["messages"])
    sent = loop_agent.client.chat.completions.create.call_args.kwargs["messages"]
    assistant_rows = [m for m in sent if m.get("role") == "assistant"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert assistant_rows[0]["content"] == REASONING
    assert "api_content" not in assistant_rows[0]


def test_stall_guard_interim_row_carries_promoted_text_as_sidecar(loop_agent):
    """Promoted reasoning that tails on an announced next action trips the stall guard; the interim
    row it appends must follow the same shape as the final row — ``content`` empty, the promoted
    text in ``api_content`` — so the continuation request replays a real assistant turn, not an
    empty one (#111761)."""
    from tests.agent.test_run_agent import _mock_response

    stalled = "The user wants the file contents. Let me now read the file."
    loop_agent.valid_tool_names = {"read_file"}
    loop_agent._stall_guards = True

    result = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=stalled),
        _mock_response(content="Here is the file.", finish_reason="stop"),
    ])

    assert result["final_response"] == "Here is the file."
    interim = [m for m in result["messages"] if m.get("role") == "assistant"][0]
    assert not interim.get("content")
    assert interim["reasoning"] == stalled
    assert interim["api_content"] == stalled

    # The continuation request carried the promoted text as the interim assistant turn.
    second_call = loop_agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    interim_wire = [m for m in second_call if m.get("role") == "assistant"][0]
    assert interim_wire["content"] == stalled
    assert "api_content" not in interim_wire


# ── planning-monologue stall: promoted reasoning must not fake a completion ──────────────────

PLAN_TAILS = [
    "Let me batch the terminal calls and run them in parallel.",
    "...Let me load the doctrine skill first, then run checks.",
    "Attempting to read and process the file. Initial hypothesis: the config is stale. I need to check the log.",
]


@pytest.mark.parametrize("tail", PLAN_TAILS)
def test_planning_tail_reasoning_only_stop_with_tools_runs_continuation_not_completion(loop_agent, tail):
    """Tools offered, zero tool calls, and the promoted reasoning ENDS on a first-person plan
    ("Let me batch...", "I need to check...") — the verbatim tails from the #111761 thread. This
    is a stalled model, not an answer: the stall-guard continuation must run (bounded by the same
    cap) instead of returning the monologue as a 'complete' final response."""
    from tests.agent.test_run_agent import _mock_response

    loop_agent.valid_tool_names = {"terminal", "read_file"}
    loop_agent._stall_guards = True

    result = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=tail),
        _mock_response(content="Ran the checks; all green.", finish_reason="stop"),
    ])

    assert result["api_calls"] == 2
    assert result["final_response"] == "Ran the checks; all green."
    interim = [m for m in result["messages"] if m.get("role") == "assistant"][0]
    assert not interim.get("content")
    assert interim["api_content"] == tail  # interim row keeps the sidecar shape


def test_planning_tail_stall_is_bounded_by_the_continuation_cap(loop_agent):
    """A model that never acts is nudged at most twice; the third planning-only stop is promoted
    so the turn still ends instead of looping."""
    from tests.agent.test_run_agent import _mock_response

    loop_agent.valid_tool_names = {"terminal"}
    loop_agent._stall_guards = True
    tail = PLAN_TAILS[0]

    result = _run(loop_agent, [
        _mock_response(content="", finish_reason="stop", reasoning_content=tail),
        _mock_response(content="", finish_reason="stop", reasoning_content=tail),
        _mock_response(content="", finish_reason="stop", reasoning_content=tail),
        _mock_response(content="NEVER REACHED", finish_reason="stop"),
    ])

    assert result["api_calls"] == 3
    assert result["final_response"] == tail


def test_genuine_reasoning_only_answer_with_tools_still_promotes_on_first_call(loop_agent):
    """The parser-compat contract survives: reasoning that states an answer (no trailing plan) is
    returned on the first call even with tools offered, and mentioning a plan BEFORE the answer
    does not count as a stall."""
    from tests.agent.test_run_agent import _mock_response

    loop_agent.valid_tool_names = {"terminal", "read_file"}
    loop_agent._stall_guards = True

    for answer in ("The answer is 42.", "Let me check the arithmetic. 6 times 7 is 42, so the answer is 42."):
        loop_agent.client.chat.completions.create.reset_mock()
        result = _run(loop_agent, [
            _mock_response(content="", finish_reason="stop", reasoning_content=answer),
            _mock_response(content="NEVER REACHED", finish_reason="stop"),
        ])
        assert result["api_calls"] == 1
        assert result["final_response"] == answer
