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
            base_url="http://127.0.0.1:8000/v1",
            model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            provider="vllm",
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
        patch("model_tools.handle_function_call", return_value="ok"),
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
    interim = next(m for m in result["messages"] if m.get("role") == "assistant")
    assert not interim.get("content")
    assert interim["reasoning"] == stalled
    assert interim["api_content"] == stalled

    # The continuation request carried the promoted text as the interim assistant turn.
    second_call = loop_agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    interim_wire = next(m for m in second_call if m.get("role") == "assistant")
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
    interim = next(m for m in result["messages"] if m.get("role") == "assistant")
    assert not interim.get("content")
    assert interim["api_content"] == tail  # interim row keeps the sidecar shape


def test_planning_tail_stall_is_bounded_by_the_continuation_cap(loop_agent):
    """A model that never acts is nudged at most twice; the third planning-only stop is promoted
    so the turn still ends — now with the resume hint attached instead of a silent stop (v4)."""
    from agent.turn_final_response import _STALL_EXHAUSTED_HINT
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
    # v4: the cap still bounds a model that never acts — the turn ends — and the reply now
    # carries the resume hint instead of silently stopping on the plan tail.
    assert result["final_response"].startswith(tail)
    assert _STALL_EXHAUSTED_HINT in result["final_response"]


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


# ── Anthropic summarized thinking is not an answer ──────────────────────────────────────────

SUMMARY = "The user asks for 391's factors; checking divisibility by 17 gives 23."


def test_anthropic_signed_thinking_only_stop_continues_instead_of_promoting(loop_agent):
    """Claude 4.7+/5.x thinking arrives as a ``display: "summarized"`` block written by a separate
    summarizer model. A thinking-only ``end_turn`` must take the empty-response continuation, not
    return the summary as the answer; the next call's text is the answer."""
    from types import SimpleNamespace

    def _anthropic(*blocks):
        return SimpleNamespace(content=list(blocks), stop_reason="end_turn", stop_details=None,
                               model="claude-opus-5-5", usage=None)

    loop_agent.api_mode = "anthropic_messages"
    loop_agent.provider = "anthropic"
    thinking_only = _anthropic(SimpleNamespace(type="thinking", thinking=SUMMARY, signature="sig-abc"))
    answer = _anthropic(SimpleNamespace(type="text", text="391 = 17 x 23."))
    with (
        patch.object(loop_agent, "_interruptible_api_call", side_effect=[thinking_only, answer]),
        patch.object(loop_agent, "_interruptible_streaming_api_call", side_effect=[thinking_only, answer]),
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("factor 391")

    assert result["final_response"] == "391 = 17 x 23."
    assert result["api_calls"] == 2


def test_native_claude_carrier_thinking_only_stop_continues_instead_of_promoting(loop_agent):
    """Same contract for a chat_completions plugin replaying native Claude turns through a
    ``<provider>.native_assistant`` reasoning_details carrier (claude-subscription-directsdk)."""
    from tests.agent.test_run_agent import _mock_response

    carrier = [{"type": "claude-subscription-directsdk-experimental.native_assistant", "version": 1,
                "messages": [{"role": "assistant", "content": [
                    {"type": "thinking", "thinking": SUMMARY, "signature": "sig-abc"}]}]}]
    result = _run(loop_agent, [
        _mock_response(content=None, finish_reason="stop", reasoning_content=SUMMARY, reasoning_details=carrier),
        _mock_response(content="391 = 17 x 23.", finish_reason="stop"),
    ])

    assert result["final_response"] == "391 = 17 x 23."
    assert result["api_calls"] == 2


# ── v4: the continuation cap is CONSECUTIVE — real progress refreshes it ────────────────────


def test_continuation_budget_refreshes_after_real_tool_progress(loop_agent):
    """Replays the user-reported mid-task silent stop: three plan-tail stops, each preceded by
    real tool work. Before v4 the third stop ran out the shared cap and the turn ended on a
    dangling plan tail ("...and downloading:"); with the consecutive budget every stop that
    follows progress is nudged, and only a model that stalls WITHOUT acting runs out."""
    from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    loop_agent.valid_tool_names = {"terminal"}
    loop_agent._stall_guards = True

    def tool_round(cid):
        return _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call(name="terminal", arguments="{}", call_id=cid)],
        )

    def plan_tail_stop():
        return _mock_response(content="步骤一完成，现在继续处理下一步：", finish_reason="stop")

    result = _run(loop_agent, [
        tool_round("c1"), plan_tail_stop(),  # stop after tool work -> nudge 1/2
        tool_round("c2"), plan_tail_stop(),  # stop after tool work -> budget refreshed, nudge again
        tool_round("c3"), plan_tail_stop(),  # stop after tool work -> budget refreshed, nudge again
        _mock_response(content="All steps complete.", finish_reason="stop"),
    ])

    assert result["final_response"] == "All steps complete."
    # Every plan-tail stop was nudged (3x), not capped at two: 3 tool rounds + 3 stalls + final.
    nudges = [m for m in result["messages"] if m.get("content") == _CODEX_ACK_CONTINUATION_NUDGE]
    assert len(nudges) == 3
    assert loop_agent.client.chat.completions.create.call_count == 7
