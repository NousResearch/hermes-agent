"""Degenerate-final recovery (#103483).

Provider-side collapse: a turn executes its tool work correctly, then ends on ``finish_reason=stop``
with an ENTIRE visible answer that is a fragment — a stray wrong-script word (``пар``), a token
starting mid-punctuation (``?warming up``). Before the fix the loop accepted the fragment as the
answer, the turn reported ``completed``, and an unattended job silently abandoned the task.

The guard rides the existing ack-continuation path: same scope knob (``agent.intent_ack_continuation``,
default ``auto`` = Responses transports), same bounded per-turn counter, same durable interim +
nudge rows. Shape alone cannot prove a collapse, so the predicate is narrow and the nudge asks for
the same answer again when it WAS complete — a false positive costs one call, never the answer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import looks_like_degenerate_final
from agent.conversation_loop import _DEGENERATE_FINAL_NUDGE


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_dropped_tool_call_recovery)."""
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
        # Explicit opt-in: the loop tests exercise the mechanism, not the transport-scoped default.
        agent._intent_ack_continuation = True
        return agent


def _run(agent, stages):
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value="ok"),
    ):
        return agent.run_conversation("build the workbook")


def _tool_round(call_id):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id=call_id)],
    )


def _final(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop")


def _user_rows_sent(agent, call_index):
    kwargs = agent.client.chat.completions.create.call_args_list[call_index].kwargs
    return [m["content"] for m in kwargs["messages"] if m.get("role") == "user"]


class TestFragmentAfterToolWork:
    def test_fragment_after_tool_work_reprompts_instead_of_exiting(self, loop_agent):
        result = _run(loop_agent, [_tool_round("call_1"), _final("пар"), _final("Workbook built: 3 sheets.")])

        assert result["final_response"] == "Workbook built: 3 sheets."
        assert loop_agent.client.chat.completions.create.call_count == 3
        # The re-prompt carried the degenerate-final nudge, not the generic ack nudge.
        assert _user_rows_sent(loop_agent, 2)[-1] == _DEGENERATE_FINAL_NUDGE
        # The fragment stays in the transcript as an interim row, the nudge right after it.
        contents = [m.get("content") for m in result["messages"]]
        assert contents.index("пар") + 1 == contents.index(_DEGENERATE_FINAL_NUDGE)

    def test_persistent_collapse_is_bounded_and_keeps_the_fragment(self, loop_agent):
        """One re-prompt per collapse: the nudge row closes the tool-work window, so a second
        fragment is the answer — today's behaviour, never a loop."""
        result = _run(loop_agent, [_tool_round("call_1"), _final("пар"), _final("?warming up")])

        assert loop_agent.client.chat.completions.create.call_count == 3
        assert result["final_response"] == "?warming up"

    def test_fragment_without_tool_work_is_a_plain_answer(self, loop_agent):
        """A chat-only turn that answers tersely is not a collapse; nothing is re-prompted."""
        result = _run(loop_agent, [_final("пар")])

        assert loop_agent.client.chat.completions.create.call_count == 1
        assert result["final_response"] == "пар"


class TestLooksLikeDegenerateFinal:
    @pytest.mark.parametrize("text", ["пар", "?warming up", ",and", "ошибка", "你好", "):"])
    def test_fragments(self, text):
        assert looks_like_degenerate_final(text)

    @pytest.mark.parametrize("text", [
        # Terse legitimate answers that reviewers showed a shape-only guard re-prompting (#111472).
        "42", "SQLite", "report.csv", "3.14159", "€12.50", "你好。", "Done.", "True", "a51143fbbe",
        "2026-09-19", "/tmp/out.log", "$5", "#123", "-1", "(a)", ".env", "~/x", "+1", "✅",
        "Reading is a skill", "The answer is 43.", "x" * 25, "",
    ])
    def test_terse_answers_are_not_fragments(self, text):
        assert not looks_like_degenerate_final(text)
