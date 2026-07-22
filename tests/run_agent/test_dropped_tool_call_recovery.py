"""Regression tests for dropped tool-call recovery.

Some providers (observed: claude-opus-4.8 / claude-sonnet-4.5 on GitHub
Copilot, ~2026-07) return ``finish_reason="tool_calls"`` while the parsed
``tool_calls`` array is empty — the model signalled it wanted to act but the
payload shipped no call. Before the fix, the conversation loop took the
no-tool-calls ``else`` branch, treated the turn's narration as the final
answer, and exited with the task unstarted. On an unattended multi-step job
(e.g. a scheduled PR reviewer) this silently did nothing.

The fix keys on the provider contract violation itself
(``finish_reason == "tool_calls"`` with zero ``tool_calls``) and re-prompts,
bounded to 3 consecutive stalls, with the budget resetting after any
successful tool round so it guards each stall rather than the whole run. A
genuine ``finish_reason="stop"`` text turn is unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)
    so we can stage a dropped-tool-call response + continuation pair on
    ``.chat.completions.create``."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
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
        return agent


def _dropped_tool_call_response(content: str):
    """A response whose finish_reason claims a tool call, but tool_calls is
    empty — the provider contract violation this fix recovers from."""
    from tests.run_agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="chatcmpl-dropped",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content, tool_calls=None),
            finish_reason="tool_calls",
        )],
        usage=None,
    )


class TestDroppedToolCallRecovery:
    def test_dropped_tool_call_reprompts_instead_of_exiting(self, loop_agent):
        """finish_reason=tool_calls with an empty tool_calls array must
        re-prompt the model to emit the call rather than exiting the loop
        with the narration as the final answer."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response("Let me verify the PR and gather evidence."),
            _mock_response(content="All checks pass. Approved.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "A dropped tool call must trigger a re-prompt (second API call), "
            "not exit the loop after one call."
        )

        # The loop must have injected a nudge user-message telling the model to
        # issue the actual tool call.
        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = second_call.kwargs.get("messages") or second_call.args[0].get("messages")
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "tool call" in (last_user.get("content") or "").lower(), (
            "The nudge must explicitly ask the model to issue the tool call."
        )
        assert "All checks pass" in result["final_response"]

    def test_empty_content_dropped_tool_call_still_reprompts(self, loop_agent):
        """The narration may live only in the reasoning field, leaving content
        empty. The recovery must still fire — it keys on the finish_reason /
        tool_calls mismatch, not on content being present."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response(""),  # no visible content at all
            _mock_response(content="Done.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "An empty-content dropped tool call must still re-prompt."
        )
        assert result["completed"] is True

    def test_clean_stop_text_turn_is_unaffected(self, loop_agent):
        """A genuine finish_reason=stop text response must exit normally — the
        recovery path must not fire on ordinary final answers."""
        from tests.run_agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Here is your answer.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A clean finish_reason=stop turn must not trigger a re-prompt."
        )
        assert "Here is your answer." in result["final_response"]

    def test_persistent_dropped_tool_calls_are_bounded(self, loop_agent):
        """If the model never emits a call, the recovery must give up after a
        bounded number of consecutive stalls instead of looping forever."""
        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response("Let me check.") for _ in range(10)
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        # 1 initial call + 3 bounded re-prompts = 4 total, then it stops.
        assert loop_agent.client.chat.completions.create.call_count <= 4, (
            "Consecutive dropped tool calls must be bounded (no infinite loop)."
        )
        assert result is not None
