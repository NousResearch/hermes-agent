"""Regression tests for repetition-dominated response handling.

A response dominated by verbatim repeated text must be discarded whether it
ends normally or at the output cap. The turn aborts with a clear user-facing
error instead of persisting and delivering the pathological fragment.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.repetition_guard import STOP_PATH_MIN_CHARS
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID

# The exact sentence from the #86581 incident.
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"


@pytest.fixture()
def loop_agent():
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        return a


def _response(
    content,
    *,
    finish_reason=FINISH_REASON_LENGTH,
    response_id=PARTIAL_STREAM_STUB_ID,
):
    from tests.agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id=response_id,
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content),
            finish_reason=finish_reason,
        )],
        usage=None,
    )


def _run(agent, message):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message)


class TestContinuationRepetitionGuard:
    def test_repetition_dominated_truncation_aborts(self, loop_agent):
        echo = _INCIDENT_ECHO * 2000
        loop_agent.client.chat.completions.create.side_effect = [_response(echo)]

        result = _run(loop_agent, "write me a long report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert echo not in (result["final_response"] or "")
        # The pathological fragment must NOT be appended to the history.
        assert not any(
            isinstance(m, dict) and m.get("_length_continuation_fragment")
            for m in result["messages"]
        )
        # Exactly one API call — no continuation was attempted.
        assert loop_agent.client.chat.completions.create.call_count == 1

    @pytest.mark.parametrize("requested_repeat", [False, True], ids=["loop", "repeat-on-request"])
    def test_repetition_dominated_stop_response_aborts(self, loop_agent, requested_repeat):
        if requested_repeat:
            # Asked-for repetition (identical lines, ~3.6k chars) is below runaway scale: a
            # completed answer must be delivered, not discarded.
            echo = "Hello world, this is a sentence the user asked me to repeat many times.\n" * 50
        else:
            paragraph = (
                "A long paragraph that should never be delivered hundreds of times "
                "when a model enters a repetition loop.\n"
                "The second line makes this a multiline repeating unit.\n"
            )
            echo = paragraph * 500
            assert len(echo) >= STOP_PATH_MIN_CHARS
        loop_agent.client.chat.completions.create.side_effect = [
            _response(echo, finish_reason="stop", response_id="completed-response")
        ]

        result = _run(loop_agent, "write me a long report")

        assert loop_agent.client.chat.completions.create.call_count == 1
        if requested_repeat:
            assert result["completed"] is True
            assert result["final_response"] == echo.strip()
            return
        assert result["completed"] is False
        assert result["partial"] is True
        assert (result["failure_reason"], result["failure_retryable"]) == ("truncated", True)
        assert "Repetition" in (result["final_response"] or "")
        assert not any(
            isinstance(m, dict) and m.get("content") == echo
            for m in result["messages"]
        )

    def test_legit_truncation_still_continues(self, loop_agent):
        # Ordinary short truncated fragments still get continuation retries.
        loop_agent.client.chat.completions.create.side_effect = [
            _response("part one "), _response("part two "),
            _response("part three "), _response("part four."),
        ]

        result = _run(loop_agent, "write me a long report")

        assert result["partial"] is True
        assert loop_agent.client.chat.completions.create.call_count == 4
