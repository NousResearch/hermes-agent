"""Stall guard on the truncation-ceiling exit.

A turn whose every response truncates ends through ``_continue_text``'s ceiling
exit, which returns ``partial_result`` directly — it never reaches
``finish_text_response``, where the said-continue-but-stopped guard lives. A
partial that TAILS with an announced next action ("let me now …") therefore
ended the turn having promised work it never started, with no stall detection.

The ceiling exit must run the same bounded guard: re-prompt with the
ack-continuation nudge (max 2 per turn, shared ``codex_ack_continuations``
budget) instead of ending silently.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH


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
        # The stall guard only arms when the agent has tools to act with.
        a.valid_tool_names = {"terminal"}
        return a


def _stub(content):
    from tests.agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content),
            finish_reason=FINISH_REASON_LENGTH,
        )],
        usage=None,
    )


def _run(agent, message, history=None):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(message, conversation_history=history)


def _stall_nudges(messages):
    from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE
    return [
        m for m in messages
        if m.get("role") == "user" and m.get("content") == _CODEX_ACK_CONTINUATION_NUDGE
    ]


def test_ceiling_partial_with_continue_intent_reprompts(loop_agent):
    """Every continuation truncates and the stitched partial still tails with
    an announced next action — the turn must re-prompt (bounded), not end on
    a promise it never acted on."""
    from tests.agent.test_run_agent import _mock_response

    loop_agent.client.chat.completions.create.side_effect = [
        _stub("I checked the first file. "),
        _stub("I checked the second file. "),
        _stub("I checked the third file. "),
        _stub("Let me now update the config."),
        _mock_response(content="Config updated.", finish_reason="stop"),
    ]
    result = _run(loop_agent, "update the config")

    assert loop_agent.client.chat.completions.create.call_count == 5, (
        "The ceiling exit must re-prompt once the stitched partial tails "
        "with announced next action instead of ending the turn."
    )
    assert result["completed"] is True
    # The committed partial is already in the transcript; the recovered
    # response must not be joined onto it a second time.
    assert result["final_response"] == "Config updated."
    assert _stall_nudges(result["messages"]), (
        "The re-prompt must carry the same ack-continuation nudge the "
        "normal stall guard appends."
    )


def test_ceiling_partial_without_continue_intent_ends_partial(loop_agent):
    """No announced next action → the ceiling exit still ends the turn as
    partial; the guard must not re-prompt a plain truncated answer."""
    loop_agent.client.chat.completions.create.side_effect = [
        _stub("part one "), _stub("part two "),
        _stub("part three "), _stub("part four."),
    ]
    result = _run(loop_agent, "write me a long report")

    assert loop_agent.client.chat.completions.create.call_count == 4
    assert result["partial"] is True
    assert "truncated after 4 continuation attempts" in (result.get("error") or "")
    assert "part four" in result["final_response"]
    assert _stall_nudges(result["messages"]) == []


def test_ceiling_stall_reprompts_bounded(loop_agent):
    """A model that keeps truncating AND keeps promising must not loop: after
    the shared 2-per-turn re-prompt budget the turn still ends partial."""
    loop_agent.client.chat.completions.create.side_effect = [
        _stub(f"Checked file {i}. Let me now continue.") for i in range(6)
    ]
    result = _run(loop_agent, "check every file")

    assert loop_agent.client.chat.completions.create.call_count == 6, (
        "4 continuation attempts + 2 bounded stall re-prompts, then the "
        "turn must end — no unbounded re-prompt loop."
    )
    assert result["partial"] is True
    assert len(_stall_nudges(result["messages"])) == 2
