"""The loop exports ``{turn_id, current_turn_user_idx}`` beside the exact ``messages`` it
addresses, and only when that row is this turn's user message verbatim.

Hosts that settle a transcript by index (hermes-webui) must not guess the current-turn
row after the loop rewrote history (alternation repair, compaction, post-turn
micro-compaction): with a repeated prompt a guessed index or a text match relabels the
historical copy and claims its old answer. The producer therefore proves the coordinate on
the final list; when it cannot, the keys are omitted and hosts fail closed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import export_current_turn_boundary


class _Agent:
    def __init__(self, turn_id="session:task:abcd1234"):
        self._current_turn_id = turn_id
        self._persist_user_message_idx = None


def test_repeated_prompt_resolves_to_the_last_verbatim_row():
    messages = [
        {"role": "user", "content": "same question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "same question"},
        {"role": "assistant", "content": "new answer"},
    ]
    agent = _Agent()
    result = export_current_turn_boundary(agent, {"messages": messages}, "same question")
    assert result["current_turn_user_idx"] == 2
    assert result["turn_id"] == agent._current_turn_id
    assert agent._persist_user_message_idx == 2


def test_missing_current_row_exports_nothing():
    messages = [
        {"role": "user", "content": "same question"},
        {"role": "assistant", "content": "old answer"},
    ]
    agent = _Agent()
    result = export_current_turn_boundary(agent, {"messages": messages}, "another question")
    assert "current_turn_user_idx" not in result and "turn_id" not in result
    assert agent._persist_user_message_idx is None


def test_rewritten_row_is_not_a_proven_boundary():
    # merge-into-tail rewrote the surviving row's content: reanchor would fall back to it,
    # but the export refuses to claim a row that is not the verbatim message.
    messages = [
        {"role": "user", "content": "summary\n\nsame question"},
        {"role": "assistant", "content": "answer"},
    ]
    result = export_current_turn_boundary(_Agent(), {"messages": messages}, "same question")
    assert "current_turn_user_idx" not in result


def test_multimodal_content_is_matched_verbatim():
    content = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}]
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": "ok"}]
    result = export_current_turn_boundary(_Agent(), {"messages": messages}, content)
    assert result["current_turn_user_idx"] == 0


def test_no_turn_id_or_non_dict_result_is_left_alone():
    assert export_current_turn_boundary(_Agent(turn_id=""), {"messages": [{"role": "user", "content": "q"}]}, "q") == {
        "messages": [{"role": "user", "content": "q"}]
    }
    assert export_current_turn_boundary(_Agent(), None, "q") is None


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


def _stub(content, finish_reason="stop"):
    from tests.run_agent.test_run_agent import _mock_assistant_msg

    return SimpleNamespace(
        id="chatcmpl-test",
        model="test/model",
        choices=[SimpleNamespace(index=0, message=_mock_assistant_msg(content=content), finish_reason=finish_reason)],
        usage=None,
    )


def test_run_conversation_exports_the_pair_on_a_success_envelope(loop_agent):
    loop_agent.client.chat.completions.create.side_effect = [_stub("new answer")]
    history = [
        {"role": "user", "content": "same question"},
        {"role": "assistant", "content": "old answer"},
    ]
    with (
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("same question", conversation_history=history)

    assert result["completed"] is True
    idx = result["current_turn_user_idx"]
    assert result["messages"][idx]["role"] == "user"
    assert result["messages"][idx]["content"] == "same question"
    assert idx > 0  # the historical identical prompt at index 0 is never the export
    assert result["turn_id"] == loop_agent._current_turn_id
    assert result["messages"][-1]["content"] == "new answer"
