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


@pytest.mark.parametrize("user_message, messages, expected_idx", [
    # a repeated prompt resolves to the LAST verbatim row, never the historical copy
    ("same question", [
        {"role": "user", "content": "same question"}, {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "same question"}, {"role": "assistant", "content": "new answer"},
    ], 2),
    # multimodal content matches by structural equality
    ([{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
     [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}]},
      {"role": "assistant", "content": "ok"}], 0),
    # a row the repair rewrote (merge-into-tail) is not a proven boundary: export nothing
    ("same question", [{"role": "user", "content": "summary\n\nsame question"}, {"role": "assistant", "content": "answer"}], None),
    # no current row at all: export nothing, persist override untouched
    ("another question", [{"role": "user", "content": "same question"}, {"role": "assistant", "content": "old answer"}], None),
])
def test_boundary_is_exported_only_for_the_verbatim_current_row(user_message, messages, expected_idx):
    agent = _Agent()
    result = export_current_turn_boundary(agent, {"messages": messages}, user_message)
    if expected_idx is None:
        assert "current_turn_user_idx" not in result and "turn_id" not in result
    else:
        assert result["current_turn_user_idx"] == expected_idx
        assert result["turn_id"] == agent._current_turn_id
    # the persist funnel has already run by the time the envelope is stamped: never re-anchor it here
    assert agent._persist_user_message_idx is None


def test_preflight_timeout_envelope_exports_nothing():
    """The preflight-timeout result carries the prior history without this turn's row (#7100);
    a repeated prompt must not be resolved to its historical copy."""
    agent = _Agent()
    result = {"messages": [{"role": "user", "content": "continue"}, {"role": "assistant", "content": "old"}],
              "turn_exit_reason": "context_compression_timeout"}
    out = export_current_turn_boundary(agent, result, "continue")
    assert "current_turn_user_idx" not in out and "turn_id" not in out


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
