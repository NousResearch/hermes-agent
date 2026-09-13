"""Turn-boundary contract v2 on every ``run_conversation`` result envelope.

The current user row is identified by the ``_turn_id`` marker stamped at append time,
never by prompt text: an identical historical prompt can never receive the live turn
id. Every envelope carries ``turn_boundary_contract``, ``turn_id`` and
``messages_projection``; ``current_turn_user_idx`` is the marked row or ``None``
(hosts treat ``None`` as capable-but-unproven and fail closed).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import (
    TURN_ROW_MARKER,
    export_current_turn_boundary,
    find_current_turn_row_idx,
    reanchor_current_turn_user_idx,
)

LIVE = "session:task:live0001"
OLD = "session:task:old00001"


class _Agent:
    def __init__(self, turn_id=LIVE):
        self._current_turn_id = turn_id
        self._persist_user_message_idx = None


def _u(text, turn_id=None):
    row = {"role": "user", "content": text}
    if turn_id:
        row[TURN_ROW_MARKER] = turn_id
    return row


def _keys(result):
    return {k: result.get(k) for k in ("turn_boundary_contract", "turn_id", "messages_projection", "current_turn_user_idx")}


def test_repeated_prompt_is_resolved_by_marker_never_by_text():
    messages = [_u("same question", OLD), {"role": "assistant", "content": "old answer"},
                _u("same question", LIVE), {"role": "assistant", "content": "new answer"}]
    out = export_current_turn_boundary(_Agent(), {"messages": messages})
    assert _keys(out) == {"turn_boundary_contract": 2, "turn_id": LIVE, "messages_projection": "full", "current_turn_user_idx": 2}


def test_rewrite_that_dropped_the_current_row_exports_a_null_index():
    # the identical historical prompt survives; its marker is another turn's (or none)
    for hist in (_u("same question", OLD), _u("same question")):
        out = export_current_turn_boundary(_Agent(), {"messages": [hist, {"role": "assistant", "content": "old answer"}]})
        assert _keys(out) == {"turn_boundary_contract": 2, "turn_id": LIVE, "messages_projection": "full", "current_turn_user_idx": None}


def test_stale_index_on_the_envelope_is_replaced_not_trusted():
    out = export_current_turn_boundary(_Agent(), {"messages": [_u("q")], "current_turn_user_idx": 0})
    assert out["current_turn_user_idx"] is None


@pytest.mark.parametrize("shape", [
    {"turn_exit_reason": "context_compression_timeout"},   # preflight timeout: prior history only
    {"completed": False, "interrupted": True},             # lease early return shape
    {"partial": True, "error": "boom"},
])
def test_every_envelope_shape_carries_the_contract(shape):
    out = export_current_turn_boundary(_Agent(), {"messages": [_u("continue", OLD)], **shape})
    assert out["turn_boundary_contract"] == 2 and out["turn_id"] == LIVE and out["messages_projection"] == "full"
    assert out["current_turn_user_idx"] is None


def test_no_turn_id_or_non_dict_result_is_left_alone():
    assert export_current_turn_boundary(_Agent(turn_id=""), {"messages": [_u("q", LIVE)]}) == {"messages": [_u("q", LIVE)]}
    assert export_current_turn_boundary(_Agent(), None) is None
    assert export_current_turn_boundary(_Agent(), {"messages": None}) == {"messages": None}


def test_find_and_reanchor_prefer_the_marker_over_text():
    messages = [_u("same question", LIVE), {"role": "assistant", "content": "a"}, _u("same question")]
    assert find_current_turn_row_idx(messages, LIVE) == 0
    assert reanchor_current_turn_user_idx(messages, "same question", turn_id=LIVE) == 0
    assert reanchor_current_turn_user_idx(messages, "same question") == 2  # text fallback: last match


def test_user_row_merges_carry_the_marker_to_the_surviving_row():
    from agent.agent_runtime_helpers import repair_message_sequence
    from agent.micro_compaction import MicroCompactionMixin

    for merge in (
        lambda rows: (repair_message_sequence(SimpleNamespace(), rows), rows)[1],
        lambda rows: MicroCompactionMixin._merge_adjacent_user_turns(rows),
    ):
        rows = [_u("todo snapshot"), _u("the question", LIVE), {"role": "assistant", "content": "a"}]
        merged = merge(rows)
        assert [m["role"] for m in merged] == ["user", "assistant"]
        assert merged[0][TURN_ROW_MARKER] == LIVE and "the question" in merged[0]["content"]
        assert find_current_turn_row_idx(merged, LIVE) == 0


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


def test_run_conversation_marks_the_row_and_exports_the_contract(loop_agent):
    from run_agent import AIAgent

    assert AIAgent.TURN_BOUNDARY_CONTRACT == 2
    loop_agent.client.chat.completions.create.side_effect = [_stub("new answer")]
    history = [{"role": "user", "content": "same question"}, {"role": "assistant", "content": "old answer"}]
    with (
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("same question", conversation_history=history)

    assert result["completed"] is True
    assert result["turn_boundary_contract"] == 2 and result["messages_projection"] == "full"
    assert result["turn_id"] == loop_agent._current_turn_id
    idx = result["current_turn_user_idx"]
    assert idx == 2  # the historical identical prompt at 0 is never the export
    assert result["messages"][idx][TURN_ROW_MARKER] == result["turn_id"]
    assert TURN_ROW_MARKER not in result["messages"][0]
    # the marker never reaches the provider
    sent = loop_agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert all(TURN_ROW_MARKER not in m for m in sent if isinstance(m, dict))
