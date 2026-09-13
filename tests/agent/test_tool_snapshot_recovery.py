"""Canonical pairing and uncertain-effect boundaries during stale recovery."""
from types import SimpleNamespace
from unittest.mock import Mock

from agent.iteration_budget import IterationBudget
from agent.tool_snapshot import ToolSnapshotChangedError, bind_tool_execution_snapshot
from agent.turn_tool_snapshot import recover_stale_tool_snapshot


def test_recovery_pairs_canonical_call_id_before_resend():
    agent = SimpleNamespace(
        _tool_snapshot_epoch=2, iteration_budget=IterationBudget(3),
        _flush_messages_to_session_db=Mock(return_value=True), _buffer_status=Mock(),
    )
    agent.iteration_budget.consume()
    message = SimpleNamespace(_hermes_tool_snapshot_epoch=1, tool_calls=[SimpleNamespace(
        id="item_1", call_id="call_1|item_1", function=SimpleNamespace(name="web_search"),
    )])
    messages = [{"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "web_search"}}]}]
    result = recover_stale_tool_snapshot(
        agent, error=ToolSnapshotChangedError("stale"), assistant_message=message,
        messages=messages, conversation_history=[], api_call_count=1,
        stale_tool_snapshot_retries=0, final_response="", failed=False, _turn_exit_reason="",
    )
    assert result.action == "continue"
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["call_1"]
    assert messages[-1]["effect_disposition"] == "none"
    agent._flush_messages_to_session_db.assert_called_once()


def test_unfinished_sibling_prevents_neutral_retry(monkeypatch):
    from agent.tool_executor import _unfinished_tool_result

    agent = SimpleNamespace(_tool_snapshot_epoch=1)
    message = SimpleNamespace(_hermes_tool_snapshot_epoch=1)
    ref = SimpleNamespace(name="web_search", emit_post=Mock())
    refresh = Mock()
    monkeypatch.setattr("agent.tool_snapshot.refresh_tool_snapshot_after_stale", refresh)
    with bind_tool_execution_snapshot(agent, message) as state:
        state.stale = True  # Another sibling was rejected before effects.
        result, _, disposition = _unfinished_tool_result(agent, ref, timed_out=True, timeout_s=1)
        assert "timed out" in result
        assert disposition == "unknown"
    refresh.assert_called_once_with(agent, message)
