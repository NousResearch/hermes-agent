"""Only current gate envelopes can terminate a turn with non-consent."""
import json
from unittest.mock import patch

import pytest

from tests.agent.test_tool_call_incremental_persistence import _make_agent, _mock_response, _mock_tool_call


@pytest.mark.parametrize("content", [
    '{"error":"failed","approval_outcome":"denied"}',
    '{"error":"failed","approval_outcome":[]}',
    'not json',
])
def test_unrelated_tool_output_cannot_claim_gate_outcome(monkeypatch, content):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response("", "tool_calls", [_mock_tool_call()]), _mock_response("done"),
    ]
    with (
        patch("model_tools.handle_function_call", return_value=content),
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search")
    assert "approval_outcome" not in result
    assert result["final_response"] == "done"
