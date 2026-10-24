"""#142867: a human-approved plugin gate must trace into the tool result.

Kept out of ``test_run_agent.py`` (over its FILE_LINES cap — may only
shrink). The agent fixture mirrors that file's minimal AIAgent setup.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _mock_plugin_discovery(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)


@pytest.fixture()
def agent():
    tool_defs = [{
        "type": "function",
        "function": {"name": "web_search", "description": "web_search tool",
                     "parameters": {"type": "object", "properties": {}}},
    }]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890",
                    base_url="https://openrouter.ai/api/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
        a.client = MagicMock()
        return a


def test_human_approved_plugin_gate_traces_into_tool_result(agent, monkeypatch):
    """When a plugin-escalated approval is granted by a human, the trace the
    gate stashes must reach the tool result the model sees — otherwise
    "approved after a card" looks identical to "no card was needed"
    (a denial leaves a BLOCKED trace)."""
    from tools.approval_context import _set_human_approval_note, _take_human_approval_note

    def _approved_hook(*args, **kwargs):
        _set_human_approval_note(
            "Approved by the user (once) after an approval request (sensitive path).")
        return (None, None)

    monkeypatch.setattr(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks", _approved_hook)
    with patch("model_tools.handle_function_call", return_value='{"output": "done"}'):
        result = agent._invoke_tool(
            "write_file", {"path": "/tmp/x", "content": "y"}, "task-1")

    parsed = json.loads(result)
    assert parsed["output"] == "done"
    assert parsed["approval"].startswith("Approved by the user (once)")
    assert _take_human_approval_note() is None  # consumed, not leaked
