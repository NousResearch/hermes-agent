"""Tests for agent/turn_finalizer.py's turn-scoped tool activity summary.

Regression for #109815: transform_llm_output could not tell whether the
turn's claims were looked up via a tool call, because it received no
tool-call evidence at all. _turn_tool_activity_summary() gives an
output-side plugin the same turn-boundary-scoped tool-name list that
post_llm_call already receives via conversation_history, without handing
it the full message history.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

from agent.turn_finalizer import _apply_output_hooks, _turn_tool_activity_summary


def test_tool_activity_stops_at_the_current_turn_boundary():
    """Tool calls from a PRIOR turn must never leak into the current turn's summary."""
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "terminal"}},
        ]},
        {"role": "tool", "content": "prior turn's tool result"},
        {"role": "assistant", "content": "prior turn's answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "web_search"}},
            {"function": {"name": "read_file"}},
        ]},
        {"role": "tool", "content": "current turn's tool result"},
    ]
    assert _turn_tool_activity_summary(messages) == ["web_search", "read_file"]


def test_tool_activity_empty_when_the_turn_called_no_tools():
    messages = [
        {"role": "user", "content": "just chatting"},
    ]
    assert _turn_tool_activity_summary(messages) == []


def test_apply_output_hooks_passes_tool_activity_to_transform_llm_output():
    """The wiring: _apply_output_hooks must forward tool_activity as a kwarg to the hook call."""
    agent = SimpleNamespace(
        session_id="s1", model="test/model", _persist_disabled=True,
    )
    messages = [
        {"role": "user", "content": "what's the answer"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]},
    ]
    captured = {}

    def fake_invoke(name, _logger, **kwargs):
        if name == "transform_llm_output":
            captured["tool_activity"] = kwargs.get("tool_activity")
        return []

    with patch("agent.turn_finalizer._invoke_hook_safely", side_effect=fake_invoke):
        _apply_output_hooks(
            agent, "final answer", logging.getLogger(__name__),
            platform="cli", effective_task_id="t1", turn_id="turn-1",
            original_user_message="what's the answer", messages=messages,
        )

    assert captured["tool_activity"] == ["terminal"]
