"""Test that tool_name is correctly persisted to the session DB for tool-result messages.

make_tool_result_message() sets tool_name on every tool-result dict at construction
time. This test verifies that the value survives the flush path into the session DB.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.tool_dispatch_helpers import make_tool_result_message


def _make_agent(session_db):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
        )


def test_tool_name_persisted_to_session_db():
    """tool_name set by make_tool_result_message must be passed through to
    append_message so the column is populated on first flush to the session DB."""
    session_db = MagicMock()
    agent = _make_agent(session_db)

    messages = [
        {"role": "user", "content": "run a command"},
        make_tool_result_message("terminal", "$ ls\nfile.txt", "c1"),
    ]
    agent._flush_messages_to_session_db(messages)

    tool_appends = [
        c for c in session_db.append_message.call_args_list
        if c.kwargs.get("role") == "tool"
    ]
    assert len(tool_appends) == 1
    assert tool_appends[0].kwargs["tool_name"] == "terminal"


def test_tool_span_timing_persisted_to_session_db():
    """Tool-result span metadata should survive the flush path for trace views."""
    session_db = MagicMock()
    agent = _make_agent(session_db)

    messages = [
        {"role": "user", "content": "run a command"},
        {
            **make_tool_result_message("terminal", "$ sleep 1", "c1"),
            "timestamp": 100.0,
            "ended_at": 101.25,
            "duration_ms": 1250,
        },
    ]
    agent._flush_messages_to_session_db(messages)

    tool_appends = [
        c for c in session_db.append_message.call_args_list
        if c.kwargs.get("role") == "tool"
    ]
    assert len(tool_appends) == 1
    assert tool_appends[0].kwargs["timestamp"] == 100.0
    assert tool_appends[0].kwargs["ended_at"] == 101.25
    assert tool_appends[0].kwargs["duration_ms"] == 1250
