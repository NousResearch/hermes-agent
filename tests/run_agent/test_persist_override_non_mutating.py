"""Regression tests for #56303 — persist override must not mutate live messages."""

from unittest.mock import MagicMock

from agent.tool_executor import _flush_session_db_after_tool_progress
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = MagicMock()
    agent.session_id = "session-56303"
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._session_db_created = True
    return agent


def test_tool_loop_incremental_flush_preserves_augmented_user_content():
    """Mid-tool-loop flush must not strip API-facing user prefixes (#56303)."""
    agent = _bare_agent()
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "hello"
    augmented = "[Group context] hello"
    messages = [
        {"role": "user", "content": augmented},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]

    _flush_session_db_after_tool_progress(agent, messages, stage="tool_result")

    assert messages[0]["content"] == augmented
    user_db_writes = [
        call.kwargs
        for call in agent._session_db.append_message.call_args_list
        if call.kwargs.get("role") == "user"
    ]
    assert user_db_writes
    assert user_db_writes[0]["content"] == "hello"
