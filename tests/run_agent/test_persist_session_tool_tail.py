"""_persist_session must close a trailing tool-result sequence.

Early-return paths in conversation_loop.py call _persist_session directly
without going through finalize_turn, which is the only place that
previously closed trailing tool results.  If the session is persisted
with messages[-1]["role"] == "tool", resuming it later creates a
tool → user alternation that strict providers (Gemini, Claude) reject
(#48879).

This test pins the contract: _persist_session always appends a synthetic
assistant message when the transcript ends at role="tool", regardless of
whether finalize_turn ran.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-persist-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(model="test-model")
    agent._save_session_log = MagicMock()
    agent._flush_messages_to_session_db = MagicMock()
    return agent


class TestPersistSessionToolTail:

    def test_tool_tail_gets_assistant_closure(self):
        """Messages ending at role=tool get a synthetic assistant appended."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "run the tool"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "search results"},
        ]

        agent._persist_session(messages)

        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Operation interrupted."

    def test_assistant_tail_unchanged(self):
        """Messages already ending at role=assistant are not modified."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        original_len = len(messages)

        agent._persist_session(messages)

        assert len(messages) == original_len
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "hi there"

    def test_user_tail_unchanged(self):
        """Messages ending at role=user are not modified."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "hello"},
        ]
        original_len = len(messages)

        agent._persist_session(messages)

        assert len(messages) == original_len
        assert messages[-1]["role"] == "user"

    def test_empty_messages_unchanged(self):
        """Empty message list does not crash."""
        agent = _make_agent()
        messages = []

        agent._persist_session(messages)

        assert len(messages) == 0

    def test_multiple_tool_results_get_single_closure(self):
        """Only one assistant closure appended even with multiple trailing tools."""
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "run tools"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tc2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result a"},
            {"role": "tool", "tool_call_id": "tc2", "content": "result b"},
        ]

        agent._persist_session(messages)

        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Operation interrupted."
        assert messages[-2]["role"] == "tool"
