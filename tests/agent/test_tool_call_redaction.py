"""Tests for tool-call argument redaction at the persistence boundary.

Verifies fix for #43083: redacting tool call arguments in build_assistant_message
caused the model to see ``***`` instead of real credentials on subsequent turns,
breaking credential-dependent commands.

The fix moves redaction from build_assistant_message (in-memory, replayed to model)
to _persist_session (storage-only, never replayed).
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class _FakeToolCall:
    """Minimal stand-in for the API response ToolCall object."""

    def __init__(self, tc_id, name, arguments, extra_content=None):
        self.id = tc_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments
        self.extra_content = extra_content


class _FakeAssistantMsg:
    """Minimal stand-in for the API response message object."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.function_call = None
        self.reasoning_content = None
        self.model_extra = None


class _FakeAgent:
    """Minimal agent with the attributes build_assistant_message reads."""

    def __init__(self):
        self.stream_delta_callback = None
        self._stream_callback = None
        self.reasoning_callback = None
        self.reasoning_effort = None

    @staticmethod
    def _strip_think_blocks(text):
        return text

    @staticmethod
    def _extract_reasoning(assistant_message):
        return None

    @staticmethod
    def _needs_thinking_reasoning_pad():
        return False

    @staticmethod
    def _deterministic_call_id(name, args, idx):
        return f"call_{idx}"

    @staticmethod
    def _split_responses_tool_id(raw_id):
        return raw_id, None

    @staticmethod
    def _derive_responses_function_call_id(call_id, response_item_id):
        return call_id


_PASSWORD = "SuperSecret123!"
_CMD_WITH_SECRET = f"PGPASSWORD='{_PASSWORD}' psql -c 'SELECT 1'"


class TestBuildAssistantMessageNoRedaction:
    """build_assistant_message must NOT redact tool call arguments.

    The in-memory messages list is replayed to the model on every turn.
    Redacting secrets to ``***`` here causes the model to copy the
    placeholder, breaking subsequent tool calls that need the real value.
    """

    def test_tool_call_arguments_preserved(self):
        """Credential in tool call arguments must survive build_assistant_message."""
        from agent.chat_completion_helpers import build_assistant_message

        args_json = json.dumps({"command": _CMD_WITH_SECRET})
        tc = _FakeToolCall("call_0", "terminal", args_json)
        assistant_msg = _FakeAssistantMsg("Running query...", [tc])

        agent = _FakeAgent()
        result = build_assistant_message(agent, assistant_msg, "tool_calls")

        assert "tool_calls" in result
        args = result["tool_calls"][0]["function"]["arguments"]
        assert _PASSWORD in args, (
            "Tool call arguments must NOT be redacted in the in-memory message. "
            "The model needs the real value to replay on subsequent turns."
        )

    def test_multiple_tool_calls_preserved(self):
        """Multiple tool calls with secrets must all be preserved."""
        from agent.chat_completion_helpers import build_assistant_message

        tc1 = _FakeToolCall(
            "call_0", "terminal",
            json.dumps({"command": "curl -H 'Authorization: Bearer *** https://api.example.com"}),
        )
        tc2 = _FakeToolCall(
            "call_1", "execute_code",
            json.dumps({"code": "import os; os.environ['DB_PASS'] = 'hunter2'"}),
        )
        assistant_msg = _FakeAssistantMsg("", [tc1, tc2])

        agent = _FakeAgent()
        result = build_assistant_message(agent, assistant_msg, "tool_calls")

        args0 = result["tool_calls"][0]["function"]["arguments"]
        args1 = result["tool_calls"][1]["function"]["arguments"]
        assert "***" in args0  # original value preserved, not redacted further
        assert "hunter2" in args1

    def test_no_tool_calls_still_works(self):
        """Assistant message without tool calls must still build correctly."""
        from agent.chat_completion_helpers import build_assistant_message

        assistant_msg = _FakeAssistantMsg("Here is the result.")

        agent = _FakeAgent()
        result = build_assistant_message(agent, assistant_msg, "stop")

        assert result["content"] == "Here is the result."
        assert result.get("tool_calls") is None or result.get("tool_calls") == []


class TestPersistSessionRedactsToolCalls:
    """_persist_session must redact tool call arguments in the session log.

    This is the safety net: even though build_assistant_message no longer
    redacts, the persisted session file must not store raw credentials.
    """

    def _make_agent(self, tmpdir):
        from pathlib import Path
        agent = object.__new__(__import__("run_agent").AIAgent)
        agent.session_id = "test-session"
        agent.model = "test-model"
        agent.base_url = ""
        agent.platform = "test"
        agent.logs_dir = Path(tmpdir)
        agent.session_start = datetime.now()
        agent._cached_system_prompt = "test"
        agent._session_db = None
        agent._last_flushed_db_idx = 0
        agent._session_json_enabled = True
        agent._session_messages = []
        agent.verbose_logging = False
        agent.tools = []
        return agent

    def test_persist_redacts_tool_call_arguments(self):
        """Session log must have redacted tool call arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            messages = [
                {
                    "role": "assistant",
                    "content": "Running query...",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "call_id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps({"command": _CMD_WITH_SECRET}),
                            },
                        }
                    ],
                }
            ]

            with patch("agent.redact._REDACT_ENABLED", True), \
                 patch.object(agent, "_flush_messages_to_session_db"):
                agent._persist_session(messages)

            log_path = os.path.join(tmpdir, "session_test-session.json")
            with open(log_path) as f:
                log = json.load(f)

            persisted_args = log["messages"][0]["tool_calls"][0]["function"]["arguments"]
            assert _PASSWORD not in persisted_args, (
                "Session log must redact credentials in tool call arguments."
            )

    def test_persist_does_not_mutate_original_messages(self):
        """The in-memory messages list must not be mutated by _persist_session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            original_args = json.dumps({"command": _CMD_WITH_SECRET})
            messages = [
                {
                    "role": "assistant",
                    "content": "Running...",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "call_id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": original_args,
                            },
                        }
                    ],
                }
            ]

            with patch("agent.redact._REDACT_ENABLED", True), \
                 patch.object(agent, "_flush_messages_to_session_db"):
                agent._persist_session(messages)

            # Original messages list must be untouched
            assert messages[0]["tool_calls"][0]["function"]["arguments"] == original_args

    def test_persist_preserves_non_string_arguments(self):
        """Non-string arguments (dict) must pass through without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(tmpdir)
            messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "call_id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "some_tool",
                                "arguments": {"key": "value"},  # dict, not str
                            },
                        }
                    ],
                }
            ]

            # Must not raise
            with patch("agent.redact._REDACT_ENABLED", True), \
                 patch.object(agent, "_flush_messages_to_session_db"):
                agent._persist_session(messages)

            log_path = os.path.join(tmpdir, "session_test-session.json")
            with open(log_path) as f:
                log = json.load(f)
            # Dict arguments should pass through unchanged
            assert log["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"key": "value"}
