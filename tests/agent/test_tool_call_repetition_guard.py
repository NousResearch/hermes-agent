"""Unit tests for repetition guard on tool-call arguments (#103599).

When a model falls into a degenerate repetition loop inside tool-call arguments,
it must be detected and flagged as a dropped/truncated tool call rather than
executing the corrupt/repeating command.
"""

from __future__ import annotations

import json
import pytest

from agent.chat_completion_helpers import _StreamingCall
from agent.repetition_guard import is_repetition_dominated


class TestToolCallRepetitionGuard:
    def test_repetition_dominated_arguments_flagged_as_truncated(self):
        # 200 repetitions of a distinct pattern in a command argument
        echo_cmd = ("cat >> /tmp/test.py <<'EOF'\necho 'amen shalom salaam peace out yo wassup'\n" * 150)
        args_json = json.dumps({"command": echo_cmd})
        assert is_repetition_dominated(args_json) is True

        tool_calls_acc = {
            0: {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": args_json,
                },
            }
        }

        mock_calls, has_truncated = _StreamingCall._assemble_tool_calls(
            tool_calls_acc, finish_reason="stop"
        )

        assert mock_calls is not None
        assert len(mock_calls) == 1
        assert has_truncated is True

    def test_clean_arguments_not_flagged(self):
        args_json = json.dumps({
            "command": "pytest tests/agent/test_tool_call_repetition_guard.py -v",
            "cwd": "/workspace/project",
        })
        assert is_repetition_dominated(args_json) is False

        tool_calls_acc = {
            0: {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": args_json,
                },
            }
        }

        mock_calls, has_truncated = _StreamingCall._assemble_tool_calls(
            tool_calls_acc, finish_reason="stop"
        )

        assert mock_calls is not None
        assert len(mock_calls) == 1
        assert has_truncated is False

    def test_unrepairable_json_still_flagged(self):
        tool_calls_acc = {
            0: {
                "id": "call_789",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": "{not valid json at all",
                },
            }
        }

        mock_calls, has_truncated = _StreamingCall._assemble_tool_calls(
            tool_calls_acc, finish_reason="stop"
        )

        assert has_truncated is True
