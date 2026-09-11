"""Unit tests for repetition guard on tool-call arguments (#103599).

When a model gets stuck in a degenerate repetition loop inside tool call arguments,
the arguments should be flagged as truncated rather than dispatched as valid commands.
"""

from __future__ import annotations

import json

from agent.chat_completion_helpers import _StreamingCall


_INCIDENT_ECHO = "amen shalom salaam ... peace out yo wassup ... "


def test_repetition_dominated_tool_call_arguments_flagged():
    """Tool call arguments dominated by repeated text are flagged as truncated."""
    # Build valid JSON with repetition-dominated command
    repeating_cmd = "cat >> /tmp/g1.py <<'PYEOF'\n" + (_INCIDENT_ECHO * 100) + "\nPYEOF"
    args_json = json.dumps({"command": repeating_cmd})

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


def test_normal_tool_call_arguments_not_flagged():
    """Normal tool call arguments (short or long unique content) are not flagged."""
    normal_cmd = "python3 -c 'import sys; print(sys.version)'"
    args_json = json.dumps({"command": normal_cmd})

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
