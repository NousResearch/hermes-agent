"""Tests for tool call argument repair in the streaming assembly path.

The streaming path (run_agent._call_chat_completions) assembles tool call
deltas into full arguments.  When a model truncates or malforms the JSON
(e.g. GLM-5.1 via Ollama), the assembly path used to pass the broken JSON
straight through — setting has_truncated_tool_args but NOT repairing it.
That triggered the truncation handler to kill the session with /new required.

The fix: repair arguments in the streaming assembly path using
_repair_tool_call_arguments() so repairable malformations (trailing commas,
unclosed brackets, Python None) don't kill the session.
"""

import json

from agent.chat_completion_helpers import _StreamingCall
from agent.message_sanitization import _repair_tool_call_arguments


def _accumulated(arguments: str) -> dict:
    return {0: {"id": "call_1", "type": "function", "function": {"name": "edit", "arguments": arguments}}}


def test_streamed_misnested_arguments_are_repaired_not_flagged_truncated():
    # deepseek-v4-flash via a portal closes an array of objects with "}}" where "}]}" is
    # required (#115061); the assembler must hand the tool a repaired call, not drop it
    # as an unrepairable "{}" flagged as truncated args.
    raw = '{"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}}'
    calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated(raw), "tool_calls")
    assert truncated is False
    assert json.loads(calls[0].function.arguments) == {
        "edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}],
    }


def test_streamed_mid_string_cut_stays_unrepairable():
    # Control: content cut inside a string is unrecoverable and must still be flagged,
    # never guessed into a wrong tool call.
    calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated('{"q": "unterminated string'), None)
    assert truncated is True
    assert calls[0].function.arguments == '{"q": "unterminated string'


class TestStreamingAssemblyRepair:
    """Verify that _repair_tool_call_arguments is applied to streaming tool
    call arguments before they're assembled into mock_tool_calls.

    These tests verify the REPAIR FUNCTION itself works correctly for the
    cases that arise during streaming assembly.  Integration tests that
    exercise the full streaming path are in run_agent.py's streaming tests.
    """

    # -- Truncation cases (most common streaming failure) --

    def test_truncated_object_no_close_brace(self):
        """Model stops mid-JSON, common with output length limits."""
        raw = '{"command": "ls -la", "timeout": 30'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed["command"] == "ls -la"
        assert parsed["timeout"] == 30



    # -- Trailing comma cases (Ollama/GLM common) --



    # -- Python None from model output --


    # -- Empty arguments (some models emit empty string) --

    def test_empty_string(self):
        assert _repair_tool_call_arguments("", "test") == "{}"


    # -- Already-valid JSON passes through unchanged --


    # -- Extra closing brackets (rare but happens) --


    # -- Real-world GLM-5.1 truncation pattern --


