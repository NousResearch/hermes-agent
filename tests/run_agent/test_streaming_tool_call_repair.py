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
import pytest

from run_agent import _repair_tool_call_arguments


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

    def test_truncated_nested_object(self):
        """Model truncates inside a nested structure."""
        raw = '{"path": "/tmp/foo", "content": "hello"'
        result = _repair_tool_call_arguments(raw, "write_file")
        parsed = json.loads(result)
        assert parsed["path"] == "/tmp/foo"

    def test_truncated_mid_value(self):
        """Model cuts off mid-string-value."""
        raw = '{"command": "git clone ht'
        result = _repair_tool_call_arguments(raw, "terminal")
        # Should produce valid JSON (even if command value is lost)
        json.loads(result)

    # -- Trailing comma cases (Ollama/GLM common) --

    def test_trailing_comma_before_close_brace(self):
        raw = '{"path": "/tmp", "content": "x",}'
        result = _repair_tool_call_arguments(raw, "write_file")
        assert json.loads(result) == {"path": "/tmp", "content": "x"}

    def test_trailing_comma_in_list(self):
        raw = '{"items": [1, 2, 3,]}'
        result = _repair_tool_call_arguments(raw, "test")
        assert json.loads(result) == {"items": [1, 2, 3]}

    # -- Python None from model output --

    def test_python_none_literal(self):
        raw = "None"
        result = _repair_tool_call_arguments(raw, "test")
        assert result == "{}"

    # -- Empty arguments (some models emit empty string) --

    def test_empty_string(self):
        assert _repair_tool_call_arguments("", "test") == "{}"

    def test_whitespace_only(self):
        assert _repair_tool_call_arguments("   \n  ", "test") == "{}"

    # -- Already-valid JSON passes through unchanged --

    def test_valid_json_passthrough(self):
        raw = '{"path": "/tmp/foo", "content": "hello"}'
        result = _repair_tool_call_arguments(raw, "write_file")
        assert json.loads(result) == {"path": "/tmp/foo", "content": "hello"}

    # -- Extra closing brackets (rare but happens) --

    def test_extra_closing_brace(self):
        raw = '{"key": "value"}}'
        result = _repair_tool_call_arguments(raw, "test")
        assert json.loads(result) == {"key": "value"}

    # -- Real-world GLM-5.1 truncation pattern --

    def test_glm_truncation_pattern(self):
        """GLM-5.1 via Ollama commonly truncates like this.

        This pattern has an unclosed colon at the end ("background":) which
        makes it unrepairable — the last-resort empty object {} is the
        safest option.  The important thing is that repairable patterns
        (trailing comma, unclosed brace WITHOUT hanging colon) DO get fixed.
        """
        raw = '{"command": "ls -la /tmp", "timeout": 30, "background":'
        result = _repair_tool_call_arguments(raw, "terminal")
        # Unrepairable — returns empty object (hanging colon can't be fixed)
        parsed = json.loads(result)
        assert parsed == {}

    def test_glm_truncation_repairable(self):
        """GLM-5.1 truncation pattern that IS repairable."""
        raw = '{"command": "ls -la /tmp", "timeout": 30'
        result = _repair_tool_call_arguments(raw, "terminal")
        parsed = json.loads(result)
        assert parsed["command"] == "ls -la /tmp"
        assert parsed["timeout"] == 30