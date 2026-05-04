"""Regression tests for agent.display._detect_tool_failure handling non-string results.

Issue #19814: custom-tools plugin tools return dict instead of str, causing
KeyError: slice(None, 500, None) when _detect_tool_failure tries result[:500].
"""

import pytest


class TestDetectToolFailure:
    """Tests for _detect_tool_failure with string and non-string results."""

    def test_string_result_success(self):
        from agent.display import _detect_tool_failure
        is_failure, suffix = _detect_tool_failure("some_tool", '{"success": true}')
        assert is_failure is False
        assert suffix == ""

    def test_string_result_error(self):
        from agent.display import _detect_tool_failure
        is_failure, suffix = _detect_tool_failure("some_tool", '{"error": "something failed"}')
        assert is_failure is True
        assert suffix == " [error]"

    def test_string_result_failed_keyword(self):
        from agent.display import _detect_tool_failure
        is_failure, suffix = _detect_tool_failure("some_tool", 'Operation "failed" due to timeout')
        assert is_failure is True
        assert suffix == " [error]"

    def test_none_result(self):
        from agent.display import _detect_tool_failure
        is_failure, suffix = _detect_tool_failure("some_tool", None)
        assert is_failure is False
        assert suffix == ""

    def test_dict_result_does_not_raise(self):
        """Regression: dict result must not cause KeyError on dict[:500]."""
        from agent.display import _detect_tool_failure
        # This was the exact crash: dict[:500] raises KeyError
        result = {"success": True, "data": "some value"}
        is_failure, suffix = _detect_tool_failure("custom_tool", result)
        assert isinstance(is_failure, bool)

    def test_dict_result_with_error_key(self):
        """Dict with 'error' key should be detected as failure."""
        from agent.display import _detect_tool_failure
        result = {"error": "connection refused"}
        is_failure, suffix = _detect_tool_failure("custom_tool", result)
        assert is_failure is True
        assert suffix == " [error]"

    def test_dict_result_with_success_key(self):
        """Dict with 'success' key but no error should not be detected as failure."""
        from agent.display import _detect_tool_failure
        result = {"success": True, "data": {"items": [1, 2, 3]}}
        is_failure, suffix = _detect_tool_failure("custom_tool", result)
        assert is_failure is False

    def test_list_result_does_not_raise(self):
        """List result should be handled gracefully."""
        from agent.display import _detect_tool_failure
        result = [1, 2, 3]
        is_failure, suffix = _detect_tool_failure("some_tool", result)
        assert isinstance(is_failure, bool)

    def test_integer_result_does_not_raise(self):
        """Integer result should be handled gracefully."""
        from agent.display import _detect_tool_failure
        is_failure, suffix = _detect_tool_failure("some_tool", 42)
        assert isinstance(is_failure, bool)

    def test_terminal_tool_with_dict_result(self):
        """Terminal tool with dict result (normal case) should still work."""
        from agent.display import _detect_tool_failure
        result = {"exit_code": 1, "output": "error message"}
        is_failure, suffix = _detect_tool_failure("terminal", result)
        # Terminal tool parses dict normally — but this dict comes in as-is
        # The function should convert it to JSON string, then safe_json_loads
        # should parse it back
        assert is_failure is True
        assert "exit 1" in suffix

    def test_memory_tool_with_dict_result(self):
        """Memory tool with dict result should be handled gracefully."""
        from agent.display import _detect_tool_failure
        result = {"success": False, "error": "You exceed the limit of 100 entries"}
        is_failure, suffix = _detect_tool_failure("memory", result)
        assert is_failure is True
        assert suffix == " [full]"

    def test_get_cute_tool_message_with_dict_result(self):
        """get_cute_tool_message should not crash when result is a dict."""
        from agent.display import get_cute_tool_message
        result = {"success": True, "data": "value"}
        # Should not raise KeyError
        msg = get_cute_tool_message("custom_tool", {"param": "val"}, 1.5, result=result)
        assert isinstance(msg, str)
