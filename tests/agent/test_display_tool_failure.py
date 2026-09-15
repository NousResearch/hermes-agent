"""Tests for _detect_tool_failure + _trim_error + get_cute_tool_message
inline failure suffix rendering.

Covers the user-visible promise: when a tool fails, the CLI shows a short,
specific reason in square brackets at the end of the completion line —
not a generic "[error]".
"""

import json

from agent.display import (
    _detect_tool_failure,
    _trim_error,
    _ERROR_SUFFIX_MAX_LEN,
    get_cute_tool_message,
)


class TestTrimError:
    """The helper that shrinks an error message for inline display."""

    def test_short_message_unchanged(self):
        assert _trim_error("nope") == "nope"


    def test_long_message_truncated_to_cap(self):
        msg = "x" * 200
        trimmed = _trim_error(msg)
        assert len(trimmed) <= _ERROR_SUFFIX_MAX_LEN
        assert trimmed.endswith("...")

    def test_file_not_found_path_collapsed_to_filename(self):
        long_path = "File not found: /home/teknium/.hermes/hermes-agent/very/deep/path/foo.py"
        assert _trim_error(long_path) == "File not found: foo.py"




class TestDetectToolFailureTerminal:
    """terminal: non-zero exit_code is the canonical failure signal."""

    def test_success_returns_no_suffix(self):
        result = json.dumps({"output": "ok\n", "exit_code": 0})
        assert _detect_tool_failure("terminal", result) == (False, "")


    def test_nonzero_exit_with_error_shows_message(self):
        result = json.dumps({
            "output": "",
            "exit_code": 127,
            "error": "ls: cannot access 'foo': No such file or directory",
        })
        is_failure, suffix = _detect_tool_failure("terminal", result)
        assert is_failure is True
        assert "cannot access" in suffix
        # Trimmed to the cap, in brackets
        assert suffix.startswith(" [")
        assert suffix.endswith("]")




class TestDetectToolFailureMemory:
    """memory: 'full' is distinct from real errors."""

    def test_memory_full_returns_full_suffix(self):
        result = json.dumps({"success": False, "error": "would exceed the limit"})
        assert _detect_tool_failure("memory", result) == (True, " [full]")

    def test_memory_other_error_returns_specific_message(self):
        # An error that's NOT a "full" overflow falls through to the
        # structured-error path and surfaces the actual message.
        result = json.dumps({"success": False, "error": "invalid action: zap"})
        is_failure, suffix = _detect_tool_failure("memory", result)
        assert is_failure is True
        assert "invalid action" in suffix


class TestDetectToolFailureStructured:
    """Generic path: any tool that returns {"error": ...} JSON."""

    def test_read_file_error_surfaced(self):
        result = json.dumps({
            "path": "/nope/missing.py",
            "success": False,
            "error": "File not found: /nope/missing.py",
        })
        is_failure, suffix = _detect_tool_failure("read_file", result)
        assert is_failure is True
        # _trim_error reduces the path to the basename.
        assert suffix == " [File not found: missing.py]"



    def test_successful_result_not_flagged(self):
        result = json.dumps({"success": True, "data": "hello"})
        assert _detect_tool_failure("web_search", result) == (False, "")

    def test_success_true_with_zero_failed_count_not_flagged(self):
        """Regression for #107581: {\"success\": true, \"failed\": 0} is success."""
        result = json.dumps({"success": True, "failed": 0, "processed": 12})
        assert _detect_tool_failure("mcp__qase__qase_ci_report", result) == (False, "")

    def test_success_true_with_null_error_not_flagged(self):
        """Regression for #107581: {\"success\": true, \"error\": null} is success."""
        result = json.dumps({"success": True, "error": None, "data": "ok"})
        assert _detect_tool_failure("read_file", result) == (False, "")

    def test_positive_failed_count_is_flagged(self):
        """Positive failed count (even without success: false) is a failure."""
        result = json.dumps({"failed": 3, "processed": 10})
        is_failure, suffix = _detect_tool_failure("tool", result)
        assert is_failure is True
        assert "failed=3" in suffix

    def test_success_false_without_error_message_flagged(self):
        """success: false alone (no error text) is a failure, reports [failed]."""
        result = json.dumps({"success": False})
        is_failure, suffix = _detect_tool_failure("tool", result)
        assert is_failure is True
        assert suffix == " [failed]"

    def test_truthy_error_string_flagged(self):
        """Truthy error strings (even with no success field) are failures."""
        result = json.dumps({"error": "Network timeout"})
        is_failure, suffix = _detect_tool_failure("web_search", result)
        assert is_failure is True
        assert "Network timeout" in suffix

    def test_nested_diagnostic_data_with_success_true_not_flagged(self):
        """Regression for #107581: nested diagnostic data doesn't trigger false positive."""
        result = json.dumps({
            "success": True,
            "data": {"diagnostics": {"error_count": 0, "failed_checks": 0}},
        })
        assert _detect_tool_failure("tool", result) == (False, "")

    def test_plain_text_error_string_fallback(self):
        """Plain text Error prefix (non-JSON) is still caught."""
        result = "Error: unexpected EOF while parsing"
        is_failure, suffix = _detect_tool_failure("tool", result)
        assert is_failure is True
        assert suffix == " [error]"



class TestGetCuteToolMessageFailureSuffix:
    """End-to-end: failure suffix is appended by get_cute_tool_message."""




    def test_memory_full_suffix(self):
        fail = json.dumps({"success": False, "error": "would exceed the limit"})
        line = get_cute_tool_message(
            "memory",
            {"action": "add", "target": "memory", "content": "x"},
            0.05,
            result=fail,
        )
        assert "[full]" in line

    def test_success_has_no_suffix(self):
        ok = json.dumps({"success": True, "data": "hi"})
        line = get_cute_tool_message("web_search", {"query": "hi"}, 0.2, result=ok)
        assert "[" not in line.split("0.2s", 1)[1]

