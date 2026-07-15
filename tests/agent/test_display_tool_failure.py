"""Tests for _detect_tool_failure + _trim_error + get_cute_tool_message
inline failure suffix rendering.

Covers the user-visible promise: when a tool fails, the CLI shows a short,
specific reason in square brackets at the end of the completion line —
not a generic "[error]".
"""

import json

# Load the exact built-in implementations whose registry-bound mechanical
# adapters the display layer is allowed to consult.
import tools.file_tools  # noqa: F401
import tools.terminal_tool  # noqa: F401

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

    def test_whitespace_stripped(self):
        assert _trim_error("  bad input  ") == "bad input"

    def test_long_message_truncated_to_cap(self):
        msg = "x" * 200
        trimmed = _trim_error(msg)
        assert len(trimmed) <= _ERROR_SUFFIX_MAX_LEN
        assert trimmed.endswith("...")

    def test_file_not_found_path_collapsed_to_filename(self):
        long_path = "File not found: /home/teknium/.hermes/hermes-agent/very/deep/path/foo.py"
        assert _trim_error(long_path) == "File not found: foo.py"

    def test_file_not_found_already_short_unchanged(self):
        assert _trim_error("File not found: foo.py") == "File not found: foo.py"

    def test_file_not_found_relative_path_unchanged(self):
        # Without a slash there's no path to trim.
        assert _trim_error("File not found: foo.py") == "File not found: foo.py"


class TestDetectToolFailureTerminal:
    """terminal: non-zero exit_code is the canonical failure signal."""

    def test_success_returns_no_suffix(self):
        result = json.dumps({"output": "ok\n", "exit_code": 0})
        assert _detect_tool_failure("terminal", result) == (False, "")

    def test_nonzero_exit_with_no_error_shows_exit_code(self):
        result = json.dumps({"output": "", "exit_code": 1})
        is_failure, suffix = _detect_tool_failure("terminal", result)
        assert is_failure is True
        assert suffix == " [exit 1]"

    def test_nonzero_exit_with_error_shows_message(self):
        result = json.dumps({
            "output": "",
            "exit_code": 127,
            "error": "ls: cannot access 'foo': No such file or directory",
        })
        is_failure, suffix = _detect_tool_failure("terminal", result)
        assert is_failure is True
        # The terminal adapter owns only the integer exit_code field. Error
        # text remains opaque data for the model rather than UI authority.
        assert suffix == " [exit 127]"

    def test_malformed_json_returns_no_suffix(self):
        # Terminal is special: only exit_code matters. Malformed JSON should
        # not crash and should not be flagged as failure.
        assert _detect_tool_failure("terminal", "not json") == (False, "")

    def test_none_result_returns_no_suffix(self):
        assert _detect_tool_failure("terminal", None) == (False, "")


class TestDetectToolFailureMemory:
    """Unregistered business-result fields remain opaque."""

    def test_memory_failure_fields_do_not_control_ui_status(self):
        result = json.dumps({"success": False, "error": "would exceed the limit"})
        assert _detect_tool_failure("memory", result) == (False, "")

    def test_memory_other_error_is_also_opaque(self):
        result = json.dumps({"success": False, "error": "invalid action: zap"})
        assert _detect_tool_failure("memory", result) == (False, "")


class TestDetectToolFailureStructured:
    """Only exact registry-bound adapters may classify structured results."""

    def test_read_file_error_surfaced(self):
        result = json.dumps({
            "path": "/nope/missing.py",
            "success": False,
            "error": "File not found: /nope/missing.py",
        })
        is_failure, suffix = _detect_tool_failure("read_file", result)
        assert is_failure is True
        assert suffix == " [error]"

    def test_error_without_registered_adapter_is_opaque(self):
        result = json.dumps({"error": "remote unavailable"})
        assert _detect_tool_failure("web_search", result) == (False, "")

    def test_status_and_success_fields_without_adapter_are_opaque(self):
        result = json.dumps({"success": False, "message": "rate limited"})
        assert _detect_tool_failure("web_search", result) == (False, "")
        assert _detect_tool_failure(
            "web_search", json.dumps({"status": "failed", "error": "boom"})
        ) == (False, "")

    def test_successful_result_not_flagged(self):
        result = json.dumps({"success": True, "data": "hello"})
        assert _detect_tool_failure("web_search", result) == (False, "")

    def test_dict_without_error_or_success_is_not_a_failure(self):
        result = json.dumps({"data": "hello"})
        is_failure, _ = _detect_tool_failure("web_search", result)
        assert is_failure is False

    def test_opaque_prose_is_not_interpreted_as_failure(self):
        for result in (
            "Error is the subject of this document",
            "the deployment failed yesterday but is healthy now",
            'literal JSON example: {"error": "boom"}',
        ):
            assert _detect_tool_failure("read_file", result) == (False, "")


class TestGetCuteToolMessageFailureSuffix:
    """End-to-end: failure suffix is appended by get_cute_tool_message."""

    def test_read_file_failure_suffix_appended(self):
        fail = json.dumps({
            "path": "/etc/missing",
            "success": False,
            "error": "File not found: /etc/missing",
        })
        line = get_cute_tool_message("read_file", {"path": "/etc/missing"}, 0.1, result=fail)
        assert "[error]" in line

    def test_terminal_exit_only_suffix(self):
        fail = json.dumps({"output": "", "exit_code": 2})
        line = get_cute_tool_message("terminal", {"command": "false"}, 0.1, result=fail)
        assert "[exit 2]" in line

    def test_terminal_with_stderr_uses_message(self):
        fail = json.dumps({
            "output": "",
            "exit_code": 127,
            "error": "command not found: notathing",
        })
        line = get_cute_tool_message("terminal", {"command": "notathing"}, 0.1, result=fail)
        assert "command not found" not in line
        assert "[exit 127]" in line

    def test_memory_business_failure_does_not_add_ui_suffix(self):
        fail = json.dumps({"success": False, "error": "would exceed the limit"})
        line = get_cute_tool_message(
            "memory",
            {"action": "add", "target": "memory", "content": "x"},
            0.05,
            result=fail,
        )
        assert "would exceed" not in line

    def test_success_has_no_suffix(self):
        ok = json.dumps({"success": True, "data": "hi"})
        line = get_cute_tool_message("web_search", {"query": "hi"}, 0.2, result=ok)
        assert "[" not in line.split("0.2s", 1)[1]

    def test_no_result_has_no_suffix(self):
        # No result passed at all — display function should not invent a
        # failure suffix.
        line = get_cute_tool_message("terminal", {"command": "ls"}, 0.2)
        assert "[" not in line.split("0.2s", 1)[1]
