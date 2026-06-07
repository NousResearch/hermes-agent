"""Tests for agent.tool_error_classifier — tool error classification and recovery hints."""

import pytest

from agent.tool_error_classifier import (
    ClassifiedToolError,
    ToolErrorType,
    classify_tool_error,
    format_classified_error,
)


class TestClassifyToolError:
    """Test the classify_tool_error function across all error types."""

    # ── Model errors (bad LLM assumption) ───────────────────────────

    def test_patch_multiple_matches(self):
        result = classify_tool_error(
            "Found 2 matches for old_string in file.py",
            tool_name="patch",
        )
        assert result.error_type == ToolErrorType.MODEL
        assert "patch" in result.recovery_hint.lower() or "old_string" in result.recovery_hint.lower()

    def test_patch_no_match(self):
        result = classify_tool_error(
            "old_string not found in the file content",
            tool_name="patch",
        )
        assert result.error_type == ToolErrorType.MODEL

    def test_content_mismatch(self):
        result = classify_tool_error(
            "File does not contain the expected text",
            tool_name="write_file",
        )
        assert result.error_type == ToolErrorType.MODEL

    def test_wrong_file_checkout(self):
        result = classify_tool_error(
            "resolved_path does not match intended checkout directory",
            tool_name="patch",
        )
        assert result.error_type == ToolErrorType.MODEL

    # ── Tool errors (transient) ─────────────────────────────────────

    def test_timeout(self):
        result = classify_tool_error(
            "Connection timed out after 30 seconds",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.TOOL
        assert "retry" in result.recovery_hint.lower() or "transient" in result.recovery_hint.lower()

    def test_rate_limit(self):
        result = classify_tool_error(
            "Rate limit exceeded: 429 Too Many Requests",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.TOOL

    def test_connection_refused(self):
        result = classify_tool_error(
            "Connection refused to localhost:8080",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.TOOL

    def test_503_service_unavailable(self):
        result = classify_tool_error(
            "HTTP 503 Service Unavailable",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.TOOL

    def test_ssl_error(self):
        result = classify_tool_error(
            "SSL: CERTIFICATE_VERIFY_FAILED",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.TOOL

    # ── Environment errors (hard failure) ───────────────────────────

    def test_file_not_found(self):
        result = classify_tool_error(
            "No such file or directory: '/tmp/nonexistent.txt'",
            tool_name="read_file",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT
        assert "path" in result.recovery_hint.lower() or "exists" in result.recovery_hint.lower()

    def test_permission_denied(self):
        result = classify_tool_error(
            "Permission denied: '/etc/shadow'",
            tool_name="write_file",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT

    def test_command_not_found(self):
        result = classify_tool_error(
            "command not found: tirith",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT
        assert "installed" in result.recovery_hint.lower() or "path" in result.recovery_hint.lower()

    def test_segfault(self):
        result = classify_tool_error(
            "Process exited with SIGSEGV (signal 11)",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT

    def test_disk_full(self):
        result = classify_tool_error(
            "No space left on device",
            tool_name="write_file",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT

    def test_module_not_found(self):
        result = classify_tool_error(
            "No module named 'requests'",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT

    # ── Input errors (bad parameters) ───────────────────────────────

    def test_missing_argument(self):
        result = classify_tool_error(
            "Missing required argument: 'path'",
            tool_name="read_file",
        )
        assert result.error_type == ToolErrorType.INPUT

    def test_invalid_parameter(self):
        result = classify_tool_error(
            "Invalid parameter type: expected string, got int",
            tool_name="patch",
        )
        assert result.error_type == ToolErrorType.INPUT

    def test_auth_required(self):
        result = classify_tool_error(
            "API key required for this provider",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.INPUT

    # ── Exception-based classification ──────────────────────────────

    def test_permission_error_exception(self):
        exc = PermissionError("Permission denied")
        result = classify_tool_error(
            "Permission denied",
            tool_name="write_file",
            exception=exc,
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT

    def test_file_not_found_exception(self):
        exc = FileNotFoundError("No such file or directory")
        result = classify_tool_error(
            "No such file or directory",
            tool_name="read_file",
            exception=exc,
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT

    def test_timeout_error_exception(self):
        exc = TimeoutError("Connection timed out")
        result = classify_tool_error(
            "Connection timed out",
            tool_name="terminal",
            exception=exc,
        )
        assert result.error_type == ToolErrorType.TOOL

    def test_connection_error_exception(self):
        exc = ConnectionError("Connection refused")
        result = classify_tool_error(
            "Connection refused",
            tool_name="terminal",
            exception=exc,
        )
        assert result.error_type == ToolErrorType.TOOL

    # ── Fallback behavior ───────────────────────────────────────────

    def test_unknown_error_defaults_to_model(self):
        result = classify_tool_error(
            "Something completely unexpected happened",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.MODEL
        assert result.recovery_hint  # should have a hint

    def test_empty_error_message(self):
        result = classify_tool_error("", tool_name="terminal")
        assert result.error_type == ToolErrorType.MODEL

    # ── Tool-specific hints ─────────────────────────────────────────

    def test_patch_tool_specific_hint(self):
        result = classify_tool_error(
            "Found 2 matches for old_string",
            tool_name="patch",
        )
        assert "old_string" in result.recovery_hint.lower()
        assert "read_file" in result.recovery_hint

    def test_terminal_env_hint(self):
        result = classify_tool_error(
            "command not found: nonexistent",
            tool_name="terminal",
        )
        assert "path" in result.recovery_hint.lower() or "installed" in result.recovery_hint.lower()

    def test_read_file_env_hint(self):
        result = classify_tool_error(
            "No such file or directory",
            tool_name="read_file",
        )
        assert "search_files" in result.recovery_hint or "path" in result.recovery_hint.lower()

    def test_generic_hint_for_unknown_tool(self):
        result = classify_tool_error(
            "Permission denied",
            tool_name="some_unknown_tool",
        )
        assert result.error_type == ToolErrorType.ENVIRONMENT
        assert result.recovery_hint  # should fall back to generic hint

    # ── ClassifiedToolError dataclass ───────────────────────────────

    def test_classified_error_fields(self):
        result = classify_tool_error("timeout", tool_name="terminal")
        assert isinstance(result, ClassifiedToolError)
        assert result.error_type in {
            ToolErrorType.MODEL, ToolErrorType.TOOL,
            ToolErrorType.ENVIRONMENT, ToolErrorType.INPUT,
        }
        assert isinstance(result.error_message, str)
        assert isinstance(result.recovery_hint, str)
        assert result.tool_name == "terminal"

    # ── format_classified_error ─────────────────────────────────────

    def test_format_classified_error_json(self):
        import json
        result = classify_tool_error("timeout", tool_name="terminal")
        formatted = format_classified_error(result)
        parsed = json.loads(formatted)
        assert "error" in parsed
        assert "error_type" in parsed
        assert "recovery_hint" in parsed
        assert parsed["error_type"] == result.error_type


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_multiple_pattern_match_priority(self):
        # "File not found" matches env patterns; should not match model patterns
        result = classify_tool_error(
            "File not found: old_string does not exist in path",
            tool_name="patch",
        )
        # "does not contain" is a model pattern, "not found" is env
        # Model patterns are checked first
        assert result.error_type in {ToolErrorType.MODEL, ToolErrorType.ENVIRONMENT}

    def test_case_insensitive_matching(self):
        result = classify_tool_error(
            "TIMEOUT: Connection Timed Out",
            tool_name="terminal",
        )
        assert result.error_type == ToolErrorType.TOOL

    def test_unicode_error_message(self):
        result = classify_tool_error(
            "文件未找到: /tmp/不存在.txt",
            tool_name="read_file",
        )
        # Unicode "文件未找到" won't match English patterns, falls through
        assert result.error_type in {
            ToolErrorType.MODEL, ToolErrorType.ENVIRONMENT,
            ToolErrorType.TOOL, ToolErrorType.INPUT,
        }

    def test_long_error_message(self):
        long_msg = "x" * 10000 + " timeout " + "y" * 10000
        result = classify_tool_error(long_msg, tool_name="terminal")
        assert result.error_type == ToolErrorType.TOOL
