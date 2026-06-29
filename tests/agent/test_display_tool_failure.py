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
        assert "cannot access" in suffix
        # Trimmed to the cap, in brackets
        assert suffix.startswith(" [")
        assert suffix.endswith("]")

    def test_malformed_json_returns_no_suffix(self):
        # Terminal is special: only exit_code matters. Malformed JSON should
        # not crash and should not be flagged as failure.
        assert _detect_tool_failure("terminal", "not json") == (False, "")

    def test_none_result_returns_no_suffix(self):
        assert _detect_tool_failure("terminal", None) == (False, "")


class TestDetectToolFailureTerminalBenign:
    """terminal: benign nonzero exits (grep/diff/test=1, interrupt=130) are
    not failures. The tool layer tags known-benign command+code pairs with
    ``exit_code_meaning``; a user interrupt returns 130 with no meaning."""

    def test_grep_no_match_exit1_with_meaning_is_benign(self):
        result = json.dumps({
            "output": "",
            "exit_code": 1,
            "error": None,
            "exit_code_meaning": "No matches found (not an error)",
        })
        assert _detect_tool_failure("terminal", result) == (False, "")

    def test_diff_differs_exit1_with_meaning_is_benign(self):
        result = json.dumps({
            "output": "< old\n> new\n",
            "exit_code": 1,
            "exit_code_meaning": "Files differ",
        })
        assert _detect_tool_failure("terminal", result) == (False, "")

    def test_user_interrupt_exit130_is_benign(self):
        # Interrupt has no exit_code_meaning — handled by the explicit 130 check.
        result = json.dumps({
            "output": "partial\n[Command interrupted]",
            "exit_code": 130,
            "error": None,
        })
        assert _detect_tool_failure("terminal", result) == (False, "")

    def test_sigpipe_exit141_is_benign(self):
        # SIGPIPE (128+13): a downstream reader closed the pipe, e.g.
        # `grep x f | head` under `set -o pipefail`. Carries no meaning —
        # handled by the explicit benign-signal check. Regression for the
        # pipefail halt reported in PR #54637.
        result = json.dumps({
            "output": "first 5 matching lines...\n",
            "exit_code": 141,
            "error": None,
        })
        assert _detect_tool_failure("terminal", result) == (False, "")

    def test_real_failure_codes_without_meaning_still_flagged(self):
        # exit 2 (grep/diff real error), 124 (timeout, stays a failure per D1),
        # 127 (command not found), and crash signals 134/137/139 (SIGABRT/
        # SIGKILL-OOM/SIGSEGV) have no meaning and are not benign signals →
        # flagged. (130/141 are the only benign signal exits.)
        for code in (2, 124, 127, 134, 137, 139):
            result = json.dumps({"output": "", "exit_code": code})
            is_failure, suffix = _detect_tool_failure("terminal", result)
            assert is_failure is True, f"exit {code} should be a failure"
            assert suffix == f" [exit {code}]"

    def test_curl_and_git_failures_carry_no_meaning_and_are_flagged(self):
        # _interpret_exit_code no longer tags curl (DNS/connect/HTTP/timeout) or
        # git (push-rejected/conflict) failures as benign, so their results
        # carry no exit_code_meaning and must flag — the failure reason is in
        # the captured stderr (output).
        for payload in (
            {"output": "curl: (7) Failed to connect to host", "exit_code": 7, "error": None},
            {"output": "! [rejected]        main -> main (non-fast-forward)", "exit_code": 1, "error": None},
        ):
            result = json.dumps(payload)
            is_failure, suffix = _detect_tool_failure("terminal", result)
            assert is_failure is True, f"{payload} should flag"
            assert suffix == f" [exit {payload['exit_code']}]"

    def test_populated_error_field_wins_over_benign_meaning(self):
        # A populated error must still flag even when exit_code_meaning is set:
        # the error-field check runs before the benign short-circuit.
        result = json.dumps({
            "output": "",
            "exit_code": 1,
            "error": "grep: invalid option -- z",
            "exit_code_meaning": "No matches found (not an error)",
        })
        is_failure, suffix = _detect_tool_failure("terminal", result)
        assert is_failure is True
        assert "invalid option" in suffix


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

    def test_error_without_success_key_still_flagged(self):
        # Some tools return {"error": "..."} with no explicit success flag.
        result = json.dumps({"error": "remote unavailable"})
        is_failure, suffix = _detect_tool_failure("web_search", result)
        assert is_failure is True
        assert suffix == " [remote unavailable]"

    def test_message_field_only_with_success_false_flagged(self):
        # When success is False and only 'message' is set, surface it.
        result = json.dumps({"success": False, "message": "rate limited"})
        is_failure, suffix = _detect_tool_failure("web_search", result)
        assert is_failure is True
        assert "rate limited" in suffix

    def test_successful_result_not_flagged(self):
        result = json.dumps({"success": True, "data": "hello"})
        assert _detect_tool_failure("web_search", result) == (False, "")

    def test_dict_without_error_or_success_uses_generic_heuristic(self):
        # Plain successful dict — should pass through the generic
        # heuristic which only fires on the string "Error" / '"error"' / etc.
        result = json.dumps({"data": "hello"})
        is_failure, _ = _detect_tool_failure("web_search", result)
        assert is_failure is False


class TestGetCuteToolMessageFailureSuffix:
    """End-to-end: failure suffix is appended by get_cute_tool_message."""

    def test_read_file_failure_suffix_appended(self):
        fail = json.dumps({
            "path": "/etc/missing",
            "success": False,
            "error": "File not found: /etc/missing",
        })
        line = get_cute_tool_message("read_file", {"path": "/etc/missing"}, 0.1, result=fail)
        assert "[File not found: missing]" in line

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
        assert "command not found" in line
        # No '[exit 127]' tag when we have a specific message
        assert "exit 127" not in line

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

    def test_no_result_has_no_suffix(self):
        # No result passed at all — display function should not invent a
        # failure suffix.
        line = get_cute_tool_message("terminal", {"command": "ls"}, 0.2)
        assert "[" not in line.split("0.2s", 1)[1]
