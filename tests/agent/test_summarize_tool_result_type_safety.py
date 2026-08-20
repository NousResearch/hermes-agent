"""Type safety tests for _summarize_tool_result.

When LLMs return non-string parameter values (e.g. bool, int, None) in tool
call arguments, _summarize_tool_result() must not crash with TypeError or
AttributeError. This caused an infinite TUI crash loop in production.
"""
import json
import pytest
from agent.context_compressor import (
    _summarize_tool_result,
    _terminal_output_head,
    _ARCHIVED_MARKER,
    ContextCompressor,
)


class TestTypeSafety:
    """Non-string tool arguments must not crash _summarize_tool_result."""

    def test_terminal_command_bool(self):
        """bool value for 'command' should not raise TypeError."""
        args = json.dumps({"command": True})
        result = _summarize_tool_result("terminal", args, '{"exit_code": 0}')
        assert "terminal" in result
        assert "True" in result or "exit" in result

    def test_terminal_command_int(self):
        """int value for 'command' should not raise TypeError."""
        args = json.dumps({"command": 42})
        result = _summarize_tool_result("terminal", args, '{"exit_code": 0}')
        assert "terminal" in result
        assert "42" in result

    def test_terminal_command_none(self):
        """None value for 'command' should not raise TypeError."""
        args = json.dumps({"command": None})
        result = _summarize_tool_result("terminal", args, '{"exit_code": 0}')
        assert "terminal" in result












class TestNormalStringArguments:
    """Normal string arguments should continue to work as before."""

    def test_terminal_normal_command(self):
        """Normal string command should be summarized correctly."""
        args = json.dumps({"command": "ls -la"})
        result = _summarize_tool_result("terminal", args, '{"exit_code": 0}')
        assert "terminal" in result
        assert "ls -la" in result
        assert "exit 0" in result

    def test_terminal_long_command_truncated(self):
        """Long commands should be truncated."""
        long_cmd = "a" * 100
        args = json.dumps({"command": long_cmd})
        result = _summarize_tool_result("terminal", args, '{"exit_code": 0}')
        assert "..." in result
        # Summary now includes output head + archived marker, so the total
        # length is longer. The command itself must still be truncated.
        assert "aaa..." in result

    def test_write_file_normal_content(self):
        """Normal string content should count lines correctly."""
        args = json.dumps({"path": "test.py", "content": "line1\nline2\nline3"})
        result = _summarize_tool_result("write_file", args, "OK")
        assert "write_file" in result
        assert "test.py" in result
        assert "3 lines" in result








class TestEdgeCases:
    """Edge cases and boundary conditions."""



    def test_null_args(self):
        """None/null args should not crash."""
        result = _summarize_tool_result("terminal", None, "output")
        assert "terminal" in result


    def test_unknown_tool_name(self):
        """Unknown tool name should return generic summary."""
        args = json.dumps({"foo": "bar"})
        result = _summarize_tool_result("unknown_tool", args, "output")
        # Should return some fallback, not crash
        assert isinstance(result, str)



class TestBackstopWrapper:
    """The outer guard: NO input shape may raise out of _summarize_tool_result.

    Compression retries on the same persisted history, so an escaping
    exception here becomes a crash loop. The wrapper returns a minimal
    '[tool] (N chars result)' summary when a branch fails.
    """

    def test_never_raises_matrix(self):
        """Fuzz the per-tool branches with hostile value shapes."""
        hostile_values = [None, True, 42, 3.14, ["a"], {"k": "v"}]
        tools = [
            "terminal", "read_file", "write_file", "search_files", "patch",
            "browser_navigate", "web_search", "web_extract", "delegate_task",
            "execute_code", "skill_view", "vision_analyze", "memory",
            "cronjob", "process", "totally_unknown_tool",
        ]
        keys = ["command", "path", "content", "pattern", "url", "query",
                "urls", "goal", "code", "name", "question", "action",
                "target", "session_id", "mode", "offset", "ref"]
        for tool in tools:
            for value in hostile_values:
                args = json.dumps({k: value for k in keys})
                result = _summarize_tool_result(tool, args, "x" * 250)
                assert isinstance(result, str) and result, (tool, value)

    def test_backstop_fallback_shape(self):
        """When a branch does fail, the fallback names the tool and size."""
        from unittest.mock import patch as _patch
        with _patch(
            "agent.context_compressor._summarize_tool_result_unguarded",
            side_effect=TypeError("boom"),
        ):
            result = _summarize_tool_result("terminal", "{}", "y" * 300)
        assert result == "[terminal] (300 chars result)"

    def test_backstop_handles_non_string_content(self):
        from unittest.mock import patch as _patch
        with _patch(
            "agent.context_compressor._summarize_tool_result_unguarded",
            side_effect=TypeError("boom"),
        ):
            result = _summarize_tool_result("terminal", "{}", None)
        assert result == "[terminal] (0 chars result)"


class TestDisplayPreviewTypeSafety:
    """Sibling site: agent/display.py previews run on the live
    tool-progress callback and crashed on non-string process args."""


    def test_process_preview_non_string_data(self):
        from agent.display import build_tool_preview
        result = build_tool_preview(
            "process", {"action": "submit", "session_id": "abc", "data": 42}
        )
        assert result == 'submit abc "42"'

    def test_process_preview_none_action(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("process", {"action": None, "session_id": "abc"})
        assert isinstance(result, str)


class TestTerminalOutputHeadPreservation:
    """The _terminal_output_head function must preserve evidence of
    application-level failures even when exit_code is 0."""

    def test_exit_zero_with_failure_body(self):
        """A curl command that exits 0 but returns a 422 body must carry
        the body head into the compressed summary."""
        content = json.dumps({
            "output": "HTTP 422 Unprocessable Entity\n{\"error\": \"Bad request\"}",
            "exit_code": 0,
            "error": None
        })
        head = _terminal_output_head(content)
        assert "422" in head
        assert "Unprocessable" in head

    def test_exit_zero_empty_output(self):
        content = json.dumps({"output": "", "exit_code": 0, "error": None})
        head = _terminal_output_head(content)
        assert head == ""

    def test_non_json_content_falls_back_to_raw(self):
        head = _terminal_output_head("plain text output line one")
        assert "plain text" in head

    def test_long_output_truncated(self):
        long_output = "A" * 500
        content = json.dumps({"output": long_output, "exit_code": 0})
        head = _terminal_output_head(content, limit=50)
        assert len(head) <= 50
        assert head.endswith("\u2026")

    def test_content_with_newlines_collapsed(self):
        content = json.dumps({
            "output": "line1\nline2\nline3",
            "exit_code": 0
        })
        head = _terminal_output_head(content)
        assert " " in head
        assert "\n" not in head


class TestArchivedMarkerIdempotence:
    """_prune_old_tool_results must not re-summarize messages that
    already contain the _ARCHIVED_MARKER."""

    def test_marker_prevents_redemotion(self):
        """A tool result that was already summarized (contains _ARCHIVED_MARKER)
        must survive a second prune pass unchanged."""
        summarized_content = (
            "[terminal] ran `npm test` -> exit 0, 47 lines, 12,000 chars "
            "| output starts: all tests passed "
            f"| {_ARCHIVED_MARKER}"
        )
        # Build a minimal message list: assistant tool_call + tool result
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"command": "npm test"})}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": summarized_content},
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "Tests passed."},
            {"role": "user", "content": "Great, now run deploy."},
        ]
        # _prune_old_tool_results doesn't access self (only module-level
        # functions and locals), so __new__ bypasses __init__ safely.
        compressor = ContextCompressor.__new__(ContextCompressor)
        pruned, count = compressor._prune_old_tool_results(
            messages, protect_tail_count=3
        )
        # The already-summarized tool result must be unchanged
        assert count == 0
        assert pruned[1]["content"] == summarized_content

    def test_large_summarized_result_not_redemoted(self):
        """Even if the summarized result exceeds min_prune_chars, the
        _ARCHIVED_MARKER guard must prevent re-summarization."""
        long_head = "A" * 300
        summarized_content = (
            f"[terminal] ran `build` -> exit 0, 500 lines, 50,000 chars "
            f"| output starts: {long_head} "
            f"| {_ARCHIVED_MARKER}"
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"command": "build"})}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": summarized_content},
            {"role": "user", "content": "Next step."},
        ]
        compressor = ContextCompressor.__new__(ContextCompressor)
        pruned, count = compressor._prune_old_tool_results(
            messages, protect_tail_count=1, min_prune_chars=200
        )
        assert count == 0
        assert pruned[1]["content"] == summarized_content

    def test_fresh_terminal_result_gets_summarized(self):
        """Verify the prune still works on fresh (non-archived) tool results."""
        large_output = json.dumps({"output": "x" * 5000, "exit_code": 0, "error": None})
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": json.dumps({"command": "test"})}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": large_output},
            {"role": "user", "content": "Done?"},
        ]
        compressor = ContextCompressor.__new__(ContextCompressor)
        pruned, count = compressor._prune_old_tool_results(
            messages, protect_tail_count=1, min_prune_chars=200
        )
        assert count == 1
        assert _ARCHIVED_MARKER in pruned[1]["content"]
        assert "exit 0" in pruned[1]["content"]


