"""Type safety tests for _summarize_tool_result.

When LLMs return non-string parameter values (e.g. bool, int, None) in tool
call arguments, _summarize_tool_result() must not crash with TypeError or
AttributeError. This caused an infinite TUI crash loop in production.
"""
import json
import pytest
from agent.context_compressor import _summarize_tool_result


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
        assert len(result) < 150

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
            "cronjob_manage", "process_manage", "totally_unknown_tool",
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
            "process_manage", {"action": "submit", "session_id": "abc", "data": 42}
        )
        assert result == 'submit abc "42"'

    def test_process_preview_none_action(self):
        from agent.display import build_tool_preview
        result = build_tool_preview("process_manage", {"action": None, "session_id": "abc"})
        assert isinstance(result, str)

