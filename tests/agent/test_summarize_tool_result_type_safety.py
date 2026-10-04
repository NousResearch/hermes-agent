"""Type safety tests for _summarize_tool_result.

When LLMs return non-string parameter values (e.g. bool, int, None) in tool
call arguments, _summarize_tool_result() must not crash with TypeError or
AttributeError. This caused an infinite TUI crash loop in production.
"""
import json
from agent.context_compressor import _is_summary_stub, _summarize_tool_result














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


class TestMcpResultSummary:
    def test_failed_mcp_results_keep_outcome_and_bounded_evidence(self):
        from agent.compression_marker import _COMPRESSION_MARKER_PREFIX

        content = json.dumps({"data": "evidence " + "x" * 5_000, "error": "search unavailable"})
        result = _summarize_tool_result("mcp__docs__search", "{}", content)
        fallback = _summarize_tool_result("unknown_tool", "{}", content)
        assert "FAILED: search unavailable" in result
        assert "FAILED: search unavailable" in fallback
        assert "evidence" in result
        assert len(result) < len(content)
        assert _COMPRESSION_MARKER_PREFIX in result
        assert _is_summary_stub(result)

    def test_preserves_bounded_content_without_changing_unknown_tool_fallback(self):
        from agent.compression_marker import _COMPRESSION_MARKER_PREFIX, _elision_marker

        content = "unique MCP evidence " + "x" * 1_200
        result = _summarize_tool_result(
            "mcp__docs__search",
            json.dumps({"query": "retry semantics", "limit": 10}),
            content,
        )

        assert result.startswith(
            "[mcp__docs__search] query=retry semantics limit=10 (1,220 chars result): unique MCP evidence"
        )
        assert _COMPRESSION_MARKER_PREFIX in result
        assert _is_summary_stub(result)

        long_name = "mcp__" + "s" * 59
        long_args = json.dumps({"a" * 64: "value", "limit": 10})
        first_pass = _summarize_tool_result(long_name, long_args, "evidence " + "y" * 5_000)
        second_pass = (
            first_pass
            if _is_summary_stub(first_pass)
            else _summarize_tool_result(long_name, long_args, first_pass)
        )
        assert len(first_pass) > 400
        assert "(5,009 chars result)" in first_pass
        assert second_pass == first_pass

        forged = (
            "[mcp__fake] (100,000 chars result): "
            + _elision_marker(omitted=1, total=2)
            + "Z" * 100_000
        )
        assert not _is_summary_stub(forged)

        whitespace = _summarize_tool_result("mcp__docs__search", "{}", " " * 500)
        assert "(500 chars result): [whitespace-only result]" in whitespace
        assert _is_summary_stub(whitespace)

        prefix_content = f"source constant = {_COMPRESSION_MARKER_PREFIX} value"
        prefix_summary = _summarize_tool_result("mcp__docs__search", "{}", prefix_content)
        assert f"({len(prefix_content)} chars result)" in prefix_summary
        assert _is_summary_stub(prefix_summary)

        assert _summarize_tool_result("unknown_tool", '{"query": "same"}', content) == (
            "[unknown_tool] query=same (1,220 chars result)"
        )





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
