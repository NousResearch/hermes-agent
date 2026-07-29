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

    def test_write_file_content_bool(self):
        """bool value for 'content' should not raise AttributeError."""
        args = json.dumps({"path": "test.txt", "content": False})
        result = _summarize_tool_result("write_file", args, "OK")
        assert "write_file" in result
        assert "test.txt" in result

    def test_write_file_content_int(self):
        """int value for 'content' should not raise AttributeError."""
        args = json.dumps({"path": "test.txt", "content": 123})
        result = _summarize_tool_result("write_file", args, "OK")
        assert "write_file" in result

    def test_delegate_task_goal_bool(self):
        """bool value for 'goal' should not raise TypeError."""
        args = json.dumps({"goal": False})
        result = _summarize_tool_result("delegate_task", args, "result")
        assert "delegate_task" in result
        assert "False" in result

    def test_delegate_task_goal_int(self):
        """int value for 'goal' should not raise TypeError."""
        args = json.dumps({"goal": 999})
        result = _summarize_tool_result("delegate_task", args, "result")
        assert "delegate_task" in result
        assert "999" in result

    def test_execute_code_code_bool(self):
        """bool value for 'code' should not raise TypeError."""
        args = json.dumps({"code": True})
        result = _summarize_tool_result("execute_code", args, "output")
        assert "execute_code" in result
        assert "True" in result

    def test_execute_code_code_int(self):
        """int value for 'code' should not raise TypeError."""
        args = json.dumps({"code": 0})
        result = _summarize_tool_result("execute_code", args, "output")
        assert "execute_code" in result

    def test_vision_analyze_question_bool(self):
        """bool value for 'question' should not raise TypeError."""
        args = json.dumps({"question": True})
        result = _summarize_tool_result("vision_analyze", args, "analysis")
        assert "vision_analyze" in result
        assert "True" in result

    def test_vision_analyze_question_int(self):
        """int value for 'question' should not raise TypeError."""
        args = json.dumps({"question": 123})
        result = _summarize_tool_result("vision_analyze", args, "analysis")
        assert "vision_analyze" in result
        assert "123" in result

    def test_vision_analyze_question_list(self):
        """list value for 'question' should not raise TypeError."""
        args = json.dumps({"question": ["a", "b"]})
        result = _summarize_tool_result("vision_analyze", args, "analysis")
        assert "vision_analyze" in result

    def test_vision_analyze_question_dict(self):
        """dict value for 'question' should not raise TypeError."""
        args = json.dumps({"question": {"key": "value"}})
        result = _summarize_tool_result("vision_analyze", args, "analysis")
        assert "vision_analyze" in result


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

    def test_delegate_task_normal_goal(self):
        """Normal string goal should be summarized correctly."""
        args = json.dumps({"goal": "Fix the bug"})
        result = _summarize_tool_result("delegate_task", args, "done")
        assert "delegate_task" in result
        assert "Fix the bug" in result

    def test_delegate_task_long_goal_truncated(self):
        """Long goals should be truncated."""
        long_goal = "x" * 100
        args = json.dumps({"goal": long_goal})
        result = _summarize_tool_result("delegate_task", args, "done")
        assert "..." in result

    def test_execute_code_normal_code(self):
        """Normal code should be previewed correctly."""
        args = json.dumps({"code": "print('hello')"})
        result = _summarize_tool_result("execute_code", args, "hello")
        assert "execute_code" in result
        assert "print" in result

    def test_execute_code_long_code_truncated(self):
        """Long code should be truncated."""
        long_code = "a = 1\n" * 20
        args = json.dumps({"code": long_code})
        result = _summarize_tool_result("execute_code", args, "output")
        assert "..." in result

    def test_vision_analyze_normal_question(self):
        """Normal question should be included."""
        args = json.dumps({"question": "What is this?"})
        result = _summarize_tool_result("vision_analyze", args, "It's a cat")
        assert "vision_analyze" in result
        assert "What is this?" in result

    def test_vision_analyze_long_question_truncated(self):
        """Long questions should be truncated."""
        long_q = "?" * 100
        args = json.dumps({"question": long_q})
        result = _summarize_tool_result("vision_analyze", args, "answer")
        assert len(result) < 150


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_args(self):
        """Empty JSON object should not crash."""
        result = _summarize_tool_result("terminal", "{}", "output")
        assert "terminal" in result

    def test_invalid_json(self):
        """Invalid JSON should not crash."""
        result = _summarize_tool_result("terminal", "not json", "output")
        assert "terminal" in result

    def test_null_args(self):
        """None/null args should not crash."""
        result = _summarize_tool_result("terminal", None, "output")
        assert "terminal" in result

    def test_non_dict_json_args(self):
        """Args that parse to a non-dict (list/scalar) should not crash."""
        result = _summarize_tool_result("terminal", json.dumps([1, 2]), "output")
        assert "terminal" in result
        result = _summarize_tool_result("terminal", json.dumps("bare"), "output")
        assert "terminal" in result

    def test_unknown_tool_name(self):
        """Unknown tool name should return generic summary."""
        args = json.dumps({"foo": "bar"})
        result = _summarize_tool_result("unknown_tool", args, "output")
        # Should return some fallback, not crash
        assert isinstance(result, str)

    def test_mixed_types_in_args(self):
        """Args with mixed types should not crash."""
        args = json.dumps({
            "command": "ls",
            "background": True,
            "timeout": 30,
            "extra": None
        })
        result = _summarize_tool_result("terminal", args, '{"exit_code": 0}')
        assert "terminal" in result
        assert "ls" in result


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

    def test_process_preview_non_string_session_id(self):
        from agent.display import build_tool_preview
        assert build_tool_preview("process", {"action": "poll", "session_id": 123}) == "poll 123"

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

    def test_process_label_non_string_session_id(self):
        from agent.display import build_tool_label
        result = build_tool_label("process", {"action": "poll", "session_id": 123})
        assert isinstance(result, str)
