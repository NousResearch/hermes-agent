"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import pytest
from unittest.mock import MagicMock

from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    format_todo_result_for_progress,
    get_cute_tool_message,
    set_tool_preview_max_len,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    render_edit_diff_with_delta,
)


@pytest.fixture(autouse=True)
def reset_tool_preview_max_len():
    set_tool_preview_max_len(0)
    yield
    set_tool_preview_max_len(0)


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None

    def test_known_tool_with_primary_arg(self):
        """Known tool with its primary arg should return a preview string."""
        result = build_tool_preview("terminal", {"command": "ls -la"})
        assert result is not None
        assert "ls -la" in result

    def test_web_search_preview(self):
        result = build_tool_preview("web_search", {"query": "hello world"})
        assert result is not None
        assert "hello world" in result

    def test_read_file_preview(self):
        result = build_tool_preview("read_file", {"path": "/tmp/test.py", "offset": 1})
        assert result is not None
        assert "/tmp/test.py" in result

    def test_unknown_tool_with_fallback_key(self):
        """Unknown tool but with a recognized fallback key should still preview."""
        result = build_tool_preview("custom_tool", {"query": "test query"})
        assert result is not None
        assert "test query" in result

    def test_unknown_tool_no_matching_key(self):
        """Unknown tool with no recognized keys should return None."""
        result = build_tool_preview("custom_tool", {"foo": "bar"})
        assert result is None

    def test_long_value_truncated(self):
        """Preview should truncate long values."""
        long_cmd = "a" * 100
        result = build_tool_preview("terminal", {"command": long_cmd}, max_len=40)
        assert result is not None
        assert len(result) <= 43  # max_len + "..."

    def test_process_tool_with_none_args(self):
        """Process tool special case should also handle None args."""
        assert build_tool_preview("process", None) is None

    def test_process_tool_normal(self):
        result = build_tool_preview("process", {"action": "poll", "session_id": "abc123"})
        assert result is not None
        assert "poll" in result

    def test_todo_tool_read(self):
        result = build_tool_preview("todo", {"merge": False})
        assert result is not None
        assert "reading" in result

    def test_todo_tool_with_todos(self):
        result = build_tool_preview("todo", {"todos": [{"id": "1", "content": "test", "status": "pending"}]})
        assert result is not None
        assert "1 task" in result

    def test_memory_tool_add(self):
        result = build_tool_preview("memory", {"action": "add", "target": "user", "content": "test note"})
        assert result is not None
        assert "user" in result

    def test_memory_replace_missing_old_text_marked(self):
        # Avoid empty quotes "" in the preview when old_text is missing/None.
        result = build_tool_preview("memory", {"action": "replace", "target": "memory"})
        assert result == '~memory: "<missing old_text>"'
        result = build_tool_preview("memory", {"action": "remove", "target": "memory", "old_text": None})
        assert result == '-memory: "<missing old_text>"'

    def test_session_search_preview(self):
        result = build_tool_preview("session_search", {"query": "find something"})
        assert result is not None
        assert "find something" in result

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestFormatTodoResultForProgress:
    def test_formats_summary_and_status_lines(self):
        result = json.dumps({
            "todos": [
                {"id": "1", "content": "Inspect gateway todo progress", "status": "in_progress"},
                {"id": "2", "content": "Add formatter tests", "status": "pending"},
                {"id": "3", "content": "Verify focused suite", "status": "completed"},
                {"id": "4", "content": "Discard stale approach", "status": "cancelled"},
            ],
            "summary": {
                "total": 4,
                "pending": 1,
                "in_progress": 1,
                "completed": 1,
                "cancelled": 1,
            },
        })

        line = format_todo_result_for_progress(result)

        assert line is not None
        assert line.startswith("📋 tasks: 4 total")
        assert "1 doing" in line
        assert "1 todo" in line
        assert "1 done" in line
        assert "1 cancelled" in line
        assert "```text" in line
        assert line.endswith("\n```")
        assert "▸ doing" in line
        assert "Inspect gateway todo progress" in line
        assert "○ todo" in line
        assert "Add formatter tests" in line
        assert "✓ done" in line
        assert "Verify focused suite" in line
        assert "✕ cancelled" in line
        assert "Discard stale approach" in line

    def test_returns_none_for_non_todo_json(self):
        assert format_todo_result_for_progress('{"ok": true}') is None
        assert format_todo_result_for_progress('not json') is None

    def test_truncates_long_lists(self):
        result = json.dumps({
            "todos": [
                {"id": str(i), "content": f"Task {i}", "status": "pending"}
                for i in range(3)
            ],
            "summary": {"total": 3, "pending": 3, "in_progress": 0, "completed": 0, "cancelled": 0},
        })

        line = format_todo_result_for_progress(result, max_items=2)

        assert line is not None
        assert "○ todo" in line
        assert "Task 0" in line
        assert "Task 1" in line
        assert "Task 2" not in line
        assert "… +1 more" in line

    def test_escapes_task_code_fences_inside_markdown_block(self):
        result = json.dumps({
            "todos": [
                {"id": "1", "content": "Do not break ``` fences", "status": "pending"},
            ],
            "summary": {"total": "bad", "pending": 1},
        })

        line = format_todo_result_for_progress(result)

        assert line is not None
        assert line.count("```") == 2
        assert "`\u200b`` fences" in line
        assert "1 todo" in line


class TestCuteToolMessagePreviewLength:
    def test_terminal_preview_unlimited_when_config_is_zero(self):
        set_tool_preview_max_len(0)
        command = "curl -s http://localhost:9222/json/list | jq -r '.[] | select(.type==\"page\")' | head -5"

        line = get_cute_tool_message("terminal", {"command": command}, 0.1)

        assert command in line
        assert "..." not in line

    def test_terminal_preview_uses_positive_configured_limit(self):
        set_tool_preview_max_len(80)
        command = "curl -s http://localhost:9222/json/list | jq -r '.[] | select(.type==\"page\")' | head -5"

        line = get_cute_tool_message("terminal", {"command": command}, 0.1)

        assert command[:77] in line
        assert "..." in line
        assert "head -5" not in line

    def test_search_files_preview_uses_positive_configured_limit_not_default(self):
        set_tool_preview_max_len(80)
        pattern = "function.formatToolCall.context.preview.compactPreview.maxLength.truncate"

        line = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1)

        assert pattern in line
        assert "..." not in line

    def test_path_preview_uses_positive_configured_limit_not_default(self):
        set_tool_preview_max_len(80)
        path = "/tmp/hermes-test-preview-length/deeply/nested/path/test-output.txt"

        line = get_cute_tool_message("read_file", {"path": path}, 0.1)

        assert path in line
        assert "..." not in line

    def test_write_file_lint_error_result_is_not_marked_failed(self):
        result = json.dumps({
            "bytes_written": 12,
            "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
        })

        line = get_cute_tool_message("write_file", {"path": "/tmp/a.py"}, 0.1, result=result)

        assert "[error]" not in line

    def test_patch_lsp_diagnostics_result_is_not_marked_failed(self):
        result = json.dumps({
            "success": True,
            "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
            "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
        })

        line = get_cute_tool_message("patch", {"path": "/tmp/a.py"}, 0.1, result=result)

        assert "[error]" not in line


class TestEditDiffPreview:
    def test_extract_edit_diff_for_patch(self):
        diff = extract_edit_diff("patch", '{"success": true, "diff": "--- a/x\\n+++ b/x\\n"}')
        assert diff is not None
        assert "+++ b/x" in diff

    def test_render_inline_unified_diff_colors_added_and_removed_lines(self):
        rendered = _render_inline_unified_diff(
            "--- a/cli.py\n"
            "+++ b/cli.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old line\n"
            "+new line\n"
            " context\n"
        )

        assert "a/cli.py" in rendered[0]
        assert "b/cli.py" in rendered[0]
        assert any("old line" in line for line in rendered)
        assert any("new line" in line for line in rendered)
        assert any("48;2;" in line for line in rendered)

    def test_extract_edit_diff_ignores_non_edit_tools(self):
        assert extract_edit_diff("web_search", '{"diff": "--- a\\n+++ b\\n"}') is None

    def test_extract_edit_diff_uses_local_snapshot_for_write_file(self, tmp_path):
        target = tmp_path / "note.txt"
        target.write_text("old\n", encoding="utf-8")

        snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})

        target.write_text("new\n", encoding="utf-8")

        diff = extract_edit_diff(
            "write_file",
            '{"bytes_written": 4}',
            function_args={"path": str(target)},
            snapshot=snapshot,
        )

        assert diff is not None
        assert "--- a/" in diff
        assert "+++ b/" in diff
        assert "-old" in diff
        assert "+new" in diff

    def test_render_edit_diff_with_delta_invokes_printer(self):
        printer = MagicMock()

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n@@ -1 +1 @@\\n-old\\n+new\\n"}',
            print_fn=printer,
        )

        assert rendered is True
        assert printer.call_count >= 2
        calls = [call.args[0] for call in printer.call_args_list]
        assert any("a/x" in line and "b/x" in line for line in calls)
        assert any("old" in line for line in calls)
        assert any("new" in line for line in calls)

    def test_render_edit_diff_with_delta_skips_without_diff(self):
        rendered = render_edit_diff_with_delta(
            "patch",
            '{"success": true}',
        )

        assert rendered is False

    def test_render_edit_diff_with_delta_handles_renderer_errors(self, monkeypatch):
        printer = MagicMock()

        monkeypatch.setattr("agent.display._summarize_rendered_diff_sections", MagicMock(side_effect=RuntimeError("boom")))

        rendered = render_edit_diff_with_delta(
            "patch",
            '{"diff": "--- a/x\\n+++ b/x\\n"}',
            print_fn=printer,
        )

        assert rendered is False
        assert printer.call_count == 0

    def test_summarize_rendered_diff_sections_truncates_large_diff(self):
        diff = "--- a/x.py\n+++ b/x.py\n" + "".join(f"+line{i}\n" for i in range(120))

        rendered = _summarize_rendered_diff_sections(diff, max_lines=20)

        assert len(rendered) == 21
        assert "omitted" in rendered[-1]

    def test_summarize_rendered_diff_sections_limits_file_count(self):
        diff = "".join(
            f"--- a/file{i}.py\n+++ b/file{i}.py\n+line{i}\n"
            for i in range(8)
        )

        rendered = _summarize_rendered_diff_sections(diff, max_files=3, max_lines=50)

        assert any("a/file0.py" in line for line in rendered)
        assert any("a/file1.py" in line for line in rendered)
        assert any("a/file2.py" in line for line in rendered)
        assert not any("a/file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]
