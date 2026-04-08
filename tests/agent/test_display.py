"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import os
import re
import pytest
from unittest.mock import MagicMock, patch

from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    _highlight_block,
    _render_inline_unified_diff,
    _result_succeeded,
    _summarize_rendered_diff_sections,
    get_cute_tool_message,
    render_edit_diff_with_delta,
    render_execute_code_preview,
    render_read_file_preview,
    render_terminal_preview,
    set_code_highlight_active,
    set_diff_limits,
    set_preview_max_lines,
)


@pytest.fixture(autouse=True)
def _disable_no_color(monkeypatch):
    """Inline diff/style assertions expect ANSI styling to be enabled."""
    monkeypatch.delenv("NO_COLOR", raising=False)


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

    def test_session_search_preview(self):
        result = build_tool_preview("session_search", {"query": "find something"})
        assert result is not None
        assert "find something" in result

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


class TestEditDiffPreview:
    def test_extract_edit_diff_for_patch(self):
        diff = extract_edit_diff("patch", '{"success": true, "diff": "--- a/x\\n+++ b/x\\n"}')
        assert diff is not None
        assert "+++ b/x" in diff

    def test_render_inline_unified_diff_colors_added_and_removed_lines(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        rendered = _render_inline_unified_diff(
            "--- a/cli.py\n"
            "+++ b/cli.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-old line\n"
            "+new line\n"
            " context\n"
        )

        import re as _re
        _strip = lambda s: _re.sub(r"\x1b\[[0-9;]*m", "", s)
        stripped = [_strip(l) for l in rendered]
        # header shows basename + change summary, not a/x → b/x
        assert "cli.py" in stripped[0]
        assert any("old line" in l for l in stripped)
        assert any("new line" in l for l in stripped)
        assert any("\x1b[" in line for line in rendered)

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
        # header shows basename + change summary
        assert any("x" in line for line in calls)
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

        # header shows basename only
        assert any("file0.py" in line for line in rendered)
        assert any("file1.py" in line for line in rendered)
        assert any("file2.py" in line for line in rendered)
        assert not any("file7.py" in line for line in rendered)
        assert "additional file" in rendered[-1]


# ---------------------------------------------------------------------------
# _highlight_block
# ---------------------------------------------------------------------------

class TestHighlightBlock:
    def _collect(self, header, content, language="python"):
        calls = []
        _highlight_block(header, content, language, calls.append)
        return calls

    def test_header_uses_pipe_prefix(self):
        calls = self._collect("📄 foo.py", "x = 1")
        assert calls[0].startswith("\033[2m  ┊ ")

    def test_header_contains_label(self):
        calls = self._collect("📄 foo.py", "x = 1")
        assert "📄 foo.py" in calls[0]

    def test_no_separator_line(self):
        calls = self._collect("📄 foo.py", "x = 1\ny = 2")
        stripped = [re.sub(r"\x1b\[[0-9;]*m", "", c).strip() for c in calls]
        # No line should consist solely of box-drawing dashes
        assert not any(c and all(ch == "─" for ch in c) for c in stripped)

    def test_returns_true_on_success(self):
        assert _highlight_block("hdr", "code", "python", MagicMock()) is True

    def test_returns_false_on_empty_content_exception(self):
        # Simulate to_ansi raising — should return False
        with patch("agent.display._rich_syntax") as mock_hl, \
             patch("agent.display._RICH_OUTPUT", True):
            mock_hl.to_ansi.side_effect = RuntimeError("boom")
            result = _highlight_block("hdr", "code", "python", MagicMock())
        assert result is False

    def test_fallback_no_rich_prints_raw_code(self):
        calls = []
        with patch("agent.display._RICH_OUTPUT", False):
            _highlight_block("hdr", "line one\nline two", "python", calls.append)
        assert any("line one" in c for c in calls)
        assert any("line two" in c for c in calls)


# ---------------------------------------------------------------------------
# render_execute_code_preview
# ---------------------------------------------------------------------------

class TestRenderExecuteCodePreview:
    def test_returns_true_and_emits_code(self):
        calls = []
        result = render_execute_code_preview("x = 42", print_fn=calls.append)
        assert result is True
        assert any("x" in c for c in calls)

    def test_empty_string_returns_false(self):
        calls = []
        assert render_execute_code_preview("", print_fn=calls.append) is False
        assert calls == []

    def test_whitespace_only_returns_false(self):
        assert render_execute_code_preview("   \n\t  ", print_fn=MagicMock()) is False

    def test_no_header_emitted(self):
        """Cute-msg already labels the tool — no header should appear in output."""
        calls = []
        render_execute_code_preview("x = 1", print_fn=calls.append)
        assert not any("execute_code" in c for c in calls)

    def test_fallback_no_rich_prints_raw_lines(self):
        calls = []
        with patch("agent.display._RICH_OUTPUT", False):
            result = render_execute_code_preview("a = 1\nb = 2", print_fn=calls.append)
        assert result is True
        assert any("a = 1" in c for c in calls)
        assert any("b = 2" in c for c in calls)

    def test_fallback_no_header_or_separator(self):
        calls = []
        with patch("agent.display._RICH_OUTPUT", False):
            render_execute_code_preview("a = 1", print_fn=calls.append)
        stripped = [re.sub(r"\x1b\[[0-9;]*m", "", c).strip() for c in calls]
        assert not any("execute_code" in c for c in stripped)
        assert not any(c and all(ch == "─" for ch in c) for c in stripped)


# ---------------------------------------------------------------------------
# render_read_file_preview
# ---------------------------------------------------------------------------

class TestRenderReadFilePreview:
    def _result(self, content):
        return json.dumps({"content": content})

    def test_known_extension_returns_true(self):
        calls = []
        result = render_read_file_preview("foo.py", self._result("x = 1"), print_fn=calls.append)
        assert result is True
        assert len(calls) >= 1

    def test_header_contains_pipe_and_filename(self):
        calls = []
        render_read_file_preview("app.ts", self._result("const x = 1;"), print_fn=calls.append)
        assert any("┊" in c and "app.ts" in c for c in calls)

    def test_unknown_extension_returns_false(self):
        calls = []
        result = render_read_file_preview("notes.log", self._result("some log"), print_fn=calls.append)
        assert result is False
        assert calls == []

    def test_empty_content_returns_false(self):
        assert render_read_file_preview("foo.py", self._result(""), print_fn=MagicMock()) is False
        assert render_read_file_preview("foo.py", self._result("  \n "), print_fn=MagicMock()) is False

    def test_empty_path_returns_false(self):
        assert render_read_file_preview("", self._result("x = 1"), print_fn=MagicMock()) is False

    def test_invalid_json_returns_false(self):
        assert render_read_file_preview("foo.py", "not json", print_fn=MagicMock()) is False


# ---------------------------------------------------------------------------
# render_terminal_preview
# ---------------------------------------------------------------------------

class TestRenderTerminalPreview:
    def _result(self, output):
        return json.dumps({"output": output})

    def test_cat_py_returns_true_and_has_header(self):
        calls = []
        result = render_terminal_preview("cat app.py", self._result("x = 1\n"), print_fn=calls.append)
        assert result is True
        assert any("┊" in c and "app.py" in c for c in calls)

    def test_no_extension_match_returns_false(self):
        assert render_terminal_preview("ls -la", self._result("file.txt"), print_fn=MagicMock()) is False

    def test_flag_tokens_skipped_sed(self):
        calls = []
        result = render_terminal_preview(
            "sed -n '1,50p' app.ts", self._result("const x = 1;"), print_fn=calls.append
        )
        assert result is True
        assert any("app.ts" in c for c in calls)

    def test_exec_command_node_suppressed(self):
        """node script.js executes the file — stdout is not source code."""
        assert render_terminal_preview(
            "node script.js", self._result("output text"), print_fn=MagicMock()
        ) is False

    def test_exec_command_python_suppressed(self):
        assert render_terminal_preview(
            "python3 analyse.py", self._result("result: 42"), print_fn=MagicMock()
        ) is False

    def test_exec_command_bash_suppressed(self):
        assert render_terminal_preview(
            "bash run.sh", self._result("done"), print_fn=MagicMock()
        ) is False

    def test_empty_output_returns_false(self):
        assert render_terminal_preview("cat foo.py", self._result(""), print_fn=MagicMock()) is False

    def test_invalid_result_json_returns_false(self):
        assert render_terminal_preview("cat foo.py", "not json", print_fn=MagicMock()) is False


# ---------------------------------------------------------------------------
# Cute-message deduplication (_code_highlight_active)
# ---------------------------------------------------------------------------

class TestCuteMessageDedup:
    def teardown_method(self):
        # Always restore flag to default after each test
        set_code_highlight_active(False)

    def test_no_snippet_when_active(self):
        set_code_highlight_active(True)
        msg = get_cute_tool_message("execute_code", {"code": "x = compute()\nreturn x"}, 1.5)
        assert "x = compute()" not in msg
        assert "return x" not in msg

    def test_snippet_present_when_inactive(self):
        set_code_highlight_active(False)
        msg = get_cute_tool_message("execute_code", {"code": "x = compute()"}, 1.5)
        assert "x = compute()" in msg

    def test_duration_always_present(self):
        for active in (True, False):
            set_code_highlight_active(active)
            msg = get_cute_tool_message("execute_code", {"code": "pass"}, 2.3)
            assert "2.3s" in msg

    def test_other_tools_unaffected_by_flag(self):
        set_code_highlight_active(True)
        msg = get_cute_tool_message("terminal", {"command": "ls -la"}, 0.5)
        assert "ls -la" in msg


# ---------------------------------------------------------------------------
# _result_succeeded gate (guards execute_code preview on error)
# ---------------------------------------------------------------------------

class TestResultSucceededGate:
    def test_error_status_fails(self):
        assert not _result_succeeded('{"status": "error", "error": "SyntaxError"}')

    def test_ok_status_passes(self):
        assert _result_succeeded('{"status": "ok", "output": "48\\n"}')

    def test_explicit_error_key_fails(self):
        assert not _result_succeeded('{"error": "something went wrong"}')

    def test_success_false_fails(self):
        assert not _result_succeeded('{"success": false}')

    def test_success_true_passes(self):
        assert _result_succeeded('{"success": true}')

    def test_invalid_json_fails(self):
        assert not _result_succeeded("not json")

    def test_none_fails(self):
        assert not _result_succeeded(None)


# ---------------------------------------------------------------------------
# set_diff_limits / set_preview_max_lines — config-exposed limit setters
# ---------------------------------------------------------------------------

class TestDisplayLimitSetters:
    """set_diff_limits and set_preview_max_lines write module-level globals
    that gate truncation in render_read_file_preview and inline diff rendering."""

    def test_set_diff_limits_updates_globals(self):
        import agent.display as _disp
        original_lines = _disp._MAX_INLINE_DIFF_LINES
        original_files = _disp._MAX_INLINE_DIFF_FILES
        try:
            set_diff_limits(max_lines=200, max_files=12)
            assert _disp._MAX_INLINE_DIFF_LINES == 200
            assert _disp._MAX_INLINE_DIFF_FILES == 12
        finally:
            set_diff_limits(max_lines=original_lines, max_files=original_files)

    def test_set_preview_max_lines_updates_global(self):
        import agent.display as _disp
        original = _disp._PREVIEW_MAX_LINES
        try:
            set_preview_max_lines(99)
            assert _disp._PREVIEW_MAX_LINES == 99
        finally:
            set_preview_max_lines(original)

    def test_set_preview_max_lines_truncates_execute_code_preview(self):
        """render_execute_code_preview honours _PREVIEW_MAX_LINES after set_preview_max_lines()."""
        import agent.display as _disp
        original = _disp._PREVIEW_MAX_LINES
        try:
            set_preview_max_lines(3)
            # Build code with 10 numbered lines
            code = "\n".join(f"x_{i} = {i}" for i in range(1, 11))
            lines_captured = []
            render_execute_code_preview(code, print_fn=lines_captured.append)
            stripped = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in lines_captured]
            content_lines = [l for l in stripped if l.strip().startswith("x_")]
            # At limit=3 only 3 content lines should appear before truncation
            assert len(content_lines) <= 3, f"expected ≤3 content lines, got {len(content_lines)}: {content_lines}"
        finally:
            set_preview_max_lines(original)

    def test_summarize_diff_sections_respects_max_lines(self):
        """_summarize_rendered_diff_sections truncates when max_lines is tight."""
        many_lines = "\n".join(f"-old{i}\n+new{i}" for i in range(10))
        diff = f"--- a/f.py\n+++ b/f.py\n@@ -1,10 +1,10 @@\n{many_lines}\n"
        result_lines = _summarize_rendered_diff_sections(diff, max_lines=2, max_files=10)
        stripped = "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", l) for l in result_lines)
        assert "omitted" in stripped.lower(), f"expected truncation notice, got: {stripped[:200]}"
