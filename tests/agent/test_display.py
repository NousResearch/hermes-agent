"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import pytest
from unittest.mock import MagicMock

from agent.display import (
    apply_tool_label,
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    format_friendly_preview,
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

    def test_delegate_task_single_goal_preview(self):
        result = build_tool_preview("delegate_task", {"goal": "Review gateway status"})
        assert result == "Review gateway status"

    def test_delegate_task_batch_goal_preview(self):
        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": "Review PR A"}, {"goal": "Review PR B"}]},
        )
        assert result == "2 tasks: Review PR A | Review PR B"

    def test_delegate_task_batch_preview_handles_missing_non_string_goals(self):
        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": None}, {"goal": 123}, "not-a-task"]},
        )
        assert result == "2 tasks: ? | 123"

    def test_delegate_task_batch_preview_respects_max_len(self):
        result = build_tool_preview(
            "delegate_task",
            {"tasks": [{"goal": "A" * 80}, {"goal": "B" * 80}]},
            max_len=30,
        )
        assert result == "2 tasks: AAAAAAAAAAAAAAAAAA..."
        assert len(result) == 30

    def test_false_like_args_zero(self):
        """Non-dict falsy values should return None, not crash."""
        assert build_tool_preview("terminal", 0) is None
        assert build_tool_preview("terminal", "") is None
        assert build_tool_preview("terminal", []) is None


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

    def test_delegate_task_batch_message_includes_goals(self):
        line = get_cute_tool_message(
            "delegate_task",
            {"tasks": [{"goal": "Review PR A"}, {"goal": "Review PR B"}]},
            1.2,
        )
        assert "2x: Review PR A | Review PR B" in line


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


class TestApplyToolLabel:
    """Operator-configured ``display.tool_labels`` mapping from raw tool names
    to human-readable labels. Default behavior (None / empty / missing entry)
    returns the raw tool name unchanged so existing operators see no change."""

    def test_none_label_map_passes_through(self):
        assert apply_tool_label("read_file", None) == "read_file"

    def test_empty_label_map_passes_through(self):
        assert apply_tool_label("read_file", {}) == "read_file"

    def test_missing_entry_passes_through(self):
        assert apply_tool_label("read_file", {"web_search": "Searching"}) == "read_file"

    def test_exact_match_returns_label(self):
        assert apply_tool_label("read_file", {"read_file": "Reading info"}) == "Reading info"

    def test_label_is_used_verbatim(self):
        """No re-casing, no quoting, no emoji injection — what the operator
        configured is what's rendered."""
        assert apply_tool_label("mcp_x_y", {"mcp_x_y": "  Hello, World!  "}) == "  Hello, World!  "


class TestFormatFriendlyPreview:
    """File-path tool previews trim to the basename when ``trim_paths`` is on
    (the operator-configured ``display.trim_path_previews``). Non-path tools
    pass through unchanged regardless. Falsy previews pass through."""

    def test_none_preview_passes_through(self):
        assert format_friendly_preview("read_file", None, True) is None

    def test_empty_preview_passes_through(self):
        assert format_friendly_preview("read_file", "", True) == ""

    def test_path_tool_trimmed(self):
        assert (
            format_friendly_preview("read_file", "/long/abs/path/STATUS_QUERY.md", True)
            == "STATUS_QUERY.md"
        )

    def test_path_tool_trim_off_keeps_full(self):
        assert (
            format_friendly_preview("read_file", "/long/abs/path/STATUS_QUERY.md", False)
            == "/long/abs/path/STATUS_QUERY.md"
        )

    def test_non_path_tool_passes_through(self):
        # query strings, slugs, URLs etc. should not be trimmed
        assert (
            format_friendly_preview("web_search", "https://example.com/a/b", True)
            == "https://example.com/a/b"
        )
        assert (
            format_friendly_preview("mcp_gbrain_query", "Project X status", True)
            == "Project X status"
        )

    def test_path_with_no_slash_passes_through(self):
        """A bare filename has no path to trim — return it intact rather than
        coercing to PurePosixPath().name (which would be the same anyway)."""
        assert format_friendly_preview("read_file", "STATUS_QUERY.md", True) == "STATUS_QUERY.md"

    def test_truncated_preview_with_ellipsis_still_handled(self):
        """The gateway may pre-truncate previews to 'first N chars...'. The
        helper should not crash on these; basename of a truncated path is
        the trailing segment of whatever survived (even if it includes the
        '...' marker)."""
        out = format_friendly_preview("read_file", "/long/path/STA...", True)
        # Don't assert exact output (depends on PurePosixPath semantics) —
        # just assert no exception and a non-empty string came back.
        assert isinstance(out, str) and out

    def test_each_path_tool_recognised(self):
        for t in ("read_file", "write_file", "edit_file", "patch", "list_files"):
            assert (
                format_friendly_preview(t, "/long/path/foo.txt", True) == "foo.txt"
            ), f"path-tool {t} should trim"


class TestApplyToolLabelWithCurated:
    """Reviewer feedback on #51869: the operator ``label_map`` must COMPOSE
    with the curated defaults on main — an empty operator map should NOT
    regress the curated humane labels. Resolution order:

        operator override → curated default → raw tool name
    """

    def test_curated_wins_when_operator_map_empty(self):
        """Empty operator map preserves curated defaults — regression guard
        for the exact bug the reviewer flagged."""
        curated = {"read_file": "Reading file"}
        assert apply_tool_label("read_file", {}, curated=curated) == "Reading file"

    def test_operator_override_beats_curated(self):
        """When both layers name the tool, operator wins — operators are
        the source of truth for their audience."""
        curated = {"read_file": "Reading file"}
        operator = {"read_file": "Fetching content"}
        assert apply_tool_label("read_file", operator, curated=curated) == "Fetching content"

    def test_curated_fills_gap_when_operator_covers_only_some_tools(self):
        """Partial operator overrides: overrides win where present, curated
        fills the rest, raw name for anything neither knows."""
        curated = {"read_file": "Reading", "write_file": "Writing"}
        operator = {"read_file": "Loading"}
        assert apply_tool_label("read_file", operator, curated=curated) == "Loading"
        assert apply_tool_label("write_file", operator, curated=curated) == "Writing"
        assert apply_tool_label("mcp_x", operator, curated=curated) == "mcp_x"

    def test_curated_none_falls_back_to_raw(self):
        """No curated map and no operator match → raw tool name, matching
        pre-PR behavior for anyone who's on neither layer."""
        assert apply_tool_label("read_file", None, curated={}) == "read_file"
        assert apply_tool_label("read_file", {}, curated={}) == "read_file"

    def test_operator_override_with_empty_curated(self):
        """Operator map still works when curated is empty — the two layers
        are independent."""
        operator = {"read_file": "Reading X"}
        assert apply_tool_label("read_file", operator, curated={}) == "Reading X"

    def test_operator_maps_to_empty_string_falls_through_to_curated(self):
        """An operator entry mapping to '' is treated as 'no override' so
        the curated default still fires. Prevents an accidentally-blank
        config from erasing labels entirely."""
        curated = {"read_file": "Reading file"}
        operator = {"read_file": ""}
        assert apply_tool_label("read_file", operator, curated=curated) == "Reading file"


class TestFormatFriendlyPreviewReversibility:
    """Reviewer feedback on #51869 — reversibility contract for
    ``trim_paths=False``. The whole point of the operator's
    ``display.trim_path_previews: false`` config knob is that it
    restores the full path. Any code path that reduces the preview to
    a basename BEFORE this function would break that contract."""

    def test_trim_off_returns_raw_preview_verbatim(self):
        """No transformation applied when trim_paths=False — even for
        path-shaped tools, even for previews that would normally trim."""
        raw = "/very/long/absolute/path/to/STATUS_QUERY.md"
        assert format_friendly_preview("read_file", raw, trim_paths=False) == raw

    def test_trim_off_preserves_pre_truncated_preview(self):
        """When ``build_tool_preview`` has already length-truncated (adding
        '...' prefix), trim_off preserves that exactly — no basename
        derivation happens."""
        raw = "...long/path/rules/STATUS_QUERY.md"
        assert format_friendly_preview("read_file", raw, trim_paths=False) == raw

    def test_trim_on_still_basenames_for_path_tools(self):
        """Positive control: with trim_paths=True (default), the path IS
        reduced to basename. This is the behavior we want to remain
        reversible via the config knob."""
        raw = "/very/long/absolute/path/to/STATUS_QUERY.md"
        assert format_friendly_preview("read_file", raw, trim_paths=True) == "STATUS_QUERY.md"

    def test_trim_off_for_all_path_tools(self):
        """Reversibility holds for every recognised path-shaped tool."""
        raw = "/deep/nested/tree/file.py"
        for tool in ("read_file", "write_file", "edit_file", "patch", "list_files"):
            assert format_friendly_preview(tool, raw, trim_paths=False) == raw, (
                f"trim_paths=False must preserve raw preview for {tool}"
            )

    def test_trim_off_still_returns_none_for_none_preview(self):
        """The `None` and empty short-circuits still apply."""
        assert format_friendly_preview("read_file", None, trim_paths=False) is None
        assert format_friendly_preview("read_file", "", trim_paths=False) == ""


class TestGatewayLevelWiring:
    """Reviewer's third ask on #51869: exercise the config → render pipeline,
    not just the helpers. These tests import the gateway's config resolver
    and drive `apply_tool_label`/`format_friendly_preview` with values it
    actually returns, so a regression in the wiring (default lookup, key
    name, resolution order) surfaces here rather than only in production."""

    def test_resolve_display_setting_defaults_for_tool_labels(self):
        """The resolver's default for `tool_labels` must be an empty dict
        (or None) so the composition layer falls back to curated + raw."""
        from gateway.display_config import resolve_display_setting

        # A minimal user_config with no display block at all.
        user_cfg = {}
        result = resolve_display_setting(user_cfg, "telegram", "tool_labels", {})
        # Must be dict-like and empty-truthy so `if label_map:` short-circuits.
        assert result in (None, {}, dict())

    def test_resolve_display_setting_defaults_for_trim_path_previews(self):
        """The resolver's default for `trim_path_previews` must be True so
        the pre-PR path-trimming behavior is preserved for opt-in-only."""
        from gateway.display_config import resolve_display_setting

        user_cfg = {}
        result = resolve_display_setting(user_cfg, "telegram", "trim_path_previews", True)
        assert result is True

    def test_resolve_and_apply_tool_label_composes_with_curated(self):
        """End-to-end wiring: pull tool_labels from config, apply via
        apply_tool_label with a non-empty curated map, verify composition."""
        from gateway.display_config import resolve_display_setting

        # Operator supplied a partial map — read_file overridden, others not.
        user_cfg = {"display": {"tool_labels": {"read_file": "Fetching info"}}}
        label_map = resolve_display_setting(user_cfg, "telegram", "tool_labels", {})
        curated = {"read_file": "Reading file", "write_file": "Writing file"}

        # Operator override wins.
        assert apply_tool_label("read_file", label_map, curated=curated) == "Fetching info"
        # Curated fills the gap for write_file.
        assert apply_tool_label("write_file", label_map, curated=curated) == "Writing file"
        # Neither layer knows this tool → raw.
        assert apply_tool_label("mcp_unknown", label_map, curated=curated) == "mcp_unknown"

    def test_resolve_and_format_preview_with_trim_off(self):
        """End-to-end: user sets trim_path_previews: false → gateway
        resolves False → format_friendly_preview returns raw path."""
        from gateway.display_config import resolve_display_setting

        user_cfg = {"display": {"trim_path_previews": False}}
        trim = resolve_display_setting(user_cfg, "telegram", "trim_path_previews", True)
        assert trim is False

        raw = "/deep/nested/path/to/rules/STATUS_QUERY.md"
        assert format_friendly_preview("read_file", raw, trim) == raw

    def test_per_platform_trim_path_override_beats_global(self):
        """Per-platform overrides win over global (documented resolution
        order in display_config.py). Verifies the platform-specific
        preview config path works end-to-end."""
        from gateway.display_config import resolve_display_setting

        user_cfg = {
            "display": {
                "trim_path_previews": True,   # global on
                "platforms": {
                    "whatsapp": {"trim_path_previews": False},  # off on WhatsApp
                },
            },
        }
        assert resolve_display_setting(
            user_cfg, "whatsapp", "trim_path_previews", True,
        ) is False
        assert resolve_display_setting(
            user_cfg, "telegram", "trim_path_previews", True,
        ) is True
