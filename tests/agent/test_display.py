"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import pytest
from unittest.mock import MagicMock

from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    register_tool_preview,
    set_tool_preview_max_len,
    _render_inline_unified_diff,
    _summarize_rendered_diff_sections,
    _unregister_tool_preview,
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

    def test_truthy_non_dict_args_returns_none(self):
        """Truthy non-dict args (e.g. ``[1, 2]``) must not raise."""
        assert build_tool_preview("terminal", [1, 2]) is None
        assert build_tool_preview("terminal", "ls") is None


# ----------------------------------------------------------------------
# Declarative tool-preview schema (#28621)
# ----------------------------------------------------------------------
#
# These tests exercise the registry primitive itself so that future
# plugin-authored templates have a documented contract: dispatch field,
# template format spec, missing-key safety, list-valued args, fallback
# resolution, and graceful degradation on bad templates.


class TestRegisterToolPreview:
    """register_tool_preview / _render_registered_preview contract."""

    def teardown_method(self) -> None:
        for name in (
            "_pytest_simple",
            "_pytest_dispatch",
            "_pytest_wildcard",
            "_pytest_list_arg",
            "_pytest_truncate",
            "_pytest_bad",
            "_pytest_collapse",
            "_pytest_lastwins",
        ):
            _unregister_tool_preview(name)

    def test_single_template_renders_args(self):
        register_tool_preview("_pytest_simple", templates="hello {who}")
        assert build_tool_preview("_pytest_simple", {"who": "world"}) == "hello world"

    def test_field_dispatch_picks_template_by_value(self):
        register_tool_preview(
            "_pytest_dispatch",
            field="action",
            templates={
                "add": "+ {content}",
                "remove": "- #{fact_id}",
            },
        )
        assert build_tool_preview(
            "_pytest_dispatch", {"action": "add", "content": "pizza"}
        ) == "+ pizza"
        assert build_tool_preview(
            "_pytest_dispatch", {"action": "remove", "fact_id": 7}
        ) == "- #7"

    def test_wildcard_fallback_used_when_value_not_listed(self):
        register_tool_preview(
            "_pytest_wildcard",
            field="action",
            templates={"add": "+ x", "*": "(other: {action})"},
        )
        assert build_tool_preview(
            "_pytest_wildcard", {"action": "ponder"}
        ) == "(other: ponder)"

    def test_missing_field_returns_none_when_no_wildcard(self):
        register_tool_preview(
            "_pytest_dispatch",
            field="action",
            templates={"add": "+ x"},
        )
        assert build_tool_preview("_pytest_dispatch", {"other": "thing"}) is None

    def test_missing_key_renders_as_empty_string(self):
        register_tool_preview("_pytest_simple", templates="[{a}|{b}]")
        assert build_tool_preview("_pytest_simple", {"a": "x"}) == "[x|]"

    def test_list_arg_joins_with_commas(self):
        register_tool_preview("_pytest_list_arg", templates="who: {entities}")
        result = build_tool_preview(
            "_pytest_list_arg", {"entities": ["alice", "bob", "carol"]}
        )
        assert result == "who: alice, bob, carol"

    def test_per_field_precision_truncates_long_strings(self):
        register_tool_preview("_pytest_simple", templates='"{content:.10}"')
        result = build_tool_preview(
            "_pytest_simple", {"content": "abcdefghij_TRUNCATED_TAIL"}
        )
        assert result == '"abcdefghij"'

    def test_per_tool_truncate_overrides_global_limit(self):
        register_tool_preview(
            "_pytest_truncate",
            templates="{content}",
            truncate=15,
        )
        set_tool_preview_max_len(80)
        result = build_tool_preview(
            "_pytest_truncate", {"content": "x" * 50}
        )
        assert result.endswith("...")
        assert len(result) == 15

    def test_global_limit_applies_when_no_per_tool_truncate(self):
        register_tool_preview("_pytest_simple", templates="{content}")
        set_tool_preview_max_len(10)
        result = build_tool_preview("_pytest_simple", {"content": "x" * 50})
        assert result.endswith("...")
        assert len(result) == 10

    def test_string_args_get_whitespace_collapsed(self):
        register_tool_preview("_pytest_collapse", templates="cmd: {command}")
        result = build_tool_preview(
            "_pytest_collapse", {"command": "echo  hello\nworld\t!"}
        )
        assert result == "cmd: echo hello world !"

    def test_none_arg_renders_as_empty(self):
        register_tool_preview("_pytest_simple", templates="[{value}]")
        assert build_tool_preview("_pytest_simple", {"value": None}) == "[]"

    def test_bad_template_falls_back_without_crashing(self, caplog):
        # ``{not_a_number:d}`` against a string value raises ValueError —
        # the registry should log and return None instead of bubbling.
        register_tool_preview("_pytest_bad", templates="{value:d}")
        with caplog.at_level("WARNING"):
            result = build_tool_preview("_pytest_bad", {"value": "abc"})
        assert result is None
        assert any("_pytest_bad" in rec.getMessage() for rec in caplog.records)

    def test_attribute_access_template_falls_back_without_crashing(self, caplog):
        # ``{value.foo}`` makes ``str.format_map`` do attribute access on the
        # arg, which raises AttributeError — a different exception family than
        # the numeric ``{value:d}`` ValueError above. The fallback must catch
        # it too so a malformed plugin template never escapes to the spinner.
        register_tool_preview("_pytest_attr", templates="{value.foo}")
        with caplog.at_level("WARNING"):
            result = build_tool_preview("_pytest_attr", {"value": "abc"})
        assert result is None
        assert any("_pytest_attr" in rec.getMessage() for rec in caplog.records)

    def test_re_registration_overwrites_previous_schema(self):
        register_tool_preview("_pytest_lastwins", templates="v1: {x}")
        register_tool_preview("_pytest_lastwins", templates="v2: {x}")
        assert build_tool_preview("_pytest_lastwins", {"x": "y"}) == "v2: y"

    def test_registry_takes_priority_over_legacy_chain(self):
        # ``terminal`` is in the legacy ``primary_args`` map — registry
        # registration should win so plugins can override even built-ins.
        try:
            register_tool_preview(
                "terminal", templates="overridden: {command}"
            )
            assert build_tool_preview(
                "terminal", {"command": "ls"}
            ) == "overridden: ls"
        finally:
            _unregister_tool_preview("terminal")
        # Sanity: legacy path restored after teardown.
        assert build_tool_preview("terminal", {"command": "ls"}) == "ls"

    def test_empty_templates_mapping_rejected(self):
        with pytest.raises(ValueError):
            register_tool_preview("_pytest_dispatch", field="action", templates={})

    def test_dispatch_mapping_without_field_rejected(self):
        with pytest.raises(ValueError):
            register_tool_preview(
                "_pytest_dispatch", templates={"add": "+ x"}
            )


class TestFactStorePluginPreview:
    """Smoke tests for the holographic-memory plugin registrations."""

    def test_fact_store_add_shows_content(self):
        # Importing the plugin triggers register_tool_preview() calls.
        import plugins.memory.holographic  # noqa: F401
        assert build_tool_preview(
            "fact_store", {"action": "add", "content": "user prefers tabs"}
        ) == '+ "user prefers tabs"'

    def test_fact_store_search_shows_query(self):
        import plugins.memory.holographic  # noqa: F401
        assert build_tool_preview(
            "fact_store", {"action": "search", "query": "editor config"}
        ) == 'search: "editor config"'

    def test_fact_store_reason_joins_entities(self):
        import plugins.memory.holographic  # noqa: F401
        result = build_tool_preview(
            "fact_store",
            {"action": "reason", "entities": ["alice", "redis"]},
        )
        assert result == "reason: alice, redis"

    def test_fact_store_update_carries_fact_id(self):
        import plugins.memory.holographic  # noqa: F401
        assert build_tool_preview(
            "fact_store", {"action": "update", "fact_id": 42}
        ) == "update: #42"

    def test_fact_store_unknown_action_falls_back_to_wildcard(self):
        import plugins.memory.holographic  # noqa: F401
        assert build_tool_preview(
            "fact_store", {"action": "experimental"}
        ) == "experimental"

    def test_fact_feedback_helpful_shows_thumbs_up(self):
        import plugins.memory.holographic  # noqa: F401
        assert build_tool_preview(
            "fact_feedback", {"action": "helpful", "fact_id": 9}
        ) == "+1 #9"

    def test_fact_feedback_unhelpful_shows_thumbs_down(self):
        import plugins.memory.holographic  # noqa: F401
        assert build_tool_preview(
            "fact_feedback", {"action": "unhelpful", "fact_id": 9}
        ) == "-1 #9"


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
