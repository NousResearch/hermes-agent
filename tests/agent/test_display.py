"""Tests for agent/display.py — build_tool_preview() and inline diff previews."""

import json
import pytest
from unittest.mock import MagicMock

import agent.display as display_module
from agent.display import (
    build_tool_preview,
    capture_local_edit_snapshot,
    extract_edit_diff,
    get_cute_tool_message,
    prepare_tool_preview,
    redact_tool_args_for_display,
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


def test_cute_tool_message_falls_back_when_renderer_raises(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("cosmetic failure")

    monkeypatch.setattr(display_module, "_get_cute_tool_message", _boom)

    assert get_cute_tool_message("web_extract", {"urls": []}, 0.25) == (
        "┊ ⚡ web_extra completed  0.2s"
    )


class TestBuildToolPreview:
    """Tests for build_tool_preview defensive handling and normal operation."""

    def test_none_args_returns_none(self):
        """PR #453: None args should not crash, should return None."""
        assert build_tool_preview("terminal", None) is None

    def test_empty_dict_returns_none(self):
        """Empty dict has no keys to preview."""
        assert build_tool_preview("terminal", {}) is None








    def test_browser_type_preview_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": secret})
        assert result is not None
        assert secret not in result
        assert "sk-pro" in result and "..." in result

    def test_browser_type_preview_keeps_normal_text(self):
        text = "hello world search query"
        result = build_tool_preview("browser_type", {"ref": "@e3", "text": text})
        assert result is not None
        assert text in result

    def test_browser_type_display_args_redact_api_key(self):
        secret = "ghp_ABCDEFGHIJ1234567890"
        safe_args = redact_tool_args_for_display(
            "browser_type", {"ref": "@e3", "text": secret}
        )
        assert secret not in str(safe_args)
        assert safe_args["ref"] == "@e3"
        assert safe_args["text"].startswith("ghp_AB")















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

    @pytest.mark.parametrize("max_len", [1, 2, 3, 4])
    def test_tiny_max_len_never_exceeded(self, max_len):
        """max_len is a hard cap on every preview path — dedicated builder (terminal), generic
        fallback key (web_search), and the cute head-truncated path (#9439)."""
        from agent.display import _cute_path, set_tool_preview_max_len
        long = "abcdefghijklmnopqrstuvwxyz"
        for tool, args in (("terminal", {"command": long}), ("web_search", {"query": long})):
            preview = build_tool_preview(tool, args, max_len=max_len)
            assert preview and len(preview) <= max_len, (tool, preview)
        set_tool_preview_max_len(max_len)
        try:
            assert len(_cute_path("/" + long + "/file.py")) <= max_len
        finally:
            set_tool_preview_max_len(0)


class TestPrepareToolPreview:
    def test_recovers_and_describes_truncated_url(self):
        url = "https://example.com/a/very/long/path/to/a/page"
        set_tool_preview_max_len(20)

        preview = prepare_tool_preview(
            "web_extract",
            {"urls": [url]},
            fallback=url[:17] + "...",
            max_len=20,
        )

        assert preview.text == url[:17] + "..."
        assert preview.truncated is True
        assert preview.url == url

    def test_untruncated_url_has_no_link_target(self):
        url = "https://example.com/page"
        preview = prepare_tool_preview(
            "browser_navigate", None, fallback=url, max_len=40
        )

        assert preview.text == url
        assert preview.truncated is False
        assert preview.url is None

    def test_truncated_non_url_has_no_link_target(self):
        preview = prepare_tool_preview(
            "web_search",
            {"query": "how to parse a URL"},
            fallback="how to parse a URL",
            max_len=12,
        )

        assert preview.truncated is True
        assert preview.url is None


class TestCuteToolMessagePreviewLength:


    def test_search_files_preview_uses_positive_configured_limit_not_default(self):
        set_tool_preview_max_len(80)
        pattern = "function.formatToolCall.context.preview.compactPreview.maxLength.truncate"

        line = get_cute_tool_message("search_files", {"pattern": pattern}, 0.1)

        assert pattern in line
        assert "..." not in line





    def test_browser_type_cute_message_redacts_api_key(self):
        secret = "sk-proj-ABCD1234567890EFGH"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@password", "text": secret},
            0.1,
            result='{"success": true, "typed": "sk-pro...EFGH"}',
        )

        assert secret not in line
        assert "sk-pro" in line

    def test_browser_type_cute_message_keeps_normal_text(self):
        text = "hello world"
        line = get_cute_tool_message(
            "browser_type",
            {"ref": "@search", "text": text},
            0.1,
            result='{"success": true, "typed": "hello world"}',
        )

        assert text in line


class TestCuteSkillManage:
    """skill_manage completion lines must name the skill that was created/changed (#52085)."""

    def test_create_success_names_the_skill_and_verb(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"action": "create", "name": "deploy-runbook"},
            0.1,
            result='{"success": true, "message": "Skill \'deploy-runbook\' created."}',
        )
        assert "created" in line
        assert "deploy-runbook" in line

    def test_patch_success_reports_updated(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"action": "patch", "name": "x"},
            0.1,
            result='{"success": true, "message": "Skill \'x\' patched."}',
        )
        assert "updated" in line
        assert " x" in line

    def test_failed_create_does_not_claim_created(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"action": "create", "name": "deploy-runbook"},
            0.1,
            result='{"success": false, "error": "A skill named deploy-runbook already exists"}',
        )
        assert "created" not in line

    def test_missing_name_still_returns_well_formed_line(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"action": "create"},
            0.1,
            result='{"success": true}',
        )
        assert line.startswith("┊")
        assert "created" in line
        assert line.endswith("0.1s")

    def test_blank_name_still_returns_well_formed_line(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"action": "delete", "name": "   "},
            0.1,
            result='{"success": true}',
        )
        assert line.startswith("┊")
        assert "deleted" in line
        assert line.endswith("0.1s")

    def test_long_name_respects_configured_preview_cap(self):
        set_tool_preview_max_len(20)
        name = "a-very-long-skill-name-that-exceeds-the-cap"

        line = get_cute_tool_message(
            "skill_manage",
            {"action": "create", "name": name},
            0.1,
            result='{"success": true}',
        )

        assert name not in line
        assert "..." in line

    # ---- advertised call shape: {"operations": [...]} (SKILL_MANAGE_SCHEMA requires it) ----

    def test_operations_array_success_names_the_skill(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"operations": [{"action": "create", "name": "deploy-runbook", "content": "..."}]},
            0.1,
            result='{"success": true, "message": "Skill \'deploy-runbook\' created."}',
        )
        assert "created" in line
        assert "deploy-runbook" in line

    def test_multi_op_batch_names_first_skill_and_marks_the_rest(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"operations": [
                {"action": "create", "name": "first-skill", "content": "..."},
                {"action": "patch", "name": "second-skill"},
            ]},
            0.1,
            result='{"success": true}',
        )
        assert "first-skill" in line
        assert "+1" in line

    def test_sole_delete_batch_reports_deleted(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"operations": [{"action": "delete", "name": "gone"}]},
            0.1,
            result='{"success": true}',
        )
        assert "deleted" in line
        assert "gone" in line
        assert "updated" not in line

    def test_failed_batch_names_intent_without_claiming_success(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"operations": [{"action": "create", "name": "dupe"}]},
            0.1,
            result='{"success": false, "error": "A skill named dupe already exists"}',
        )
        assert "created" not in line
        assert "skill skill" not in line
        assert "already exists" in line  # failure suffix proves it went through the real path

    def test_staged_write_reports_staged_not_created(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"action": "create", "name": "staged-one"},
            0.1,
            result='{"success": true, "staged": true, "pending_id": "p1", "message": "Queued for approval; not yet saved."}',
        )
        assert "staged" in line
        assert "created" not in line
        assert "staged-one" in line

    def test_unknown_action_verb_passes_through(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"operations": [{"action": "move", "name": "renamed-skill"}]},
            0.1,
            result='{"success": true}',
        )
        assert "move" in line
        assert "renamed-skill" in line

    def test_failed_op_without_action_falls_back_to_updated_not_skill_skill(self):
        line = get_cute_tool_message(
            "skill_manage",
            {"operations": [{"name": "x"}]},
            0.1,
            result='{"success": false, "error": "boom"}',
        )
        assert "skill skill" not in line  # the round-1 bug: verb and name fallbacks collided
        assert "updated x" in line        # neutral fallback verb + the op's name
        assert "boom" in line             # failure marker survives


class TestEditDiffPreview:



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


class TestBuildToolLabel:
    """Friendly human-phrased tool labels for built-in tools."""

    @pytest.fixture(autouse=True)
    def _enable_friendly(self):
        from agent.display import set_friendly_tool_labels
        set_friendly_tool_labels(True)
        yield
        set_friendly_tool_labels(True)

    def test_web_search_uses_for_connector(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_search", {"query": "weather in NYC"})
        assert label == 'Searching the web for weather in NYC'

    def test_web_extract_reads_url(self):
        from agent.display import build_tool_label
        label = build_tool_label("web_extract", {"urls": ["https://example.com/page"]})
        assert label is not None
        assert label.startswith("Reading ")
        assert "example.com/page" in label







    def test_disabled_falls_back_to_preview(self):
        from agent.display import (
            build_tool_label,
            build_tool_preview,
            set_friendly_tool_labels,
        )
        set_friendly_tool_labels(False)
        args = {"query": "weather in NYC"}
        label = build_tool_label("web_search", args)
        # With the feature off, must match the raw preview exactly
        assert label == build_tool_preview("web_search", args)
        assert "Searching the web" not in (label or "")



class TestBuildStatusPhrase:
    """build_status_phrase — live working-state text for Slack's status line."""



    def test_verb_only_when_args_none(self):
        # live_status: "verb" mode passes args=None to suppress previews.
        from agent.display import build_status_phrase
        assert build_status_phrase("terminal", None) == "is running…"
        assert build_status_phrase("read_file", None) == "is reading…"



    def test_caps_length_for_slack_status_line(self):
        from agent.display import build_status_phrase
        phrase = build_status_phrase(
            "terminal", {"command": "x" * 300}, max_len=49
        )
        assert phrase is not None and len(phrase) <= 49
        assert phrase.endswith("…")


    def test_respects_friendly_labels_toggle(self):
        from agent.display import build_status_phrase, set_friendly_tool_labels
        set_friendly_tool_labels(False)
        try:
            assert build_status_phrase("terminal", {"command": "ls"}) is None
        finally:
            set_friendly_tool_labels(True)
