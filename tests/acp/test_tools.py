"""Tests for acp_adapter.tools — tool kind mapping and ACP content building."""


from acp_adapter.edit_approval import EditProposal
from acp_adapter.tools import (
    TOOL_KIND_MAP,
    build_tool_complete,
    build_tool_start,
    build_tool_title,
    extract_locations,
    get_tool_kind,
    make_tool_call_id,
)
from acp.schema import (
    FileEditToolCallContent,
    ContentToolCallContent,
    ToolCallLocation,
    ToolCallStart,
    ToolCallProgress,
)


# ---------------------------------------------------------------------------
# TOOL_KIND_MAP coverage
# ---------------------------------------------------------------------------


COMMON_HERMES_TOOLS = ["read_file", "search_files", "terminal", "patch", "write_file", "process"]


class TestToolKindMap:
    def test_all_hermes_tools_have_kind(self):
        """Every common hermes tool should appear in TOOL_KIND_MAP."""
        for tool in COMMON_HERMES_TOOLS:
            assert tool in TOOL_KIND_MAP, f"{tool} missing from TOOL_KIND_MAP"

    def test_tool_kind_read_file(self):
        assert get_tool_kind("read_file") == "read"

    def test_tool_kind_terminal(self):
        assert get_tool_kind("terminal") == "execute"

    def test_tool_kind_patch(self):
        assert get_tool_kind("patch") == "edit"

    def test_tool_kind_write_file(self):
        assert get_tool_kind("write_file") == "edit"

    def test_tool_kind_web_search(self):
        assert get_tool_kind("web_search") == "fetch"

    def test_tool_kind_execute_code(self):
        assert get_tool_kind("execute_code") == "execute"

    def test_tool_kind_todo(self):
        assert get_tool_kind("todo") == "other"

    def test_tool_kind_skill_view(self):
        assert get_tool_kind("skill_view") == "read"

    def test_tool_kind_browser_navigate(self):
        assert get_tool_kind("browser_navigate") == "fetch"

    def test_unknown_tool_returns_other_kind(self):
        assert get_tool_kind("nonexistent_tool_xyz") == "other"


# ---------------------------------------------------------------------------
# make_tool_call_id
# ---------------------------------------------------------------------------


class TestMakeToolCallId:
    def test_returns_string(self):
        tc_id = make_tool_call_id()
        assert isinstance(tc_id, str)

    def test_starts_with_tc_prefix(self):
        tc_id = make_tool_call_id()
        assert tc_id.startswith("tc-")

    def test_ids_are_unique(self):
        ids = {make_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# build_tool_title
# ---------------------------------------------------------------------------


class TestBuildToolTitle:
    def test_terminal_title_includes_command(self):
        title = build_tool_title("terminal", {"command": "ls -la /tmp"})
        assert "ls -la /tmp" in title

    def test_terminal_title_truncates_long_command(self):
        long_cmd = "x" * 200
        title = build_tool_title("terminal", {"command": long_cmd})
        assert len(title) < 120
        assert "..." in title

    def test_read_file_title(self):
        title = build_tool_title("read_file", {"path": "/etc/hosts"})
        assert "/etc/hosts" in title

    def test_patch_title(self):
        title = build_tool_title("patch", {"path": "main.py", "mode": "replace"})
        assert "main.py" in title

    def test_search_title(self):
        title = build_tool_title("search_files", {"pattern": "TODO"})
        assert "TODO" in title

    def test_web_search_title(self):
        title = build_tool_title("web_search", {"query": "python asyncio"})
        assert "python asyncio" in title

    def test_web_extract_title_unwraps_search_result_object(self):
        title = build_tool_title("web_extract", {
            "urls": [
                {"url": "https://example.com/a", "title": "A"},
                {"href": "https://example.org/b"},
            ]
        })
        assert title == "extract: https://example.com/a (+1)"

    def test_web_extract_title_handles_malformed_object(self):
        assert build_tool_title("web_extract", {"urls": [{"title": "missing"}]}) == "extract: ?"

    def test_skill_view_title_includes_skill_name(self):
        title = build_tool_title("skill_view", {"name": "github-pitfalls"})
        assert title == "skill view (github-pitfalls)"

    def test_skill_view_title_includes_linked_file(self):
        title = build_tool_title("skill_view", {"name": "github-pitfalls", "file_path": "references/api.md"})
        assert title == "skill view (github-pitfalls/references/api.md)"

    def test_execute_code_title_includes_first_code_line(self):
        title = build_tool_title("execute_code", {"code": "\nfrom hermes_tools import terminal\nprint('done')"})
        assert title == "python: from hermes_tools import terminal"

    def test_skill_manage_title_includes_action_and_target(self):
        title = build_tool_title(
            "skill_manage",
            {"action": "patch", "name": "hermes-agent-operations", "file_path": "references/acp.md"},
        )
        assert title == "skill patch: hermes-agent-operations/references/acp.md"

    def test_unknown_tool_uses_name(self):
        title = build_tool_title("some_new_tool", {"foo": "bar"})
        assert title == "some_new_tool"


# ---------------------------------------------------------------------------
# build_tool_start
# ---------------------------------------------------------------------------


class TestBuildToolStart:
    def test_build_tool_start_for_patch(self):
        """patch start should not duplicate the edit-approval diff."""
        args = {
            "path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('world')",
        }
        result = build_tool_start("tc-1", "patch", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) >= 1
        item = result.content[0]
        assert isinstance(item, ContentToolCallContent)
        assert "Approval prompt shows the diff" in item.content.text
        assert "src/main.py" in item.content.text

    def test_build_tool_start_for_write_file(self):
        """write_file start should not duplicate the edit-approval diff."""
        args = {"path": "new_file.py", "content": "print('hello')"}
        result = build_tool_start("tc-w1", "write_file", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) >= 1
        item = result.content[0]
        assert isinstance(item, ContentToolCallContent)
        assert "Approval prompt shows the diff" in item.content.text
        assert "new_file.py" in item.content.text

    def test_auto_approved_edit_start_shows_diff_content(self):
        """Auto-approved edit starts need the diff because no approval card exists."""
        args = {"path": "/tmp/acp.txt", "old_string": "old", "new_string": "new"}
        result = build_tool_start(
            "tc-auto-edit",
            "patch",
            args,
            edit_diff=EditProposal("patch", "/tmp/acp.txt", "old\n", "new\n", args),
        )

        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) == 1
        item = result.content[0]
        assert isinstance(item, FileEditToolCallContent)
        assert item.path == "/tmp/acp.txt"
        assert item.old_text == "old\n"
        assert item.new_text == "new\n"

    def test_build_tool_start_for_terminal(self):
        """terminal should produce text content with the command."""
        args = {"command": "ls -la /tmp"}
        result = build_tool_start("tc-2", "terminal", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "execute"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        # The wrapped text block should contain the command
        text = content_item.content.text
        assert "ls -la /tmp" in text

    def test_build_tool_start_for_read_file(self):
        """read_file start should stay compact; completion carries file contents."""
        args = {"path": "/etc/hosts", "offset": 1, "limit": 50}
        result = build_tool_start("tc-3", "read_file", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "read"
        assert result.content is None
        assert result.raw_input is None

    def test_build_tool_start_survives_non_string_command(self):
        """A malformed (non-string) terminal command previously raised
        TypeError in build_tool_title (len(None)) and aborted the render."""
        result = build_tool_start("tc-bad-cmd", "terminal", {"command": None})
        assert isinstance(result, ToolCallStart)
        assert result.kind == "execute"  # tool identity preserved in the fallback

    def test_build_tool_start_survives_non_string_path(self):
        """A non-string read_file path previously raised a ToolCallLocation
        pydantic ValidationError in extract_locations and aborted the render."""
        result = build_tool_start("tc-bad-path", "read_file", {"path": {"p": "x"}})
        assert isinstance(result, ToolCallStart)
        assert result.kind == "read"

    def test_build_tool_start_survives_non_string_goal(self):
        """A non-string delegate_task goal previously raised TypeError
        (len(123)) in build_tool_title and aborted the render."""
        result = build_tool_start("tc-bad-goal", "delegate_task", {"goal": 123})
        assert isinstance(result, ToolCallStart)

    def test_build_tool_start_for_web_extract_is_compact(self):
        """web_extract start should stay compact; title identifies URLs."""
        args = {"urls": ["https://example.com/docs"]}
        result = build_tool_start("tc-web-start", "web_extract", args)
        assert isinstance(result, ToolCallStart)
        assert result.title == "extract: https://example.com/docs"
        assert result.kind == "fetch"
        assert result.content is None
        assert result.raw_input is None

    def test_build_tool_start_for_browser_navigate(self):
        """browser_navigate should emit a polished start event."""
        args = {"url": "https://x.com"}
        result = build_tool_start("tc-browser-start", "browser_navigate", args)
        assert isinstance(result, ToolCallStart)
        assert result.title == "navigate: https://x.com"
        assert result.kind == "fetch"
        assert result.content[0].content.text == '{\n  "url": "https://x.com"\n}'
        assert result.raw_input is None

    def test_build_tool_start_for_search(self):
        """search_files should include pattern in content."""
        args = {"pattern": "TODO", "target": "content"}
        result = build_tool_start("tc-4", "search_files", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "search"
        assert "TODO" in result.content[0].content.text
        assert result.raw_input is None

    def test_build_tool_start_for_todo_is_human_readable(self):
        args = {"todos": [{"id": "one", "content": "Fix ACP rendering", "status": "in_progress"}]}
        result = build_tool_start("tc-todo", "todo", args)
        assert result.title == "todo (1 item)"
        assert "Fix ACP rendering" in result.content[0].content.text
        assert result.raw_input is None

    def test_build_tool_start_for_skill_view_is_human_readable(self):
        result = build_tool_start("tc-skill", "skill_view", {"name": "github-pitfalls"})
        assert result.title == "skill view (github-pitfalls)"
        assert "github-pitfalls" in result.content[0].content.text
        assert result.raw_input is None

    def test_build_tool_start_for_execute_code_shows_code_preview(self):
        result = build_tool_start("tc-code", "execute_code", {"code": "print('hello')"})
        assert result.kind == "execute"
        assert result.title == "python: print('hello')"
        assert "```python" in result.content[0].content.text
        assert "print('hello')" in result.content[0].content.text
        assert result.raw_input is None

    def test_build_tool_start_for_skill_manage_patch_shows_diff(self):
        result = build_tool_start(
            "tc-skill-manage",
            "skill_manage",
            {
                "action": "patch",
                "name": "hermes-agent-operations",
                "file_path": "references/acp.md",
                "old_string": "old advice",
                "new_string": "new advice",
            },
        )
        assert result.kind == "edit"
        assert result.title == "skill patch: hermes-agent-operations/references/acp.md"
        assert isinstance(result.content[0], FileEditToolCallContent)
        assert result.content[0].path == "skills/hermes-agent-operations/references/acp.md"
        assert result.content[0].old_text == "old advice"
        assert result.content[0].new_text == "new advice"
        assert result.raw_input is None

    def test_build_tool_start_generic_fallback(self):
        """Unknown tools should get a generic text representation."""
        args = {"foo": "bar", "baz": 42}
        result = build_tool_start("tc-5", "some_tool", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "other"


# ---------------------------------------------------------------------------
# build_tool_complete
# ---------------------------------------------------------------------------


class TestBuildToolComplete:
    def test_arbitrary_raw_result_never_reaches_acp_payload(self):
        marker = "PRIVATE_OUTPUT_xzx_7e9"
        result = build_tool_complete(
            "call-1",
            "mcp__server__tool",
            result=marker,
            summary="safe summary",
        )

        assert result.raw_output is None
        assert marker not in repr(result.model_dump())
        assert result.content[0].content.text == "safe summary"

    def test_summary_is_resanitized_at_the_acp_boundary(self):
        marker = "sk-proj-ABCDEF1234567890"
        result = build_tool_complete(
            "call-1",
            "custom_tool",
            result="ignored",
            summary=f"failed with Bearer {marker}",
        )

        assert marker not in repr(result.model_dump())
        assert result.content[0].content.text == "failed with Bearer [REDACTED]"

    def test_completion_without_summary_has_no_presentation_content(self):
        marker = "PRIVATE_OUTPUT_WITHOUT_SUMMARY"
        result = build_tool_complete("tc-2", "terminal", marker)

        assert isinstance(result, ToolCallProgress)
        assert result.status == "completed"
        assert result.content is None
        assert result.raw_output is None
        assert marker not in repr(result.model_dump())

    def test_empty_sanitized_summary_is_preserved_without_raw_fallback(self):
        marker = "PRIVATE_OUTPUT_EMPTY_SUMMARY"
        result = build_tool_complete("tc-empty", "read_file", marker, summary="")

        assert result.content[0].content.text == ""
        assert result.raw_output is None
        assert marker not in repr(result.model_dump())

    def test_summary_is_the_only_completion_content(self):
        marker = "PRIVATE_OUTPUT_WITH_ARGS_AND_SNAPSHOT"
        result = build_tool_complete(
            "tc-safe",
            "write_file",
            marker,
            function_args={"path": "private.txt", "content": marker},
            snapshot=object(),
            summary="write_file: completed",
        )

        assert [item.content.text for item in result.content] == ["write_file: completed"]
        assert marker not in repr(result.model_dump())

    def test_explicit_lifecycle_error_overrides_ambiguous_raw_output(self):
        result = build_tool_complete(
            "tc-error",
            "some_plugin_tool",
            "plain output with no structured error marker",
            status="failed",
            is_error=True,
        )

        assert result.status == "failed"
        assert result.content is None
        assert result.raw_output is None

    def test_legacy_failure_classification_does_not_expose_result(self):
        cases = [
            ("skill_manage", '{"success": false, "error": "PRIVATE_ONE"}'),
            ("some_tool", '{"ok": false, "error": "PRIVATE_TWO"}'),
            ("terminal", '{"output": "PRIVATE_THREE", "exit_code": 2}'),
            ("execute_code", '{"output": "PRIVATE_FOUR", "returncode": 2}'),
            ("patch", "Error executing tool 'patch': PRIVATE_FIVE"),
            ("read_file", '{"error": "PRIVATE_SIX"}'),
        ]

        for tool_name, raw in cases:
            result = build_tool_complete("tc-fail", tool_name, raw)
            assert result.status == "failed"
            assert result.content is None
            assert result.raw_output is None
            assert raw not in repr(result.model_dump())

    def test_plain_error_word_and_unflagged_json_remain_completed(self):
        for raw in [
            "Error: pytest collected 0 items",
            '{"error": "timeout while reading optional source"}',
        ]:
            result = build_tool_complete("tc-ok", "some_tool", raw)
            assert result.status == "completed"
            assert result.content is None
            assert result.raw_output is None


# ---------------------------------------------------------------------------
# extract_locations
# ---------------------------------------------------------------------------


class TestExtractLocations:
    def test_extract_locations_with_path(self):
        args = {"path": "src/app.py", "offset": 42}
        locs = extract_locations(args)
        assert len(locs) == 1
        assert isinstance(locs[0], ToolCallLocation)
        assert locs[0].path == "src/app.py"
        assert locs[0].line == 42

    def test_extract_locations_without_path(self):
        args = {"command": "echo hi"}
        locs = extract_locations(args)
        assert locs == []
