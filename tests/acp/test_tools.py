"""Tests for acp_adapter.tools — tool kind mapping and ACP content building."""

import pytest

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
        """patch should produce a FileEditToolCallContent (diff)."""
        args = {
            "path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('world')",
        }
        result = build_tool_start("tc-1", "patch", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        # The first content item should be a diff
        assert len(result.content) >= 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path == "src/main.py"
        assert diff_item.new_text == "print('world')"
        assert diff_item.old_text == "print('hello')"

    def test_build_tool_start_for_write_file(self):
        """write_file should produce a FileEditToolCallContent (diff)."""
        args = {"path": "new_file.py", "content": "print('hello')"}
        result = build_tool_start("tc-w1", "write_file", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) >= 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path == "new_file.py"

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
    def test_build_tool_complete_for_terminal(self):
        """Completed terminal call should include output text."""
        result = build_tool_complete("tc-2", "terminal", "total 42\ndrwxr-xr-x 2 root root 4096 ...")
        assert isinstance(result, ToolCallProgress)
        assert result.status == "completed"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        assert "total 42" in content_item.content.text
        assert result.raw_output is None

    def test_build_tool_complete_for_todo_is_checklist(self):
        result = build_tool_complete(
            "tc-todo",
            "todo",
            '{"todos":[{"id":"a","content":"Inspect ACP","status":"completed"},{"id":"b","content":"Patch renderers","status":"in_progress"}],"summary":{"total":2,"pending":0,"in_progress":1,"completed":1,"cancelled":0}}',
        )
        text = result.content[0].content.text
        assert "✅ Inspect ACP" in text
        assert "- 🔄 Patch renderers" in text
        assert "**Progress:** 1 completed, 1 in progress, 0 pending" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_skill_view_summarizes_content_without_raw_json(self):
        result = build_tool_complete(
            "tc-skill",
            "skill_view",
            '{"success":true,"name":"github-pitfalls","description":"GitHub gotchas","content":"# GitHub Pitfalls\\nUse gh carefully.","path":"github/github-pitfalls/SKILL.md"}',
        )
        text = result.content[0].content.text
        assert "**Skill loaded**" in text
        assert "`github-pitfalls`" in text
        assert "GitHub gotchas" in text
        assert "GitHub Pitfalls" in text
        assert "Use gh carefully" not in text
        assert "Full skill content is available to the agent" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_execute_code_formats_output(self):
        result = build_tool_complete("tc-code", "execute_code", '{"output":"hello\\n","exit_code":0}')
        text = result.content[0].content.text
        assert "Exit code: 0" in text
        assert "hello" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_skill_manage_summarizes_without_raw_json(self):
        result = build_tool_complete(
            "tc-skill-manage",
            "skill_manage",
            '{"success":true,"message":"Patched references/hermes-acp-zed-rendering.md in skill \'hermes-agent-operations\' (1 replacement)."}',
            function_args={
                "action": "patch",
                "name": "hermes-agent-operations",
                "file_path": "references/hermes-acp-zed-rendering.md",
            },
        )
        text = result.content[0].content.text
        assert "**✅ Skill updated**" in text
        assert "`patch`" in text
        assert "`hermes-agent-operations`" in text
        assert "references/hermes-acp-zed-rendering.md" in text
        assert "{\"success\"" not in text
        assert result.raw_output is None

    def test_build_tool_complete_for_read_file_formats_content(self):
        result = build_tool_complete(
            "tc-read",
            "read_file",
            '{"content":"1|hello\\n2|world","total_lines":2}',
            function_args={"path":"README.md","offset":1,"limit":20},
        )
        text = result.content[0].content.text
        assert "Read README.md" in text
        assert "1|hello" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_search_files_formats_matches(self):
        result = build_tool_complete(
            "tc-search",
            "search_files",
            '{"total_count":2,"matches":[{"path":"README.md","line":3,"content":"TODO: fix this"},{"path":"src/app.py","line":9,"content":"needle"}],"truncated":true}\n\n[Hint: Results truncated. Use offset=12 to see more.]',
        )
        text = result.content[0].content.text
        assert "Search results" in text
        assert "Found 2 matches" in text
        assert "README.md:3" in text
        assert "TODO: fix this" in text
        assert "Results truncated" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_process_list_formats_table(self):
        result = build_tool_complete(
            "tc-process",
            "process",
            '{"processes":[{"session_id":"p1","status":"running","pid":123,"command":"npm run dev"}]}',
            function_args={"action":"list"},
        )
        text = result.content[0].content.text
        assert "Processes: 1" in text
        assert "`p1`" in text
        assert "npm run dev" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_delegate_task_summarizes_children(self):
        result = build_tool_complete(
            "tc-delegate",
            "delegate_task",
            '{"results":[{"task_index":0,"status":"completed","summary":"Reviewed ACP rendering.","model":"gpt-5.5","duration_seconds":3.2,"tool_trace":[{"tool":"read_file"}]}],"total_duration_seconds":3.4}',
        )
        text = result.content[0].content.text
        assert "Delegation results: 1 task" in text
        assert "Reviewed ACP rendering" in text
        assert "gpt-5.5" in text
        assert "Tools: read_file" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_session_search_recent(self):
        result = build_tool_complete(
            "tc-session",
            "session_search",
            '{"success":true,"mode":"recent","results":[{"session_id":"s1","title":"ACP work","last_active":"2026-05-02","message_count":12,"preview":"Polished tool rendering."}],"count":1}',
        )
        text = result.content[0].content.text
        assert "Recent sessions" in text
        assert "ACP work" in text
        assert "Polished tool rendering" in text
        assert result.raw_output is None

    def test_build_tool_complete_for_memory_avoids_dumping_entries(self):
        result = build_tool_complete(
            "tc-memory",
            "memory",
            '{"success":true,"target":"user","entries":["private long memory"],"usage":"1% — 19/2000 chars","entry_count":1,"message":"Entry added."}',
            function_args={"action":"add","target":"user","content":"User likes concise ACP rendering."},
        )
        text = result.content[0].content.text
        assert "Memory add saved" in text
        assert "User likes concise ACP rendering" in text
        assert "private long memory" not in text
        assert result.raw_output is None

    def test_build_tool_complete_for_web_extract_summarizes_urls(self):
        result = build_tool_complete(
            "tc-web-extract",
            "web_extract",
            '{"results":[{"url":"https://example.com","title":"Example","content":"# Intro\\nThis is extracted content."}]}',
        )
        text = result.content[0].content.text
        assert "Web extract: 1 URL" in text
        assert "Example" in text
        assert "Content:" in text
        assert "Intro" in text
        assert result.raw_output is None

    def test_build_tool_complete_truncates_large_output(self):
        """Very large outputs should be truncated."""
        big_output = "x" * 10000
        result = build_tool_complete("tc-6", "read_file", big_output)
        assert isinstance(result, ToolCallProgress)
        display_text = result.content[0].content.text
        assert len(display_text) < 6000
        assert "truncated" in display_text

    def test_build_tool_complete_for_patch_uses_diff_blocks(self):
        """Completed patch calls should keep structured diff content for Zed."""
        patch_result = (
            '{"success": true, "diff": "--- a/README.md\\n+++ b/README.md\\n@@ -1 +1,2 @@\\n old line\\n+new line\\n", '
            '"files_modified": ["README.md"]}'
        )
        result = build_tool_complete("tc-p1", "patch", patch_result)
        assert isinstance(result, ToolCallProgress)
        assert len(result.content) == 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path == "README.md"
        assert diff_item.old_text == "old line"
        assert diff_item.new_text == "old line\nnew line"

    def test_build_tool_complete_for_patch_falls_back_to_text_when_no_diff(self):
        result = build_tool_complete("tc-p2", "patch", '{"success": true}')
        assert isinstance(result, ToolCallProgress)
        assert isinstance(result.content[0], ContentToolCallContent)

    def test_build_tool_complete_for_write_file_uses_snapshot_diff(self, tmp_path):
        target = tmp_path / "diff-test.txt"
        snapshot = type("Snapshot", (), {"paths": [target], "before": {str(target): None}})()
        target.write_text("hello from hermes\n", encoding="utf-8")

        result = build_tool_complete(
            "tc-wf1",
            "write_file",
            '{"bytes_written": 18, "dirs_created": false}',
            function_args={"path": str(target), "content": "hello from hermes\n"},
            snapshot=snapshot,
        )
        assert isinstance(result, ToolCallProgress)
        assert len(result.content) == 1
        diff_item = result.content[0]
        assert isinstance(diff_item, FileEditToolCallContent)
        assert diff_item.path.endswith("diff-test.txt")
        assert diff_item.old_text is None
        assert diff_item.new_text == "hello from hermes"


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
