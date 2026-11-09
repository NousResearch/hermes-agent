"""Tests for acp_adapter.tools — tool kind mapping and ACP content building."""


import pytest

from acp_adapter.edit_approval import EditProposal
from acp_adapter.tools import (
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
from gateway.session_tool_events import bounded_result
from tools.browser_use_cli import _native_screenshot_result
from tools.registry import tool_error, tool_result
from toolsets import resolve_toolset


# ---------------------------------------------------------------------------
# TOOL_KIND_MAP coverage
# ---------------------------------------------------------------------------


COMMON_HERMES_TOOLS = ["read_file", "search_files", "terminal", "patch", "write_file", "process"]


class TestToolKindMap:










    def test_unknown_tool_returns_other_kind(self):
        assert get_tool_kind("nonexistent_tool_xyz") == "other"


# ---------------------------------------------------------------------------
# make_tool_call_id
# ---------------------------------------------------------------------------


class TestMakeToolCallId:


    def test_ids_are_unique(self):
        ids = {make_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# build_tool_title
# ---------------------------------------------------------------------------


class TestBuildToolTitle:

    def test_terminal_title_truncates_long_command(self):
        long_cmd = "x" * 200
        title = build_tool_title("terminal", {"command": long_cmd})
        assert len(title) < 120
        assert "..." in title





    def test_unknown_tool_uses_name(self):
        title = build_tool_title("some_new_tool", {"foo": "bar"})
        assert title == "some_new_tool"

    @pytest.mark.parametrize(
        "tool_name, args",
        [
            ("terminal", {"command": "git status --short"}),
            ("read_file", {"path": "/etc/hosts", "offset": 10}),
            ("search_files", {"pattern": "TODO", "path": "src"}),
            ("web_search", {"query": "hermes agent acp"}),
            ("execute_code", {"code": "\nfrom hermes_tools import terminal\nprint('done')"}),
            ("skill_view", {"name": "github", "file_path": "references/x.md"}),
        ],
    )
    def test_title_derives_from_display_preview(self, tool_name, args):
        """ACP titles are the shared agent.display preview, not a parallel per-tool table."""
        from agent.display import build_tool_preview

        assert build_tool_preview(tool_name, args, max_len=80) in build_tool_title(tool_name, args)


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
        assert "src/main.py" in item.content.text


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







    def test_build_tool_start_for_browser_navigate(self):
        """browser_navigate should emit a polished start event."""
        args = {"url": "https://x.com"}
        result = build_tool_start("tc-browser-start", "browser_navigate", args)
        assert isinstance(result, ToolCallStart)
        assert "https://x.com" in result.title
        assert result.kind == "fetch"
        assert "https://x.com" in result.content[0].content.text
        assert result.raw_input is None

    def test_every_browser_tool_in_the_acp_toolset_reports_execute(self):
        """A browser tool in the ACP toolset is typed at start, and a refused call ends failed."""
        browser_tools = [name for name in resolve_toolset("hermes-acp") if name.startswith("browser_")]
        assert browser_tools

        for name in browser_tools:
            started = build_tool_start("tc-browser-start", name, {"code": "print(page_info())"})
            refused = build_tool_complete("tc-browser-done", name, tool_error("Blocked: unsafe URL"))
            assert (started.kind, started.raw_input) == ("execute", None), name
            assert (refused.kind, refused.status) == ("execute", "failed"), name
            assert "Blocked: unsafe URL" in refused.content[0].content.text, name








# ---------------------------------------------------------------------------
# build_tool_complete
# ---------------------------------------------------------------------------


_BROWSER_STDOUT = "o" * 2_000 + "STDOUT-KEPT"
_BROWSER_RUN_CARD = ("STDOUT-KEPT", "STDERR-END", "/tmp/bu/shot.png", "/tmp/bu/workspace")
_BROWSER_CUT_RUN_CARD = ("STDOUT-KEPT", "event size limit")


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








    def test_build_tool_complete_marks_returncode_nonzero_as_failed(self):
        result = build_tool_complete("tc-fail", "execute_code", '{"output": "bad", "returncode": 2}')
        assert result.status == "failed"

    @pytest.mark.parametrize(
        "exit_code, stdout, image_chars, card",
        [
            pytest.param(1, _BROWSER_STDOUT + "o" * 5_000, None, _BROWSER_RUN_CARD, id="plain_result"),
            pytest.param(1, _BROWSER_STDOUT + "o" * 70_000, None, _BROWSER_CUT_RUN_CARD, id="plain_result_over_the_event_cap"),
            pytest.param(0, _BROWSER_STDOUT + "o" * 70_000, None, _BROWSER_CUT_RUN_CARD, id="successful_run_over_the_event_cap"),
            pytest.param(1, _BROWSER_STDOUT + "o" * 5_000, 1_000, _BROWSER_RUN_CARD, id="screenshot_envelope"),
            pytest.param(1, _BROWSER_STDOUT + "o" * 5_000, 200_000, _BROWSER_RUN_CARD, id="envelope_cut_at_the_event_cap"),
            pytest.param(1, _BROWSER_STDOUT + "o" * 70_000, 200_000, _BROWSER_CUT_RUN_CARD, id="stdout_fills_the_cut"),
            # One pass of the pattern is 16 envelope characters, so these shifts land the cut on each of them.
            *[pytest.param(1, "STDOUT-KEPT" + "x" * shift + '🐴"' * 3_000, 200_000, _BROWSER_CUT_RUN_CARD,
                           id=f"cut_lands_{shift}_into_the_escapes") for shift in range(16)],
        ],
    )
    def test_build_tool_complete_for_browser_exec_shows_the_run(self, monkeypatch, exit_code, stdout, image_chars, card):
        """The card shows the run itself, not the envelope a screenshot or the gateway's event cap wraps it in."""
        run = {
            "success": exit_code == 0, "exit_code": exit_code, "output": stdout,
            "workspace": "/tmp/bu/workspace", "stderr": "e" * 2_000 + "STDERR-END", "screenshot_path": "/tmp/bu/shot.png",
        }
        delivered = tool_result(run)
        if image_chars is not None:
            monkeypatch.setattr("tools.vision_tools._should_use_native_vision_fast_path", lambda: True)
            monkeypatch.setattr(
                "tools.vision_tools._resize_image_for_vision",
                lambda *_args, **_kwargs: "data:image/jpeg;base64," + "A" * image_chars,
            )
            delivered = _native_screenshot_result(run, run["screenshot_path"])

        update = build_tool_complete("tc-browser", "browser_exec", bounded_result(delivered, False), {"code": "print(1)"})

        text = update.content[0].content.text
        text.encode("utf-8")  # half a surrogate pair would not survive the editor's JSON parser
        assert [shown for shown in (f"Exit code: {exit_code}", *card) if shown not in text] == []
        assert [leaked for leaked in ("_multimodal", "base64", '"exit_code"', '"truncated"', "\\") if leaked in text] == []
        assert (update.kind, update.status, update.raw_output) == ("execute", "failed" if exit_code else "completed", None)








    def test_build_tool_complete_for_search_files_formats_matches(self):
        result = build_tool_complete(
            "tc-search",
            "search_files",
            '{"total_count":2,"matches":[{"path":"README.md","line":3,"content":"TODO: fix this"},{"path":"src/app.py","line":9,"content":"needle"}],"truncated":true}\n\n[Hint: Results truncated. Use offset=12 to see more.]',
        )
        text = result.content[0].content.text
        assert "README.md:3" in text
        assert "TODO: fix this" in text
        assert "Results truncated" in text
        assert result.raw_output is None







    def test_build_tool_complete_generically_formats_unknown_json_dict_without_raw_output(self):
        result = build_tool_complete(
            "tc-recall-search",
            "memory_archive_search",
            '{"results":[{"id":"obs-1","status":"active","content":"Recall should render as a readable summary."}],"trust":"lower-trust archive evidence"}',
        )
        text = result.content[0].content.text
        assert "lower-trust archive evidence" in text
        assert "Recall should render as a readable summary" in text
        assert "{\"results\"" not in text
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
