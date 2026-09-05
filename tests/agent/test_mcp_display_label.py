"""MCP tool calls must not render their wire scaffolding to the user.

A tool with no curated verb falls back to printing its own name, which is
the one display path where a raw name reaches the user. On a runtime that
routes Hermes' tools through the ``hermes-tools`` MCP server, that produced::

    ⚙️ mcp__hermes-tools__update_active_task: "mcp__hermes-tools__update_active_task"

— the namespace spelled out, and then the name repeated as its own preview
because the tool had no preview rule and the caller's fallback is the name.

These tests pin the rendering contract, not the wording of any one verb.
"""

import pytest

from agent.display import (
    build_tool_preview,
    get_tool_verb,
    tool_verb_connector,
    verb_drops_preview,
)
from agent.tool_identity import (
    MCP_SERVER_SEPARATOR,
    canonical_tool_name,
    display_tool_label,
    strip_mcp_namespace,
)

INTERNAL = "mcp__hermes-tools__"


class TestInternalToolsLoseTheNamespace:
    """Hermes' own tools are Hermes tools whichever road they arrive by."""

    @pytest.mark.parametrize(
        "tool",
        ["update_active_task", "memory", "session_search", "skill_view", "read_file"],
    )
    def test_underscore_convention_is_stripped(self, tool):
        assert display_tool_label(INTERNAL + tool) == tool

    @pytest.mark.parametrize(
        "tool", ["update_active_task", "memory", "skill_manage"]
    )
    def test_dot_convention_is_stripped(self, tool):
        assert display_tool_label(f"mcp.hermes-tools.{tool}") == tool

    def test_label_never_contains_wire_scaffolding(self):
        label = display_tool_label(INTERNAL + "update_active_task")
        assert "mcp__" not in label
        assert "hermes-tools" not in label

    def test_an_internal_tool_renders_identically_to_its_bare_form(self):
        """The whole point: same tool, same label, whichever runtime called."""
        assert display_tool_label(INTERNAL + "memory") == display_tool_label("memory")


class TestThirdPartyServersStayVisible:
    """Hiding which server ran a tool would be misleading."""

    def test_server_is_preserved(self):
        assert display_tool_label("mcp__linear__get_issue") == (
            f"linear{MCP_SERVER_SEPARATOR}get_issue"
        )

    def test_dot_convention_too(self):
        assert display_tool_label("mcp.linear.get_issue") == (
            f"linear{MCP_SERVER_SEPARATOR}get_issue"
        )

    def test_scaffolding_is_still_dropped(self):
        label = display_tool_label("mcp__linear__get_issue")
        assert "mcp__" not in label
        assert "__" not in label

    def test_server_name_survives_verbatim(self):
        assert display_tool_label("mcp__github__create_pr").startswith("github")


class TestNonMcpNamesAreUnaffected:
    def test_native_names_pass_through(self):
        assert display_tool_label("terminal") == "terminal"

    def test_foreign_names_are_canonicalized(self):
        """A label is still a canonical label: Bash is terminal."""
        assert display_tool_label("Bash") == "terminal"
        assert display_tool_label("exec_command") == "terminal"

    def test_unmapped_tools_keep_their_own_name(self):
        assert display_tool_label("ToolSearch") == "ToolSearch"

    @pytest.mark.parametrize(
        "malformed", ["mcp__", "mcp__onlyserver", "mcp.", "mcp.onlyserver", "mcp"]
    )
    def test_malformed_mcp_names_do_not_crash(self, malformed):
        assert isinstance(display_tool_label(malformed), str)

    def test_empty_name_is_returned_unchanged(self):
        assert display_tool_label("") == ""


class TestStripMcpNamespaceUnchanged:
    """The refactor behind display_tool_label must not move this behaviour."""

    def test_internal_is_stripped(self):
        assert strip_mcp_namespace(INTERNAL + "memory") == "memory"

    def test_third_party_is_returned_untouched(self):
        name = "mcp__linear__get_issue"
        assert strip_mcp_namespace(name) == name

    @pytest.mark.parametrize("malformed", ["mcp__", "mcp__onlyserver", "mcp.x"])
    def test_malformed_returned_untouched(self, malformed):
        assert strip_mcp_namespace(malformed) == malformed

    def test_plain_name_untouched(self):
        assert strip_mcp_namespace("read_file") == "read_file"


class TestTheTwoToolsThatHadNoVerb:
    """Both fell to the raw-name branch, which is what looked unrefined."""

    @pytest.mark.parametrize("name", ["update_active_task", INTERNAL + "update_active_task"])
    def test_update_active_task_has_a_verb(self, name):
        assert get_tool_verb(name)

    def test_update_active_task_drops_its_preview(self):
        """The whole record is the argument; a first-line echo would mislead."""
        assert verb_drops_preview(INTERNAL + "update_active_task")

    def test_tool_search_has_a_verb(self):
        assert get_tool_verb("ToolSearch")

    def test_tool_search_previews_its_query(self):
        preview = build_tool_preview(
            "ToolSearch", {"query": "select:Read,Edit"}, max_len=None
        )
        assert preview == "select:Read,Edit"

    def test_tool_search_reads_as_a_search(self):
        assert tool_verb_connector("ToolSearch") == " for "


class TestNoToolRendersAsNameColonName:
    """The doubled-name form is the defect; assert it cannot recur."""

    @pytest.mark.parametrize(
        "raw",
        [
            INTERNAL + "update_active_task",
            INTERNAL + "memory",
            "mcp__linear__get_issue",
            "ToolSearch",
            "Bash",
        ],
    )
    def test_label_is_never_the_raw_wire_name_when_namespaced(self, raw):
        label = display_tool_label(raw)
        assert label
        if raw.startswith("mcp"):
            assert label != raw, "namespaced name reached the user verbatim"

    def test_canonical_and_label_agree_for_internal_tools(self):
        raw = INTERNAL + "read_file"
        assert display_tool_label(raw) == canonical_tool_name(raw)
