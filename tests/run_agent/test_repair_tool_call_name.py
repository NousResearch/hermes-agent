"""Tests for AIAgent._repair_tool_call — tool-name normalization.

Regression guard for #14784: Claude-style models sometimes emit
class-like tool-call names (``TodoTool_tool``, ``Patch_tool``,
``BrowserClick_tool``, ``PatchTool``). Before the fix they returned
"Unknown tool" even though the target tool was registered under a
snake_case name. The repair routine now normalizes CamelCase,
strips trailing ``_tool`` / ``-tool`` / ``tool`` suffixes (up to
twice to handle double-tacked suffixes like ``TodoTool_tool``), and
falls back to fuzzy match.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


VALID = {
    "todo",
    "patch",
    "browser_click",
    "browser_navigate",
    "web_search",
    "read_file",
    "write_file",
    "terminal",
}


@pytest.fixture
def repair():
    """Return a bound _repair_tool_call built on a minimal shell agent.

    We avoid constructing a real AIAgent (which pulls in credential
    resolution, session DB, etc.) because the repair routine only
    reads self.valid_tool_names. A SimpleNamespace stub is enough to
    bind the unbound function.
    """
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestExistingBehaviorStillWorks:
    """Pre-existing repairs must keep working (no regressions)."""

    def test_lowercase_already_matches(self, repair):
        assert repair("browser_click") == "browser_click"

    def test_uppercase_simple(self, repair):
        assert repair("TERMINAL") == "terminal"

    def test_dash_to_underscore(self, repair):
        assert repair("web-search") == "web_search"

    def test_space_to_underscore(self, repair):
        assert repair("write file") == "write_file"

    def test_fuzzy_near_miss(self, repair):
        # One-character typo — fuzzy match at 0.7 cutoff
        assert repair("terminall") == "terminal"

    def test_unknown_returns_none(self, repair):
        assert repair("xyz_no_such_tool") is None


class TestClassLikeEmissions:
    """Regression coverage for #14784 — CamelCase + _tool suffix variants."""

    def test_camel_case_no_suffix(self, repair):
        assert repair("BrowserClick") == "browser_click"

    def test_camel_case_with_underscore_tool_suffix(self, repair):
        assert repair("BrowserClick_tool") == "browser_click"

    def test_camel_case_with_Tool_class_suffix(self, repair):
        assert repair("PatchTool") == "patch"

    def test_double_tacked_class_and_snake_suffix(self, repair):
        # Hardest case from the report: TodoTool_tool — strip both
        # '_tool' (trailing) and 'Tool' (CamelCase embedded) to reach 'todo'.
        assert repair("TodoTool_tool") == "todo"

    def test_simple_name_with_tool_suffix(self, repair):
        assert repair("Patch_tool") == "patch"

    def test_simple_name_with_dash_tool_suffix(self, repair):
        assert repair("patch-tool") == "patch"

    def test_camel_case_preserves_multi_word_match(self, repair):
        assert repair("ReadFile_tool") == "read_file"
        assert repair("WriteFileTool") == "write_file"

    def test_mixed_separators_and_suffix(self, repair):
        assert repair("write-file_Tool") == "write_file"


class TestEdgeCases:
    """Edge inputs that must not crash or produce surprising results."""

    def test_empty_string(self, repair):
        assert repair("") is None

    def test_only_tool_suffix(self, repair):
        # '_tool' by itself is not a valid tool name — must not match
        # anything plausible.
        assert repair("_tool") is None

    def test_none_passed_as_name(self, repair):
        # Defensive: real callers always pass str, but guard against
        # a bug upstream that sends None.
        assert repair(None) is None

    def test_very_long_name_does_not_match_by_accident(self, repair):
        # Fuzzy match should not claim a tool for something obviously unrelated.
        assert repair("ThisIsNotRemotelyARealToolName_tool") is None


MCP_VALID = {
    "mcp_APE_meta",
    "mcp_APE_search",
    "mcp_APE_add",
    "mcp_APE_contextualize",
    "mcp_APE_graph_nearby",
    "mcp_APE_graph_connect",
    "mcp_discord_send",
    "terminal",
    "read_file",
}


@pytest.fixture
def mcp_repair():
    """Repair fixture with MCP-prefixed tool names registered."""
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=MCP_VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestMcpPrefixDrop:
    """Regression guard: models drop the ``mcp_`` prefix Hermes adds when
    registering MCP server tools (e.g. ``APE_search`` instead of
    ``mcp_APE_search``).  difflib scores ~0.60 for this pattern — just
    below the 0.7 cutoff — so the fuzzy path misses it.  The explicit
    prefix-prepend check introduced alongside these tests catches it."""

    def test_ape_meta_without_prefix(self, mcp_repair):
        assert mcp_repair("APE_meta") == "mcp_APE_meta"

    def test_ape_search_without_prefix(self, mcp_repair):
        assert mcp_repair("APE_search") == "mcp_APE_search"

    def test_ape_add_without_prefix(self, mcp_repair):
        assert mcp_repair("APE_add") == "mcp_APE_add"

    def test_ape_contextualize_without_prefix(self, mcp_repair):
        assert mcp_repair("APE_contextualize") == "mcp_APE_contextualize"

    def test_ape_graph_nearby_without_prefix(self, mcp_repair):
        assert mcp_repair("APE_graph_nearby") == "mcp_APE_graph_nearby"

    def test_other_mcp_server_without_prefix(self, mcp_repair):
        # Not APE-specific — any mcp_<server>_<tool> pattern should resolve.
        assert mcp_repair("discord_send") == "mcp_discord_send"

    def test_already_correct_name_unchanged(self, mcp_repair):
        # Already has prefix — fast-path returns it directly, no double-prefix.
        assert mcp_repair("mcp_APE_meta") == "mcp_APE_meta"

    def test_uppercase_without_prefix(self, mcp_repair):
        # Case-insensitive: APE_SEARCH -> mcp_APE_search via lowered candidate.
        assert mcp_repair("APE_SEARCH") == "mcp_APE_search"

    def test_non_mcp_tool_unaffected(self, mcp_repair):
        # Non-MCP tools in the valid set must still resolve normally.
        assert mcp_repair("TERMINAL") == "terminal"

    def test_prefix_drop_does_not_invent_tools(self, mcp_repair):
        # Prepending mcp_ to a non-existent name must not return a wrong match.
        assert mcp_repair("APE_nonexistent") is None

