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


# MCP-prefixed tool names registered under the canonical ``mcp_<server>_<tool>``
# scheme. Qwen3-class tool-callers token-split and corrupt the ``mcp_`` prefix.
MCP_VALID = {
    "mcp_knowledge_kb_get",
    "mcp_knowledge_kb_search",
    "mcp_proxmox_get_nodes",
}


@pytest.fixture
def mcp_repair():
    """Bind _repair_tool_call to a stub exposing MCP-prefixed tool names."""
    from run_agent import AIAgent

    stub = SimpleNamespace(valid_tool_names=MCP_VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestNormaliseMcpToolPrefix:
    """Unit coverage for the prefix-normalisation helper in isolation.

    Qwen3-235B token-split emits ``mmcp_knowledge_kb_get`` (extra leading
    'm') or ``mcp_mcp_knowledge_kb_get`` (doubled prefix). The helper must
    collapse those to a single ``mcp_`` prefix and leave everything else
    untouched.
    """

    def test_strips_spurious_leading_m(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("mmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"
        )

    def test_collapses_doubled_prefix(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("mcp_mcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_collapses_triple_nested_prefix(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # Three layers must collapse all the way to a single ``mcp_``.
        assert (
            normalise_mcp_tool_prefix("mcp_mcp_mcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_collapses_quadruple_nested_prefix(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # Arbitrary depth normalises to a single ``mcp_`` prefix.
        assert (
            normalise_mcp_tool_prefix("mcp_mcp_mcp_mcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_strips_leading_junk_before_mcp(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("xmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"
        )
        assert (
            normalise_mcp_tool_prefix("abmcp_knowledge_kb_get")
            == "mcp_knowledge_kb_get"
        )

    def test_legit_name_passes_through_unchanged(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert (
            normalise_mcp_tool_prefix("mcp_knowledge_kb_get") == "mcp_knowledge_kb_get"
        )

    def test_non_mcp_name_passes_through_unchanged(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # No ``mcp_`` anywhere near the front — must not be touched.
        assert normalise_mcp_tool_prefix("kb_search") == "kb_search"
        assert normalise_mcp_tool_prefix("4pkoPc7Uh") == "4pkoPc7Uh"

    def test_junk_too_far_from_front_not_normalised(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        # ``mcp_`` more than 5 chars in is not treated as a corrupted prefix.
        assert normalise_mcp_tool_prefix("toolongprefix_mcp_x") == "toolongprefix_mcp_x"

    def test_empty_string_passes_through(self):
        from agent.agent_runtime_helpers import normalise_mcp_tool_prefix

        assert normalise_mcp_tool_prefix("") == ""


class TestMcpPrefixRepairFastPath:
    """End-to-end: corrupted MCP names repair to the canonical name via the
    direct-match fast-path; legit and truly-garbled names behave correctly.
    """

    def test_mmcp_repairs_to_canonical(self, mcp_repair):
        assert mcp_repair("mmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_doubled_prefix_repairs_to_canonical(self, mcp_repair):
        assert mcp_repair("mcp_mcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_leading_junk_repairs_to_canonical(self, mcp_repair):
        assert mcp_repair("xmcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_legit_mcp_name_unchanged(self, mcp_repair):
        assert mcp_repair("mcp_knowledge_kb_get") == "mcp_knowledge_kb_get"

    def test_bare_name_without_prefix_not_falsely_normalised(self, mcp_repair):
        # ``kb_search`` has no ``mcp_`` prefix — normalisation is a no-op, and
        # it must NOT be silently rewritten to mcp_knowledge_kb_search by
        # the prefix fast-path. (Fuzzy match against an mcp_-prefixed name is
        # well below the 0.7 cutoff, so this returns None.)
        assert mcp_repair("kb_search") is None

    def test_truly_garbled_token_returns_none(self, mcp_repair):
        # Random token from a token-split must still fall through, not match.
        assert mcp_repair("4pkoPc7Uh") is None
