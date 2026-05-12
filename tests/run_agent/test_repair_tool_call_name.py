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


class TestCCCanonicalAliasFastPath:
    """Anthropic OAuth path emits CC canonical names (Bash, Read, Edit,
    Write, Grep) because cc_aliases.replace_with_cc_canonical substitutes
    them on the outbound side to satisfy the plan-budget billing
    classifier. Validation runs before dispatch, so _repair_tool_call
    must translate these back to their hermes equivalents — exact match,
    case-sensitive, no normalization.
    """

    def test_repairs_cc_bash_to_terminal(self):
        from run_agent import AIAgent
        stub = SimpleNamespace(valid_tool_names={"terminal", "read_file"})
        repair = AIAgent._repair_tool_call.__get__(stub, AIAgent)
        assert repair("Bash") == "terminal"

    def test_repairs_cc_read_to_read_file(self, repair):
        assert repair("Read") == "read_file"

    def test_repairs_cc_edit_to_patch(self, repair):
        assert repair("Edit") == "patch"

    def test_repairs_cc_write_to_write_file(self, repair):
        assert repair("Write") == "write_file"

    def test_repairs_cc_grep_to_search_files(self):
        # search_files isn't in the default VALID set; build a fixture
        # that includes it so we can verify the alias resolves.
        from run_agent import AIAgent
        stub = SimpleNamespace(valid_tool_names=VALID | {"search_files"})
        repair = AIAgent._repair_tool_call.__get__(stub, AIAgent)
        assert repair("Grep") == "search_files"

    def test_cc_alias_only_when_hermes_name_valid(self):
        # If the mapped hermes name isn't registered, the fast-path must
        # NOT return it — fall through to the rest of the repair logic
        # (which has nothing matching "Bash" → returns None here).
        from run_agent import AIAgent
        stub = SimpleNamespace(valid_tool_names={"read_file", "patch"})
        repair = AIAgent._repair_tool_call.__get__(stub, AIAgent)
        assert repair("Bash") is None
