"""Tests for AIAgent._repair_tool_call — tool-name normalization.

Regression guard for #14784: Claude-style models sometimes emit
class-like tool-call names (``TodoTool_tool``, ``Patch_tool``,
``BrowserClick_tool``, ``PatchTool``). Before the fix they returned
"Unknown tool" even though the target tool was registered under a
snake_case name. The repair routine now normalizes CamelCase,
strips trailing ``_tool`` / ``-tool`` / ``tool`` suffixes (up to
twice to handle double-tacked suffixes like ``TodoTool_tool``). A name
that is merely *close* to a registered tool is never repaired (see
``TestSimilarNamesAreNotExecuted``).
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
    "execute_code",
    "session_search",
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







class TestClassLikeEmissions:
    """Regression coverage for #14784 — CamelCase + _tool suffix variants."""

    def test_camel_case_no_suffix(self, repair):
        assert repair("BrowserClick") == "browser_click"









class TestEdgeCases:
    """Edge inputs that must not crash or produce surprising results."""

    def test_empty_string(self, repair):
        assert repair("") is None





class TestVolcEngineXmlPollution:
    """Regression coverage for #33007 — VolcEngine ``api/plan`` endpoint
    leaks raw XML attribute fragments into ``tool_use.name``.

    Observed in production with the ``anthropic_messages`` API mode:

        terminal" parameter="command" string="true
        execute_code" parameter="code" string="true
        session_search" parameter="session_id" string="true

    The fix trims at the first ``"``/``'``/``<``/``>`` so the rest of
    the repair pipeline can resolve the cleaned name to a real tool.
    """

    def test_terminal_with_xml_attribute_pollution(self, repair):
        # Exact pattern from the bug report (terminal call).
        polluted = 'terminal" parameter="command" string="true'
        assert repair(polluted) == "terminal"




    def test_tool_name_with_trailing_quote_only(self, repair):
        # Minimal leak — just a stray trailing quote, no full attribute.
        assert repair('terminal"') == "terminal"



    def test_clean_tool_name_unaffected_by_sanitizer(self, repair):
        # Pure passthrough — no XML/quote chars, no change.
        assert repair("execute_code") == "execute_code"
        assert repair("session_search") == "session_search"

    def test_space_separated_name_still_normalizes(self, repair):
        # Critical: the XML strip must NOT consume whitespace, or the
        # legitimate ``"write file" -> write_file`` repair path breaks.
        assert repair("write file") == "write_file"


    def test_leading_quote_is_not_repaired(self, repair):
        # Sanitizer only trims when the XML char is at idx > 0 — a
        # name that *starts* with a quote is left untouched. It is no
        # longer fuzzy-matched onto "terminal" either: it is reported
        # as unknown with a "Did you mean" hint
        # (tests/agent/test_tool_name_suggestion.py).
        assert repair('"terminal"') is None


def _repair_against(valid):
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=valid)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestSimilarNamesAreNotExecuted:
    """A name that is only *similar* to a registered tool (typo, sibling,
    inverse) is not repaired, so the call never runs under a name the model
    did not emit; it takes the unknown-tool path with a "Did you mean" hint
    instead. Spelling variants of the SAME name keep repairing."""

    @pytest.mark.parametrize("name", [
        "terminl", "read_fle", "web_serch", "browser_clik", "execute_cod", "session_serch",
    ])
    def test_typos_are_not_repaired(self, repair, name):
        assert repair(name) is None

    def test_xml_polluted_typo_is_not_repaired(self, repair):
        assert repair('terminl" parameter="command" string="true') is None

    def test_sibling_of_an_unavailable_tool_is_not_substituted(self):
        # kanban_list is withheld for this session; kanban_link is not an alias for it.
        assert _repair_against({"kanban_link", "kanban_block"})("kanban_list") is None
        # An operation must not resolve to its own inverse.
        assert _repair_against({"kanban_block"})("kanban_unblok") is None

    def test_different_tool_with_a_similar_name_is_not_substituted(self):
        assert _repair_against({"web_search", "search_files"})("search_web") is None

    @pytest.mark.parametrize("name, expected", [
        ("READ_FILE", "read_file"),
        ("Read-File", "read_file"),
        ("write file", "write_file"),
        ("BrowserNavigate", "browser_navigate"),
        ("browser-click", "browser_click"),
        ("TodoTool_tool", "todo"),
        ("PatchTool", "patch"),
        ("Patch_tool", "patch"),
        ("web_search-tool", "web_search"),
        ('terminal" parameter="command" string="true', "terminal"),
    ])
    def test_spelling_variants_of_the_same_name_still_repair(self, repair, name, expected):
        assert repair(name) == expected
