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


# Simulate a dispatcher-spawned kanban worker (issue #94506): the
# orchestrator-mode tools (kanban_list, kanban_unblock, kanban_show, etc.)
# are withheld by check_fn, while the worker-facing tools remain available.
# A gated tool name (or a misspelling of one) must resolve to None, not be
# fuzzy-remapped onto an available sibling.
GATED_KANBAN_VALID = VALID | {
    "kanban_link",
    "kanban_block",
    "kanban_comment",
    "kanban_create",
}


@pytest.fixture
def repair_gated_kanban():
    """repair() bound to a stub whose valid_tool_names simulates kanban
    worker gating (kanban_list / kanban_unblock withheld, siblings present)."""
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=GATED_KANBAN_VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestGatedToolNameNotRemapped:
    """Regression for #94506 — check_fn-gated tool names must not be
    fuzzy-remapped onto an available sibling (a read becoming a write, or
    an operation becoming its own inverse)."""

    def test_gated_exact_name_not_remapped(self, repair_gated_kanban):
        # kanban_list is withheld; kanban_link is available and similar.
        # Must fail closed, not silently dispatch a read onto a write.
        assert repair_gated_kanban("kanban_list") is None

    def test_gated_camelcase_name_not_remapped(self, repair_gated_kanban):
        assert repair_gated_kanban("KanbanList") is None

    def test_misspelled_gated_name_not_remapped(self, repair_gated_kanban):
        # Maintainer-confirmed gap (kiminofate): a misspelling of a gated
        # tool (kanban_unblock) must fail closed even though the *inverse*
        # operation kanban_block is available and highly similar.
        assert repair_gated_kanban("kanban_unblok") is None

    def test_misspelled_gated_list_not_remapped(self, repair_gated_kanban):
        # Another misspelling of the gated kanban_list: must not resolve to
        # the available kanban_link sibling.
        assert repair_gated_kanban("kanban_lsit") is None

    def test_typo_of_available_tool_still_repairs(self, repair_gated_kanban):
        # Control: a misspelling of an AVAILABLE tool must still repair.
        assert repair_gated_kanban("kanban_blck") == "kanban_block"


class TestSessionInjectedToolTypoRepairs:
    """Regression for #94506 review: context-engine / memory-provider
    schemas are added directly to valid_tool_names without being registered
    in tools.registry. A typo of such an injected tool must still repair,
    since the fuzzy candidate set unions registry + valid names."""

    def test_typo_of_session_injected_tool_repairs(self):
        # Simulate a session-injected tool (present in valid_tool_names,
        # absent from the registry).
        from run_agent import AIAgent
        from types import SimpleNamespace
        injected = VALID | {"lcm_grep"}
        stub = SimpleNamespace(valid_tool_names=injected)
        repair = AIAgent._repair_tool_call.__get__(stub, AIAgent)
        assert repair("lcm_grepp") == "lcm_grep"


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


    def test_leading_quote_falls_through_to_fuzzy_match(self, repair):
        # Sanitizer only trims when the XML char is at idx > 0 — a
        # name that *starts* with a quote is left untouched so the
        # rest of the pipeline (fuzzy match at 0.7 cutoff) can still
        # recover the obvious target.
        assert repair('"terminal"') == "terminal"
