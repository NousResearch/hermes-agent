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


# -- MCP name bridging (#100807) --------------------------------------------
#
# Prompts teach the BARE MCP catalog name; the runtime registers
# ``mcp__<server>__<tool>``. The old bare-fuzzy failed both directions:
# bare names scored below cutoff (whole batch voided), and hallucinated
# prefixed names silently fuzzy-resolved to a DIFFERENT real MCP tool
# (measured 12/12 mis-routes on a 51-tool server).

MCP_VALID = {
    "mcp__example_server__entity_search",
    "mcp__example_server__entity_update",
    "mcp__example_server__drive_read",
    "mcp__example_server__calendar_update",
    "mcp__other_server__drive_read",  # same catalog name on 2 servers
} | VALID  # core tools still registered alongside

MIXED_STUB = SimpleNamespace(valid_tool_names=MCP_VALID)


@pytest.fixture
def mcp_repair():
    from run_agent import AIAgent
    return AIAgent._repair_tool_call.__get__(MIXED_STUB, AIAgent)


class TestMcpNameBridging:
    def test_bare_catalog_name_bridges_to_registered_mcp_tool(self, mcp_repair):
        """entity_search -> mcp__example_server__entity_search (exactly one
        server exposes it)."""
        assert mcp_repair("entity_search") == "mcp__example_server__entity_search"

    def test_ambiguous_bare_name_returns_none_not_fuzzy(self, mcp_repair):
        """drive_read is exposed by TWO servers -> ambiguous -> None. Never
        guess the server; let the model self-correct."""
        assert mcp_repair("drive_read") is None

    def test_unknown_bare_name_returns_none_not_fuzzy(self, mcp_repair):
        """A bare name no MCP server exposes (and no core match) -> None,
        the existing 'does not exist' error the model self-corrects from."""
        assert mcp_repair("entity_delete") is None

    def test_hallucinated_prefixed_name_never_mis_routes(self, mcp_repair):
        """The 12/12 mis-route class: mcp__example_server__entity_delete does
        NOT exist; under old fuzzy it resolved to entity_update (0.906).
        Now: None."""
        assert mcp_repair("mcp__example_server__entity_delete") is None

    def test_prefixed_core_tool_strips_to_core(self, mcp_repair):
        """A model prefixed a CORE tool with mcp__. Strip to the bare
        suffix and match core tools exactly."""
        assert mcp_repair("mcp__todo") == "todo"
        assert mcp_repair("mcp__web_search") == "web_search"

    def test_exact_mcp_name_passes_through(self, mcp_repair):
        """An exact registered mcp name is a direct hit, unchanged."""
        assert (
            mcp_repair("mcp__example_server__entity_search")
            == "mcp__example_server__entity_search"
        )

    def test_core_fuzzy_still_works_alongside_mcp(self, mcp_repair):
        """Core-only fuzzy behavior is unchanged by the MCP machinery
        (browser_click variants still repair)."""
        assert mcp_repair("browser-clic") == "browser_click"

    def test_bare_name_not_registered_anywhere_is_none(self, mcp_repair):
        assert mcp_repair("totally_unknown") is None
