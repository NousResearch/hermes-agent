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


# Claude Code tool-name aliasing — regression guard for #84222.
#
# Under an Anthropic OAuth token the request carries the Claude Code system
# identity, and Claude models trained on that identity reach for Claude Code's
# own tools (Glob/Grep/Read). Those are real tools, just not ours, and none of
# them survive the existing pipeline: `read` vs `read_file` scores 0.61 against
# difflib's 0.7 cutoff, and `glob`/`grep` have no near neighbour at all. Every
# such call burned one of the 3 agent-correction strikes.

CLAUDE_CODE_VALID = VALID | {"search_files"}


@pytest.fixture
def repair_cc():
    """``repair`` bound to a tool set that includes ``search_files``."""
    from run_agent import AIAgent
    stub = SimpleNamespace(valid_tool_names=CLAUDE_CODE_VALID)
    return AIAgent._repair_tool_call.__get__(stub, AIAgent)


class TestClaudeCodeToolAliases:
    @pytest.mark.parametrize(
        ("emitted", "expected"),
        [
            ("Glob", "search_files"),
            ("Grep", "search_files"),
            ("Read", "read_file"),
            # Case is the model's choice, not a contract.
            ("glob", "search_files"),
            ("GREP", "search_files"),
        ],
    )
    def test_read_only_claude_code_names_are_aliased(self, repair_cc, emitted, expected):
        assert repair_cc(emitted) == expected

    def test_alias_does_not_fire_when_target_is_not_registered(self):
        """An alias must never invent a tool the agent does not have."""
        from run_agent import AIAgent
        stub = SimpleNamespace(valid_tool_names={"terminal"})
        repair = AIAgent._repair_tool_call.__get__(stub, AIAgent)

        assert repair("Glob") is None

    def test_destructive_claude_code_names_are_not_aliased(self, repair_cc):
        """Write/Edit/Bash are deliberately excluded.

        Aliasing a read-only name turns a failed call into a successful one.
        Aliasing a destructive name turns a failed call into an *executed*
        one, under argument schemas the model did not write for. Those stay
        unknown-tool errors until someone maps the arguments too.
        """
        assert repair_cc("Write") is None
        assert repair_cc("Edit") is None
        assert repair_cc("Bash") is None

    def test_real_tool_name_still_wins_over_alias(self, repair_cc):
        """The alias table must not shadow a genuinely registered tool."""
        assert repair_cc("read_file") == "read_file"
        assert repair_cc("search_files") == "search_files"
