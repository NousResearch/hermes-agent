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


def _tool_call(name: str):
    return SimpleNamespace(function=SimpleNamespace(name=name))


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

    @pytest.mark.parametrize(
        ("legacy_name", "canonical_name"),
        sorted(__import__("model_tools", fromlist=["_LEGACY_TOOL_ALIASES"])._LEGACY_TOOL_ALIASES.items()),
    )
    def test_declared_legacy_alias_repairs_before_tool_name_validation(
        self,
        legacy_name,
        canonical_name,
    ):
        from run_agent import AIAgent

        stub = SimpleNamespace(valid_tool_names={canonical_name})
        repair_alias = AIAgent._repair_tool_call.__get__(stub, AIAgent)

        assert repair_alias(legacy_name) == canonical_name

    def test_declared_legacy_alias_repairs_when_canonical_tool_is_enabled_but_deferred(self):
        from run_agent import AIAgent

        stub = SimpleNamespace(
            valid_tool_names={"tool_search", "tool_describe", "tool_call"},
            enabled_toolsets=["todo"],
            disabled_toolsets=None,
        )
        repair_alias = AIAgent._repair_tool_call.__get__(stub, AIAgent)

        assert repair_alias("todo") == "todo_list"

    def test_declared_legacy_alias_does_not_widen_a_disabled_toolset(self):
        from run_agent import AIAgent

        stub = SimpleNamespace(
            valid_tool_names={"tool_search", "tool_describe", "tool_call"},
            enabled_toolsets=["todo"],
            disabled_toolsets=["todo"],
        )
        repair_alias = AIAgent._repair_tool_call.__get__(stub, AIAgent)

        assert repair_alias("todo") is None


class TestConversationValidationBoundary:
    @staticmethod
    def _agent(*, disabled_toolsets=None, valid_tool_names=None):
        from run_agent import AIAgent

        agent = SimpleNamespace(
            valid_tool_names=valid_tool_names
            or {"tool_search", "tool_describe", "tool_call"},
            enabled_toolsets=["todo"],
            disabled_toolsets=disabled_toolsets,
            log_prefix="",
        )
        agent._repair_tool_call = AIAgent._repair_tool_call.__get__(agent, AIAgent)
        return agent

    def test_legacy_alias_to_deferred_canonical_is_valid_for_that_call_only(self):
        from agent.agent_runtime_helpers import repair_and_classify_tool_call_names

        call = _tool_call("todo")
        invalid, exemptions = repair_and_classify_tool_call_names(
            self._agent(),
            [call],
        )

        assert invalid == []
        assert call.function.name == "todo_list"
        assert exemptions == {id(call)}

    def test_direct_deferred_canonical_name_still_requires_tool_call_bridge(self):
        from agent.agent_runtime_helpers import repair_and_classify_tool_call_names

        call = _tool_call("todo_list")
        invalid, exemptions = repair_and_classify_tool_call_names(
            self._agent(),
            [call],
        )

        assert invalid == ["todo_list"]
        assert exemptions == set()

    def test_disabled_legacy_alias_remains_invalid(self):
        from agent.agent_runtime_helpers import repair_and_classify_tool_call_names

        call = _tool_call("todo")
        invalid, exemptions = repair_and_classify_tool_call_names(
            self._agent(disabled_toolsets=["todo"]),
            [call],
        )

        assert invalid == ["todo"]
        assert exemptions == set()

    def test_visible_canonical_alias_needs_no_deferred_exemption(self):
        from agent.agent_runtime_helpers import repair_and_classify_tool_call_names

        call = _tool_call("todo")
        invalid, exemptions = repair_and_classify_tool_call_names(
            self._agent(valid_tool_names={"todo_list"}),
            [call],
        )

        assert invalid == []
        assert call.function.name == "todo_list"
        assert exemptions == set()







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
