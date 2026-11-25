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


@pytest.mark.parametrize("name, repaired", [
    ("read_fiel", "read_file"),
    ('terminal" parameter="command" string="true', "terminal"),  # #33007 shape, anthropic_messages
])
def test_repaired_name_reaches_the_native_replay(name, repaired):
    """Signed-thinking turns replay from the ordered sidecars, so the repair must rename them too."""
    import json

    from agent.anthropic_message_convert import _convert_assistant_message
    from agent.bedrock_adapter import _assistant_blocks
    from agent.turn_tool_validation import validate_tool_calls
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "anthropic", "m", "s1", []
    agent.valid_tool_names = set(VALID)
    agent.log_prefix, agent._base_url, agent.api_mode = "", "https://api.example.invalid", "anthropic_messages"
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    agent._vprint = lambda *a, **k: None
    args = {"path": "a.txt"}
    provider_data = {
        "anthropic_content_blocks": [{"type": "thinking", "thinking": "read it", "signature": "sig"},
                                     {"type": "tool_use", "id": "toolu_1", "name": name, "input": args}],
        "bedrock_content_blocks": [{"reasoningContent": {"text": "read it", "signature": "sig"}},
                                   {"toolUse": {"toolUseId": "toolu_1", "name": name, "input": args}}],
    }
    tc = SimpleNamespace(id="toolu_1", type="function",
                         function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    message = SimpleNamespace(content="", tool_calls=[tc], provider_data=provider_data)
    verdict = validate_tool_calls(agent, message, "tool_calls", messages=[], conversation_history=[],
                                  api_call_count=1, effective_task_id="t1")
    stored = {"role": "assistant", "content": "", **provider_data, "tool_calls": [
        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]}

    assert verdict.action == "ok" and tc.function.name == repaired
    assert [b["name"] for b in _convert_assistant_message(stored)["content"] if b["type"] == "tool_use"] == [repaired]
    assert [b["toolUse"]["name"] for b in _assistant_blocks(stored, "") if "toolUse" in b] == [repaired]
