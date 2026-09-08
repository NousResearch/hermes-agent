"""Regression tests for ``AIAgent._toolguard_controlled_halt_response`` (#105404).

The user-visible final reply must explain the safely-known, code-derived
blocker rather than pointing at the last tool result (invisible on messaging
surfaces such as Telegram), stay bounded and fail-closed for unknown codes,
and never copy raw arguments, payloads, credentials or private URLs.
"""
import re

import pytest

from agent.tool_guardrails import ToolGuardrailDecision
from run_agent import AIAgent


def _halt(code: str, *, tool_name: str = "example_tool", count: int = 5) -> ToolGuardrailDecision:
    return ToolGuardrailDecision(
        action="halt", code=code, tool_name=tool_name, count=count)


def _response(code: str, **kw) -> str:
    agent = AIAgent.__new__(AIAgent)
    return agent._toolguard_controlled_halt_response(_halt(code, **kw))


def test_identical_call_streak_explains_repeated_result():
    msg = _response("identical_call_streak_halt", count=5)
    assert "5 times" in msg
    assert "example_tool" in msg


def test_loop_web_search_cap_explains_limit():
    msg = _response("loop_web_search_cap", count=50)
    assert "web search limit" in msg
    assert "50" in msg


def test_loop_subagent_cap_explains_limit():
    msg = _response("loop_subagent_cap", count=10)
    assert "subagent limit" in msg
    assert "10" in msg


def test_unknown_code_falls_back_to_no_progress():
    msg = _response("some_future_halt_code", count=3)
    assert "no progress" in msg
    # Count is not surfaced for unknown codes (no safe template knows what it means).
    assert "3" not in msg


@pytest.mark.parametrize("code", [
    "identical_call_streak_halt", "loop_web_search_cap", "loop_subagent_cap", "some_future_code",
])
def test_reply_does_not_reference_invisible_tool_result(code):
    """Regression for #105404: the reply must not point at the last tool result."""
    assert "last tool result" not in _response(code)


@pytest.mark.parametrize("code", [
    "identical_call_streak_halt", "loop_web_search_cap", "loop_subagent_cap", "some_future_code",
])
def test_reply_does_not_claim_completion(code):
    assert not re.search(r"\bcomplete[ds]?\b", _response(code), re.IGNORECASE)
