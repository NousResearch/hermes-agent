"""The shared read behind "cut off, or malformed generation?" for tool-call arguments."""

from types import SimpleNamespace

from agent.tool_call_truncation_evidence import (
    ASSUMED_DEFAULT_OUTPUT_CAP,
    output_budget_for_request,
    truncation_verdict,
)


def test_verdict_disproves_only_with_usage_well_under_budget():
    assert truncation_verdict(185, 4096) == "disproved"
    assert truncation_verdict(4096, 4096) == "corroborated"
    # No usage is never a disproof: a dropped stream delivers no usage chunk.
    assert truncation_verdict(None, 4096) == "inconclusive"
    assert truncation_verdict(0, 4096) == "inconclusive"


def test_budget_prefers_wire_cap_then_agent_then_assumed_default():
    agent = SimpleNamespace(max_tokens=None, provider="custom", model="fake", api_mode="chat_completions")
    assert output_budget_for_request(agent, {"max_tokens": 8192}) == 8192
    agent.max_tokens = 2048
    assert output_budget_for_request(agent, {}) == 2048
    agent.max_tokens = None
    assert output_budget_for_request(agent, {}) == ASSUMED_DEFAULT_OUTPUT_CAP
