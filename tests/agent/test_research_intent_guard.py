"""Contract tests for the explicit per-intent research collection guard."""

import pytest

from agent.tool_guardrails import (
    RESEARCH_BUDGET_EXHAUSTED,
    RESEARCH_INTENT_INVALID,
    RESEARCH_INTENT_REQUIRED,
    RESEARCH_SYNTHESIS_STATE,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    normalize_research_budget,
    normalize_research_intent,
)


def _controller(**budget):
    config = ToolCallGuardrailConfig.from_mapping({"research_budget": budget})
    return ToolCallGuardrailController(config)


def _successful_call(controller, *, query, label, tool_name="web_search"):
    args = {"query": query, "research_intent": label}
    decision = controller.before_call(tool_name, args)
    assert decision.allows_execution
    controller.after_call(tool_name, args, '{"data": "evidence"}', failed=False)


def test_research_budget_types_and_intent_normalization_are_strict():
    assert normalize_research_budget({"repeated_intent_max": 2}) == {"repeated_intent_max": 2}
    with pytest.raises(ValueError, match="positive integer"):
        normalize_research_budget({"repeated_intent_max": 0})
    with pytest.raises(ValueError, match="positive integer"):
        normalize_research_budget({"repeated_intent_max": True})

    assert normalize_research_intent("  Ｅｖｅｎｉｎｇ   WEATHER  ") == "evening weather"
    assert normalize_research_intent(42) is None
    assert normalize_research_intent("\n\t") is None


def test_reworded_queries_share_one_normalized_intent_cap():
    controller = _controller(repeated_intent_max=2)

    _successful_call(controller, query="Kobe evening weather", label="  Evening WEATHER ")
    _successful_call(controller, query="weather forecast near Sannomiya", label="evening   weather")

    blocked = controller.before_call(
        "web_search",
        {"query": "Kobe forecast today", "research_intent": "EVENING WEATHER"},
    )
    assert blocked.action == "block"
    assert blocked.code == RESEARCH_BUDGET_EXHAUSTED
    assert blocked.state == RESEARCH_SYNTHESIS_STATE
    assert controller.research_budget_metadata["intent_counts"] == {"evening weather": 2}


def test_different_intent_labels_have_independent_attempt_counts():
    controller = _controller(repeated_intent_max=1)

    _successful_call(controller, query="Kobe weather", label="weather")
    _successful_call(controller, query="Kobe station hours", label="hours")

    blocked = controller.before_call(
        "web_search",
        {"query": "another weather query", "research_intent": "weather"},
    )
    assert blocked.code == RESEARCH_BUDGET_EXHAUSTED
    assert controller.research_budget_metadata["intent_counts"] == {"weather": 1, "hours": 1}


def test_missing_or_malformed_intent_fails_closed_without_consuming_collection():
    controller = _controller(repeated_intent_max=2)

    missing = controller.before_call("web_search", {"query": "no label"})
    assert missing.action == "block"
    assert missing.code == RESEARCH_INTENT_REQUIRED
    assert missing.state == "COLLECT"
    assert controller.research_budget_metadata["web_search_count"] == 0
    assert controller.research_budget_metadata["intent_counts"] == {}

    malformed = controller.before_call(
        "web_search",
        {"query": "bad label", "research_intent": 123},
    )
    assert malformed.action == "block"
    assert malformed.code == RESEARCH_INTENT_INVALID
    assert controller.research_budget_metadata["web_search_count"] == 0
    assert controller.research_budget_metadata["intent_counts"] == {}


def test_unconfigured_policy_keeps_collection_behavior_and_finalization_tools_available():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping({}))

    assert controller.before_call("web_search", {"query": "legacy"}).allows_execution
    assert controller.research_budget_metadata is None
    for tool_name in ("terminal", "read_file", "write_file", "kanban_complete"):
        assert controller.before_call(tool_name, {}).allows_execution


def test_intent_cap_transition_preserves_synthesis_tools():
    controller = _controller(repeated_intent_max=1)
    _successful_call(controller, query="first", label="facts")

    blocked = controller.before_call(
        "web_extract",
        {"urls": ["https://example.test/second"], "research_intent": "facts"},
    )
    assert blocked.code == RESEARCH_BUDGET_EXHAUSTED
    assert blocked.state == RESEARCH_SYNTHESIS_STATE
    for tool_name in ("terminal", "read_file", "write_file", "kanban_complete"):
        assert controller.before_call(tool_name, {}).allows_execution
