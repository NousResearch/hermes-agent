"""Invariant tests for the deterministic bounded-research runtime fixture."""

from __future__ import annotations

import json

import pytest

from agent.tool_guardrails import (
    RESEARCH_BUDGET_EXHAUSTED,
    RESEARCH_SYNTHESIS_ONLY,
    RESEARCH_SYNTHESIS_STATE,
    RESEARCH_TERMINAL_STATE,
)
from evals.research_runtime.fixture import TASK_POLICY, render_markdown, run_fixture


@pytest.fixture(scope="module")
def report() -> dict:
    """Run the real offline integration fixture once for this test module."""
    return run_fixture()


def test_fixture_is_offline_and_exercises_the_typed_task_policy(report):
    assert report["fixture"]["external_network_calls"] == 0
    assert report["fixture"]["source_mode"] == "synthetic_fixture_only"
    policy_path = report["task_policy_path"]
    assert policy_path["native_create_readback_matches"] is True
    assert policy_path["worker_envelope_readback_matches"] is True
    assert policy_path["active_guardrail_policy"]["repeated_intent_max"] == TASK_POLICY[
        "repeated_intent_max"
    ]
    assert policy_path["active_guardrail_policy"]["web_search_max"] == TASK_POLICY[
        "web_search_max"
    ]


def test_before_after_metrics_show_bounded_completion_without_fabricated_quality(report):
    before = report["scenarios"]["before"]
    after = report["scenarios"]["after"]

    assert before["terminal_outcome"] == "timeout_before_synthesis"
    assert before["terminal_tool_calls"] == 0
    assert after["terminal_outcome"] == "completed"
    assert after["terminal_tool_calls"] == 1
    assert after["logical_runtime_seconds"] < before["logical_runtime_seconds"]
    assert after["web_search_calls"] < before["web_search_calls"]
    assert after["browser_extract_calls"] < before["browser_extract_calls"]
    assert after["retry_calls"] < before["retry_calls"]
    assert set(report["quality_statuses"]) <= {
        "VERIFIED",
        "UNCERTAIN",
        "NOT_CHECKED",
        "NOT_FOUND_WITHIN_BUDGET",
    }
    assert "NOT_CHECKED" in report["quality_statuses"]
    assert "UNCERTAIN" in report["quality_statuses"]


def test_repeated_intent_rewrites_force_the_synthesis_transition(report):
    after = report["scenarios"]["after"]
    weather_events = [
        event for event in after["events"] if event["tool"] == "web_search" and event["intent"] == "weather"
    ]

    assert [event["status"] for event in weather_events] == ["executed", "executed", "blocked"]
    assert all(event["retry"] for event in weather_events[1:])
    assert weather_events[-1]["decision_code"] == RESEARCH_BUDGET_EXHAUSTED
    assert weather_events[-1]["state"] == RESEARCH_SYNTHESIS_STATE
    assert after["web_search_calls"] <= TASK_POLICY["repeated_intent_max"]
    assert after["lifecycle_states"] == ["COLLECT", RESEARCH_SYNTHESIS_STATE, RESEARCH_TERMINAL_STATE]
    assert after["guardrail_state"] == RESEARCH_TERMINAL_STATE
    assert after["collection_disabled_after_transition"] is True
    assert set(after["finalization_tools_available"]) == {
        "terminal",
        "read_file",
        "write_file",
        "kanban_complete",
    }


def test_search_and_extract_caps_are_independently_enforced(report):
    probes = report["budget_probes"]
    search = probes["web_search"]
    extract = probes["browser_extract"]

    assert search["web_search_calls"] <= TASK_POLICY["web_search_max"]
    assert extract["browser_extract_calls"] <= TASK_POLICY["browser_extract_max"]
    assert search["collection_calls_attempted"] > search["collection_calls_executed"]
    assert extract["collection_calls_attempted"] > extract["collection_calls_executed"]
    assert search["guardrail_code"] == RESEARCH_BUDGET_EXHAUSTED
    assert extract["guardrail_code"] == RESEARCH_BUDGET_EXHAUSTED
    assert search["lifecycle_states"] == extract["lifecycle_states"] == [
        "COLLECT",
        RESEARCH_SYNTHESIS_STATE,
        RESEARCH_TERMINAL_STATE,
    ]


def test_evidence_checkpoint_recovery_is_synthesis_only_and_not_a_full_replay(report):
    recovery = report["scenarios"]["recovery"]
    dispatch = report["recovery_dispatch"]

    assert dispatch["failure_class"] == "timeout_before_synthesis"
    assert dispatch["checkpoint_evidence_present"] is True
    assert dispatch["checkpoint"]["evidence_count"] > 0
    assert dispatch["worker_recovery_mode"] == "synthesis_only"
    assert dispatch["worker_env_recovery_mode"] == "synthesis_only"
    assert dispatch["task_policy_still_present"] is True
    assert recovery["recovery_mode"] == "synthesis_only"
    assert recovery["guardrail_code"] == RESEARCH_SYNTHESIS_ONLY
    assert recovery["collection_calls_executed"] == 0
    assert recovery["blocked_collection_calls"] == 1
    assert recovery["preserved_evidence"] is True
    assert recovery["terminal_outcome"] == "completed"
    assert recovery["finalization_tools_available"]
    assert recovery["collection_calls_attempted"] < report["scenarios"]["before"]["collection_calls_attempted"]


def test_report_render_is_json_safe_and_contains_auditable_sections(report):
    assert json.loads(json.dumps(report)) == report
    markdown = render_markdown(report)
    assert "Before / after / recovery metrics" in markdown
    assert "Independent collection caps" in markdown
    assert "SYNTHESIZE_REQUIRED" in markdown
    assert "synthesis_only" in markdown
    assert "external network calls: 0" in markdown
