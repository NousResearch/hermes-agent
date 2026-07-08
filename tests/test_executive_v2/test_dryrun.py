"""Tests for the dry-run renderer."""

from __future__ import annotations

import pytest

from agent.executive.dryrun import render_dry_run
from agent.executive.types import ObjectiveState, ObjectiveStateData


def _make_state(oid="oid-test", text="hello", state=ObjectiveState.DRAFT) -> ObjectiveStateData:
    return ObjectiveStateData(
        objective_id=oid,
        state=state,
        objective_text=text,
        constraints=[],
        user_id="u",
        created_at="2026-01-01",
    )


def test_render_includes_objective_id():
    s = _make_state(oid="oid-x")
    out = render_dry_run(s)
    assert "oid-x" in out


def test_render_includes_state_name():
    s = _make_state(state=ObjectiveState.NORMALIZED)
    out = render_dry_run(s)
    assert "NORMALIZED" in out


def test_render_includes_goal_class_when_normalized_present():
    s = _make_state()
    s.normalized = {"goal_class": "STRATEGIC", "risk_profile": "high", "estimated_complexity": "L", "success_criteria": ["x is done"]}
    out = render_dry_run(s)
    assert "STRATEGIC" in out
    assert "high" in out
    assert "L" in out


def test_render_includes_risk_components_when_contract_present():
    s = _make_state()
    s.contract = {
        "risk_components": {
            "financial": 1.0, "regulatory": 0.0, "customer_facing": 1.0,
            "irreversibility": 0.0, "data_sensitivity": 0.0,
        },
        "risk_score": 0.5,
        "approval_requirements": [{"gate": "HIGH_RISK_DRAFT", "approver": "user", "ttl_hours": 24}],
        "budget": {"policy": "standard", "max_iterations": 10, "max_duration_minutes": 30, "max_cost_usd": 5.0},
    }
    out = render_dry_run(s)
    assert "financial=1.0" in out
    assert "HIGH_RISK_DRAFT" in out
    assert "policy=standard" in out


def test_render_handles_no_normalized_no_discovered_no_contract():
    s = _make_state()
    out = render_dry_run(s)
    # Should still produce output with objective_id and state.
    assert "oid-test" in out
    assert "DRAFT" in out


def test_render_handles_empty_capability_discovery():
    s = _make_state()
    s.discovered = {
        "candidates": [],
        "reuse_decision": "generate",
        "gaps": ("no_capability",),
    }
    out = render_dry_run(s)
    assert "generate" in out
    assert "no_capability" in out
