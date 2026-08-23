"""Safety regressions for compute-class fallback and route persistence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hermes_cli.compute_routing import route_task


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "compute_routing_contract_v1.json"
ROUTE_IDENTITY_FIELDS = (
    "compute_class",
    "policy_version",
    "provider",
    "model",
    "reasoning_effort",
)


def _common_input() -> dict:
    return copy.deepcopy(
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["common_input"]
    )


def _route(case_input: dict) -> dict:
    return route_task(
        task_context=case_input["task_context"],
        role_decision=case_input["role_decision"],
        capability_snapshot=case_input["capability_snapshot"],
        role_constraints=case_input["role_constraints"],
        policy=case_input["policy"],
        execution=case_input["execution"],
        unattended=case_input["unattended"],
    )


def test_availability_error_without_distinct_candidate_exhausts_fallback():
    case_input = _common_input()
    case_input["capability_snapshot"]["candidates"] = [
        {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "reasoning_efforts": ["medium"],
        }
    ]
    case_input["execution"].update(
        attempted=True,
        error="model_unavailable",
        fallback_index=0,
    )

    result = _route(case_input)

    assert result["status"] == "blocked"
    assert result["spawn"] is False
    assert result["outcome"] == "routing_unavailable"
    assert result["event_kind"] == "fallback_exhausted"
    assert result["fallback_index"] == 0


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("compute_class", "deep"),
        ("policy_version", "stale-policy"),
        ("provider", "other-provider"),
        ("model", "other-model"),
        ("reasoning_effort", "high"),
    ],
)
def test_attempted_execution_rejects_persisted_route_mismatch(field, mismatch):
    case_input = _common_input()
    initial = _route(case_input)
    persisted_route = copy.deepcopy(initial["persisted_route"])
    assert set(ROUTE_IDENTITY_FIELDS) <= set(persisted_route)

    persisted_route[field] = mismatch
    case_input["task_context"]["persisted_route"] = persisted_route
    case_input["execution"]["attempted"] = True

    result = _route(case_input)

    assert result["status"] == "rejected"
    assert result["spawn"] is False
    assert result["reason_code"] == "route_persistence_mismatch"
