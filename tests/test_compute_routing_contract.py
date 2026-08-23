"""Fixed RED oracle for compute-class model routing.

Every normal and known-bad fixture enters the same pure public route_task
interface. Missing production API is a failure, never a skip.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "compute_routing_contract_v1.json"
BOUNDARY_FIELDS = (
    "role",
    "assignee",
    "write_scope",
    "workspace",
    "approval",
    "toolsets",
    "master_forbidden",
)
REQUIRED_ROUTE_METADATA = (
    "compute_class",
    "route_decision_id",
    "policy_version",
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _deep_merge(base: dict, patch: Mapping) -> dict:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _materialize(case: Mapping, *, variant: Mapping | None = None) -> dict:
    materialized = _deep_merge(_fixture()["common_input"], case.get("input_patch", {}))
    if variant is not None:
        materialized = _deep_merge(materialized, variant.get("input_patch", {}))
    return materialized


def _route_task(case_input: dict):
    contract = _fixture()["public_interface"]
    try:
        module = importlib.import_module(contract["module"])
    except ModuleNotFoundError:
        pytest.fail(
            f"required public routing module {contract['module']!r} is missing",
            pytrace=False,
        )
    route_task = getattr(module, contract["callable"], None)
    if not callable(route_task):
        pytest.fail(
            f"{contract['module']}.{contract['callable']} must be callable",
            pytrace=False,
        )
    return route_task(**{name: case_input[name] for name in contract["arguments"]})


def _assert_subset(actual: Mapping, expected: Mapping, path: str = "result") -> None:
    assert isinstance(actual, Mapping), (
        f"{path} must be a mapping, got {type(actual).__name__}"
    )
    for key, expected_value in expected.items():
        assert key in actual, f"{path}.{key} is missing"
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            _assert_subset(actual_value, expected_value, f"{path}.{key}")
        else:
            assert actual_value == expected_value, (
                f"{path}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )


def _assert_boundary_unchanged(case_input: Mapping, result: Mapping) -> None:
    contract = result.get("task_contract")
    assert isinstance(contract, Mapping), (
        "result.task_contract must expose role-owned fields"
    )
    original = case_input["role_decision"]
    for field in BOUNDARY_FIELDS:
        assert contract.get(field) == original[field], (
            f"compute routing changed role boundary {field}"
        )


def test_fixture_is_closed_complete_and_structurally_valid():
    data = _fixture()
    assert data["schema_version"] == "compute_routing_contract/v1"
    assert data["common_input"]["policy"]["precedence"] == [
        "specialist",
        "architect",
        "deep",
        "standard",
        "quick",
    ]
    assert [
        case["expected"]["compute_class"] for case in data["normal_cases"]
    ] == ["specialist", "architect", "deep", "standard", "quick"]
    assert [case["invariant"] for case in data["mutants"]] == [
        f"I{i}" for i in range(1, 13)
    ]
    assert len({case["id"] for case in data["mutants"]}) == 12

    required = set(data["public_interface"]["arguments"])
    materialized_cases = [_materialize(case) for case in data["normal_cases"]]
    materialized_cases.extend(
        _materialize(case, variant=variant)
        for case in data["mutants"]
        for variant in (case.get("variants") or [None])
    )
    for case_input in materialized_cases:
        assert set(case_input) == required
        assert set(case_input["role_decision"]) >= set(BOUNDARY_FIELDS)
        assert (
            case_input["policy"]["schema_version"]
            == "compute_routing_policy/v1"
        )


@pytest.mark.parametrize(
    "case",
    _fixture()["normal_cases"],
    ids=lambda case: case["id"],
)
def test_normal_routes_use_precedence_and_preserve_role_boundary(case):
    case_input = _materialize(case)
    result = _route_task(case_input)

    _assert_subset(result, case["expected"])
    _assert_boundary_unchanged(case_input, result)
    persisted = result.get("persisted_route")
    assert isinstance(persisted, Mapping), (
        "result.persisted_route must be explicit before dispatch"
    )
    assert persisted.get("compute_class") == case["expected"]["compute_class"]
    assert (
        persisted.get("policy_version")
        == case_input["policy"]["policy_version"]
    )
    assert persisted.get("route_decision_id")
    for key in REQUIRED_ROUTE_METADATA:
        assert key in persisted


@pytest.mark.parametrize(
    "case",
    [case for case in _fixture()["mutants"] if "variants" not in case],
    ids=lambda case: case["id"],
)
def test_known_bad_mutant_is_dropped_through_route_task(case):
    case_input = _materialize(case)
    result = _route_task(case_input)

    _assert_subset(result, case["expected"])
    _assert_boundary_unchanged(case_input, result)
    assert result.get("outcome") != "verified" or result.get("actual_route"), (
        "verified requires actual provider/model/reasoning read-back"
    )


@pytest.mark.parametrize(
    "variant",
    _fixture()["mutants"][-1]["variants"],
    ids=lambda variant: f"I12-{variant['name']}",
)
def test_quota_and_work_failures_remain_distinct(variant):
    case = _fixture()["mutants"][-1]
    case_input = _materialize(case, variant=variant)
    result = _route_task(case_input)

    _assert_subset(result, variant["expected"])
    _assert_boundary_unchanged(case_input, result)


def test_i12_expected_contracts_are_not_aliases():
    variants = _fixture()["mutants"][-1]["variants"]
    quota, work = (variant["expected"] for variant in variants)
    assert quota["outcome"] == "routing_unavailable"
    assert work["outcome"] == "work_failed"
    assert quota["event_kind"] != work["event_kind"]
    assert quota["spawn"] is True
    assert work["spawn"] is False
