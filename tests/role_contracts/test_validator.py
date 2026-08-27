from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "docs/role-contracts/role-contracts.json"
SPEC = importlib.util.spec_from_file_location(
    "validate_role_contracts", ROOT / "scripts/validate_role_contracts.py"
)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


@pytest.fixture
def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def assert_rejected(registry: dict, message: str) -> None:
    errors = validator.validate_registry(registry)
    assert any(message in error for error in errors), errors


def test_canonical_registry_conforms(registry: dict) -> None:
    assert validator.validate_registry(registry) == []


def test_missing_role_contract_is_rejected(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    broken["roles"] = [role for role in broken["roles"] if role["role_id"] != "role.statistician"]
    assert_rejected(broken, "missing required role contract: role.statistician")


def test_duplicate_role_ids_are_rejected(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    broken["roles"].append(copy.deepcopy(broken["roles"][0]))
    assert_rejected(broken, "duplicate role_id: role.ares_supervisor")


def test_required_artifact_fields_are_rejected(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    explorer = next(role for role in broken["roles"] if role["role_id"] == "role.explorer")
    explorer["required_artifacts"][0]["required_fields"].remove("falsifier")
    assert_rejected(broken, "artifact explorer_dissent missing field falsifier")


def test_explorer_dissent_cannot_be_summarized_away(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    explorer = next(role for role in broken["roles"] if role["role_id"] == "role.explorer")
    explorer["dissent_policy"]["preservation"] = "summary_only"
    assert_rejected(broken, "Explorer dissent must preserve the dissent artifact")


def test_public_ready_with_blockers_is_rejected(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    decision = broken["publication_decision_examples"][0]
    decision["readiness"] = "public_ready"
    decision["blocking_evidence_gaps"] = ["gap-1"]
    assert_rejected(broken, "public_ready decision has blocking evidence gaps")


def test_lane_authority_widening_is_rejected(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    lane = next(role for role in broken["roles"] if role["role_id"] == "role.data_evidence")
    lane["authority"]["can_promote"].append("publication_readiness")
    assert_rejected(broken, "forbidden promotion in role.data_evidence: publication_readiness")


@pytest.mark.parametrize(
    "promotion",
    [
        "publication_readiness",
        "runtime_enforcement",
        "unverified_claim",
        "final_role_authority",
        "source_truth",
    ],
)
def test_data_evidence_required_forbidden_promotions_are_required(registry: dict, promotion: str) -> None:
    broken = copy.deepcopy(registry)
    lane = next(role for role in broken["roles"] if role["role_id"] == "role.data_evidence")
    lane["forbidden_promotions"].remove(promotion)
    lane["authority"]["forbidden_promotions"].remove(promotion)
    assert_rejected(broken, f"role.data_evidence must forbid promotion: {promotion}")


def test_broken_finding_implementation_verification_link_is_rejected(registry: dict) -> None:
    broken = copy.deepcopy(registry)
    broken["fiv_examples"][0]["implementation_id"] = "impl.missing"
    assert_rejected(broken, "FIV link implementation_id -> impl.missing is unresolved")


def test_cli_accepts_canonical_registry() -> None:
    result = validator.main([str(REGISTRY_PATH)])
    assert result == 0
