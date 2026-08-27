from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("validate_role_contracts", ROOT / "scripts/validate_role_contracts.py")
assert SPEC and SPEC.loader
validator_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator_module)


@pytest.fixture
def validator():
    return validator_module



def valid_instances() -> dict:
    return {
        "evidence": [{"evidence_id": "ev-1", "evidence_state": "verified"}],
        "claim_ledgers": [
            {
                "claim_id": "claim-1",
                "claim_text": "The validator checks concrete artifacts.",
                "evidence_ids": ["ev-1"],
                "evidence_state": "verified",
                "blocking_gaps": [],
                "provenance": {"source": "test"},
            }
        ],
        "publication_decisions": [
            {"decision_id": "decision-1", "readiness": "public_ready", "claim_ids": ["claim-1"], "blocking_evidence_gaps": []}
        ],
        "explorer_dissents": [
            {
                "dissent_id": "dissent-1",
                "alternatives": ["alternative"],
                "evidence": ["ev-1"],
                "falsifier": "A failing test",
                "kill_criterion": "No reproducible artifact",
                "reconciliation_disposition": "retained as a bounded alternative",
                "preserved_artifact_ref": "artifact://dissent-1",
            }
        ],
        "fiv_chains": [
            {
                "chain_id": "fiv-1",
                "finding": {"id": "find-1", "kind": "finding", "evidence_state": "verified"},
                "implementation": {"id": "impl-1", "kind": "implementation", "evidence_state": "derived"},
                "verification": {"id": "verify-1", "kind": "verification", "evidence_state": "verified"},
            }
        ],
    }


def assert_rejected(validator, instances: dict, message: str) -> None:
    errors = validator.validate_artifact_instances({}, instances)
    assert any(message in error for error in errors), errors


def test_valid_concrete_claim_decision_dissent_and_fiv_are_accepted(validator) -> None:
    assert validator.validate_artifact_instances({}, valid_instances()) == []


def test_invalid_readiness_and_public_ready_blockers_are_rejected(validator) -> None:
    broken = valid_instances()
    broken["publication_decisions"][0]["readiness"] = "ready-ish"
    assert_rejected(validator, broken, "invalid readiness")

    broken = valid_instances()
    broken["publication_decisions"][0]["readiness"] = "public_ready"
    broken["claim_ledgers"][0]["blocking_gaps"] = ["gap-1"]
    assert_rejected(validator, broken, "unresolved or blocked evidence")


def test_claim_ledger_requires_concrete_evidence_reference(validator) -> None:
    broken = valid_instances()
    broken["claim_ledgers"][0]["evidence_ids"] = ["ev-missing"]
    assert_rejected(validator, broken, "unresolved evidence")


def test_empty_or_summary_only_dissent_requires_preserved_reference(validator) -> None:
    for mutation in ("empty", "summary_only"):
        broken = valid_instances()
        if mutation == "empty":
            broken["explorer_dissents"][0]["alternatives"] = []
        else:
            broken["explorer_dissents"][0]["preservation"] = "summary_only"
            broken["explorer_dissents"][0].pop("preserved_artifact_ref")
        assert_rejected(validator, broken, "preserved artifact reference" if mutation == "summary_only" else "empty alternatives")


def test_f_only_and_f_to_i_are_rejected(validator) -> None:
    for remove in ("implementation", "verification"):
        broken = valid_instances()
        broken["fiv_chains"][0].pop(remove)
        assert_rejected(validator, broken, "must contain finding, implementation, and verification")


def test_invalid_verification_states_are_rejected(validator) -> None:
    for state in ("blocked", "unknown", "superseded"):
        broken = copy.deepcopy(valid_instances())
        broken["fiv_chains"][0]["verification"]["evidence_state"] = state
        assert_rejected(validator, broken, "verification is not verifiable")
