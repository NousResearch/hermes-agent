from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.dispatch_contracts import (
    BudgetPolicy,
    DispatchReceipt,
    TaskClassification,
    WorkerAttemptMetadata,
    dispatch_contract_json_schemas,
    validate_dispatch_receipt,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dispatch_contracts"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize(
    "fixture_name",
    [
        "local_file_task.json",
        "github_ops_task.json",
        "research_task.json",
        "repo_code_task.json",
    ],
)
def test_valid_dispatch_receipt_fixtures_parse(fixture_name: str):
    receipt = validate_dispatch_receipt(_load_fixture(fixture_name))

    assert isinstance(receipt, DispatchReceipt)
    assert receipt.schema_version == "dispatch-contracts/v1"
    assert receipt.worker_attempt.work_order_id == receipt.work_order_id
    assert receipt.worker_attempt.attempt_id == receipt.attempt_id
    assert receipt.evidence.items


def test_contracts_export_machine_readable_json_schemas():
    schemas = dispatch_contract_json_schemas()

    assert set(schemas) == {
        "TaskClassification",
        "BudgetPolicy",
        "WorkerAttemptMetadata",
        "EvidencePacket",
        "DispatchReceipt",
    }
    assert schemas["DispatchReceipt"]["type"] == "object"
    assert "required" in schemas["WorkerAttemptMetadata"]


def test_task_classification_rejects_unknown_policy_values():
    with pytest.raises(ValidationError):
        TaskClassification(
            task_complexity="huge",
            stakes="medium",
            reversibility="reversible",
            tool_reach="github",
            memory_dependency="local_context",
            verification_need="deterministic",
        )


def test_budget_policy_enforces_positive_bounds_and_approval_threshold():
    with pytest.raises(ValidationError):
        BudgetPolicy(
            max_worker_sessions_per_issue=0,
            max_parallel_workers_default=2,
            max_worker_turns=4,
            max_worker_runtime_minutes=30,
            max_escalations_to_sota=1,
            stop_after_failed_attempts=2,
            require_ceo_approval_above_stakes="irreversible",
        )

    with pytest.raises(ValidationError):
        BudgetPolicy(
            max_worker_sessions_per_issue=3,
            max_parallel_workers_default=4,
            max_worker_turns=4,
            max_worker_runtime_minutes=30,
            max_escalations_to_sota=1,
            stop_after_failed_attempts=2,
            require_ceo_approval_above_stakes="high",
        )


def test_worker_attempt_rejects_completion_claim_without_evidence_path():
    payload = _load_fixture("local_file_task.json")["worker_attempt"]
    payload["status"] = "succeeded"
    payload["evidence_packet_path"] = None

    with pytest.raises(ValidationError, match="evidence_packet_path"):
        WorkerAttemptMetadata(**payload)


def test_worker_attempt_rejects_external_account_without_approval():
    payload = _load_fixture("github_ops_task.json")["worker_attempt"]
    payload["allowed_tools"] = ["browser", "gmail_send"]
    payload["assigned_scope"] = "send an email to a third party"
    payload["external_approval_id"] = None

    with pytest.raises(ValidationError, match="external_approval_id"):
        WorkerAttemptMetadata(**payload)


def test_receipt_rejects_mismatched_attempt_ids():
    payload = _load_fixture("repo_code_task.json")
    payload["worker_attempt"]["attempt_id"] = "different-attempt"

    with pytest.raises(ValidationError, match="same work_order_id and attempt_id"):
        validate_dispatch_receipt(payload)


def test_receipt_rejects_success_without_verification_items():
    payload = _load_fixture("research_task.json")
    payload["evidence"]["items"] = []

    with pytest.raises(ValidationError, match="at least one evidence item"):
        validate_dispatch_receipt(payload)


def test_receipt_rejects_claimed_touched_files_not_listed_in_evidence():
    payload = _load_fixture("local_file_task.json")
    payload["worker_attempt"]["touched_files"] = ["/tmp/output.md", "/tmp/unverified.md"]

    with pytest.raises(ValidationError, match="touched file"):
        validate_dispatch_receipt(payload)
