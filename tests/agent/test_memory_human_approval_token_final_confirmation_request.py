from copy import deepcopy

from agent.memory_human_approval_token_final_confirmation_request import (
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_ROUTING,
    create_human_approval_token_final_confirmation_request,
    explain_human_approval_token_final_confirmation_request,
    recommend_human_approval_token_final_confirmation_request_action,
    summarize_human_approval_token_final_confirmation_requests,
    validate_human_approval_token_final_confirmation_request,
)
from agent.memory_human_approval_token_write_lock_gate import create_human_approval_token_write_lock_gate
from tests.agent.test_memory_human_approval_token_write_lock_gate import _dry_run


def _gate(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    return create_human_approval_token_write_lock_gate(
        _dry_run(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        operator="token-write-lock-operator",
    )


def test_valid_token_write_lock_gate_creates_final_confirmation_review_required_request():
    gate = _gate()

    request = create_human_approval_token_final_confirmation_request(gate, requester="final-confirmer")

    assert request["request_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_KIND
    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_REQUIRED
    assert request["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_ROUTING
    assert request["lock_reason"] is None
    assert request["source_token_write_lock_gate_id"] == gate["gate_id"]
    assert request["source_token_issuance_dry_run_id"] == gate["source_token_issuance_dry_run_id"]
    assert request["source_token_issuance_plan_id"] == gate["source_token_issuance_plan_id"]
    assert request["source_review_outcome_id"] == gate["source_review_outcome_id"]
    assert request["source_request_id"] == gate["source_request_id"]
    assert request["source_gate_id"] == gate["source_gate_id"]
    assert request["source_dry_run_id"] == gate["source_dry_run_id"]
    assert request["source_plan_id"] == gate["source_plan_id"]
    assert request["source_outcome_id"] == gate["source_outcome_id"]
    assert request["source_packet_id"] == gate["source_packet_id"]
    assert request["source_submission_id"] == gate["source_submission_id"]
    assert request["source_draft_id"] == gate["source_draft_id"]
    assert request["source_decision_id"] == gate["source_decision_id"]
    assert request["source_queue_item_id"] == gate["source_queue_item_id"]
    assert request["requester"] == "final-confirmer"
    assert request["token_write_lock_gate_validation"] == {"valid": True, "errors": []}
    assert request["request_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_final_confirmation_request(request) == {"valid": True, "errors": []}
    assert request["next_step_recommendation"]["action"] == "route_to_manual_final_confirmation_review_before_any_token_write"
    assert request["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert request["next_step_recommendation"]["persists_approvals"] is False
    assert request["next_step_recommendation"]["creates_real_proposals"] is False
    assert request["next_step_recommendation"]["writes_operation_ledger"] is False
    assert request["next_step_recommendation"]["writes_token_files"] is False
    assert request["next_step_recommendation"]["writes_approval_audit"] is False


def test_locked_token_write_lock_gate_creates_locked_request():
    gate = _gate(outcome="reject")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "token_review_rejected"
    assert request["token_write_lock_gate_validation"] == {"valid": True, "errors": []}
    assert request["request_validation"] == {"valid": True, "errors": []}


def test_locked_token_write_lock_gate_without_lock_reason_uses_default_lock_reason():
    gate = _gate(outcome="reject")
    gate["lock_reason"] = None

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "token_write_lock_gate_locked"


def test_invalid_token_write_lock_gate_creates_locked_request():
    gate = _gate()
    gate["gate_kind"] = "wrong"

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "invalid_token_write_lock_gate"
    assert request["token_write_lock_gate_validation"]["valid"] is False
    assert request["request_validation"] == {"valid": True, "errors": []}


def test_missing_approval_token_record_preview_locks_request():
    gate = _gate()
    gate.pop("approval_token_record_preview")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_approval_token_record_preview"


def test_missing_approval_audit_record_preview_locks_request():
    gate = _gate()
    gate.pop("approval_audit_record_preview")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_approval_audit_record_preview"


def test_missing_token_target_paths_preview_locks_request():
    gate = _gate()
    gate.pop("token_target_paths_preview")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_token_target_paths_preview"


def test_missing_proposal_record_preview_locks_request():
    gate = _gate()
    gate.pop("proposal_record_preview")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_proposal_record_preview"


def test_missing_operation_ledger_preview_locks_request():
    gate = _gate()
    gate.pop("operation_ledger_preview")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_operation_ledger_preview"


def test_missing_target_paths_preview_locks_request():
    gate = _gate()
    gate.pop("target_paths_preview")

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_target_paths_preview"


def test_missing_source_evidence_locks_request():
    gate = _gate(source_pattern_ids=[], source_fact_ids=[])

    request = create_human_approval_token_final_confirmation_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_source_evidence"


def test_preview_integrity_failed_when_preview_only_false_or_written_created_token_approved_persisted_flags_true():
    fields = (
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
    )
    cases = []
    for field in fields:
        preview_only_false = _gate()
        preview_only_false[field]["preview_only"] = False
        cases.append(preview_only_false)

        for flag in ("written", "created_real_proposal", "created_operation_event", "token_issued", "approved", "persisted"):
            flagged = _gate()
            flagged[field][flag] = True
            cases.append(flagged)

    for gate in cases:
        request = create_human_approval_token_final_confirmation_request(gate)
        assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_LOCKED
        assert request["lock_reason"] == "preview_integrity_failed"


def test_final_confirmation_checklist_is_deterministic():
    request_a = create_human_approval_token_final_confirmation_request(_gate())
    request_b = create_human_approval_token_final_confirmation_request(_gate())

    assert request_a["final_confirmation_checklist"] == request_b["final_confirmation_checklist"]
    assert [check["id"] for check in request_a["final_confirmation_checklist"]] == [
        "verify_token_write_lock_gate_validation",
        "verify_token_previews_only",
        "verify_no_token_approval_or_write_flags",
        "verify_payload_and_source_evidence",
        "require_manual_final_confirmation_review",
    ]


def test_input_token_write_lock_gate_is_not_mutated():
    gate = _gate()
    before = deepcopy(gate)

    request = create_human_approval_token_final_confirmation_request(gate)
    request["source_token_write_lock_gate_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert gate == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_or_approval_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    request = create_human_approval_token_final_confirmation_request(_gate())
    explanation = explain_human_approval_token_final_confirmation_request(request)
    recommendation = recommend_human_approval_token_final_confirmation_request_action(request)

    assert request["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY
    assert request["policy"]["read_only"] is True
    assert request["policy"]["would_write_memory"] is False
    assert request["policy"]["would_modify_config"] is False
    assert request["policy"]["would_write_graph"] is False
    assert request["policy"]["does_not_create_operation_events"] is True
    assert request["policy"]["creates_final_confirmation_requests_only"] is True
    assert request["policy"]["issues_real_approval_tokens"] is False
    assert request["policy"]["persists_approvals"] is False
    assert request["policy"]["creates_real_proposals"] is False
    assert request["policy"]["writes_proposal_files"] is False
    assert request["policy"]["writes_operation_ledger"] is False
    assert request["policy"]["writes_token_files"] is False
    assert request["policy"]["writes_approval_audit"] is False
    assert request["policy"]["applies_proposals"] is False
    assert request["policy"]["submits_to_governance"] is False
    assert request["policy"]["converts_to_real_proposal"] is False
    for key in (
        "token_issued",
        "approved",
        "persisted",
        "submitted",
        "written",
        "created_real_proposal",
        "created_operation_event",
        "writes_proposal_files",
        "writes_operation_ledger",
        "writes_token_files",
        "writes_approval_audit",
        "converted_to_real_proposal",
    ):
        assert explanation[key] is False
    assert recommendation["issues_real_approval_tokens"] is False
    assert recommendation["persists_approvals"] is False
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["writes_proposal_files"] is False
    assert recommendation["writes_operation_ledger"] is False
    assert recommendation["writes_token_files"] is False
    assert recommendation["writes_approval_audit"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()
    assert not (hermes_home / "memory" / "approvals" / "human_approval_tokens.jsonl").exists()
    assert not (hermes_home / "memory" / "audit" / "human_approval_token_audit.jsonl").exists()


def test_request_must_never_be_marked_token_issued_approved_persisted_submitted_written_created_or_converted():
    request = create_human_approval_token_final_confirmation_request(_gate())

    for forbidden_key in (
        "token_issued",
        "approved",
        "persisted",
        "submitted",
        "written",
        "created_real_proposal",
        "created_operation_event",
        "writes_proposal_files",
        "writes_operation_ledger",
        "writes_token_files",
        "writes_approval_audit",
        "converted_to_real_proposal",
    ):
        mutated = deepcopy(request)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_final_confirmation_request(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_final_confirmation_required_requests_and_by_block_type_status():
    requests = [
        create_human_approval_token_final_confirmation_request(_gate("procedural_rules")),
        create_human_approval_token_final_confirmation_request(_gate("project_context")),
        create_human_approval_token_final_confirmation_request(_gate("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_final_confirmation_requests(requests)

    assert summary["total"] == 3
    assert summary["locked_count"] == 1
    assert summary["final_confirmation_review_required_count"] == 2
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {
        "final_confirmation_review_required": 2,
        "locked": 1,
    }
    assert summary["by_lock_reason"] == {"None": 2, "token_review_rejected": 1}
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REQUEST_POLICY
