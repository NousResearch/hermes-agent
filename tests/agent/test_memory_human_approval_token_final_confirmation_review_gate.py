from copy import deepcopy

from agent.memory_human_approval_token_final_confirmation_review_gate import (
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_STATUS,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING,
    create_human_approval_token_final_confirmation_review_outcome,
    evaluate_human_approval_token_final_confirmation_request,
    explain_human_approval_token_final_confirmation_review_outcome,
    recommend_human_approval_token_final_confirmation_review_action,
    summarize_human_approval_token_final_confirmation_review_outcomes,
    validate_human_approval_token_final_confirmation_review_outcome,
)
from agent.memory_human_approval_token_final_confirmation_request import (
    create_human_approval_token_final_confirmation_request,
)
from tests.agent.test_memory_human_approval_token_final_confirmation_request import _gate


def _request(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    return create_human_approval_token_final_confirmation_request(
        _gate(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        requester="final-confirmation-requester",
    )


def test_valid_final_confirmation_review_required_request_becomes_confirm_token_write_outcome_candidate():
    request = _request()

    outcome = create_human_approval_token_final_confirmation_review_outcome(request, confirmer="human-confirmer")

    assert outcome["review_outcome_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_KIND
    assert outcome["review_outcome_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_OUTCOME_STATUS
    assert outcome["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_ROUTING
    assert outcome["source_final_confirmation_request_id"] == request["request_id"]
    assert outcome["source_token_write_lock_gate_id"] == request["source_token_write_lock_gate_id"]
    assert outcome["source_token_issuance_dry_run_id"] == request["source_token_issuance_dry_run_id"]
    assert outcome["source_token_issuance_plan_id"] == request["source_token_issuance_plan_id"]
    assert outcome["source_review_outcome_id"] == request["source_review_outcome_id"]
    assert outcome["source_request_id"] == request["source_request_id"]
    assert outcome["source_gate_id"] == request["source_gate_id"]
    assert outcome["source_dry_run_id"] == request["source_dry_run_id"]
    assert outcome["source_plan_id"] == request["source_plan_id"]
    assert outcome["source_outcome_id"] == request["source_outcome_id"]
    assert outcome["source_packet_id"] == request["source_packet_id"]
    assert outcome["source_submission_id"] == request["source_submission_id"]
    assert outcome["source_draft_id"] == request["source_draft_id"]
    assert outcome["source_decision_id"] == request["source_decision_id"]
    assert outcome["source_queue_item_id"] == request["source_queue_item_id"]
    assert outcome["confirmer"] == "human-confirmer"
    assert outcome["outcome"] == "confirm_token_write"
    assert outcome["final_confirmation_request_validation"] == {"valid": True, "errors": []}
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_final_confirmation_review_outcome(outcome) == {"valid": True, "errors": []}
    assert outcome["next_step_recommendation"]["action"] == "route_to_manual_token_write_without_issuing_token_in_review_gate"
    assert outcome["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert outcome["next_step_recommendation"]["persists_approvals"] is False
    assert outcome["next_step_recommendation"]["creates_real_proposals"] is False
    assert outcome["next_step_recommendation"]["writes_operation_ledger"] is False
    assert outcome["next_step_recommendation"]["writes_token_files"] is False
    assert outcome["next_step_recommendation"]["writes_approval_audit"] is False


def test_valid_final_confirmation_review_outcome_id_matches_v0_1_baseline():
    outcome = create_human_approval_token_final_confirmation_review_outcome(
        _request(),
        confirmer="human-confirmer",
    )

    assert (
        outcome["review_outcome_id"]
        == "memory-human-approval-token-final-confirmation-review-outcome:v0.1:13650d00f3a3edc4"
    )


def test_invalid_request_candidate_rejects():
    request = _request()
    request["request_kind"] = "wrong"

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "reject"
    assert outcome["final_confirmation_request_validation"]["valid"] is False
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}


def test_locked_request_candidate_rejects():
    request = _request(outcome="reject")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert request["request_status"] == "locked"
    assert outcome["outcome"] == "reject"
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}


def test_missing_approval_token_record_preview_requests_changes():
    request = _request()
    request.pop("approval_token_record_preview")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_missing_approval_audit_record_preview_requests_changes():
    request = _request()
    request.pop("approval_audit_record_preview")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_missing_token_target_paths_preview_requests_changes():
    request = _request()
    request.pop("token_target_paths_preview")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_missing_proposal_record_preview_requests_changes():
    request = _request()
    request.pop("proposal_record_preview")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_missing_operation_ledger_preview_requests_changes():
    request = _request()
    request.pop("operation_ledger_preview")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_missing_target_paths_preview_requests_changes():
    request = _request()
    request.pop("target_paths_preview")

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_missing_source_evidence_requests_changes():
    request = _request(source_pattern_ids=[], source_fact_ids=[])

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_preview_integrity_failed_requests_changes():
    request = _request()
    request["approval_token_record_preview"]["token_issued"] = True

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)

    assert outcome["outcome"] == "request_changes"


def test_explicit_supported_outcome_override_is_accepted_but_not_issued_applied_persisted_written_or_submitted():
    request = _request()

    outcome = create_human_approval_token_final_confirmation_review_outcome(
        request,
        outcome="defer",
        rationale="Manual reviewer chose to defer.",
    )
    explanation = explain_human_approval_token_final_confirmation_review_outcome(outcome)
    recommendation = recommend_human_approval_token_final_confirmation_review_action(outcome)

    assert outcome["outcome"] == "defer"
    assert outcome["rationale"] == "Manual reviewer chose to defer."
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert explanation["token_issued"] is False
    assert explanation["approved"] is False
    assert explanation["persisted"] is False
    assert explanation["submitted"] is False
    assert explanation["written"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert explanation["writes_proposal_files"] is False
    assert explanation["writes_operation_ledger"] is False
    assert explanation["writes_token_files"] is False
    assert explanation["writes_approval_audit"] is False
    assert explanation["converted_to_real_proposal"] is False
    assert recommendation["issues_real_approval_tokens"] is False
    assert recommendation["persists_approvals"] is False
    assert recommendation["applies_proposals"] is False
    assert recommendation["submits_to_governance"] is False


def test_unsupported_outcome_fails_validation():
    outcome = create_human_approval_token_final_confirmation_review_outcome(_request())
    outcome["outcome"] = "approve_and_issue"

    validation = validate_human_approval_token_final_confirmation_review_outcome(outcome)

    assert validation["valid"] is False
    assert "outcome_must_be_supported" in validation["errors"]


def test_input_request_candidate_is_not_mutated():
    request = _request()
    before = deepcopy(request)

    outcome = create_human_approval_token_final_confirmation_review_outcome(request)
    outcome["source_final_confirmation_request_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert request == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_or_approval_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    outcome = create_human_approval_token_final_confirmation_review_outcome(_request())
    explanation = explain_human_approval_token_final_confirmation_review_outcome(outcome)
    recommendation = recommend_human_approval_token_final_confirmation_review_action(outcome)

    assert outcome["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY
    assert outcome["policy"]["read_only"] is True
    assert outcome["policy"]["would_write_memory"] is False
    assert outcome["policy"]["would_modify_config"] is False
    assert outcome["policy"]["would_write_graph"] is False
    assert outcome["policy"]["does_not_create_operation_events"] is True
    assert outcome["policy"]["creates_final_confirmation_review_outcomes_only"] is True
    assert outcome["policy"]["issues_real_approval_tokens"] is False
    assert outcome["policy"]["persists_approvals"] is False
    assert outcome["policy"]["creates_real_proposals"] is False
    assert outcome["policy"]["writes_proposal_files"] is False
    assert outcome["policy"]["writes_operation_ledger"] is False
    assert outcome["policy"]["writes_token_files"] is False
    assert outcome["policy"]["writes_approval_audit"] is False
    assert outcome["policy"]["applies_proposals"] is False
    assert outcome["policy"]["submits_to_governance"] is False
    assert outcome["policy"]["converts_to_real_proposal"] is False
    assert explanation["token_issued"] is False
    assert explanation["approved"] is False
    assert explanation["persisted"] is False
    assert explanation["submitted"] is False
    assert explanation["written"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert explanation["writes_proposal_files"] is False
    assert explanation["writes_operation_ledger"] is False
    assert explanation["writes_token_files"] is False
    assert explanation["writes_approval_audit"] is False
    assert explanation["converted_to_real_proposal"] is False
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


def test_outcome_candidate_must_never_be_marked_token_issued_approved_persisted_submitted_written_created_or_converted():
    outcome = create_human_approval_token_final_confirmation_review_outcome(_request())

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
        mutated = deepcopy(outcome)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_final_confirmation_review_outcome(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_by_outcome_block_type_and_status():
    outcomes = [
        create_human_approval_token_final_confirmation_review_outcome(_request("procedural_rules")),
        create_human_approval_token_final_confirmation_review_outcome(_request("project_context")),
        create_human_approval_token_final_confirmation_review_outcome(_request("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_final_confirmation_review_outcomes(outcomes)

    assert summary["total"] == 3
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_outcome"] == {"confirm_token_write": 2, "reject": 1}
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {"final_confirmation_review_outcome_candidate": 3}
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_FINAL_CONFIRMATION_REVIEW_POLICY


def test_evaluate_exposes_same_deterministic_decision_without_creating_writes():
    evaluation = evaluate_human_approval_token_final_confirmation_request(_request(), confirmer="reviewer")

    assert evaluation["confirmer"] == "reviewer"
    assert evaluation["outcome"] == "confirm_token_write"
    assert evaluation["validation"] == {"valid": True, "errors": []}
    assert evaluation["policy"]["issues_real_approval_tokens"] is False
