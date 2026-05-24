from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_approval_token_request import (
    create_human_approval_token_request,
)
from agent.memory_human_approval_token_review_gate import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_ROUTING,
    MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_STATUS,
    create_human_approval_token_review_outcome,
    evaluate_human_approval_token_request,
    explain_human_approval_token_review_outcome,
    recommend_human_approval_token_review_action,
    summarize_human_approval_token_review_outcomes,
    validate_human_approval_token_review_outcome,
)
from agent.memory_human_review_outcome_gate import create_human_review_outcome_candidate
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_real_proposal_creation_plan import create_real_proposal_creation_plan
from agent.memory_real_proposal_dry_run import create_real_proposal_dry_run
from agent.memory_real_proposal_write_lock_gate import create_real_proposal_write_lock_gate
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _plan(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create approval token review outcome candidates only."], "nested": {"value": "preserved"}},
        project_scope="memory-fabric",
        source_pattern_ids=source_pattern_ids if source_pattern_ids is not None else ["pattern-1"],
        source_fact_ids=source_fact_ids if source_fact_ids is not None else ["fact-1"],
        metadata={"source": "test"},
    )
    queue_item = create_review_queue_item(block, reviewer="memory-reviewer")
    decision = evaluate_review_queue_item(queue_item, reviewer="memory-reviewer")
    draft = create_memory_proposal_draft(decision, author="proposal-drafter")
    submission = create_governance_submission_candidate(draft, reviewer="governance-reviewer")
    packet = create_governance_submission_packet(submission, reviewer="packet-reviewer")
    outcome_candidate = create_human_review_outcome_candidate(packet, reviewer="human-reviewer", outcome=outcome)
    return create_real_proposal_creation_plan(outcome_candidate, planner="plan-reviewer")


def _request(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    dry_run = create_real_proposal_dry_run(
        _plan(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        operator="final-preflight-operator",
    )
    gate = create_real_proposal_write_lock_gate(dry_run, operator="write-lock-operator")
    return create_human_approval_token_request(gate, requester="token-requester")


def test_valid_review_required_request_becomes_approve_token_issuance_outcome_candidate():
    request = _request()

    outcome = create_human_approval_token_review_outcome(request, reviewer="token-reviewer")

    assert outcome["review_outcome_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_KIND
    assert outcome["review_outcome_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_STATUS
    assert outcome["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_OUTCOME_ROUTING
    assert outcome["source_request_id"] == request["request_id"]
    assert outcome["source_gate_id"] == request["source_gate_id"]
    assert outcome["source_dry_run_id"] == request["source_dry_run_id"]
    assert outcome["source_plan_id"] == request["source_plan_id"]
    assert outcome["source_outcome_id"] == request["source_outcome_id"]
    assert outcome["source_packet_id"] == request["source_packet_id"]
    assert outcome["source_submission_id"] == request["source_submission_id"]
    assert outcome["source_draft_id"] == request["source_draft_id"]
    assert outcome["source_decision_id"] == request["source_decision_id"]
    assert outcome["source_queue_item_id"] == request["source_queue_item_id"]
    assert outcome["reviewer"] == "token-reviewer"
    assert outcome["outcome"] == "approve_token_issuance"
    assert outcome["request_validation"] == {"valid": True, "errors": []}
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_review_outcome(outcome) == {"valid": True, "errors": []}
    assert outcome["next_step_recommendation"]["action"] == "manual_token_issuance_still_required"
    assert outcome["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert outcome["next_step_recommendation"]["creates_real_proposals"] is False
    assert outcome["next_step_recommendation"]["writes_operation_ledger"] is False


def test_valid_human_approval_token_review_outcome_id_matches_v0_1_baseline():
    outcome = create_human_approval_token_review_outcome(_request(), reviewer="token-reviewer")

    assert (
        outcome["review_outcome_id"]
        == "memory-human-approval-token-review-outcome:v0.1:b58709b83a4aaeb5"
    )


def test_invalid_request_candidate_rejects():
    request = _request()
    request["request_kind"] = "invalid"

    outcome = create_human_approval_token_review_outcome(request)

    assert outcome["outcome"] == "reject"
    assert outcome["request_validation"]["valid"] is False
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert evaluate_human_approval_token_request(request)["outcome"] == "reject"


def test_locked_request_candidate_rejects():
    request = _request(outcome="reject")

    outcome = create_human_approval_token_review_outcome(request)

    assert request["request_status"] == "locked"
    assert outcome["outcome"] == "reject"


def test_missing_proposal_record_preview_requests_changes():
    request = _request()
    request.pop("proposal_record_preview")

    outcome = create_human_approval_token_review_outcome(request)

    assert outcome["outcome"] == "request_changes"
    assert "missing_proposal_record_preview" in outcome["request_validation"]["errors"]


def test_missing_operation_ledger_preview_requests_changes():
    request = _request()
    request.pop("operation_ledger_preview")

    outcome = create_human_approval_token_review_outcome(request)

    assert outcome["outcome"] == "request_changes"
    assert "missing_operation_ledger_preview" in outcome["request_validation"]["errors"]


def test_missing_target_paths_preview_requests_changes():
    request = _request()
    request.pop("target_paths_preview")

    outcome = create_human_approval_token_review_outcome(request)

    assert outcome["outcome"] == "request_changes"
    assert "missing_target_paths_preview" in outcome["request_validation"]["errors"]


def test_missing_source_evidence_requests_changes():
    request = _request(source_pattern_ids=[], source_fact_ids=[])

    outcome = create_human_approval_token_review_outcome(request)

    assert outcome["outcome"] == "request_changes"
    assert "missing_source_evidence" in outcome["request_validation"]["errors"]


def test_explicit_supported_outcome_override_is_accepted_but_not_issued_applied_or_persisted():
    request = _request()

    outcome = create_human_approval_token_review_outcome(
        request,
        reviewer="token-reviewer",
        outcome="defer",
    )
    recommendation = recommend_human_approval_token_review_action(outcome)

    assert outcome["outcome"] == "defer"
    assert outcome["review_outcome_validation"] == {"valid": True, "errors": []}
    assert "Explicit supported" in outcome["rationale"]
    assert recommendation["issues_real_approval_tokens"] is False
    assert recommendation["persists_approvals"] is False
    assert recommendation["applies_proposals"] is False
    assert recommendation["submits_to_governance"] is False


def test_unsupported_outcome_fails_validation():
    outcome = create_human_approval_token_review_outcome(_request(), outcome="ship_it")

    validation = validate_human_approval_token_review_outcome(outcome)

    assert validation["valid"] is False
    assert "unsupported_human_approval_token_review_outcome" in validation["errors"]
    assert outcome["next_step_recommendation"]["action"] == "do_not_use_token_review_outcome_candidate"


def test_input_request_candidate_is_not_mutated():
    request = _request()
    before = deepcopy(request)

    outcome = create_human_approval_token_review_outcome(request)
    outcome["source_request_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert request == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_or_approval_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    outcome = create_human_approval_token_review_outcome(_request())
    explanation = explain_human_approval_token_review_outcome(outcome)
    recommendation = recommend_human_approval_token_review_action(outcome)

    assert outcome["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY
    assert outcome["policy"]["read_only"] is True
    assert outcome["policy"]["would_write_memory"] is False
    assert outcome["policy"]["would_modify_config"] is False
    assert outcome["policy"]["would_write_graph"] is False
    assert outcome["policy"]["does_not_create_operation_events"] is True
    assert outcome["policy"]["creates_review_outcome_candidates_only"] is True
    assert outcome["policy"]["issues_real_approval_tokens"] is False
    assert outcome["policy"]["persists_approvals"] is False
    assert outcome["policy"]["creates_real_proposals"] is False
    assert outcome["policy"]["writes_proposal_files"] is False
    assert outcome["policy"]["writes_operation_ledger"] is False
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
    assert explanation["converted_to_real_proposal"] is False
    assert recommendation["issues_real_approval_tokens"] is False
    assert recommendation["persists_approvals"] is False
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["writes_proposal_files"] is False
    assert recommendation["writes_operation_ledger"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_outcome_candidate_must_never_be_marked_token_issued_approved_persisted_or_written():
    outcome = create_human_approval_token_review_outcome(_request())

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
        "converted_to_real_proposal",
    ):
        mutated = deepcopy(outcome)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_review_outcome(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_by_outcome_block_type_and_status():
    outcomes = [
        create_human_approval_token_review_outcome(_request("procedural_rules")),
        create_human_approval_token_review_outcome(_request("project_context")),
        create_human_approval_token_review_outcome(_request("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_review_outcomes(outcomes)

    assert summary["total"] == 3
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_outcome"] == {
        "approve_token_issuance": 2,
        "reject": 1,
    }
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {"token_review_outcome_candidate": 3}
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_REVIEW_GATE_POLICY
