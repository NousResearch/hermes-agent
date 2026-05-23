from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_approval_token_request import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_ROUTING,
    create_human_approval_token_request,
    explain_human_approval_token_request,
    recommend_human_approval_token_request_action,
    summarize_human_approval_token_requests,
    validate_human_approval_token_request,
)
from agent.memory_human_review_outcome_gate import create_human_review_outcome_candidate
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_real_proposal_creation_plan import create_real_proposal_creation_plan
from agent.memory_real_proposal_dry_run import create_real_proposal_dry_run
from agent.memory_real_proposal_write_lock_gate import (
    MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED,
    create_real_proposal_write_lock_gate,
)
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _plan(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create approval token request candidates only."], "nested": {"value": "preserved"}},
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


def _gate(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    dry_run = create_real_proposal_dry_run(
        _plan(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        operator="final-preflight-operator",
    )
    return create_real_proposal_write_lock_gate(dry_run, operator="write-lock-operator")


def test_eligible_write_lock_gate_creates_review_required_request():
    gate = _gate()

    request = create_human_approval_token_request(gate, requester="token-requester")

    assert request["request_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_KIND
    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_REVIEW_REQUIRED
    assert request["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_ROUTING
    assert request["lock_reason"] is None
    assert request["source_gate_id"] == gate["gate_id"]
    assert request["source_dry_run_id"] == gate["source_dry_run_id"]
    assert request["source_plan_id"] == gate["source_plan_id"]
    assert request["source_outcome_id"] == gate["source_outcome_id"]
    assert request["source_packet_id"] == gate["source_packet_id"]
    assert request["source_submission_id"] == gate["source_submission_id"]
    assert request["source_draft_id"] == gate["source_draft_id"]
    assert request["source_decision_id"] == gate["source_decision_id"]
    assert request["source_queue_item_id"] == gate["source_queue_item_id"]
    assert request["requester"] == "token-requester"
    assert request["write_lock_gate_validation"] == {"valid": True, "errors": []}
    assert request["request_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_request(request) == {"valid": True, "errors": []}
    assert request["next_step_recommendation"]["action"] == "perform_manual_human_approval_token_review_without_issuing_token"
    assert request["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert request["next_step_recommendation"]["creates_real_proposals"] is False
    assert request["next_step_recommendation"]["writes_operation_ledger"] is False


def test_invalid_write_lock_gate_creates_locked_request():
    gate = _gate()
    gate["gate_kind"] = "invalid"

    request = create_human_approval_token_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    assert request["lock_reason"] == "invalid_write_lock_gate"
    assert request["write_lock_gate_validation"]["valid"] is False
    assert request["request_validation"] == {"valid": True, "errors": []}
    assert request["next_step_recommendation"]["action"] == "keep_approval_token_request_locked"


def test_locked_write_lock_gate_creates_locked_request():
    gate = _gate(outcome="reject")

    request = create_human_approval_token_request(gate)

    assert gate["gate_status"] == MEMORY_REAL_PROPOSAL_WRITE_LOCK_GATE_LOCKED
    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    assert request["lock_reason"] == gate["lock_reason"]


def test_missing_proposal_record_preview_locks_request():
    gate = _gate()
    gate.pop("proposal_record_preview")

    request = create_human_approval_token_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_proposal_record_preview"


def test_missing_operation_ledger_preview_locks_request():
    gate = _gate()
    gate.pop("operation_ledger_preview")

    request = create_human_approval_token_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_operation_ledger_preview"


def test_missing_target_paths_preview_locks_request():
    gate = _gate()
    gate.pop("target_paths_preview")

    request = create_human_approval_token_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_target_paths_preview"


def test_missing_source_evidence_locks_request():
    gate = _gate(source_pattern_ids=[], source_fact_ids=[])

    request = create_human_approval_token_request(gate)

    assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
    assert request["lock_reason"] == "missing_source_evidence"


def test_preview_integrity_failed_when_preview_only_false_or_written_created_flags_true():
    cases = []
    for field in ("proposal_record_preview", "operation_ledger_preview", "target_paths_preview"):
        preview_only_false = _gate()
        preview_only_false[field]["preview_only"] = False
        cases.append(preview_only_false)

        written_true = _gate()
        written_true[field]["written"] = True
        cases.append(written_true)

        created_true = _gate()
        created_true[field]["created_real_proposal"] = True
        cases.append(created_true)

    for gate in cases:
        request = create_human_approval_token_request(gate)
        assert request["request_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_LOCKED
        assert request["lock_reason"] == "preview_integrity_failed"


def test_approval_checklist_and_token_requirements_are_deterministic():
    request_a = create_human_approval_token_request(_gate())
    request_b = create_human_approval_token_request(_gate())

    assert request_a["approval_request_checklist"] == request_b["approval_request_checklist"]
    assert request_a["approval_token_requirements"] == request_b["approval_token_requirements"]
    assert [check["id"] for check in request_a["approval_request_checklist"]] == [
        "verify_write_lock_gate_validation",
        "verify_preview_artifacts_only",
        "verify_no_token_or_approval_persistence",
        "verify_no_real_proposal_or_ledger_write",
        "route_to_manual_human_approval_token_review",
    ]
    assert [requirement["id"] for requirement in request_a["approval_token_requirements"]] == [
        "human_reviewer_required",
        "approval_token_must_be_external_to_request",
        "real_write_requires_later_governed_step",
    ]


def test_input_write_lock_gate_is_not_mutated():
    gate = _gate()
    before = deepcopy(gate)

    request = create_human_approval_token_request(gate)
    request["source_write_lock_gate_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert gate == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_or_token_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    request = create_human_approval_token_request(_gate())
    explanation = explain_human_approval_token_request(request)
    recommendation = recommend_human_approval_token_request_action(request)

    assert request["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY
    assert request["policy"]["read_only"] is True
    assert request["policy"]["would_write_memory"] is False
    assert request["policy"]["would_modify_config"] is False
    assert request["policy"]["would_write_graph"] is False
    assert request["policy"]["does_not_create_operation_events"] is True
    assert request["policy"]["creates_approval_token_requests_only"] is True
    assert request["policy"]["issues_real_approval_tokens"] is False
    assert request["policy"]["persists_approvals"] is False
    assert request["policy"]["creates_real_proposals"] is False
    assert request["policy"]["writes_proposal_files"] is False
    assert request["policy"]["writes_operation_ledger"] is False
    assert request["policy"]["applies_proposals"] is False
    assert request["policy"]["submits_to_governance"] is False
    assert request["policy"]["converts_to_real_proposal"] is False
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


def test_request_must_never_be_marked_token_issued_approved_persisted_submitted_or_written():
    request = create_human_approval_token_request(_gate())

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
        mutated = deepcopy(request)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_request(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_review_required_requests_and_by_block_type_status():
    requests = [
        create_human_approval_token_request(_gate("procedural_rules")),
        create_human_approval_token_request(_gate("project_context")),
        create_human_approval_token_request(_gate("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_requests(requests)

    assert summary["total"] == 3
    assert summary["locked_count"] == 1
    assert summary["review_required_count"] == 2
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {
        "approval_token_review_required": 2,
        "locked": 1,
    }
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_REQUEST_POLICY
