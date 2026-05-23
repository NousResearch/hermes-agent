from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_approval_token_issuance_plan import (
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_ROUTING,
    create_human_approval_token_issuance_plan,
    explain_human_approval_token_issuance_plan,
    recommend_human_approval_token_issuance_plan_action,
    summarize_human_approval_token_issuance_plans,
    validate_human_approval_token_issuance_plan,
)
from agent.memory_human_approval_token_request import create_human_approval_token_request
from agent.memory_human_approval_token_review_gate import create_human_approval_token_review_outcome
from agent.memory_human_review_outcome_gate import create_human_review_outcome_candidate
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_real_proposal_creation_plan import create_real_proposal_creation_plan
from agent.memory_real_proposal_dry_run import create_real_proposal_dry_run
from agent.memory_real_proposal_write_lock_gate import create_real_proposal_write_lock_gate
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _creation_plan(block_type="procedural_rules", source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create approval token issuance plan candidates only."], "nested": {"value": "preserved"}},
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
    outcome_candidate = create_human_review_outcome_candidate(packet, reviewer="human-reviewer")
    return create_real_proposal_creation_plan(outcome_candidate, planner="plan-reviewer")


def _review_outcome(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    dry_run = create_real_proposal_dry_run(
        _creation_plan(
            block_type=block_type,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        operator="final-preflight-operator",
    )
    gate = create_real_proposal_write_lock_gate(dry_run, operator="write-lock-operator")
    request = create_human_approval_token_request(gate, requester="token-requester")
    return create_human_approval_token_review_outcome(request, reviewer="token-reviewer", outcome=outcome)


def test_approve_token_issuance_outcome_creates_manual_token_issuance_plan_required():
    outcome = _review_outcome()

    plan = create_human_approval_token_issuance_plan(outcome, planner="token-plan-reviewer")

    assert plan["plan_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_KIND
    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_REQUIRED
    assert plan["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_ROUTING
    assert plan["lock_reason"] is None
    assert plan["source_review_outcome_id"] == outcome["review_outcome_id"]
    assert plan["source_request_id"] == outcome["source_request_id"]
    assert plan["source_gate_id"] == outcome["source_gate_id"]
    assert plan["source_dry_run_id"] == outcome["source_dry_run_id"]
    assert plan["source_plan_id"] == outcome["source_plan_id"]
    assert plan["source_outcome_id"] == outcome["source_outcome_id"]
    assert plan["source_packet_id"] == outcome["source_packet_id"]
    assert plan["source_submission_id"] == outcome["source_submission_id"]
    assert plan["source_draft_id"] == outcome["source_draft_id"]
    assert plan["source_decision_id"] == outcome["source_decision_id"]
    assert plan["source_queue_item_id"] == outcome["source_queue_item_id"]
    assert plan["planner"] == "token-plan-reviewer"
    assert plan["outcome"] == "approve_token_issuance"
    assert plan["review_outcome_validation"] == {"valid": True, "errors": []}
    assert plan["plan_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_issuance_plan(plan) == {"valid": True, "errors": []}
    assert plan["next_step_recommendation"]["action"] == "run_manual_token_issuance_dry_run_before_any_token_write"
    assert plan["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert plan["next_step_recommendation"]["creates_real_proposals"] is False
    assert plan["next_step_recommendation"]["writes_operation_ledger"] is False


def test_request_changes_creates_locked_plan():
    plan = create_human_approval_token_issuance_plan(_review_outcome(outcome="request_changes"))

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "token_review_requested_changes"


def test_reject_creates_locked_plan():
    plan = create_human_approval_token_issuance_plan(_review_outcome(outcome="reject"))

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "token_review_rejected"


def test_defer_creates_locked_plan():
    plan = create_human_approval_token_issuance_plan(_review_outcome(outcome="defer"))

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "token_review_deferred"


def test_invalid_review_outcome_creates_locked_plan():
    outcome = _review_outcome(outcome="ship_it")

    plan = create_human_approval_token_issuance_plan(outcome)

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "invalid_token_review_outcome"
    assert plan["review_outcome_validation"]["valid"] is False


def test_missing_proposal_record_preview_locks_plan():
    outcome = _review_outcome()
    outcome.pop("proposal_record_preview")

    plan = create_human_approval_token_issuance_plan(outcome)

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "missing_proposal_record_preview"


def test_missing_operation_ledger_preview_locks_plan():
    outcome = _review_outcome()
    outcome.pop("operation_ledger_preview")

    plan = create_human_approval_token_issuance_plan(outcome)

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "missing_operation_ledger_preview"


def test_missing_target_paths_preview_locks_plan():
    outcome = _review_outcome()
    outcome.pop("target_paths_preview")

    plan = create_human_approval_token_issuance_plan(outcome)

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "missing_target_paths_preview"


def test_missing_source_evidence_locks_plan():
    outcome = _review_outcome(source_pattern_ids=[], source_fact_ids=[])

    plan = create_human_approval_token_issuance_plan(outcome)

    assert plan["plan_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_LOCKED
    assert plan["lock_reason"] == "missing_source_evidence"


def test_token_issuance_steps_and_preflight_checks_are_deterministic():
    plan_a = create_human_approval_token_issuance_plan(_review_outcome())
    plan_b = create_human_approval_token_issuance_plan(_review_outcome())

    assert plan_a["token_issuance_steps"] == plan_b["token_issuance_steps"]
    assert plan_a["token_preflight_checks"] == plan_b["token_preflight_checks"]
    assert plan_a["plan_id"] == plan_b["plan_id"]


def test_input_review_outcome_candidate_is_not_mutated():
    outcome = _review_outcome()
    before = deepcopy(outcome)

    plan = create_human_approval_token_issuance_plan(outcome)
    plan["source_review_outcome_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert outcome == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_or_approval_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    plan = create_human_approval_token_issuance_plan(_review_outcome())
    explanation = explain_human_approval_token_issuance_plan(plan)
    recommendation = recommend_human_approval_token_issuance_plan_action(plan)

    assert plan["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY
    assert plan["policy"]["read_only"] is True
    assert plan["policy"]["would_write_memory"] is False
    assert plan["policy"]["would_modify_config"] is False
    assert plan["policy"]["would_write_graph"] is False
    assert plan["policy"]["does_not_create_operation_events"] is True
    assert plan["policy"]["creates_token_issuance_plan_candidates_only"] is True
    assert plan["policy"]["issues_real_approval_tokens"] is False
    assert plan["policy"]["persists_approvals"] is False
    assert plan["policy"]["creates_real_proposals"] is False
    assert plan["policy"]["writes_proposal_files"] is False
    assert plan["policy"]["writes_operation_ledger"] is False
    assert plan["policy"]["applies_proposals"] is False
    assert plan["policy"]["submits_to_governance"] is False
    assert plan["policy"]["converts_to_real_proposal"] is False
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


def test_plan_must_never_be_marked_token_issued_approved_persisted_or_written():
    plan = create_human_approval_token_issuance_plan(_review_outcome())

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
        mutated = deepcopy(plan)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_issuance_plan(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_plan_required_candidates_and_by_block_type_status():
    plans = [
        create_human_approval_token_issuance_plan(_review_outcome("procedural_rules")),
        create_human_approval_token_issuance_plan(_review_outcome("project_context")),
        create_human_approval_token_issuance_plan(_review_outcome("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_issuance_plans(plans)

    assert summary["total"] == 3
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["plan_required_count"] == 2
    assert summary["locked_count"] == 1
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {
        "locked": 1,
        "manual_token_issuance_plan_required": 2,
    }
    assert summary["by_lock_reason"] == {"None": 2, "token_review_rejected": 1}
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_PLAN_POLICY
