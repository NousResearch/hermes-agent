from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_review_outcome_gate import create_human_review_outcome_candidate
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_real_proposal_creation_plan import (
    MEMORY_REAL_PROPOSAL_CREATION_PLAN_KIND,
    MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY,
    MEMORY_REAL_PROPOSAL_CREATION_PLAN_ROUTING,
    MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS,
    create_real_proposal_creation_plan,
    explain_real_proposal_creation_plan,
    recommend_real_proposal_creation_plan_action,
    summarize_real_proposal_creation_plans,
    validate_real_proposal_creation_plan,
)
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _outcome(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create real proposal creation plans only."], "nested": {"value": "preserved"}},
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
    return create_human_review_outcome_candidate(packet, reviewer="human-reviewer", outcome=outcome)


def test_approve_real_proposal_creation_outcome_creates_valid_manual_plan():
    outcome = _outcome()

    plan = create_real_proposal_creation_plan(outcome, planner="plan-reviewer")

    assert plan["plan_kind"] == MEMORY_REAL_PROPOSAL_CREATION_PLAN_KIND
    assert plan["plan_status"] == MEMORY_REAL_PROPOSAL_CREATION_PLAN_STATUS
    assert plan["routing"] == MEMORY_REAL_PROPOSAL_CREATION_PLAN_ROUTING
    assert plan["source_outcome_id"] == outcome["outcome_id"]
    assert plan["source_packet_id"] == outcome["source_packet_id"]
    assert plan["source_submission_id"] == outcome["source_submission_id"]
    assert plan["source_draft_id"] == outcome["source_draft_id"]
    assert plan["source_decision_id"] == outcome["source_decision_id"]
    assert plan["source_queue_item_id"] == outcome["source_queue_item_id"]
    assert plan["planner"] == "plan-reviewer"
    assert plan["plan_validation"] == {"valid": True, "errors": []}
    assert validate_real_proposal_creation_plan(plan) == {"valid": True, "errors": []}
    assert plan["next_step_recommendation"]["action"] == "perform_manual_real_proposal_creation_preflight"
    assert plan["next_step_recommendation"]["creates_real_proposals"] is False
    assert plan["next_step_recommendation"]["converts_to_real_proposal"] is False


def test_valid_manual_plan_id_matches_v0_1_baseline():
    plan = create_real_proposal_creation_plan(_outcome(), planner="plan-reviewer")

    assert plan["plan_id"] == "memory-real-proposal-creation-plan:v0.1:ec09132b407b8a96"


def test_request_changes_creates_invalid_plan_with_human_review_requested_changes():
    plan = create_real_proposal_creation_plan(_outcome(outcome="request_changes"))

    assert plan["plan_validation"]["valid"] is False
    assert "human_review_requested_changes" in plan["plan_validation"]["errors"]
    assert plan["invalid_reason"] == "human_review_requested_changes"
    assert plan["next_step_recommendation"]["action"] == "do_not_create_real_proposal"


def test_reject_creates_invalid_plan_with_human_review_rejected():
    plan = create_real_proposal_creation_plan(_outcome(outcome="reject"))

    assert plan["plan_validation"]["valid"] is False
    assert "human_review_rejected" in plan["plan_validation"]["errors"]
    assert plan["invalid_reason"] == "human_review_rejected"


def test_defer_creates_invalid_plan_with_human_review_deferred():
    plan = create_real_proposal_creation_plan(_outcome(outcome="defer"))

    assert plan["plan_validation"]["valid"] is False
    assert "human_review_deferred" in plan["plan_validation"]["errors"]
    assert plan["invalid_reason"] == "human_review_deferred"


def test_invalid_outcome_candidate_creates_invalid_plan():
    outcome = _outcome(outcome="ship_it")

    plan = create_real_proposal_creation_plan(outcome)

    assert plan["plan_validation"]["valid"] is False
    assert "invalid_human_review_outcome_candidate" in plan["plan_validation"]["errors"]
    assert plan["invalid_reason"] == "invalid_human_review_outcome_candidate"


def test_missing_payload_preview_creates_invalid_plan():
    outcome = _outcome()
    outcome["payload_preview"] = None

    plan = create_real_proposal_creation_plan(outcome)

    assert plan["plan_validation"]["valid"] is False
    assert "missing_payload_preview" in plan["plan_validation"]["errors"]
    assert plan["invalid_reason"] == "missing_payload_preview"


def test_missing_source_evidence_creates_invalid_plan():
    outcome = _outcome()
    outcome["source_pattern_ids"] = []
    outcome["source_fact_ids"] = []

    plan = create_real_proposal_creation_plan(outcome)

    assert plan["plan_validation"]["valid"] is False
    assert "missing_source_evidence" in plan["plan_validation"]["errors"]
    assert plan["invalid_reason"] == "missing_source_evidence"


def test_creation_steps_and_preflight_checks_are_deterministic():
    plan_a = create_real_proposal_creation_plan(_outcome())
    plan_b = create_real_proposal_creation_plan(_outcome())

    assert plan_a["creation_steps"] == plan_b["creation_steps"]
    assert plan_a["preflight_checks"] == plan_b["preflight_checks"]
    assert [step["id"] for step in plan_a["creation_steps"]] == [
        "verify_human_review_approval",
        "verify_payload_preview",
        "verify_source_evidence",
        "manual_real_proposal_creation",
    ]
    assert [check["id"] for check in plan_a["preflight_checks"]] == [
        "no_memory_write",
        "no_graph_write",
        "no_config_change",
        "no_proposal_or_ledger_record",
    ]


def test_input_outcome_candidate_is_not_mutated():
    outcome = _outcome()
    before = deepcopy(outcome)

    plan = create_real_proposal_creation_plan(outcome)
    plan["source_outcome_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert outcome == before


def test_policy_proves_no_memory_config_graph_proposal_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    plan = create_real_proposal_creation_plan(_outcome())
    explanation = explain_real_proposal_creation_plan(plan)
    recommendation = recommend_real_proposal_creation_plan_action(plan)

    assert plan["policy"] == MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY
    assert plan["policy"]["read_only"] is True
    assert plan["policy"]["would_write_memory"] is False
    assert plan["policy"]["would_modify_config"] is False
    assert plan["policy"]["would_write_graph"] is False
    assert plan["policy"]["does_not_create_operation_events"] is True
    assert plan["policy"]["creates_plan_candidates_only"] is True
    assert plan["policy"]["creates_real_proposals"] is False
    assert plan["policy"]["applies_proposals"] is False
    assert plan["policy"]["persists_approvals"] is False
    assert plan["policy"]["submits_to_governance"] is False
    assert plan["policy"]["converts_to_real_proposal"] is False
    assert explanation["submitted"] is False
    assert explanation["applied"] is False
    assert explanation["persisted"] is False
    assert explanation["approved"] is False
    assert explanation["converted_to_real_proposal"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert explanation["created_proposal_record"] is False
    assert explanation["submitted_to_governance"] is False
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["submits_to_governance"] is False
    assert recommendation["converts_to_real_proposal"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_plan_must_never_be_marked_submitted_applied_persisted_approved_or_converted():
    plan = create_real_proposal_creation_plan(_outcome())

    for forbidden_key in (
        "submitted",
        "applied",
        "persisted",
        "approved",
        "converted_to_real_proposal",
        "created_real_proposal",
        "created_operation_event",
        "created_proposal_record",
        "submitted_to_governance",
    ):
        mutated = deepcopy(plan)
        mutated[forbidden_key] = True
        validation = validate_real_proposal_creation_plan(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_valid_invalid_plans_and_by_block_type_status():
    plans = [
        create_real_proposal_creation_plan(_outcome("procedural_rules")),
        create_real_proposal_creation_plan(_outcome("project_context")),
        create_real_proposal_creation_plan(_outcome("procedural_rules", outcome="reject")),
    ]

    summary = summarize_real_proposal_creation_plans(plans)

    assert summary["total"] == 3
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 1
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {"manual_creation_plan_required": 3}
    assert summary["policy"] == MEMORY_REAL_PROPOSAL_CREATION_PLAN_POLICY
