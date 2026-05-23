from copy import deepcopy

from agent.memory_block_review_queue import create_review_queue_item
from agent.memory_blocks import create_memory_block_candidate
from agent.memory_governance_submission_packet import create_governance_submission_packet
from agent.memory_human_review_outcome_gate import create_human_review_outcome_candidate
from agent.memory_proposal_draft_builder import create_memory_proposal_draft
from agent.memory_proposal_governance_gate import create_governance_submission_candidate
from agent.memory_real_proposal_creation_plan import create_real_proposal_creation_plan
from agent.memory_real_proposal_dry_run import (
    MEMORY_REAL_PROPOSAL_DRY_RUN_KIND,
    MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY,
    MEMORY_REAL_PROPOSAL_DRY_RUN_ROUTING,
    MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS,
    create_real_proposal_dry_run,
    explain_real_proposal_dry_run,
    recommend_real_proposal_dry_run_action,
    summarize_real_proposal_dry_runs,
    validate_real_proposal_dry_run,
)
from agent.memory_review_decision_gate import evaluate_review_queue_item


def _plan(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    block = create_memory_block_candidate(
        block_type,
        {"rules": ["Create dry-run previews only."], "nested": {"value": "preserved"}},
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


def test_valid_plan_creates_valid_manual_final_preflight_dry_run():
    plan = _plan()

    dry_run = create_real_proposal_dry_run(plan, operator="final-preflight-operator")

    assert dry_run["dry_run_kind"] == MEMORY_REAL_PROPOSAL_DRY_RUN_KIND
    assert dry_run["dry_run_status"] == MEMORY_REAL_PROPOSAL_DRY_RUN_STATUS
    assert dry_run["routing"] == MEMORY_REAL_PROPOSAL_DRY_RUN_ROUTING
    assert dry_run["source_plan_id"] == plan["plan_id"]
    assert dry_run["source_outcome_id"] == plan["source_outcome_id"]
    assert dry_run["source_packet_id"] == plan["source_packet_id"]
    assert dry_run["source_submission_id"] == plan["source_submission_id"]
    assert dry_run["source_draft_id"] == plan["source_draft_id"]
    assert dry_run["source_decision_id"] == plan["source_decision_id"]
    assert dry_run["source_queue_item_id"] == plan["source_queue_item_id"]
    assert dry_run["operator"] == "final-preflight-operator"
    assert dry_run["plan_validation"] == {"valid": True, "errors": []}
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}
    assert validate_real_proposal_dry_run(dry_run) == {"valid": True, "errors": []}
    assert dry_run["next_step_recommendation"]["action"] == "perform_manual_final_preflight_before_real_proposal_write"
    assert dry_run["next_step_recommendation"]["creates_real_proposals"] is False
    assert dry_run["next_step_recommendation"]["writes_operation_ledger"] is False


def test_invalid_plan_creates_invalid_dry_run():
    plan = _plan(outcome="reject")

    dry_run = create_real_proposal_dry_run(plan)

    assert dry_run["dry_run_validation"]["valid"] is False
    assert "invalid_real_proposal_creation_plan" in dry_run["dry_run_validation"]["errors"]
    assert dry_run["invalid_reason"] == "invalid_real_proposal_creation_plan"
    assert dry_run["next_step_recommendation"]["action"] == "do_not_create_real_proposal"


def test_missing_payload_preview_creates_invalid_dry_run():
    plan = _plan()
    plan["payload_preview"] = None

    dry_run = create_real_proposal_dry_run(plan)

    assert dry_run["dry_run_validation"]["valid"] is False
    assert "missing_payload_preview" in dry_run["dry_run_validation"]["errors"]
    assert dry_run["invalid_reason"] == "missing_payload_preview"


def test_missing_source_evidence_creates_invalid_dry_run():
    plan = _plan()
    plan["source_pattern_ids"] = []
    plan["source_fact_ids"] = []

    dry_run = create_real_proposal_dry_run(plan)

    assert dry_run["dry_run_validation"]["valid"] is False
    assert "missing_source_evidence" in dry_run["dry_run_validation"]["errors"]
    assert dry_run["invalid_reason"] == "missing_source_evidence"


def test_missing_creation_steps_or_preflight_checks_creates_invalid_dry_run():
    plan_without_steps = _plan()
    plan_without_steps["creation_steps"] = []
    plan_without_checks = _plan()
    plan_without_checks["preflight_checks"] = []

    dry_run_without_steps = create_real_proposal_dry_run(plan_without_steps)
    dry_run_without_checks = create_real_proposal_dry_run(plan_without_checks)

    assert dry_run_without_steps["dry_run_validation"]["valid"] is False
    assert "missing_plan_controls" in dry_run_without_steps["dry_run_validation"]["errors"]
    assert dry_run_without_steps["invalid_reason"] == "missing_plan_controls"
    assert dry_run_without_checks["dry_run_validation"]["valid"] is False
    assert "missing_plan_controls" in dry_run_without_checks["dry_run_validation"]["errors"]
    assert dry_run_without_checks["invalid_reason"] == "missing_plan_controls"


def test_proposal_record_preview_contains_payload_and_source_evidence():
    plan = _plan()

    dry_run = create_real_proposal_dry_run(plan)
    preview = dry_run["proposal_record_preview"]

    assert preview["preview_only"] is True
    assert preview["written"] is False
    assert preview["block_id"] == plan["block_id"]
    assert preview["block_type"] == plan["block_type"]
    assert preview["project_scope"] == plan["project_scope"]
    assert preview["payload_preview"] == plan["payload_preview"]
    assert preview["source_pattern_ids"] == plan["source_pattern_ids"]
    assert preview["source_fact_ids"] == plan["source_fact_ids"]
    assert preview["created_real_proposal"] is False
    assert preview["submitted_to_governance"] is False


def test_operation_ledger_preview_is_preview_only_and_not_written():
    dry_run = create_real_proposal_dry_run(_plan())
    preview = dry_run["operation_ledger_preview"]

    assert preview["preview_only"] is True
    assert preview["written"] is False
    assert preview["created_operation_event"] is False
    assert preview["would_write_memory"] is False
    assert preview["would_write_graph"] is False
    assert preview["would_modify_config"] is False
    assert preview["would_write_proposal_file"] is False
    assert preview["would_write_operation_ledger"] is False


def test_target_paths_preview_is_preview_only_and_deterministic():
    plan = _plan()

    dry_run_a = create_real_proposal_dry_run(plan)
    dry_run_b = create_real_proposal_dry_run(plan)

    assert dry_run_a["target_paths_preview"] == dry_run_b["target_paths_preview"]
    assert dry_run_a["target_paths_preview"]["preview_only"] is True
    assert dry_run_a["target_paths_preview"]["base"] == "${HERMES_HOME}"
    assert dry_run_a["target_paths_preview"]["proposal_file"] == "memory/proposals/memory_write_proposals.jsonl"
    assert dry_run_a["target_paths_preview"]["operation_ledger_file"] == "memory/audit/memory_operation_ledger.jsonl"
    assert dry_run_a["target_paths_preview"]["created_real_proposal"] is False
    assert dry_run_a["target_paths_preview"]["created_operation_event"] is False


def test_final_preflight_checklist_is_deterministic():
    dry_run_a = create_real_proposal_dry_run(_plan())
    dry_run_b = create_real_proposal_dry_run(_plan())

    assert dry_run_a["final_preflight_checklist"] == dry_run_b["final_preflight_checklist"]
    assert [check["id"] for check in dry_run_a["final_preflight_checklist"]] == [
        "verify_plan_validation",
        "verify_payload_and_evidence",
        "verify_preview_records_only",
        "verify_no_memory_graph_config_write",
        "manual_operator_final_preflight",
    ]


def test_input_plan_is_not_mutated():
    plan = _plan()
    before = deepcopy(plan)

    dry_run = create_real_proposal_dry_run(plan)
    dry_run["source_plan_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert plan == before


def test_policy_proves_no_memory_config_graph_proposal_or_ledger_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    dry_run = create_real_proposal_dry_run(_plan())
    explanation = explain_real_proposal_dry_run(dry_run)
    recommendation = recommend_real_proposal_dry_run_action(dry_run)

    assert dry_run["policy"] == MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY
    assert dry_run["policy"]["read_only"] is True
    assert dry_run["policy"]["would_write_memory"] is False
    assert dry_run["policy"]["would_modify_config"] is False
    assert dry_run["policy"]["would_write_graph"] is False
    assert dry_run["policy"]["does_not_create_operation_events"] is True
    assert dry_run["policy"]["creates_dry_run_candidates_only"] is True
    assert dry_run["policy"]["creates_real_proposals"] is False
    assert dry_run["policy"]["writes_proposal_files"] is False
    assert dry_run["policy"]["writes_operation_ledger"] is False
    assert dry_run["policy"]["applies_proposals"] is False
    assert dry_run["policy"]["persists_approvals"] is False
    assert dry_run["policy"]["submits_to_governance"] is False
    assert dry_run["policy"]["converts_to_real_proposal"] is False
    assert explanation["written"] is False
    assert explanation["submitted"] is False
    assert explanation["applied"] is False
    assert explanation["persisted"] is False
    assert explanation["approved"] is False
    assert explanation["created_real_proposal"] is False
    assert explanation["created_operation_event"] is False
    assert explanation["converted_to_real_proposal"] is False
    assert recommendation["creates_real_proposals"] is False
    assert recommendation["writes_proposal_files"] is False
    assert recommendation["writes_operation_ledger"] is False
    assert not (hermes_home / "memory" / "graph" / "memory_graph.sqlite").exists()
    assert not (hermes_home / "memory" / "audit" / "memory_operation_ledger.jsonl").exists()
    assert not (hermes_home / "memory" / "proposals" / "memory_write_proposals.jsonl").exists()


def test_dry_run_must_never_be_marked_written_submitted_applied_persisted_approved_or_converted():
    dry_run = create_real_proposal_dry_run(_plan())

    for forbidden_key in (
        "written",
        "submitted",
        "applied",
        "persisted",
        "approved",
        "created_real_proposal",
        "created_operation_event",
        "converted_to_real_proposal",
        "created_proposal_record",
        "submitted_to_governance",
        "persisted_approval",
    ):
        mutated = deepcopy(dry_run)
        mutated[forbidden_key] = True
        validation = validate_real_proposal_dry_run(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_valid_invalid_dry_runs_and_by_block_type_status():
    dry_runs = [
        create_real_proposal_dry_run(_plan("procedural_rules")),
        create_real_proposal_dry_run(_plan("project_context")),
        create_real_proposal_dry_run(_plan("procedural_rules", outcome="reject")),
    ]

    summary = summarize_real_proposal_dry_runs(dry_runs)

    assert summary["total"] == 3
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 1
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {"manual_final_preflight_required": 3}
    assert summary["policy"] == MEMORY_REAL_PROPOSAL_DRY_RUN_POLICY
