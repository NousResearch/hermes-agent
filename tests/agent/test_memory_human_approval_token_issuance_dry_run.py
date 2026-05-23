from copy import deepcopy

from agent.memory_human_approval_token_issuance_dry_run import (
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_ROUTING,
    create_human_approval_token_issuance_dry_run,
    explain_human_approval_token_issuance_dry_run,
    recommend_human_approval_token_issuance_dry_run_action,
    summarize_human_approval_token_issuance_dry_runs,
    validate_human_approval_token_issuance_dry_run,
)
from agent.memory_human_approval_token_issuance_plan import create_human_approval_token_issuance_plan
from tests.agent.test_memory_human_approval_token_issuance_plan import _review_outcome


def _plan(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    return create_human_approval_token_issuance_plan(
        _review_outcome(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        planner="token-plan-reviewer",
    )


def test_valid_token_issuance_plan_creates_final_preflight_required_dry_run():
    plan = _plan()

    dry_run = create_human_approval_token_issuance_dry_run(plan, operator="token-dry-run-operator")

    assert dry_run["dry_run_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_KIND
    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_FINAL_PREFLIGHT_REQUIRED
    assert dry_run["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_ROUTING
    assert dry_run["lock_reason"] is None
    assert dry_run["source_token_issuance_plan_id"] == plan["plan_id"]
    assert dry_run["source_review_outcome_id"] == plan["source_review_outcome_id"]
    assert dry_run["source_request_id"] == plan["source_request_id"]
    assert dry_run["source_gate_id"] == plan["source_gate_id"]
    assert dry_run["source_dry_run_id"] == plan["source_dry_run_id"]
    assert dry_run["source_plan_id"] == plan["source_plan_id"]
    assert dry_run["source_outcome_id"] == plan["source_outcome_id"]
    assert dry_run["source_packet_id"] == plan["source_packet_id"]
    assert dry_run["source_submission_id"] == plan["source_submission_id"]
    assert dry_run["source_draft_id"] == plan["source_draft_id"]
    assert dry_run["source_decision_id"] == plan["source_decision_id"]
    assert dry_run["source_queue_item_id"] == plan["source_queue_item_id"]
    assert dry_run["operator"] == "token-dry-run-operator"
    assert dry_run["token_issuance_plan_validation"] == {"valid": True, "errors": []}
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_issuance_dry_run(dry_run) == {"valid": True, "errors": []}
    assert dry_run["next_step_recommendation"]["action"] == "request_manual_final_human_confirmation_before_token_write"
    assert dry_run["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert dry_run["next_step_recommendation"]["persists_approvals"] is False
    assert dry_run["next_step_recommendation"]["creates_real_proposals"] is False
    assert dry_run["next_step_recommendation"]["writes_operation_ledger"] is False


def test_locked_plan_creates_locked_dry_run():
    plan = _plan(outcome="reject")

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "token_review_rejected"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_locked_plan_without_lock_reason_uses_default_lock_reason():
    plan = _plan(outcome="reject")
    plan["lock_reason"] = None

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "token_issuance_plan_locked"


def test_invalid_plan_creates_locked_dry_run():
    plan = _plan()
    plan["plan_kind"] = "wrong"

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "invalid_token_issuance_plan"
    assert dry_run["token_issuance_plan_validation"]["valid"] is False


def test_missing_proposal_record_preview_locks_dry_run():
    plan = _plan()
    plan.pop("proposal_record_preview")

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "missing_proposal_record_preview"


def test_missing_operation_ledger_preview_locks_dry_run():
    plan = _plan()
    plan.pop("operation_ledger_preview")

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "missing_operation_ledger_preview"


def test_missing_target_paths_preview_locks_dry_run():
    plan = _plan()
    plan.pop("target_paths_preview")

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "missing_target_paths_preview"


def test_missing_source_evidence_locks_dry_run():
    plan = _plan(source_pattern_ids=[], source_fact_ids=[])

    dry_run = create_human_approval_token_issuance_dry_run(plan)

    assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
    assert dry_run["lock_reason"] == "missing_source_evidence"


def test_missing_token_issuance_steps_or_preflight_checks_locks_dry_run():
    for field in ("token_issuance_steps", "token_preflight_checks"):
        plan = _plan()
        plan.pop(field)

        dry_run = create_human_approval_token_issuance_dry_run(plan)

        assert dry_run["dry_run_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_LOCKED
        assert dry_run["lock_reason"] == "missing_token_plan_controls"


def test_preview_records_and_final_preflight_checklist_are_deterministic_and_preview_only():
    dry_run_a = create_human_approval_token_issuance_dry_run(_plan(), operator="token-dry-run-operator")
    dry_run_b = create_human_approval_token_issuance_dry_run(_plan(), operator="token-dry-run-operator")

    assert dry_run_a["approval_token_record_preview"] == dry_run_b["approval_token_record_preview"]
    assert dry_run_a["approval_audit_record_preview"] == dry_run_b["approval_audit_record_preview"]
    assert dry_run_a["token_target_paths_preview"] == dry_run_b["token_target_paths_preview"]
    assert dry_run_a["final_token_preflight_checklist"] == dry_run_b["final_token_preflight_checklist"]
    assert dry_run_a["dry_run_id"] == dry_run_b["dry_run_id"]
    assert dry_run_a["approval_token_record_preview"]["preview_only"] is True
    assert dry_run_a["approval_token_record_preview"]["token_issued"] is False
    assert dry_run_a["approval_token_record_preview"]["approved"] is False
    assert dry_run_a["approval_token_record_preview"]["persisted"] is False
    assert dry_run_a["approval_token_record_preview"]["written"] is False
    assert dry_run_a["approval_audit_record_preview"]["preview_only"] is True
    assert dry_run_a["approval_audit_record_preview"]["created_operation_event"] is False
    assert dry_run_a["approval_audit_record_preview"]["writes_approval_audit"] is False
    assert dry_run_a["token_target_paths_preview"]["preview_only"] is True
    assert dry_run_a["token_target_paths_preview"]["writes_token_files"] is False
    assert dry_run_a["token_target_paths_preview"]["writes_approval_audit"] is False


def test_input_plan_is_not_mutated():
    plan = _plan()
    before = deepcopy(plan)

    dry_run = create_human_approval_token_issuance_dry_run(plan)
    dry_run["source_token_issuance_plan_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert plan == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_or_approval_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    dry_run = create_human_approval_token_issuance_dry_run(_plan())
    explanation = explain_human_approval_token_issuance_dry_run(dry_run)
    recommendation = recommend_human_approval_token_issuance_dry_run_action(dry_run)

    assert dry_run["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY
    assert dry_run["policy"]["read_only"] is True
    assert dry_run["policy"]["would_write_memory"] is False
    assert dry_run["policy"]["would_modify_config"] is False
    assert dry_run["policy"]["would_write_graph"] is False
    assert dry_run["policy"]["does_not_create_operation_events"] is True
    assert dry_run["policy"]["creates_token_issuance_dry_run_candidates_only"] is True
    assert dry_run["policy"]["issues_real_approval_tokens"] is False
    assert dry_run["policy"]["persists_approvals"] is False
    assert dry_run["policy"]["creates_real_proposals"] is False
    assert dry_run["policy"]["writes_proposal_files"] is False
    assert dry_run["policy"]["writes_operation_ledger"] is False
    assert dry_run["policy"]["writes_token_files"] is False
    assert dry_run["policy"]["writes_approval_audit"] is False
    assert dry_run["policy"]["applies_proposals"] is False
    assert dry_run["policy"]["submits_to_governance"] is False
    assert dry_run["policy"]["converts_to_real_proposal"] is False
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


def test_dry_run_must_never_be_marked_token_issued_approved_persisted_or_written():
    dry_run = create_human_approval_token_issuance_dry_run(_plan())

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
        mutated = deepcopy(dry_run)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_issuance_dry_run(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_preview_records_must_never_be_marked_written_or_persisted():
    dry_run = create_human_approval_token_issuance_dry_run(_plan())

    for field, forbidden_key in (
        ("approval_token_record_preview", "token_issued"),
        ("approval_token_record_preview", "persisted"),
        ("approval_audit_record_preview", "created_operation_event"),
        ("approval_audit_record_preview", "writes_approval_audit"),
        ("token_target_paths_preview", "writes_token_files"),
        ("token_target_paths_preview", "writes_approval_audit"),
    ):
        mutated = deepcopy(dry_run)
        mutated[field][forbidden_key] = True
        validation = validate_human_approval_token_issuance_dry_run(mutated)
        assert validation["valid"] is False
        assert f"{field}_{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_final_preflight_required_candidates_and_by_block_type_status():
    dry_runs = [
        create_human_approval_token_issuance_dry_run(_plan("procedural_rules")),
        create_human_approval_token_issuance_dry_run(_plan("project_context")),
        create_human_approval_token_issuance_dry_run(_plan("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_issuance_dry_runs(dry_runs)

    assert summary["total"] == 3
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["final_preflight_required_count"] == 2
    assert summary["locked_count"] == 1
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {
        "locked": 1,
        "manual_token_issuance_final_preflight_required": 2,
    }
    assert summary["by_lock_reason"] == {"None": 2, "token_review_rejected": 1}
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_ISSUANCE_DRY_RUN_POLICY
