from copy import deepcopy

import pytest

from agent.memory_human_approval_token_real_write_executor_implementation_dry_run import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_ROUTING,
    create_human_approval_token_real_write_executor_implementation_dry_run,
    explain_human_approval_token_real_write_executor_implementation_dry_run,
    recommend_human_approval_token_real_write_executor_implementation_dry_run_action,
    summarize_human_approval_token_real_write_executor_implementation_dry_runs,
    validate_human_approval_token_real_write_executor_implementation_dry_run,
)
from agent.memory_human_approval_token_real_write_executor_implementation_plan import (
    create_human_approval_token_real_write_executor_implementation_plan,
)
from tests.agent.test_memory_human_approval_token_real_write_executor_implementation_plan import (
    _outcome,
)


FORBIDDEN_TRUE_KEYS = (
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
    "invokes_real_token_write_executor",
    "implements_real_token_write_executor",
    "creates_executor_source_files",
    "converted_to_real_proposal",
)


def _plan(outcome=None, block_type="procedural_rules"):
    return create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(outcome=outcome, block_type=block_type),
        implementer="implementation-planner",
    )


def test_valid_implementation_plan_creates_code_review_plan_required_dry_run():
    plan = _plan()

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan,
        operator="implementation-dry-run-operator",
    )

    assert (
        dry_run["dry_run_kind"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_KIND
    )
    assert (
        dry_run["dry_run_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_REQUIRED
    )
    assert (
        dry_run["routing"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_ROUTING
    )
    assert dry_run["lock_reason"] is None
    assert dry_run["source_implementation_plan_id"] == plan["plan_id"]
    assert dry_run["source_contract_review_outcome_id"] == plan[
        "source_contract_review_outcome_id"
    ]
    assert dry_run["source_contract_id"] == plan["source_contract_id"]
    assert dry_run["source_write_final_gate_id"] == plan["source_write_final_gate_id"]
    assert dry_run["source_write_execution_dry_run_id"] == plan[
        "source_write_execution_dry_run_id"
    ]
    assert dry_run["source_write_execution_plan_id"] == plan[
        "source_write_execution_plan_id"
    ]
    assert dry_run["source_final_confirmation_review_outcome_id"] == plan[
        "source_final_confirmation_review_outcome_id"
    ]
    assert dry_run["source_final_confirmation_request_id"] == plan[
        "source_final_confirmation_request_id"
    ]
    assert dry_run["source_token_write_lock_gate_id"] == plan[
        "source_token_write_lock_gate_id"
    ]
    assert dry_run["source_token_issuance_dry_run_id"] == plan[
        "source_token_issuance_dry_run_id"
    ]
    assert dry_run["source_token_issuance_plan_id"] == plan[
        "source_token_issuance_plan_id"
    ]
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
    assert dry_run["operator"] == "implementation-dry-run-operator"
    assert dry_run["outcome"] == "approve_executor_contract"
    assert dry_run["implementation_plan_validation"] == {"valid": True, "errors": []}
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_real_write_executor_implementation_dry_run(
        dry_run
    ) == {"valid": True, "errors": []}
    assert dry_run["next_step_recommendation"]["action"] == (
        "route_to_real_token_write_executor_code_review_plan_without_executor_code_or_invocation"
    )
    assert dry_run["next_step_recommendation"]["invokes_real_token_write_executor"] is False
    assert dry_run["next_step_recommendation"]["implements_real_token_write_executor"] is False
    assert dry_run["next_step_recommendation"]["creates_executor_source_files"] is False


def test_locked_implementation_plan_creates_locked_dry_run():
    plan = _plan(outcome="request_contract_changes")

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert (
        dry_run["dry_run_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
    )
    assert dry_run["lock_reason"] == "contract_review_requested_changes"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_locked_implementation_plan_without_reason_uses_default_lock_reason():
    plan = _plan(outcome="request_contract_changes")
    plan["lock_reason"] = None

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert dry_run["lock_reason"] == "real_write_executor_implementation_plan_locked"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_invalid_implementation_plan_creates_locked_dry_run():
    plan = _plan()
    plan["plan_kind"] = "wrong"

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert (
        dry_run["dry_run_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
    )
    assert dry_run["lock_reason"] == "invalid_real_write_executor_implementation_plan"
    assert dry_run["implementation_plan_validation"]["valid"] is False
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("approval_token_record_preview", "missing_approval_token_record_preview"),
        ("approval_audit_record_preview", "missing_approval_audit_record_preview"),
        ("token_target_paths_preview", "missing_token_target_paths_preview"),
        ("proposal_record_preview", "missing_proposal_record_preview"),
        ("operation_ledger_preview", "missing_operation_ledger_preview"),
        ("target_paths_preview", "missing_target_paths_preview"),
        (
            "approval_token_write_payload_preview",
            "missing_approval_token_write_payload_preview",
        ),
        (
            "approval_audit_write_payload_preview",
            "missing_approval_audit_write_payload_preview",
        ),
        ("token_write_target_paths_preview", "missing_token_write_target_paths_preview"),
        ("executor_contract_inputs", "missing_executor_contract_inputs"),
        ("executor_hard_lock_checks", "missing_executor_hard_lock_checks"),
        ("executor_audit_fields", "missing_executor_audit_fields"),
        ("executor_rollback_rules", "missing_executor_rollback_rules"),
        ("executor_forbidden_side_effects", "missing_executor_forbidden_side_effects"),
        ("executor_contract_checklist", "missing_executor_contract_checklist"),
        ("contract_review_checklist", "missing_contract_review_checklist"),
        ("implementation_plan_interfaces", "missing_implementation_plan_interfaces"),
        ("implementation_plan_files", "missing_implementation_plan_files"),
        (
            "implementation_plan_idempotency_strategy",
            "missing_implementation_plan_idempotency_strategy",
        ),
        (
            "implementation_plan_filesystem_safety_model",
            "missing_implementation_plan_filesystem_safety_model",
        ),
        (
            "implementation_plan_audit_strategy",
            "missing_implementation_plan_audit_strategy",
        ),
        (
            "implementation_plan_rollback_strategy",
            "missing_implementation_plan_rollback_strategy",
        ),
        ("implementation_plan_test_plan", "missing_implementation_plan_test_plan"),
        (
            "implementation_plan_forbidden_actions",
            "missing_implementation_plan_forbidden_actions",
        ),
    ],
)
def test_missing_required_implementation_plan_fields_lock_dry_run(field, reason):
    plan = _plan()
    plan.pop(field)

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert dry_run["lock_reason"] == reason
    assert (
        dry_run["dry_run_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_LOCKED
    )
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_missing_source_evidence_locks_dry_run():
    plan = _plan()
    plan["source_pattern_ids"] = []
    plan["source_fact_ids"] = []

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert dry_run["lock_reason"] == "missing_source_evidence"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_preview_integrity_failed_locks_dry_run():
    plan = _plan()
    plan["approval_token_record_preview"]["token_issued"] = True

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert dry_run["lock_reason"] == "preview_integrity_failed"
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_implementation_plan_integrity_failed_locks_dry_run():
    plan = _plan()
    plan["implementation_plan_interfaces"][0]["implemented_in_v0_1"] = True

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )

    assert dry_run["lock_reason"] == "implementation_plan_integrity_failed"
    assert dry_run["implementation_plan_validation"]["valid"] is False
    assert dry_run["dry_run_validation"] == {"valid": True, "errors": []}


def test_implementation_dry_run_module_boundary_preview_is_deterministic_and_no_code():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_module_boundary_preview"]
        == dry_run_b["implementation_dry_run_module_boundary_preview"]
    )
    preview = dry_run_a["implementation_dry_run_module_boundary_preview"]
    assert preview["module_created_in_v0_1"] is False
    assert preview["creates_executor_source_files"] is False
    assert preview["implements_real_token_write_executor"] is False


def test_implementation_dry_run_interface_preview_is_deterministic_and_no_invocation():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_interface_preview"]
        == dry_run_b["implementation_dry_run_interface_preview"]
    )
    assert all(
        interface["implemented_in_v0_1"] is False
        for interface in dry_run_a["implementation_dry_run_interface_preview"]
    )
    assert all(
        interface["invoked_in_v0_1"] is False
        for interface in dry_run_a["implementation_dry_run_interface_preview"]
    )


def test_implementation_dry_run_idempotency_check_preview_is_deterministic():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_idempotency_check_preview"]
        == dry_run_b["implementation_dry_run_idempotency_check_preview"]
    )
    assert (
        dry_run_a["implementation_dry_run_idempotency_check_preview"]["v0_1_effect"]
        == "no_idempotency_state_written"
    )


def test_implementation_dry_run_filesystem_safety_check_preview_is_preview_only():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_filesystem_safety_check_preview"]
        == dry_run_b["implementation_dry_run_filesystem_safety_check_preview"]
    )
    preview = dry_run_a["implementation_dry_run_filesystem_safety_check_preview"]
    assert preview["preview_only"] is True
    assert preview["writes_token_files"] is False
    assert preview["writes_approval_audit"] is False
    assert preview["creates_executor_source_files"] is False


def test_implementation_dry_run_audit_check_preview_is_preview_only():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_audit_check_preview"]
        == dry_run_b["implementation_dry_run_audit_check_preview"]
    )
    preview = dry_run_a["implementation_dry_run_audit_check_preview"]
    assert preview["preview_only"] is True
    assert preview["created_operation_event"] is False
    assert preview["writes_approval_audit"] is False
    assert preview["writes_operation_ledger"] is False


def test_implementation_dry_run_rollback_check_preview_is_no_write():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_rollback_check_preview"]
        == dry_run_b["implementation_dry_run_rollback_check_preview"]
    )
    preview = dry_run_a["implementation_dry_run_rollback_check_preview"]
    assert preview["v0_1_effect"] == "no_rollback_action_because_no_write_is_performed"
    assert preview["writes_token_files"] is False
    assert preview["writes_approval_audit"] is False


def test_implementation_dry_run_test_harness_preview_does_not_create_files():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_test_harness_preview"]
        == dry_run_b["implementation_dry_run_test_harness_preview"]
    )
    assert all(
        test_case["creates_files"] is False
        for test_case in dry_run_a["implementation_dry_run_test_harness_preview"]
    )
    assert all(
        test_case["writes"] is False
        for test_case in dry_run_a["implementation_dry_run_test_harness_preview"]
    )


def test_implementation_dry_run_readiness_checklist_is_deterministic():
    dry_run_a = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    dry_run_b = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )

    assert (
        dry_run_a["implementation_dry_run_readiness_checklist"]
        == dry_run_b["implementation_dry_run_readiness_checklist"]
    )
    assert "executor_source_files_not_created" in dry_run_a[
        "implementation_dry_run_readiness_checklist"
    ]


def test_input_implementation_plan_is_not_mutated():
    plan = _plan()
    before = deepcopy(plan)

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        plan
    )
    dry_run["source_implementation_plan_snapshot"]["payload_preview"]["content"][
        "nested"
    ]["value"] = "changed"

    assert plan == before


def test_policy_proves_no_writes_or_real_executor_code_or_invocation(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan()
    )
    explanation = explain_human_approval_token_real_write_executor_implementation_dry_run(
        dry_run
    )
    recommendation = (
        recommend_human_approval_token_real_write_executor_implementation_dry_run_action(
            dry_run
        )
    )

    assert (
        dry_run["policy"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY
    )
    assert dry_run["policy"]["read_only"] is True
    assert dry_run["policy"]["would_write_memory"] is False
    assert dry_run["policy"]["would_modify_config"] is False
    assert dry_run["policy"]["would_write_graph"] is False
    assert dry_run["policy"]["does_not_create_operation_events"] is True
    assert dry_run["policy"][
        "creates_real_write_executor_implementation_dry_run_candidates_only"
    ] is True
    assert dry_run["policy"]["invokes_real_token_write_executor"] is False
    assert dry_run["policy"]["implements_real_token_write_executor"] is False
    assert dry_run["policy"]["creates_executor_source_files"] is False
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
    for forbidden_key in FORBIDDEN_TRUE_KEYS:
        assert explanation[forbidden_key] is False
    assert recommendation["invokes_real_token_write_executor"] is False
    assert recommendation["implements_real_token_write_executor"] is False
    assert recommendation["creates_executor_source_files"] is False
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
    assert not (
        hermes_home
        / "agent"
        / "memory_human_approval_token_real_write_executor.py"
    ).exists()


def test_dry_run_must_never_be_marked_written_or_executor_invoked_or_implemented():
    for forbidden_key in FORBIDDEN_TRUE_KEYS:
        dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
            _plan()
        )
        dry_run[forbidden_key] = True

        validation = validate_human_approval_token_real_write_executor_implementation_dry_run(
            dry_run
        )

        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_code_review_plan_required_dry_runs_and_by_block_type_status():
    required_dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan(block_type="procedural_rules")
    )
    locked_dry_run = create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan(block_type="preference", outcome="request_contract_changes")
    )

    summary = summarize_human_approval_token_real_write_executor_implementation_dry_runs(
        [required_dry_run, locked_dry_run]
    )

    assert summary["total"] == 2
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 0
    assert summary["real_token_write_executor_code_review_plan_required_count"] == 1
    assert summary["locked_count"] == 1
    assert summary["by_block_type"] == {"preference": 1, "procedural_rules": 1}
    assert summary["by_status"] == {
        "locked": 1,
        "real_token_write_executor_code_review_plan_required": 1,
    }
    assert (
        summary["policy"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_DRY_RUN_POLICY
    )
