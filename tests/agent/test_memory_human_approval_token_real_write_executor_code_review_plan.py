from copy import deepcopy

import pytest

from agent.memory_human_approval_token_real_write_executor_code_review_plan import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_ROUTING,
    create_human_approval_token_real_write_executor_code_review_plan,
    explain_human_approval_token_real_write_executor_code_review_plan,
    recommend_human_approval_token_real_write_executor_code_review_plan_action,
    summarize_human_approval_token_real_write_executor_code_review_plans,
    validate_human_approval_token_real_write_executor_code_review_plan,
)
from agent.memory_human_approval_token_real_write_executor_implementation_dry_run import (
    create_human_approval_token_real_write_executor_implementation_dry_run,
)
from tests.agent.test_memory_human_approval_token_real_write_executor_implementation_dry_run import (
    _plan,
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
    "creates_executor_tests",
    "converted_to_real_proposal",
)


def _dry_run(outcome=None, block_type="procedural_rules"):
    return create_human_approval_token_real_write_executor_implementation_dry_run(
        _plan(outcome=outcome, block_type=block_type),
        operator="implementation-dry-run-operator",
    )


def test_valid_implementation_dry_run_creates_code_review_gate_required_plan():
    dry_run = _dry_run()

    plan = create_human_approval_token_real_write_executor_code_review_plan(
        dry_run,
        reviewer="code-review-planner",
    )

    assert (
        plan["plan_kind"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_KIND
    )
    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_GATE_REQUIRED
    )
    assert (
        plan["routing"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_ROUTING
    )
    assert plan["lock_reason"] is None
    assert plan["source_implementation_dry_run_id"] == dry_run["dry_run_id"]
    assert plan["source_implementation_plan_id"] == dry_run[
        "source_implementation_plan_id"
    ]
    assert plan["source_contract_review_outcome_id"] == dry_run[
        "source_contract_review_outcome_id"
    ]
    assert plan["source_contract_id"] == dry_run["source_contract_id"]
    assert plan["source_write_final_gate_id"] == dry_run["source_write_final_gate_id"]
    assert plan["source_write_execution_dry_run_id"] == dry_run[
        "source_write_execution_dry_run_id"
    ]
    assert plan["source_write_execution_plan_id"] == dry_run[
        "source_write_execution_plan_id"
    ]
    assert plan["source_final_confirmation_review_outcome_id"] == dry_run[
        "source_final_confirmation_review_outcome_id"
    ]
    assert plan["source_final_confirmation_request_id"] == dry_run[
        "source_final_confirmation_request_id"
    ]
    assert plan["source_token_write_lock_gate_id"] == dry_run[
        "source_token_write_lock_gate_id"
    ]
    assert plan["source_token_issuance_dry_run_id"] == dry_run[
        "source_token_issuance_dry_run_id"
    ]
    assert plan["source_token_issuance_plan_id"] == dry_run[
        "source_token_issuance_plan_id"
    ]
    assert plan["source_review_outcome_id"] == dry_run["source_review_outcome_id"]
    assert plan["source_request_id"] == dry_run["source_request_id"]
    assert plan["source_gate_id"] == dry_run["source_gate_id"]
    assert plan["source_dry_run_id"] == dry_run["source_dry_run_id"]
    assert plan["source_plan_id"] == dry_run["source_plan_id"]
    assert plan["source_outcome_id"] == dry_run["source_outcome_id"]
    assert plan["source_packet_id"] == dry_run["source_packet_id"]
    assert plan["source_submission_id"] == dry_run["source_submission_id"]
    assert plan["source_draft_id"] == dry_run["source_draft_id"]
    assert plan["source_decision_id"] == dry_run["source_decision_id"]
    assert plan["source_queue_item_id"] == dry_run["source_queue_item_id"]
    assert plan["reviewer"] == "code-review-planner"
    assert plan["outcome"] == "approve_executor_contract"
    assert plan["implementation_dry_run_validation"] == {"valid": True, "errors": []}
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_real_write_executor_code_review_plan(
        plan
    ) == {"valid": True, "errors": []}
    assert plan["next_step_recommendation"]["action"] == (
        "route_to_real_token_write_executor_code_review_gate_without_executor_source_creation"
    )
    assert plan["next_step_recommendation"]["invokes_real_token_write_executor"] is False
    assert plan["next_step_recommendation"]["implements_real_token_write_executor"] is False
    assert plan["next_step_recommendation"]["creates_executor_source_files"] is False
    assert plan["next_step_recommendation"]["creates_executor_tests"] is False


def test_locked_implementation_dry_run_creates_locked_plan():
    dry_run = _dry_run(outcome="request_contract_changes")

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED
    )
    assert plan["lock_reason"] == "contract_review_requested_changes"
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


def test_locked_implementation_dry_run_without_reason_uses_default_lock_reason():
    dry_run = _dry_run(outcome="request_contract_changes")
    dry_run["lock_reason"] = None

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert plan["lock_reason"] == "real_write_executor_implementation_dry_run_locked"
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


def test_invalid_implementation_dry_run_creates_locked_plan():
    dry_run = _dry_run()
    dry_run["dry_run_kind"] = "wrong"

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED
    )
    assert plan["lock_reason"] == "invalid_real_write_executor_implementation_dry_run"
    assert plan["implementation_dry_run_validation"]["valid"] is False
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


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
        (
            "implementation_dry_run_module_boundary_preview",
            "missing_implementation_dry_run_module_boundary_preview",
        ),
        (
            "implementation_dry_run_interface_preview",
            "missing_implementation_dry_run_interface_preview",
        ),
        (
            "implementation_dry_run_idempotency_check_preview",
            "missing_implementation_dry_run_idempotency_check_preview",
        ),
        (
            "implementation_dry_run_filesystem_safety_check_preview",
            "missing_implementation_dry_run_filesystem_safety_check_preview",
        ),
        (
            "implementation_dry_run_audit_check_preview",
            "missing_implementation_dry_run_audit_check_preview",
        ),
        (
            "implementation_dry_run_rollback_check_preview",
            "missing_implementation_dry_run_rollback_check_preview",
        ),
        (
            "implementation_dry_run_test_harness_preview",
            "missing_implementation_dry_run_test_harness_preview",
        ),
        (
            "implementation_dry_run_readiness_checklist",
            "missing_implementation_dry_run_readiness_checklist",
        ),
    ],
)
def test_missing_required_inherited_fields_lock_plan_with_explicit_reasons(
    field,
    reason,
):
    dry_run = _dry_run()
    dry_run.pop(field)

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert plan["lock_reason"] == reason
    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_LOCKED
    )
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


def test_missing_source_evidence_locks_plan():
    dry_run = _dry_run()
    dry_run["source_pattern_ids"] = []
    dry_run["source_fact_ids"] = []

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert plan["lock_reason"] == "missing_source_evidence"
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


def test_preview_integrity_failed_locks_plan():
    dry_run = _dry_run()
    dry_run["approval_token_record_preview"]["token_issued"] = True

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert plan["lock_reason"] == "preview_integrity_failed"
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


def test_implementation_dry_run_integrity_failed_locks_plan():
    dry_run = _dry_run()
    dry_run["implementation_dry_run_interface_preview"][0]["implemented_in_v0_1"] = True

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)

    assert plan["lock_reason"] == "implementation_dry_run_integrity_failed"
    assert plan["implementation_dry_run_validation"]["valid"] is False
    assert plan["code_review_plan_validation"] == {"valid": True, "errors": []}


def test_code_review_scope_is_deterministic_and_does_not_create_executor_code():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert plan_a["code_review_scope"] == plan_b["code_review_scope"]
    assert plan_a["code_review_scope"]["creates_executor_source_files"] is False
    assert plan_a["code_review_scope"]["creates_executor_tests"] is False
    assert plan_a["code_review_scope"]["implements_real_token_write_executor"] is False
    assert plan_a["code_review_scope"]["invokes_real_token_write_executor"] is False


def test_code_review_required_files_are_deterministic_and_not_created():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert plan_a["code_review_required_files"] == plan_b["code_review_required_files"]
    assert all(
        file_plan["create_in_v0_1"] is False
        for file_plan in plan_a["code_review_required_files"]
    )
    assert all(
        file_plan["contains_executor_code_in_v0_1"] is False
        for file_plan in plan_a["code_review_required_files"]
    )
    assert all(
        file_plan["writes_files_in_v0_1"] is False
        for file_plan in plan_a["code_review_required_files"]
    )


def test_code_review_static_analysis_checks_are_deterministic():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert (
        plan_a["code_review_static_analysis_checks"]
        == plan_b["code_review_static_analysis_checks"]
    )
    assert {check["id"] for check in plan_a["code_review_static_analysis_checks"]} == {
        "no_executor_source_created_by_plan",
        "public_surface_is_read_only_candidate_builder",
        "no_dynamic_write_dispatch",
        "deterministic_identifiers",
    }


def test_code_review_security_checks_are_deterministic():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert plan_a["code_review_security_checks"] == plan_b["code_review_security_checks"]
    assert all(
        check.get("creates_executor_source_files", False) is False
        for check in plan_a["code_review_security_checks"]
    )


def test_code_review_write_safety_checks_include_no_write_assertions():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert (
        plan_a["code_review_write_safety_checks"]
        == plan_b["code_review_write_safety_checks"]
    )
    checks = {check["id"]: check for check in plan_a["code_review_write_safety_checks"]}
    assert checks["assert_no_token_file_write_in_v0_1"]["writes_token_files"] is False
    assert checks["assert_no_approval_audit_write_in_v0_1"][
        "writes_approval_audit"
    ] is False
    assert checks["assert_no_proposal_file_write_in_v0_1"][
        "writes_proposal_files"
    ] is False
    assert checks["assert_no_operation_ledger_write_in_v0_1"][
        "writes_operation_ledger"
    ] is False
    assert checks["assert_no_operation_ledger_write_in_v0_1"][
        "created_operation_event"
    ] is False


def test_code_review_test_matrix_is_deterministic_and_every_test_has_writes_false():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert plan_a["code_review_test_matrix"] == plan_b["code_review_test_matrix"]
    assert all(test_case["writes"] is False for test_case in plan_a["code_review_test_matrix"])
    assert all(
        test_case["creates_executor_source_files"] is False
        for test_case in plan_a["code_review_test_matrix"]
    )
    assert all(
        test_case["creates_executor_tests"] is False
        for test_case in plan_a["code_review_test_matrix"]
    )


def test_code_review_acceptance_criteria_require_no_executor_source_creation():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert (
        plan_a["code_review_acceptance_criteria"]
        == plan_b["code_review_acceptance_criteria"]
    )
    criteria = {
        criterion["id"]: criterion
        for criterion in plan_a["code_review_acceptance_criteria"]
    }
    assert criteria["no_executor_source_creation_in_v0_1"][
        "creates_executor_source_files"
    ] is False
    assert criteria["no_executor_source_creation_in_v0_1"][
        "creates_executor_tests"
    ] is False


def test_code_review_forbidden_actions_include_no_executor_or_write_actions():
    plan = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    for forbidden in (
        "implement_real_token_write_executor",
        "invoke_real_token_write_executor",
        "create_executor_source_files",
        "create_executor_tests",
        "write_token_files",
        "write_approval_audit_files",
        "write_proposal_files",
        "write_operation_ledger",
    ):
        assert forbidden in plan["code_review_forbidden_actions"]


def test_code_review_plan_checklist_is_deterministic():
    plan_a = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    plan_b = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())

    assert plan_a["code_review_plan_checklist"] == plan_b["code_review_plan_checklist"]
    assert "executor_source_files_not_created" in plan_a["code_review_plan_checklist"]
    assert "executor_tests_not_created" in plan_a["code_review_plan_checklist"]


def test_input_implementation_dry_run_is_not_mutated():
    dry_run = _dry_run()
    before = deepcopy(dry_run)

    plan = create_human_approval_token_real_write_executor_code_review_plan(dry_run)
    plan["source_implementation_dry_run_snapshot"]["payload_preview"]["content"][
        "nested"
    ]["value"] = "changed"

    assert dry_run == before


def test_policy_proves_no_writes_or_real_executor_code_or_invocation(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    plan = create_human_approval_token_real_write_executor_code_review_plan(_dry_run())
    explanation = explain_human_approval_token_real_write_executor_code_review_plan(
        plan
    )
    recommendation = (
        recommend_human_approval_token_real_write_executor_code_review_plan_action(
            plan
        )
    )

    assert (
        plan["policy"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY
    )
    assert plan["policy"]["read_only"] is True
    assert plan["policy"]["would_write_memory"] is False
    assert plan["policy"]["would_modify_config"] is False
    assert plan["policy"]["would_write_graph"] is False
    assert plan["policy"]["does_not_create_operation_events"] is True
    assert plan["policy"][
        "creates_real_write_executor_code_review_plan_candidates_only"
    ] is True
    assert plan["policy"]["invokes_real_token_write_executor"] is False
    assert plan["policy"]["implements_real_token_write_executor"] is False
    assert plan["policy"]["creates_executor_source_files"] is False
    assert plan["policy"]["creates_executor_tests"] is False
    assert plan["policy"]["issues_real_approval_tokens"] is False
    assert plan["policy"]["persists_approvals"] is False
    assert plan["policy"]["creates_real_proposals"] is False
    assert plan["policy"]["writes_proposal_files"] is False
    assert plan["policy"]["writes_operation_ledger"] is False
    assert plan["policy"]["writes_token_files"] is False
    assert plan["policy"]["writes_approval_audit"] is False
    assert plan["policy"]["applies_proposals"] is False
    assert plan["policy"]["submits_to_governance"] is False
    assert plan["policy"]["converts_to_real_proposal"] is False
    for forbidden_key in FORBIDDEN_TRUE_KEYS:
        assert explanation[forbidden_key] is False
    assert recommendation["invokes_real_token_write_executor"] is False
    assert recommendation["implements_real_token_write_executor"] is False
    assert recommendation["creates_executor_source_files"] is False
    assert recommendation["creates_executor_tests"] is False
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


def test_plan_must_never_be_marked_written_or_executor_invoked_or_implemented():
    for forbidden_key in FORBIDDEN_TRUE_KEYS:
        plan = create_human_approval_token_real_write_executor_code_review_plan(
            _dry_run()
        )
        plan[forbidden_key] = True

        validation = validate_human_approval_token_real_write_executor_code_review_plan(
            plan
        )

        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_code_review_gate_required_plans_and_by_block_type_status():
    required_plan = create_human_approval_token_real_write_executor_code_review_plan(
        _dry_run(block_type="procedural_rules")
    )
    locked_plan = create_human_approval_token_real_write_executor_code_review_plan(
        _dry_run(block_type="preference", outcome="request_contract_changes")
    )

    summary = summarize_human_approval_token_real_write_executor_code_review_plans(
        [required_plan, locked_plan]
    )

    assert summary["total"] == 2
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 0
    assert summary["real_token_write_executor_code_review_gate_required_count"] == 1
    assert summary["locked_count"] == 1
    assert summary["by_block_type"] == {"preference": 1, "procedural_rules": 1}
    assert summary["by_status"] == {
        "locked": 1,
        "real_token_write_executor_code_review_gate_required": 1,
    }
    assert (
        summary["policy"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_CODE_REVIEW_PLAN_POLICY
    )
