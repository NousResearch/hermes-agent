from copy import deepcopy

import pytest

from agent.memory_human_approval_token_real_write_executor_contract_review_gate import (
    create_human_approval_token_real_write_executor_contract_review_outcome,
)
from agent.memory_human_approval_token_real_write_executor_implementation_plan import (
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED,
    MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_ROUTING,
    create_human_approval_token_real_write_executor_implementation_plan,
    explain_human_approval_token_real_write_executor_implementation_plan,
    recommend_human_approval_token_real_write_executor_implementation_plan_action,
    summarize_human_approval_token_real_write_executor_implementation_plans,
    validate_human_approval_token_real_write_executor_implementation_plan,
)
from tests.agent.test_memory_human_approval_token_real_write_executor_contract_review_gate import (
    _contract,
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
    "converted_to_real_proposal",
)


def _outcome(outcome=None, block_type="procedural_rules"):
    return create_human_approval_token_real_write_executor_contract_review_outcome(
        _contract(block_type=block_type),
        reviewer="contract-reviewer",
        outcome=outcome,
        rationale=f"Explicit {outcome} outcome." if outcome else None,
    )


def test_valid_approve_executor_contract_outcome_creates_implementation_plan_required():
    outcome = _outcome()

    plan = create_human_approval_token_real_write_executor_implementation_plan(
        outcome,
        implementer="implementation-planner",
    )

    assert plan["plan_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_KIND
    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_REQUIRED
    )
    assert (
        plan["routing"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_ROUTING
    )
    assert plan["lock_reason"] is None
    assert plan["source_contract_review_outcome_id"] == outcome["review_outcome_id"]
    assert plan["source_contract_id"] == outcome["source_contract_id"]
    assert plan["source_write_final_gate_id"] == outcome["source_write_final_gate_id"]
    assert plan["source_write_execution_dry_run_id"] == outcome[
        "source_write_execution_dry_run_id"
    ]
    assert plan["source_write_execution_plan_id"] == outcome[
        "source_write_execution_plan_id"
    ]
    assert plan["source_final_confirmation_review_outcome_id"] == outcome[
        "source_final_confirmation_review_outcome_id"
    ]
    assert plan["source_final_confirmation_request_id"] == outcome[
        "source_final_confirmation_request_id"
    ]
    assert plan["source_token_write_lock_gate_id"] == outcome[
        "source_token_write_lock_gate_id"
    ]
    assert plan["source_token_issuance_dry_run_id"] == outcome[
        "source_token_issuance_dry_run_id"
    ]
    assert plan["source_token_issuance_plan_id"] == outcome[
        "source_token_issuance_plan_id"
    ]
    assert plan["source_review_outcome_id"] == outcome["source_review_outcome_id"]
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
    assert plan["implementer"] == "implementation-planner"
    assert plan["outcome"] == "approve_executor_contract"
    assert plan["contract_review_outcome_validation"] == {"valid": True, "errors": []}
    assert plan["plan_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_real_write_executor_implementation_plan(
        plan
    ) == {"valid": True, "errors": []}
    assert plan["next_step_recommendation"]["action"] == (
        "route_to_real_token_write_executor_implementation_dry_run_without_code_or_invocation"
    )
    assert plan["next_step_recommendation"]["invokes_real_token_write_executor"] is False
    assert plan["next_step_recommendation"]["implements_real_token_write_executor"] is False


def test_implementation_plan_id_is_canonical_for_valid_contract_review_outcome():
    plan = create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(),
        implementer="implementation-planner",
    )

    assert plan["plan_id"] == (
        "memory-human-approval-token-real-write-executor-implementation-plan:"
        "v0.1:20b98e1aeed54cf6"
    )
    assert (
        create_human_approval_token_real_write_executor_implementation_plan(
            _outcome(),
            implementer="implementation-planner",
        )["plan_id"]
        == plan["plan_id"]
    )


def test_request_contract_changes_creates_locked_plan():
    plan = create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(outcome="request_contract_changes")
    )

    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
    )
    assert plan["lock_reason"] == "contract_review_requested_changes"
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_reject_contract_creates_locked_plan():
    plan = create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(outcome="reject_contract")
    )

    assert plan["lock_reason"] == "contract_review_rejected"
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_defer_contract_review_creates_locked_plan():
    plan = create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(outcome="defer_contract_review")
    )

    assert plan["lock_reason"] == "contract_review_deferred"
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_invalid_contract_review_outcome_creates_locked_plan():
    outcome = _outcome()
    outcome["review_outcome_kind"] = "wrong"

    plan = create_human_approval_token_real_write_executor_implementation_plan(outcome)

    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
    )
    assert plan["lock_reason"] == "invalid_contract_review_outcome"
    assert plan["contract_review_outcome_validation"]["valid"] is False
    assert plan["plan_validation"] == {"valid": True, "errors": []}


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
    ],
)
def test_missing_required_contract_review_fields_lock_plan(field, reason):
    outcome = _outcome()
    outcome.pop(field)

    plan = create_human_approval_token_real_write_executor_implementation_plan(outcome)

    assert plan["lock_reason"] == reason
    assert (
        plan["plan_status"]
        == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_LOCKED
    )
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_missing_source_evidence_locks_plan():
    outcome = _outcome()
    outcome["source_pattern_ids"] = []
    outcome["source_fact_ids"] = []

    plan = create_human_approval_token_real_write_executor_implementation_plan(outcome)

    assert plan["lock_reason"] == "missing_source_evidence"
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_preview_integrity_failed_locks_plan():
    outcome = _outcome()
    outcome["approval_token_record_preview"]["token_issued"] = True

    plan = create_human_approval_token_real_write_executor_implementation_plan(outcome)

    assert plan["lock_reason"] == "preview_integrity_failed"
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_contract_review_integrity_failed_locks_plan():
    outcome = _outcome()
    outcome["contract_review_checklist"].remove("real_token_write_executor_not_invoked")

    plan = create_human_approval_token_real_write_executor_implementation_plan(outcome)

    assert plan["lock_reason"] == "contract_review_integrity_failed"
    assert plan["contract_review_outcome_validation"]["valid"] is False
    assert plan["plan_validation"] == {"valid": True, "errors": []}


def test_implementation_plan_interfaces_are_deterministic():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert plan_a["implementation_plan_interfaces"] == plan_b["implementation_plan_interfaces"]
    assert all(
        interface["implemented_in_v0_1"] is False
        for interface in plan_a["implementation_plan_interfaces"]
    )
    assert all(
        interface["invoked_in_v0_1"] is False
        for interface in plan_a["implementation_plan_interfaces"]
    )


def test_implementation_plan_files_are_deterministic_and_do_not_create_executor_code():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert plan_a["implementation_plan_files"] == plan_b["implementation_plan_files"]
    assert all(file_plan["create_in_v0_1"] is False for file_plan in plan_a["implementation_plan_files"])
    assert all(
        file_plan["contains_executor_code_in_v0_1"] is False
        for file_plan in plan_a["implementation_plan_files"]
    )


def test_implementation_plan_idempotency_strategy_is_deterministic():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert (
        plan_a["implementation_plan_idempotency_strategy"]
        == plan_b["implementation_plan_idempotency_strategy"]
    )
    assert (
        plan_a["implementation_plan_idempotency_strategy"]["v0_1_effect"]
        == "plan_only_no_idempotency_state_written"
    )


def test_implementation_plan_filesystem_safety_model_is_deterministic():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert (
        plan_a["implementation_plan_filesystem_safety_model"]
        == plan_b["implementation_plan_filesystem_safety_model"]
    )
    assert (
        plan_a["implementation_plan_filesystem_safety_model"]["v0_1_effect"]
        == "no_filesystem_write_or_directory_creation"
    )


def test_implementation_plan_audit_strategy_is_deterministic_and_preview_only():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert (
        plan_a["implementation_plan_audit_strategy"]
        == plan_b["implementation_plan_audit_strategy"]
    )
    assert (
        plan_a["implementation_plan_audit_strategy"]["v0_1_effect"]
        == "no_approval_audit_file_write_and_no_operation_ledger_event"
    )


def test_implementation_plan_rollback_strategy_is_deterministic_and_no_write():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert (
        plan_a["implementation_plan_rollback_strategy"]
        == plan_b["implementation_plan_rollback_strategy"]
    )
    assert (
        plan_a["implementation_plan_rollback_strategy"]["v0_1_effect"]
        == "no_rollback_action_because_no_write_is_performed"
    )


def test_implementation_plan_test_plan_is_deterministic():
    plan_a = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    plan_b = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    assert plan_a["implementation_plan_test_plan"] == plan_b["implementation_plan_test_plan"]
    assert all(test_case["writes"] is False for test_case in plan_a["implementation_plan_test_plan"])


def test_implementation_plan_forbidden_actions_include_no_executor_or_write_actions():
    plan = create_human_approval_token_real_write_executor_implementation_plan(_outcome())

    for forbidden in (
        "implement_real_token_write_executor",
        "invoke_real_token_write_executor",
        "write_token_files",
        "write_approval_audit_files",
        "write_proposal_files",
        "write_operation_ledger",
    ):
        assert forbidden in plan["implementation_plan_forbidden_actions"]


def test_input_contract_review_outcome_is_not_mutated():
    outcome = _outcome()
    before = deepcopy(outcome)

    plan = create_human_approval_token_real_write_executor_implementation_plan(outcome)
    plan["source_contract_review_outcome_snapshot"]["payload_preview"]["content"][
        "nested"
    ]["value"] = "changed"

    assert outcome == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_approval_writes_or_executor(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    plan = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
    explanation = explain_human_approval_token_real_write_executor_implementation_plan(plan)
    recommendation = (
        recommend_human_approval_token_real_write_executor_implementation_plan_action(plan)
    )

    assert plan["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY
    assert plan["policy"]["read_only"] is True
    assert plan["policy"]["would_write_memory"] is False
    assert plan["policy"]["would_modify_config"] is False
    assert plan["policy"]["would_write_graph"] is False
    assert plan["policy"]["does_not_create_operation_events"] is True
    assert plan["policy"]["creates_real_write_executor_implementation_plan_candidates_only"] is True
    assert plan["policy"]["invokes_real_token_write_executor"] is False
    assert plan["policy"]["implements_real_token_write_executor"] is False
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


def test_plan_must_never_be_marked_token_issued_approved_written_or_executor_invoked():
    for forbidden_key in FORBIDDEN_TRUE_KEYS:
        plan = create_human_approval_token_real_write_executor_implementation_plan(_outcome())
        plan[forbidden_key] = True

        validation = validate_human_approval_token_real_write_executor_implementation_plan(
            plan
        )

        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_implementation_plan_required_candidates_and_by_block_type_status():
    required_plan = create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(block_type="procedural_rules")
    )
    locked_plan = create_human_approval_token_real_write_executor_implementation_plan(
        _outcome(block_type="preference", outcome="request_contract_changes")
    )

    summary = summarize_human_approval_token_real_write_executor_implementation_plans(
        [required_plan, locked_plan]
    )

    assert summary["total"] == 2
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 0
    assert summary["real_token_write_executor_implementation_plan_required_count"] == 1
    assert summary["locked_count"] == 1
    assert summary["by_block_type"] == {"preference": 1, "procedural_rules": 1}
    assert summary["by_status"] == {
        "locked": 1,
        "real_token_write_executor_implementation_plan_required": 1,
    }
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_REAL_WRITE_EXECUTOR_IMPLEMENTATION_PLAN_POLICY
