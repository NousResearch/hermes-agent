from copy import deepcopy

from agent.memory_human_approval_token_issuance_dry_run import create_human_approval_token_issuance_dry_run
from agent.memory_human_approval_token_write_lock_gate import (
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE,
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_KIND,
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED,
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY,
    MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ROUTING,
    create_human_approval_token_write_lock_gate,
    explain_human_approval_token_write_lock_gate,
    recommend_human_approval_token_write_lock_action,
    summarize_human_approval_token_write_lock_gates,
    validate_human_approval_token_write_lock_gate,
)
from tests.agent.test_memory_human_approval_token_issuance_dry_run import _plan


def _dry_run(block_type="procedural_rules", outcome=None, source_pattern_ids=None, source_fact_ids=None):
    return create_human_approval_token_issuance_dry_run(
        _plan(
            block_type=block_type,
            outcome=outcome,
            source_pattern_ids=source_pattern_ids,
            source_fact_ids=source_fact_ids,
        ),
        operator="token-dry-run-operator",
    )


def test_valid_token_issuance_dry_run_creates_eligible_final_human_confirmation_gate():
    dry_run = _dry_run()

    gate = create_human_approval_token_write_lock_gate(dry_run, operator="token-write-lock-operator")

    assert gate["gate_kind"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_KIND
    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ELIGIBLE
    assert gate["routing"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_ROUTING
    assert gate["lock_reason"] is None
    assert gate["source_token_issuance_dry_run_id"] == dry_run["dry_run_id"]
    assert gate["source_token_issuance_plan_id"] == dry_run["source_token_issuance_plan_id"]
    assert gate["source_review_outcome_id"] == dry_run["source_review_outcome_id"]
    assert gate["source_request_id"] == dry_run["source_request_id"]
    assert gate["source_gate_id"] == dry_run["source_gate_id"]
    assert gate["source_dry_run_id"] == dry_run["source_dry_run_id"]
    assert gate["source_plan_id"] == dry_run["source_plan_id"]
    assert gate["source_outcome_id"] == dry_run["source_outcome_id"]
    assert gate["source_packet_id"] == dry_run["source_packet_id"]
    assert gate["source_submission_id"] == dry_run["source_submission_id"]
    assert gate["source_draft_id"] == dry_run["source_draft_id"]
    assert gate["source_decision_id"] == dry_run["source_decision_id"]
    assert gate["source_queue_item_id"] == dry_run["source_queue_item_id"]
    assert gate["operator"] == "token-write-lock-operator"
    assert gate["token_dry_run_validation"] == {"valid": True, "errors": []}
    assert gate["gate_validation"] == {"valid": True, "errors": []}
    assert validate_human_approval_token_write_lock_gate(gate) == {"valid": True, "errors": []}
    assert gate["next_step_recommendation"]["action"] == "request_separate_final_human_confirmation_before_token_write"
    assert gate["next_step_recommendation"]["issues_real_approval_tokens"] is False
    assert gate["next_step_recommendation"]["persists_approvals"] is False
    assert gate["next_step_recommendation"]["creates_real_proposals"] is False
    assert gate["next_step_recommendation"]["writes_operation_ledger"] is False
    assert gate["next_step_recommendation"]["writes_token_files"] is False
    assert gate["next_step_recommendation"]["writes_approval_audit"] is False


def test_locked_token_dry_run_creates_locked_gate():
    dry_run = _dry_run(outcome="reject")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "token_review_rejected"
    assert gate["token_dry_run_validation"] == {"valid": True, "errors": []}
    assert gate["gate_validation"] == {"valid": True, "errors": []}


def test_locked_token_dry_run_without_lock_reason_uses_default_lock_reason():
    dry_run = _dry_run(outcome="reject")
    dry_run["lock_reason"] = None

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "token_issuance_dry_run_locked"


def test_invalid_token_dry_run_creates_locked_gate():
    dry_run = _dry_run()
    dry_run["dry_run_kind"] = "wrong"

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "invalid_token_issuance_dry_run"
    assert gate["token_dry_run_validation"]["valid"] is False
    assert gate["gate_validation"] == {"valid": True, "errors": []}


def test_missing_approval_token_record_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("approval_token_record_preview")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_approval_token_record_preview"


def test_missing_approval_audit_record_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("approval_audit_record_preview")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_approval_audit_record_preview"


def test_missing_token_target_paths_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("token_target_paths_preview")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_token_target_paths_preview"


def test_missing_proposal_record_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("proposal_record_preview")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_proposal_record_preview"


def test_missing_operation_ledger_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("operation_ledger_preview")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_operation_ledger_preview"


def test_missing_target_paths_preview_locks_gate():
    dry_run = _dry_run()
    dry_run.pop("target_paths_preview")

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_target_paths_preview"


def test_missing_source_evidence_locks_gate():
    dry_run = _dry_run(source_pattern_ids=[], source_fact_ids=[])

    gate = create_human_approval_token_write_lock_gate(dry_run)

    assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
    assert gate["lock_reason"] == "missing_source_evidence"


def test_preview_integrity_failed_when_preview_only_false_or_written_created_token_approved_persisted_flags_true():
    fields = (
        "approval_token_record_preview",
        "approval_audit_record_preview",
        "token_target_paths_preview",
        "proposal_record_preview",
        "operation_ledger_preview",
        "target_paths_preview",
    )
    cases = []
    for field in fields:
        preview_only_false = _dry_run()
        preview_only_false[field]["preview_only"] = False
        cases.append(preview_only_false)

        for flag in ("written", "created_real_proposal", "created_operation_event", "token_issued", "approved", "persisted"):
            flagged = _dry_run()
            flagged[field][flag] = True
            cases.append(flagged)

    for dry_run in cases:
        gate = create_human_approval_token_write_lock_gate(dry_run)
        assert gate["gate_status"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_LOCKED
        assert gate["lock_reason"] == "preview_integrity_failed"


def test_write_lock_checklist_is_deterministic():
    gate_a = create_human_approval_token_write_lock_gate(_dry_run())
    gate_b = create_human_approval_token_write_lock_gate(_dry_run())

    assert gate_a["write_lock_checklist"] == gate_b["write_lock_checklist"]
    assert [check["id"] for check in gate_a["write_lock_checklist"]] == [
        "verify_token_dry_run_validation",
        "verify_token_previews_only",
        "verify_no_token_approval_or_write_flags",
        "verify_payload_and_source_evidence",
        "require_separate_final_human_confirmation",
    ]


def test_input_token_dry_run_is_not_mutated():
    dry_run = _dry_run()
    before = deepcopy(dry_run)

    gate = create_human_approval_token_write_lock_gate(dry_run)
    gate["source_token_issuance_dry_run_snapshot"]["payload_preview"]["content"]["nested"]["value"] = "changed"

    assert dry_run == before


def test_policy_proves_no_memory_config_graph_proposal_ledger_token_or_approval_writes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    gate = create_human_approval_token_write_lock_gate(_dry_run())
    explanation = explain_human_approval_token_write_lock_gate(gate)
    recommendation = recommend_human_approval_token_write_lock_action(gate)

    assert gate["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY
    assert gate["policy"]["read_only"] is True
    assert gate["policy"]["would_write_memory"] is False
    assert gate["policy"]["would_modify_config"] is False
    assert gate["policy"]["would_write_graph"] is False
    assert gate["policy"]["does_not_create_operation_events"] is True
    assert gate["policy"]["creates_token_write_lock_candidates_only"] is True
    assert gate["policy"]["issues_real_approval_tokens"] is False
    assert gate["policy"]["persists_approvals"] is False
    assert gate["policy"]["creates_real_proposals"] is False
    assert gate["policy"]["writes_proposal_files"] is False
    assert gate["policy"]["writes_operation_ledger"] is False
    assert gate["policy"]["writes_token_files"] is False
    assert gate["policy"]["writes_approval_audit"] is False
    assert gate["policy"]["applies_proposals"] is False
    assert gate["policy"]["submits_to_governance"] is False
    assert gate["policy"]["converts_to_real_proposal"] is False
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


def test_gate_must_never_be_marked_token_issued_approved_persisted_submitted_written_created_or_converted():
    gate = create_human_approval_token_write_lock_gate(_dry_run())

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
        mutated = deepcopy(gate)
        mutated[forbidden_key] = True
        validation = validate_human_approval_token_write_lock_gate(mutated)
        assert validation["valid"] is False
        assert f"{forbidden_key}_must_be_false_or_absent" in validation["errors"]


def test_summary_counts_locked_eligible_gates_and_by_block_type_status():
    gates = [
        create_human_approval_token_write_lock_gate(_dry_run("procedural_rules")),
        create_human_approval_token_write_lock_gate(_dry_run("project_context")),
        create_human_approval_token_write_lock_gate(_dry_run("procedural_rules", outcome="reject")),
    ]

    summary = summarize_human_approval_token_write_lock_gates(gates)

    assert summary["total"] == 3
    assert summary["locked_count"] == 1
    assert summary["eligible_count"] == 2
    assert summary["valid_count"] == 3
    assert summary["invalid_count"] == 0
    assert summary["by_block_type"] == {"procedural_rules": 2, "project_context": 1}
    assert summary["by_status"] == {
        "eligible_for_final_human_confirmation": 2,
        "locked": 1,
    }
    assert summary["by_lock_reason"] == {"None": 2, "token_review_rejected": 1}
    assert summary["policy"] == MEMORY_HUMAN_APPROVAL_TOKEN_WRITE_LOCK_GATE_POLICY
