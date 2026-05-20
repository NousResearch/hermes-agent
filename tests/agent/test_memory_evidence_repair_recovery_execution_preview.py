from agent.memory_evidence_repair_recovery_decision_gate import (
    build_evidence_repair_recovery_decision_gate,
)
from agent.memory_evidence_repair_recovery_execution_preview import (
    RECOVERY_EXECUTION_STATUS_BLOCKED,
    RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_EXECUTION_STATUS_READY,
    build_evidence_repair_recovery_execution_preview,
    empty_evidence_repair_recovery_execution_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_decision_gate import (
    _failure_drill,
    _preparedness_drill,
)


def _preparedness_decision():
    return build_evidence_repair_recovery_decision_gate(
        rollback_drill=_preparedness_drill()
    ).to_dict()


def _manual_rollback_decision():
    return build_evidence_repair_recovery_decision_gate(
        rollback_drill=_failure_drill()
    ).to_dict()


def test_preparedness_decision_creates_non_mutating_execution_preview():
    payload = build_evidence_repair_recovery_execution_preview(
        recovery_decision=_preparedness_decision()
    ).to_dict()

    preview = payload["previews"][0]
    actions = {step["action"] for step in preview["steps"]}
    assert payload["status"] == RECOVERY_EXECUTION_STATUS_READY
    assert payload["summary"]["recovery_execution_ready"] is True
    assert payload["summary"]["preparedness_preview_count"] == 1
    assert payload["summary"]["future_mutation_step_count"] == 0
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert preview["future_would_mutate_memory"] is False
    assert preview["route"] == "preparedness_review_only"
    assert "keep_rollback_drill_available" in actions
    assert "rerun_post_commit_audit_after_future_commit" in actions


def test_manual_rollback_decision_creates_ordered_execution_preview():
    payload = build_evidence_repair_recovery_execution_preview(
        recovery_decision=_manual_rollback_decision()
    ).to_dict()

    preview = payload["previews"][0]
    actions = [step["action"] for step in preview["steps"]]
    restore_steps = [
        step for step in preview["steps"] if step["action"] == "apply_manual_rollback_restore"
    ]
    assert payload["status"] == RECOVERY_EXECUTION_STATUS_READY
    assert payload["summary"]["manual_rollback_preview_count"] == 1
    assert payload["summary"]["future_mutation_step_count"] == 1
    assert preview["future_would_mutate_memory"] is True
    assert preview["human_approval_required"] is True
    assert preview["operation_ids"] == ["dryrun-op-123"]
    assert actions == [
        "verify_recovery_decision_gate",
        "preserve_failed_audit_context",
        "isolate_impacted_memory_records",
        "verify_snapshot_or_rollback_plan",
        "apply_manual_rollback_restore",
        "reconcile_receipt_token_lock_state",
        "rerun_post_commit_audit",
    ]
    assert restore_steps
    assert restore_steps[0]["future_would_mutate_memory"] is True
    assert restore_steps[0]["human_approval_required"] is True


def test_tampered_decision_without_controls_blocks_execution_preview():
    decision = _manual_rollback_decision()
    decision["decisions"][0]["blocked_actions"] = []

    payload = build_evidence_repair_recovery_execution_preview(
        recovery_decision=decision
    ).to_dict()

    assert payload["status"] == RECOVERY_EXECUTION_STATUS_BLOCKED
    assert payload["previews"] == []
    assert "regenerate_recovery_decision_gate" in payload["required_actions"]


def test_blocked_decision_gate_blocks_execution_preview():
    payload = build_evidence_repair_recovery_execution_preview(
        recovery_decision={
            "status": "blocked",
            "blocking_reasons": ["rollback drill unavailable"],
            "required_actions": ["produce_ready_rollback_drill_preview"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_EXECUTION_STATUS_BLOCKED
    assert payload["previews"] == []
    assert "produce_ready_rollback_drill_preview" in payload["required_actions"]


def test_no_action_decision_does_not_create_execution_preview():
    payload = build_evidence_repair_recovery_execution_preview(
        recovery_decision={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["recovery_execution_preview_available"] is False


def test_empty_recovery_execution_preview_is_read_only():
    payload = empty_evidence_repair_recovery_execution_preview().to_dict()

    assert payload["status"] == RECOVERY_EXECUTION_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []
