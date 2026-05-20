from agent.memory_evidence_repair_recovery_decision_gate import (
    RECOVERY_DECISION_STATUS_BLOCKED,
    RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_DECISION_STATUS_READY,
    RECOVERY_ROUTE_MANUAL_ROLLBACK,
    RECOVERY_ROUTE_PREPAREDNESS_ONLY,
    build_evidence_repair_recovery_decision_gate,
    empty_evidence_repair_recovery_decision_gate,
)
from agent.memory_evidence_repair_rollback_drill_preview import (
    build_evidence_repair_rollback_drill_preview,
)
from tests.agent.test_memory_evidence_repair_rollback_drill_preview import (
    _failed_post_commit_audit,
    _planned_post_commit_audit,
)


def _preparedness_drill():
    return build_evidence_repair_rollback_drill_preview(
        post_commit_audit=_planned_post_commit_audit()
    ).to_dict()


def _failure_drill():
    return build_evidence_repair_rollback_drill_preview(
        post_commit_audit=_failed_post_commit_audit()
    ).to_dict()


def test_preparedness_drill_creates_preparedness_decision():
    payload = build_evidence_repair_recovery_decision_gate(
        rollback_drill=_preparedness_drill()
    ).to_dict()

    decision = payload["decisions"][0]
    assert payload["status"] == RECOVERY_DECISION_STATUS_READY
    assert payload["summary"]["recovery_decision_ready"] is True
    assert payload["summary"]["preparedness_review_count"] == 1
    assert payload["summary"]["manual_rollback_required_count"] == 0
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert decision["route"] == RECOVERY_ROUTE_PREPAREDNESS_ONLY
    assert decision["priority"] == "p2"
    assert decision["future_would_mutate_memory"] is False
    assert "rollback_drill_kept_read_only" in decision["required_preconditions"]


def test_failure_drill_creates_manual_rollback_decision():
    payload = build_evidence_repair_recovery_decision_gate(
        rollback_drill=_failure_drill()
    ).to_dict()

    decision = payload["decisions"][0]
    assert payload["status"] == RECOVERY_DECISION_STATUS_READY
    assert payload["summary"]["manual_rollback_required_count"] == 1
    assert decision["route"] == RECOVERY_ROUTE_MANUAL_ROLLBACK
    assert decision["priority"] == "p0"
    assert decision["operation_ids"] == ["dryrun-op-123"]
    assert "request_explicit_human_recovery_approval" in decision["recommended_actions"]
    assert "snapshot_or_rollback_plan_verified" in decision["required_preconditions"]
    assert "automatic_memory_rollback" in decision["blocked_actions"]


def test_tampered_drill_with_non_planned_step_blocks_decision():
    drill = _failure_drill()
    drill["previews"][0]["steps"][0]["status"] = "done"

    payload = build_evidence_repair_recovery_decision_gate(
        rollback_drill=drill
    ).to_dict()

    assert payload["status"] == RECOVERY_DECISION_STATUS_BLOCKED
    assert payload["decisions"] == []
    assert "regenerate_rollback_drill_preview" in payload["required_actions"]


def test_blocked_drill_blocks_decision_gate():
    payload = build_evidence_repair_recovery_decision_gate(
        rollback_drill={
            "status": "blocked",
            "blocking_reasons": ["post commit audit unavailable"],
            "required_actions": ["produce_post_commit_audit_preview"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_DECISION_STATUS_BLOCKED
    assert payload["decisions"] == []
    assert "produce_post_commit_audit_preview" in payload["required_actions"]


def test_no_action_drill_does_not_create_decision():
    payload = build_evidence_repair_recovery_decision_gate(
        rollback_drill={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED
    assert payload["decisions"] == []
    assert payload["summary"]["recovery_decision_available"] is False


def test_empty_recovery_decision_gate_is_read_only():
    payload = empty_evidence_repair_recovery_decision_gate().to_dict()

    assert payload["status"] == RECOVERY_DECISION_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["decisions"] == []
