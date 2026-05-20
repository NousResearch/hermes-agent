from agent.memory_evidence_repair_post_commit_audit_preview import (
    build_evidence_repair_post_commit_audit_preview,
)
from agent.memory_evidence_repair_rollback_drill_preview import (
    ROLLBACK_DRILL_STATUS_BLOCKED,
    ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED,
    ROLLBACK_DRILL_STATUS_READY,
    ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE,
    ROLLBACK_DRILL_TRIGGER_PREPAREDNESS,
    build_evidence_repair_rollback_drill_preview,
    empty_evidence_repair_rollback_drill_preview,
)
from tests.agent.test_memory_evidence_repair_post_commit_audit_preview import (
    _executor_preview,
)


def _planned_post_commit_audit():
    return build_evidence_repair_post_commit_audit_preview(
        executor_preview=_executor_preview()
    ).to_dict()


def _failed_post_commit_audit():
    executor = _executor_preview()
    preview = executor["previews"][0]
    return build_evidence_repair_post_commit_audit_preview(
        executor_preview=executor,
        observed_memory_patches={
            "dryrun-op-123": {"applied": True, "patch_digest": "a" * 64}
        },
        recorded_receipts=[{"id": preview["receipt_id"]}],
        used_token_ids=[],
        released_lock_ids=[preview["lock_id"]],
        rollback_status={"status": "snapshot_available"},
    ).to_dict()


def test_planned_post_commit_audit_creates_preparedness_drill():
    payload = build_evidence_repair_rollback_drill_preview(
        post_commit_audit=_planned_post_commit_audit()
    ).to_dict()

    drill = payload["previews"][0]
    actions = {step["action"] for step in drill["steps"]}
    assert payload["status"] == ROLLBACK_DRILL_STATUS_READY
    assert payload["summary"]["rollback_drill_ready"] is True
    assert payload["summary"]["failed_audit_step_count"] == 0
    assert payload["summary"]["future_restore_step_count"] == 0
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert drill["trigger"] == ROLLBACK_DRILL_TRIGGER_PREPAREDNESS
    assert drill["rollback_mode"] == "preparedness_rehearsal"
    assert drill["id"].startswith("rollback-drill-")
    assert "verify_pre_commit_snapshot_or_rollback_plan" in actions
    assert "rerun_post_commit_audit" in actions


def test_failed_post_commit_audit_creates_failure_response_drill():
    payload = build_evidence_repair_rollback_drill_preview(
        post_commit_audit=_failed_post_commit_audit()
    ).to_dict()

    drill = payload["previews"][0]
    actions = [step["action"] for step in drill["steps"]]
    assert payload["status"] == ROLLBACK_DRILL_STATUS_READY
    assert payload["summary"]["failed_audit_step_count"] >= 1
    assert payload["summary"]["future_restore_step_count"] == 1
    assert drill["trigger"] == ROLLBACK_DRILL_TRIGGER_AUDIT_FAILURE
    assert drill["rollback_mode"] == "failure_response"
    assert drill["operation_ids"] == ["dryrun-op-123"]
    assert "isolate_impacted_memory_records" in actions
    assert "restore_from_pre_commit_snapshot_or_rollback_plan" in actions


def test_rollback_plan_can_be_used_as_drill_source():
    payload = build_evidence_repair_rollback_drill_preview(
        post_commit_audit=_failed_post_commit_audit(),
        rollback_plan={
            "steps": [
                {
                    "operation_id": "dryrun-op-123",
                    "patch_digest": "a" * 64,
                }
            ]
        },
    ).to_dict()

    restore_steps = [
        step
        for step in payload["previews"][0]["steps"]
        if step["action"] == "restore_from_pre_commit_snapshot_or_rollback_plan"
    ]
    assert restore_steps
    assert restore_steps[0]["rollback_source"] == "provided_rollback_plan"


def test_blocked_audit_without_failure_context_blocks_drill():
    payload = build_evidence_repair_rollback_drill_preview(
        post_commit_audit={
            "status": "blocked",
            "blocking_reasons": ["executor preview is invalid"],
            "required_actions": ["regenerate_executor_preview"],
        }
    ).to_dict()

    assert payload["status"] == ROLLBACK_DRILL_STATUS_BLOCKED
    assert payload["previews"] == []
    assert "regenerate_executor_preview" in payload["required_actions"]


def test_no_action_post_commit_audit_does_not_create_drill():
    payload = build_evidence_repair_rollback_drill_preview(
        post_commit_audit={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["rollback_drill_preview_available"] is False


def test_empty_rollback_drill_preview_is_read_only():
    payload = empty_evidence_repair_rollback_drill_preview().to_dict()

    assert payload["status"] == ROLLBACK_DRILL_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []
