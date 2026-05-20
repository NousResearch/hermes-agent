from agent.memory_evidence_repair_recovery_approval_token_gate import (
    build_evidence_repair_recovery_approval_token_gate,
)
from agent.memory_evidence_repair_recovery_executor_preview import (
    CHECK_RECOVERY_EXECUTION_STEPS,
    CHECK_RECOVERY_WRITE_LOCK_EXPIRY,
    CHECK_RECOVERY_WRITE_LOCK_INTEGRITY,
    RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED,
    RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED,
    RECOVERY_EXECUTOR_PREVIEW_STATUS_READY,
    build_evidence_repair_recovery_executor_preview,
    empty_evidence_repair_recovery_executor_preview,
)
from agent.memory_evidence_repair_recovery_human_approval_token import (
    build_evidence_repair_recovery_human_approval_token,
)
from agent.memory_evidence_repair_recovery_write_lock_gate import (
    build_evidence_repair_recovery_write_lock_gate,
)
from tests.agent.test_memory_evidence_repair_recovery_human_approval_token import (
    _manual_rollback_execution,
)


def _recovery_write_lock_gate():
    token_report = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_manual_rollback_execution(),
        approver="han",
        approval_reason="manual recovery approval",
        expires_in_minutes=20,
    ).to_dict()
    token_report["generated_at"] = "2026-05-11T00:00:00+00:00"
    token = token_report["tokens"][0]
    token_gate = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        confirmation_text=token["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()
    return build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate,
        lock_owner="han",
        lock_ttl_minutes=15,
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()


def test_ready_recovery_write_lock_creates_recovery_executor_preview_steps():
    payload = build_evidence_repair_recovery_executor_preview(
        recovery_write_lock_gate=_recovery_write_lock_gate(),
        current_time="2026-05-11T00:12:00+00:00",
    ).to_dict()

    preview = payload["previews"][0]
    actions = [step["action"] for step in preview["steps"]]
    assert payload["status"] == RECOVERY_EXECUTOR_PREVIEW_STATUS_READY
    assert payload["summary"]["manual_recovery_executor_preview_ready"] is True
    assert payload["summary"]["future_mutation_step_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert preview["id"].startswith("recovery-executor-preview-")
    assert preview["would_execute_manual_recovery"] is True
    assert preview["would_apply_memory_recovery"] is True
    assert preview["would_mark_recovery_token_used"] is True
    assert preview["would_release_recovery_write_lock"] is True
    assert preview["would_rerun_post_commit_audit"] is True
    assert actions[0] == "verify_recovery_write_lock_token_and_execution"
    assert "apply_manual_rollback_restore" in actions
    assert actions[-2:] == ["mark_recovery_token_used", "release_recovery_write_lock"]
    mutation_steps = [
        step for step in preview["steps"] if step["future_would_mutate_memory"]
    ]
    assert mutation_steps[0]["action"] == "apply_manual_rollback_restore"
    assert preview["steps"][-1]["stop_on_failure"] is False


def test_expired_recovery_write_lock_blocks_recovery_executor_preview():
    payload = build_evidence_repair_recovery_executor_preview(
        recovery_write_lock_gate=_recovery_write_lock_gate(),
        current_time="2026-05-11T00:26:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED
    assert CHECK_RECOVERY_WRITE_LOCK_EXPIRY in failed_checks
    assert "regenerate_recovery_write_lock_gate" in payload["required_actions"]


def test_tampered_recovery_write_lock_digest_blocks_recovery_executor_preview():
    gate = _recovery_write_lock_gate()
    gate["locks"][0]["lock_digest"] = "bad"

    payload = build_evidence_repair_recovery_executor_preview(
        recovery_write_lock_gate=gate,
        current_time="2026-05-11T00:12:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED
    assert CHECK_RECOVERY_WRITE_LOCK_INTEGRITY in failed_checks


def test_mismatched_recovery_execution_steps_block_recovery_executor_preview():
    gate = _recovery_write_lock_gate()
    execution = (
        gate["locks"][0]["source_recovery_token_gate"][
            "source_recovery_execution_preview"
        ]
    )
    execution["steps"][0]["id"] = "tampered-step-id"

    payload = build_evidence_repair_recovery_executor_preview(
        recovery_write_lock_gate=gate,
        current_time="2026-05-11T00:12:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_EXECUTOR_PREVIEW_STATUS_BLOCKED
    assert CHECK_RECOVERY_EXECUTION_STEPS in failed_checks


def test_no_action_recovery_write_lock_gate_does_not_create_preview():
    payload = build_evidence_repair_recovery_executor_preview(
        recovery_write_lock_gate={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["recovery_executor_preview_available"] is False


def test_empty_recovery_executor_preview_is_read_only():
    payload = empty_evidence_repair_recovery_executor_preview().to_dict()

    assert payload["status"] == RECOVERY_EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []
