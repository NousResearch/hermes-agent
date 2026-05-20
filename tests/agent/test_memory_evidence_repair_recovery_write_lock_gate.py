from agent.memory_evidence_repair_recovery_approval_token_gate import (
    build_evidence_repair_recovery_approval_token_gate,
)
from agent.memory_evidence_repair_recovery_human_approval_token import (
    build_evidence_repair_recovery_human_approval_token,
)
from agent.memory_evidence_repair_recovery_write_lock_gate import (
    CHECK_RECOVERY_ACTIVE_LOCKS,
    CHECK_RECOVERY_TOKEN_GATE_INTEGRITY,
    CHECK_RECOVERY_TOKEN_REUSE,
    RECOVERY_LOCK_DRAFT_STATUS_READY,
    RECOVERY_WRITE_LOCK_SCOPE,
    RECOVERY_WRITE_LOCK_STATUS_BLOCKED,
    RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED,
    RECOVERY_WRITE_LOCK_STATUS_READY,
    RECOVERY_WRITE_LOCK_TYPE,
    build_evidence_repair_recovery_write_lock_gate,
    empty_evidence_repair_recovery_write_lock_gate,
)
from tests.agent.test_memory_evidence_repair_recovery_human_approval_token import (
    _manual_rollback_execution,
    _preparedness_execution,
)


def _verified_recovery_token_gate():
    token_report = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_manual_rollback_execution(),
        approver="han",
        approval_reason="manual recovery approval",
        expires_in_minutes=20,
    ).to_dict()
    token_report["generated_at"] = "2026-05-11T00:00:00+00:00"
    token = token_report["tokens"][0]
    return build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        confirmation_text=token["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()


def test_verified_recovery_token_gate_creates_recovery_write_lock_draft():
    token_gate = _verified_recovery_token_gate()

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate,
        lock_owner="han",
        lock_ttl_minutes=15,
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    lock = payload["locks"][0]
    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_READY
    assert payload["summary"]["recovery_executor_preview_allowed"] is True
    assert payload["summary"]["recovery_write_lock_available"] is True
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert lock["id"].startswith("recovery-write-lock-")
    assert lock["lock_type"] == RECOVERY_WRITE_LOCK_TYPE
    assert lock["status"] == RECOVERY_LOCK_DRAFT_STATUS_READY
    assert lock["scope"] == RECOVERY_WRITE_LOCK_SCOPE
    assert lock["owner"] == "han"
    assert lock["token_id"] == token_gate["token_id"]
    assert lock["operation_ids"] == ["dryrun-op-123"]
    assert lock["future_mutation_step_ids"] == [
        "recovery-execution-step-5-apply_manual_rollback_restore"
    ]
    assert lock["would_acquire_write_lock"] is True
    assert lock["would_release_after_recovery"] is True
    assert lock["expires_at"] == "2026-05-11T00:25:00+00:00"


def test_active_recovery_lock_conflict_blocks_gate():
    token_gate = _verified_recovery_token_gate()

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate,
        active_locks=[
            {
                "id": "recovery-lock-active",
                "status": "active",
                "scope": RECOVERY_WRITE_LOCK_SCOPE,
                "owner": "han",
                "operation_ids": ["dryrun-op-123"],
                "expires_at": "2026-05-11T00:20:00+00:00",
            }
        ],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_BLOCKED
    assert CHECK_RECOVERY_ACTIVE_LOCKS in failed_checks
    assert payload["active_lock_conflicts"][0]["reason"] == "overlapping_operation_ids"
    assert payload["locks"] == []


def test_expired_recovery_lock_does_not_block_gate():
    token_gate = _verified_recovery_token_gate()

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate,
        active_locks=[
            {
                "id": "recovery-lock-expired",
                "status": "active",
                "scope": RECOVERY_WRITE_LOCK_SCOPE,
                "operation_ids": ["dryrun-op-123"],
                "expires_at": "2026-05-11T00:09:00+00:00",
            }
        ],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_READY
    assert payload["active_lock_conflicts"] == []
    assert len(payload["locks"]) == 1


def test_used_recovery_token_blocks_write_lock_gate():
    token_gate = _verified_recovery_token_gate()

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate,
        already_used_token_ids=[token_gate["token_id"]],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_BLOCKED
    assert CHECK_RECOVERY_TOKEN_REUSE in failed_checks
    assert "regenerate_recovery_human_approval_token" in payload["required_actions"]


def test_tampered_recovery_token_gate_integrity_blocks_gate():
    token_gate = _verified_recovery_token_gate()
    token_gate["verified_future_mutation_step_ids"] = []

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate,
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_BLOCKED
    assert CHECK_RECOVERY_TOKEN_GATE_INTEGRITY in failed_checks
    assert "regenerate_recovery_approval_token_gate" in payload["required_actions"]


def test_no_action_recovery_token_gate_does_not_create_lock():
    token_report = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_preparedness_execution()
    ).to_dict()
    token_gate = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report
    ).to_dict()

    payload = build_evidence_repair_recovery_write_lock_gate(
        recovery_token_gate=token_gate
    ).to_dict()

    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED
    assert payload["locks"] == []
    assert payload["checks"] == []


def test_empty_recovery_write_lock_gate_is_read_only():
    payload = empty_evidence_repair_recovery_write_lock_gate().to_dict()

    assert payload["status"] == RECOVERY_WRITE_LOCK_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["locks"] == []
