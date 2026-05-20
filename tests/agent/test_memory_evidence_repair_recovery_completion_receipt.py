from agent.memory_evidence_repair_recovery_completion_receipt import (
    RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED,
    RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_COMPLETION_RECEIPT_STATUS_READY,
    RECOVERY_COMPLETION_RECEIPT_TYPE,
    build_evidence_repair_recovery_completion_receipt,
    empty_evidence_repair_recovery_completion_receipt,
)
from agent.memory_evidence_repair_recovery_executor_preview import (
    build_evidence_repair_recovery_executor_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_executor_preview import (
    _recovery_write_lock_gate,
)


def _recovery_executor_preview():
    return build_evidence_repair_recovery_executor_preview(
        recovery_write_lock_gate=_recovery_write_lock_gate(),
        current_time="2026-05-11T00:12:00+00:00",
    ).to_dict()


def test_ready_recovery_executor_preview_creates_completion_receipt():
    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=_recovery_executor_preview(),
        actor="han",
        recovery_reason="manual post-audit recovery complete",
    ).to_dict()

    receipt = payload["receipts"][0]
    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_READY
    assert payload["summary"]["recovery_completion_receipt_available"] is True
    assert payload["summary"]["would_record_receipt_count"] == 1
    assert payload["summary"]["would_mark_recovery_token_used_count"] == 1
    assert payload["summary"]["would_release_recovery_write_lock_count"] == 1
    assert payload["summary"]["would_trigger_recovery_audit_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert receipt["id"].startswith("recovery-completion-receipt-")
    assert receipt["receipt_type"] == RECOVERY_COMPLETION_RECEIPT_TYPE
    assert receipt["actor"] == "han"
    assert receipt["recovery_reason"] == "manual post-audit recovery complete"
    assert receipt["operation_ids"] == ["dryrun-op-123"]
    assert receipt["future_mutation_step_ids"] == [
        "recovery-execution-step-5-apply_manual_rollback_restore"
    ]
    assert receipt["used_recovery_token_id"] in payload["used_recovery_token_ids"]
    assert receipt["released_recovery_lock_id"] in payload["released_recovery_lock_ids"]
    assert receipt["would_trigger_recovery_audit"] is True


def test_blocked_recovery_executor_preview_does_not_create_receipt():
    executor_preview = _recovery_executor_preview()
    executor_preview["status"] = "blocked"
    executor_preview["blocking_reasons"] = ["Recovery write lock expired."]
    executor_preview["required_actions"] = ["regenerate_recovery_write_lock_gate"]

    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=executor_preview
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED
    assert payload["receipts"] == []
    assert payload["blocking_reasons"] == ["Recovery write lock expired."]
    assert payload["required_actions"] == ["regenerate_recovery_executor_preview"]


def test_existing_used_recovery_token_blocks_completion_receipt():
    executor_preview = _recovery_executor_preview()
    token_id = executor_preview["previews"][0]["token_id"]

    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=executor_preview,
        used_token_ids=[token_id],
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED
    assert payload["receipts"] == []
    assert "regenerate_recovery_human_approval_token" in payload["required_actions"]
    assert token_id in payload["used_recovery_token_ids"]


def test_existing_released_recovery_lock_blocks_completion_receipt():
    executor_preview = _recovery_executor_preview()
    lock_id = executor_preview["previews"][0]["lock_id"]

    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=executor_preview,
        released_lock_ids=[lock_id],
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED
    assert payload["receipts"] == []
    assert "regenerate_recovery_write_lock_gate" in payload["required_actions"]
    assert lock_id in payload["released_recovery_lock_ids"]


def test_tampered_recovery_executor_preview_digest_blocks_completion_receipt():
    executor_preview = _recovery_executor_preview()
    executor_preview["previews"][0]["preview_digest"] = "bad"

    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=executor_preview
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_BLOCKED
    assert payload["receipts"] == []
    assert "regenerate_recovery_executor_preview" in payload["required_actions"]


def test_recovery_completion_receipt_id_and_digest_are_stable():
    executor_preview = _recovery_executor_preview()
    first = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=executor_preview,
        actor="han",
        recovery_reason="manual post-audit recovery complete",
    ).to_dict()
    second = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=executor_preview,
        actor="han",
        recovery_reason="manual post-audit recovery complete",
    ).to_dict()

    assert first["receipts"][0]["id"] == second["receipts"][0]["id"]
    assert first["receipts"][0]["receipt_digest"] == second["receipts"][0]["receipt_digest"]


def test_no_action_recovery_executor_preview_does_not_create_receipt():
    payload = build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED
    assert payload["receipts"] == []
    assert payload["summary"]["recovery_completion_receipt_available"] is False


def test_empty_recovery_completion_receipt_is_read_only():
    payload = empty_evidence_repair_recovery_completion_receipt().to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_RECEIPT_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["receipts"] == []
