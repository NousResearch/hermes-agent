from agent.memory_evidence_repair_recovery_completion_audit_preview import (
    CHECK_OBSERVED_RECOVERY_COMPLETION_STATE,
    CHECK_RECOVERY_COMPLETION_RECEIPT_INTEGRITY,
    RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED,
    RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_COMPLETION_AUDIT_STATUS_READY,
    build_evidence_repair_recovery_completion_audit_preview,
    empty_evidence_repair_recovery_completion_audit_preview,
)
from agent.memory_evidence_repair_recovery_completion_receipt import (
    build_evidence_repair_recovery_completion_receipt,
)
from tests.agent.test_memory_evidence_repair_recovery_completion_receipt import (
    _recovery_executor_preview,
)


def _recovery_completion_receipt():
    return build_evidence_repair_recovery_completion_receipt(
        recovery_executor_preview=_recovery_executor_preview(),
        actor="han",
        recovery_reason="manual post-audit recovery complete",
    ).to_dict()


def _observed_recovery_steps(receipt):
    return {
        step_id: {"status": "completed"}
        for step_id in receipt["receipts"][0]["recovery_step_ids"]
    }


def test_ready_recovery_completion_receipt_creates_planned_audit():
    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=_recovery_completion_receipt()
    ).to_dict()

    audit = payload["previews"][0]
    categories = {step["category"] for step in audit["audit_steps"]}
    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_READY
    assert payload["summary"]["recovery_completion_audit_ready"] is True
    assert payload["summary"]["planned_audit_step_count"] == 12
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert audit["id"].startswith("recovery-completion-audit-")
    assert categories == {
        "completion_receipt",
        "token",
        "lock",
        "recovery_step",
        "post_recovery_audit",
        "contamination_guard",
    }
    assert audit["would_verify_completion_receipt_recorded"] is True
    assert audit["would_verify_no_secondary_memory_contamination"] is True


def test_observed_recovery_completion_state_can_pass():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]

    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=receipt,
        observed_recovery_steps=_observed_recovery_steps(receipt),
        recorded_receipts=[{"id": row["id"]}],
        used_token_ids=[row["token_id"]],
        released_lock_ids=[row["lock_id"]],
        post_recovery_audit_status={"status": "pass"},
        contamination_status={"contaminated": False},
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_READY
    assert payload["summary"]["observed_pass_step_count"] == 12
    assert payload["summary"]["observed_fail_step_count"] == 0
    assert payload["previews"][0]["observed_state_supplied"] is True


def test_observed_missing_recovery_token_blocks_audit():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]

    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=receipt,
        observed_recovery_steps=_observed_recovery_steps(receipt),
        recorded_receipts=[{"id": row["id"]}],
        used_token_ids=[],
        released_lock_ids=[row["lock_id"]],
        post_recovery_audit_status={"status": "pass"},
        contamination_status={"contaminated": False},
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED
    assert CHECK_OBSERVED_RECOVERY_COMPLETION_STATE in failed_checks
    assert "investigate_recovery_completion_audit_failures" in payload["required_actions"]


def test_observed_contamination_blocks_audit():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]

    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=receipt,
        observed_recovery_steps=_observed_recovery_steps(receipt),
        recorded_receipts=[{"id": row["id"]}],
        used_token_ids=[row["token_id"]],
        released_lock_ids=[row["lock_id"]],
        post_recovery_audit_status={"status": "pass"},
        contamination_status={"contaminated": True},
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED
    assert CHECK_OBSERVED_RECOVERY_COMPLETION_STATE in failed_checks


def test_tampered_recovery_completion_receipt_blocks_audit():
    receipt = _recovery_completion_receipt()
    receipt["receipts"][0]["receipt_digest"] = "bad"

    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=receipt
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_COMPLETION_RECEIPT_INTEGRITY in failed_checks
    assert "regenerate_recovery_completion_receipt" in payload["required_actions"]


def test_no_action_recovery_completion_receipt_does_not_create_audit():
    payload = build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["recovery_completion_audit_preview_available"] is False


def test_empty_recovery_completion_audit_preview_is_read_only():
    payload = empty_evidence_repair_recovery_completion_audit_preview().to_dict()

    assert payload["status"] == RECOVERY_COMPLETION_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []
