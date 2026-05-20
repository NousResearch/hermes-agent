from agent.memory_evidence_repair_recovery_closure_gate import (
    CHECK_RECOVERY_COMPLETION_AUDIT_INTEGRITY,
    CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS,
    RECOVERY_CLOSURE_DRAFT_STATUS_READY,
    RECOVERY_CLOSURE_STATUS_BLOCKED,
    RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_STATUS_READY,
    build_evidence_repair_recovery_closure_gate,
    empty_evidence_repair_recovery_closure_gate,
)
from agent.memory_evidence_repair_recovery_completion_audit_preview import (
    build_evidence_repair_recovery_completion_audit_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_completion_audit_preview import (
    _observed_recovery_steps,
    _recovery_completion_receipt,
)


def _passing_completion_audit():
    receipt = _recovery_completion_receipt()
    row = receipt["receipts"][0]
    return build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=receipt,
        observed_recovery_steps=_observed_recovery_steps(receipt),
        recorded_receipts=[{"id": row["id"]}],
        used_token_ids=[row["token_id"]],
        released_lock_ids=[row["lock_id"]],
        post_recovery_audit_status={"status": "pass"},
        contamination_status={"contaminated": False},
    ).to_dict()


def _planned_completion_audit():
    return build_evidence_repair_recovery_completion_audit_preview(
        recovery_completion_receipt=_recovery_completion_receipt()
    ).to_dict()


def test_passing_completion_audit_creates_recovery_closure_draft():
    payload = build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit=_passing_completion_audit()
    ).to_dict()

    closure = payload["closures"][0]
    assert payload["status"] == RECOVERY_CLOSURE_STATUS_READY
    assert payload["summary"]["recovery_closure_ready"] is True
    assert payload["summary"]["would_close_recovery_loop_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert closure["id"].startswith("recovery-closure-")
    assert closure["status"] == RECOVERY_CLOSURE_DRAFT_STATUS_READY
    assert closure["closure_decision"] == "close_recovery_loop"
    assert closure["operation_ids"] == ["dryrun-op-123"]
    assert closure["would_close_recovery_loop"] is True
    assert closure["would_mark_recovery_resolved"] is True
    assert closure["would_preserve_audit_evidence"] is True


def test_planned_completion_audit_blocks_recovery_closure():
    payload = build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit=_planned_completion_audit()
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_STATUS_BLOCKED
    assert CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS in failed_checks
    assert "provide_observed_recovery_completion_state" in payload["required_actions"]
    assert payload["closures"] == []


def test_failed_completion_audit_step_blocks_recovery_closure():
    audit = _passing_completion_audit()
    audit["previews"][0]["audit_steps"][0]["status"] = "fail"

    payload = build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit=audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_STATUS_BLOCKED
    assert CHECK_RECOVERY_COMPLETION_AUDIT_OBSERVED_PASS in failed_checks
    assert "investigate_recovery_completion_audit_failures" in payload["required_actions"]


def test_tampered_completion_audit_digest_blocks_recovery_closure():
    audit = _passing_completion_audit()
    audit["previews"][0]["audit_digest"] = "bad"

    payload = build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit=audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_STATUS_BLOCKED
    assert CHECK_RECOVERY_COMPLETION_AUDIT_INTEGRITY in failed_checks
    assert "regenerate_recovery_completion_audit_preview" in payload["required_actions"]


def test_no_action_completion_audit_does_not_create_closure():
    payload = build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED
    assert payload["closures"] == []
    assert payload["summary"]["recovery_closure_available"] is False


def test_empty_recovery_closure_gate_is_read_only():
    payload = empty_evidence_repair_recovery_closure_gate().to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["closures"] == []
