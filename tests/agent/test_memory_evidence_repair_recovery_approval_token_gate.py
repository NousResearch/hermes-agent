from agent.memory_evidence_repair_recovery_approval_token_gate import (
    CHECK_EXECUTION_PREVIEW_DIGEST,
    CHECK_RECOVERY_CONFIRMATION_TEXT,
    CHECK_RECOVERY_EXPIRY,
    CHECK_RECOVERY_REUSE,
    RECOVERY_TOKEN_GATE_STATUS_BLOCKED,
    RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED,
    RECOVERY_TOKEN_GATE_STATUS_VERIFIED,
    build_evidence_repair_recovery_approval_token_gate,
    empty_evidence_repair_recovery_approval_token_gate,
)
from agent.memory_evidence_repair_recovery_human_approval_token import (
    build_evidence_repair_recovery_human_approval_token,
)
from tests.agent.test_memory_evidence_repair_recovery_human_approval_token import (
    _manual_rollback_execution,
    _preparedness_execution,
)


def _recovery_token_report():
    payload = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_manual_rollback_execution(),
        approver="han",
        approval_reason="manual recovery approval",
        expires_in_minutes=20,
    ).to_dict()
    payload["generated_at"] = "2026-05-11T00:00:00+00:00"
    return payload


def test_valid_recovery_token_and_confirmation_are_verified():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        confirmation_text=token["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_VERIFIED
    assert payload["summary"]["manual_recovery_verified"] is True
    assert payload["summary"]["verified_operation_count"] == 1
    assert payload["summary"]["verified_future_mutation_step_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["token_id"] == token["id"]
    assert payload["verified_operation_ids"] == ["dryrun-op-123"]


def test_wrong_confirmation_blocks_recovery_gate():
    token_report = _recovery_token_report()

    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        confirmation_text="wrong confirmation",
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_RECOVERY_CONFIRMATION_TEXT in failed_checks
    assert "provide_exact_recovery_confirmation_text" in payload["required_actions"]


def test_expired_recovery_token_blocks_gate():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        confirmation_text=token["required_confirmation_text"],
        current_time="2026-05-11T00:25:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_RECOVERY_EXPIRY in failed_checks
    assert "regenerate_recovery_human_approval_token" in payload["required_actions"]


def test_used_recovery_token_blocks_gate():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]

    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        confirmation_text=token["required_confirmation_text"],
        used_token_ids=[token["id"]],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_RECOVERY_REUSE in failed_checks


def test_mismatched_execution_preview_blocks_gate():
    token_report = _recovery_token_report()
    token = token_report["tokens"][0]
    execution = _manual_rollback_execution()["previews"][0]
    execution["preview_digest"] = "bad"

    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report,
        recovery_execution=execution,
        confirmation_text=token["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_EXECUTION_PREVIEW_DIGEST in failed_checks


def test_no_action_recovery_token_report_does_not_create_gate():
    token_report = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_preparedness_execution()
    ).to_dict()

    payload = build_evidence_repair_recovery_approval_token_gate(
        recovery_approval_token=token_report
    ).to_dict()

    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED
    assert payload["checks"] == []


def test_empty_recovery_token_gate_is_read_only():
    payload = empty_evidence_repair_recovery_approval_token_gate().to_dict()

    assert payload["status"] == RECOVERY_TOKEN_GATE_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
