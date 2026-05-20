from agent.memory_evidence_repair_approval_token_gate import (
    CHECK_CONFIRMATION_TEXT,
    CHECK_EXPIRY,
    CHECK_PATCH_DIGESTS,
    CHECK_REUSE,
    TOKEN_GATE_STATUS_BLOCKED,
    TOKEN_GATE_STATUS_NO_ACTION_NEEDED,
    TOKEN_GATE_STATUS_VERIFIED,
    build_evidence_repair_approval_token_gate,
    empty_evidence_repair_approval_token_gate,
)
from agent.memory_evidence_repair_human_approval_token import (
    build_evidence_repair_human_approval_token,
)


def _ready_dry_run():
    return {
        "status": "ready_for_manual_commit",
        "summary": {
            "manual_commit_allowed": True,
            "operation_count": 1,
            "has_blocks": False,
        },
        "checks": [{"id": "commit_gate", "status": "pass"}],
        "operations": [
            {
                "id": "dryrun-op-123",
                "ledger_entry_id": "ledger-123",
                "candidate_id": "repair-123",
                "preview_id": "preview-123",
                "provider": "builtin",
                "repair_action": "attach_provenance",
                "patch_digest": "a" * 64,
                "target": {"candidate_id": "repair-123"},
                "evidence_fields": ["source_url"],
            }
        ],
        "blocking_reasons": [],
        "required_actions": [],
    }


def _token_report():
    payload = build_evidence_repair_human_approval_token(
        dry_run=_ready_dry_run(),
        approver="han",
        approval_reason="final manual approval",
        expires_in_minutes=30,
    ).to_dict()
    payload["generated_at"] = "2026-05-11T00:00:00+00:00"
    return payload


def test_exact_confirmation_verifies_token_gate():
    token_report = _token_report()
    confirmation_text = token_report["tokens"][0]["required_confirmation_text"]

    payload = build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        confirmation_text=confirmation_text,
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    assert payload["status"] == TOKEN_GATE_STATUS_VERIFIED
    assert payload["summary"]["manual_commit_verified"] is True
    assert payload["summary"]["fail_count"] == 0
    assert payload["verified_operation_ids"] == ["dryrun-op-123"]
    assert payload["verified_candidate_ids"] == ["repair-123"]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False


def test_wrong_confirmation_blocks_token_gate():
    payload = build_evidence_repair_approval_token_gate(
        approval_token=_token_report(),
        confirmation_text="APPROVE the wrong thing",
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_CONFIRMATION_TEXT in failed_checks
    assert "provide_exact_confirmation_text" in payload["required_actions"]


def test_changed_patch_digest_blocks_token_gate():
    token_report = _token_report()
    dry_run = _ready_dry_run()
    dry_run["operations"][0]["patch_digest"] = "b" * 64
    confirmation_text = token_report["tokens"][0]["required_confirmation_text"]

    payload = build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        dry_run=dry_run,
        confirmation_text=confirmation_text,
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_PATCH_DIGESTS in failed_checks
    assert "regenerate_human_approval_token_for_current_patch_set" in payload["required_actions"]


def test_expired_token_blocks_token_gate():
    token_report = _token_report()
    confirmation_text = token_report["tokens"][0]["required_confirmation_text"]

    payload = build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        confirmation_text=confirmation_text,
        current_time="2026-05-11T00:31:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_EXPIRY in failed_checks
    assert "Approval token has expired." in payload["blocking_reasons"]


def test_used_token_id_blocks_token_gate():
    token_report = _token_report()
    token_id = token_report["tokens"][0]["id"]
    confirmation_text = token_report["tokens"][0]["required_confirmation_text"]

    payload = build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        confirmation_text=confirmation_text,
        used_token_ids=[token_id],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == TOKEN_GATE_STATUS_BLOCKED
    assert CHECK_REUSE in failed_checks
    assert "Approval token is already marked as used." in payload["blocking_reasons"]


def test_empty_token_gate_is_read_only():
    payload = empty_evidence_repair_approval_token_gate().to_dict()

    assert payload["status"] == TOKEN_GATE_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["checks"] == []
