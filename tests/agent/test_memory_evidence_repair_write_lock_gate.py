from agent.memory_evidence_repair_approval_token_gate import (
    build_evidence_repair_approval_token_gate,
)
from agent.memory_evidence_repair_commit_receipt import (
    build_evidence_repair_commit_receipt,
)
from agent.memory_evidence_repair_human_approval_token import (
    build_evidence_repair_human_approval_token,
)
from agent.memory_evidence_repair_write_lock_gate import (
    CHECK_ACTIVE_LOCKS,
    CHECK_RECEIPT_INTEGRITY,
    CHECK_TOKEN_REUSE,
    WRITE_LOCK_STATUS_BLOCKED,
    WRITE_LOCK_STATUS_NO_ACTION_NEEDED,
    WRITE_LOCK_STATUS_READY,
    WRITE_LOCK_TYPE,
    build_evidence_repair_write_lock_gate,
    empty_evidence_repair_write_lock_gate,
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


def _verified_gate():
    token_report = build_evidence_repair_human_approval_token(
        dry_run=_ready_dry_run(),
        approver="han",
        approval_reason="final manual approval",
        expires_in_minutes=30,
    ).to_dict()
    token_report["generated_at"] = "2026-05-11T00:00:00+00:00"
    return build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        confirmation_text=token_report["tokens"][0]["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()


def _commit_receipt():
    return build_evidence_repair_commit_receipt(
        token_gate=_verified_gate(),
        actor="han",
        commit_reason="manual memory evidence repair",
    ).to_dict()


def test_ready_receipt_creates_write_lock_draft():
    payload = build_evidence_repair_write_lock_gate(
        commit_receipt=_commit_receipt(),
        lock_owner="han",
        lock_ttl_minutes=15,
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()

    lock = payload["locks"][0]
    assert payload["status"] == WRITE_LOCK_STATUS_READY
    assert payload["summary"]["executor_dry_run_allowed"] is True
    assert payload["summary"]["fail_count"] == 0
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert lock["id"].startswith("write-lock-")
    assert lock["lock_type"] == WRITE_LOCK_TYPE
    assert lock["owner"] == "han"
    assert lock["operation_ids"] == ["dryrun-op-123"]
    assert lock["would_acquire_write_lock"] is True
    assert lock["would_release_after_commit"] is True
    assert lock["expires_at"] == "2026-05-11T00:35:00+00:00"


def test_active_lock_conflict_blocks_write_lock():
    receipt = _commit_receipt()

    payload = build_evidence_repair_write_lock_gate(
        commit_receipt=receipt,
        active_locks=[
            {
                "id": "write-lock-existing",
                "status": "active",
                "scope": "manual_memory_evidence_repair_commit",
                "owner": "other",
                "operation_ids": ["dryrun-op-123"],
                "expires_at": "2026-05-11T00:40:00+00:00",
            }
        ],
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == WRITE_LOCK_STATUS_BLOCKED
    assert CHECK_ACTIVE_LOCKS in failed_checks
    assert payload["summary"]["active_lock_conflict_count"] == 1
    assert payload["active_lock_conflicts"][0]["reason"] == "overlapping_operation_ids"


def test_expired_active_lock_does_not_block_write_lock():
    payload = build_evidence_repair_write_lock_gate(
        commit_receipt=_commit_receipt(),
        active_locks=[
            {
                "id": "write-lock-expired",
                "status": "active",
                "scope": "manual_memory_evidence_repair_commit",
                "operation_ids": ["dryrun-op-123"],
                "expires_at": "2026-05-11T00:10:00+00:00",
            }
        ],
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()

    assert payload["status"] == WRITE_LOCK_STATUS_READY
    assert payload["summary"]["active_lock_conflict_count"] == 0
    assert payload["locks"]


def test_already_used_token_blocks_write_lock():
    receipt = _commit_receipt()
    token_id = receipt["receipts"][0]["token_id"]

    payload = build_evidence_repair_write_lock_gate(
        commit_receipt=receipt,
        already_used_token_ids=[token_id],
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == WRITE_LOCK_STATUS_BLOCKED
    assert CHECK_TOKEN_REUSE in failed_checks
    assert "regenerate_human_approval_token" in payload["required_actions"]


def test_tampered_receipt_digest_blocks_write_lock():
    receipt = _commit_receipt()
    receipt["receipts"][0]["receipt_digest"] = "bad"

    payload = build_evidence_repair_write_lock_gate(
        commit_receipt=receipt,
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == WRITE_LOCK_STATUS_BLOCKED
    assert CHECK_RECEIPT_INTEGRITY in failed_checks
    assert "regenerate_commit_receipt" in payload["required_actions"]


def test_no_action_receipt_does_not_create_write_lock():
    payload = build_evidence_repair_write_lock_gate(
        commit_receipt={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == WRITE_LOCK_STATUS_NO_ACTION_NEEDED
    assert payload["locks"] == []
    assert payload["summary"]["write_lock_available"] is False


def test_empty_write_lock_gate_is_read_only():
    payload = empty_evidence_repair_write_lock_gate().to_dict()

    assert payload["status"] == WRITE_LOCK_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["locks"] == []
