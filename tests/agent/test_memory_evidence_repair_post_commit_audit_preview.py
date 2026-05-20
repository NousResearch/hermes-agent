from agent.memory_evidence_repair_approval_token_gate import (
    build_evidence_repair_approval_token_gate,
)
from agent.memory_evidence_repair_commit_receipt import (
    build_evidence_repair_commit_receipt,
)
from agent.memory_evidence_repair_executor_preview import (
    build_evidence_repair_executor_preview,
)
from agent.memory_evidence_repair_human_approval_token import (
    build_evidence_repair_human_approval_token,
)
from agent.memory_evidence_repair_post_commit_audit_preview import (
    CHECK_EXECUTOR_PREVIEW_INTEGRITY,
    CHECK_OBSERVED_POST_STATE,
    POST_COMMIT_AUDIT_STATUS_BLOCKED,
    POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED,
    POST_COMMIT_AUDIT_STATUS_READY,
    build_evidence_repair_post_commit_audit_preview,
    empty_evidence_repair_post_commit_audit_preview,
)
from agent.memory_evidence_repair_write_lock_gate import (
    build_evidence_repair_write_lock_gate,
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


def _executor_preview():
    token_report = build_evidence_repair_human_approval_token(
        dry_run=_ready_dry_run(),
        approver="han",
        approval_reason="final manual approval",
        expires_in_minutes=30,
    ).to_dict()
    token_report["generated_at"] = "2026-05-11T00:00:00+00:00"
    token_gate = build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        confirmation_text=token_report["tokens"][0]["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()
    receipt = build_evidence_repair_commit_receipt(
        token_gate=token_gate,
        actor="han",
        commit_reason="manual memory evidence repair",
    ).to_dict()
    write_lock = build_evidence_repair_write_lock_gate(
        commit_receipt=receipt,
        lock_owner="han",
        lock_ttl_minutes=15,
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()
    return build_evidence_repair_executor_preview(
        write_lock_gate=write_lock,
        current_time="2026-05-11T00:25:00+00:00",
    ).to_dict()


def test_ready_executor_preview_creates_planned_post_commit_audit():
    payload = build_evidence_repair_post_commit_audit_preview(
        executor_preview=_executor_preview()
    ).to_dict()

    audit = payload["previews"][0]
    categories = {step["category"] for step in audit["audit_steps"]}
    assert payload["status"] == POST_COMMIT_AUDIT_STATUS_READY
    assert payload["summary"]["post_commit_audit_ready"] is True
    assert payload["summary"]["planned_audit_step_count"] == 5
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert audit["id"].startswith("post-commit-audit-")
    assert categories == {"memory_patch", "receipt", "token", "lock", "rollback"}


def test_observed_post_commit_state_can_pass():
    executor = _executor_preview()
    preview = executor["previews"][0]

    payload = build_evidence_repair_post_commit_audit_preview(
        executor_preview=executor,
        observed_memory_patches={
            "dryrun-op-123": {"applied": True, "patch_digest": "a" * 64}
        },
        recorded_receipts=[{"id": preview["receipt_id"]}],
        used_token_ids=[preview["token_id"]],
        released_lock_ids=[preview["lock_id"]],
        rollback_status={"status": "snapshot_available"},
    ).to_dict()

    assert payload["status"] == POST_COMMIT_AUDIT_STATUS_READY
    assert payload["summary"]["observed_pass_step_count"] == 5
    assert payload["summary"]["observed_fail_step_count"] == 0
    assert payload["previews"][0]["observed_state_supplied"] is True


def test_observed_missing_token_blocks_audit():
    executor = _executor_preview()
    preview = executor["previews"][0]

    payload = build_evidence_repair_post_commit_audit_preview(
        executor_preview=executor,
        observed_memory_patches={
            "dryrun-op-123": {"applied": True, "patch_digest": "a" * 64}
        },
        recorded_receipts=[{"id": preview["receipt_id"]}],
        used_token_ids=[],
        released_lock_ids=[preview["lock_id"]],
        rollback_status={"status": "snapshot_available"},
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == POST_COMMIT_AUDIT_STATUS_BLOCKED
    assert CHECK_OBSERVED_POST_STATE in failed_checks
    assert "investigate_post_commit_audit_failures" in payload["required_actions"]


def test_tampered_executor_preview_blocks_audit():
    executor = _executor_preview()
    executor["previews"][0]["preview_digest"] = "bad"

    payload = build_evidence_repair_post_commit_audit_preview(
        executor_preview=executor
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == POST_COMMIT_AUDIT_STATUS_BLOCKED
    assert CHECK_EXECUTOR_PREVIEW_INTEGRITY in failed_checks
    assert "regenerate_executor_preview" in payload["required_actions"]


def test_no_action_executor_preview_does_not_create_audit():
    payload = build_evidence_repair_post_commit_audit_preview(
        executor_preview={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["post_commit_audit_preview_available"] is False


def test_empty_post_commit_audit_preview_is_read_only():
    payload = empty_evidence_repair_post_commit_audit_preview().to_dict()

    assert payload["status"] == POST_COMMIT_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []
