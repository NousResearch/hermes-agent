from agent.memory_evidence_repair_approval_token_gate import (
    build_evidence_repair_approval_token_gate,
)
from agent.memory_evidence_repair_commit_receipt import (
    build_evidence_repair_commit_receipt,
)
from agent.memory_evidence_repair_executor_preview import (
    CHECK_EXECUTION_OPERATIONS,
    CHECK_WRITE_LOCK_EXPIRY,
    CHECK_WRITE_LOCK_INTEGRITY,
    EXECUTOR_PREVIEW_STATUS_BLOCKED,
    EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED,
    EXECUTOR_PREVIEW_STATUS_READY,
    build_evidence_repair_executor_preview,
    empty_evidence_repair_executor_preview,
)
from agent.memory_evidence_repair_human_approval_token import (
    build_evidence_repair_human_approval_token,
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


def _write_lock_gate():
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
    return build_evidence_repair_write_lock_gate(
        commit_receipt=receipt,
        lock_owner="han",
        lock_ttl_minutes=15,
        current_time="2026-05-11T00:20:00+00:00",
    ).to_dict()


def test_ready_write_lock_creates_executor_preview_steps():
    payload = build_evidence_repair_executor_preview(
        write_lock_gate=_write_lock_gate(),
        current_time="2026-05-11T00:25:00+00:00",
    ).to_dict()

    preview = payload["previews"][0]
    steps = preview["steps"]
    assert payload["status"] == EXECUTOR_PREVIEW_STATUS_READY
    assert payload["summary"]["manual_executor_preview_ready"] is True
    assert payload["summary"]["future_write_step_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert preview["id"].startswith("executor-preview-")
    assert preview["would_apply_memory_patches"] is True
    assert preview["would_release_write_lock"] is True
    assert [step["action"] for step in steps] == [
        "verify_write_lock_and_receipt",
        "apply_evidence_metadata_patch",
        "record_commit_receipt",
        "mark_approval_token_used",
        "release_write_lock",
    ]
    assert steps[1]["operation_id"] == "dryrun-op-123"
    assert steps[1]["future_would_write_memory"] is True
    assert steps[-1]["stop_on_failure"] is False


def test_expired_write_lock_blocks_executor_preview():
    payload = build_evidence_repair_executor_preview(
        write_lock_gate=_write_lock_gate(),
        current_time="2026-05-11T00:36:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == EXECUTOR_PREVIEW_STATUS_BLOCKED
    assert CHECK_WRITE_LOCK_EXPIRY in failed_checks
    assert "regenerate_write_lock_gate" in payload["required_actions"]


def test_tampered_write_lock_digest_blocks_executor_preview():
    gate = _write_lock_gate()
    gate["locks"][0]["lock_digest"] = "bad"

    payload = build_evidence_repair_executor_preview(
        write_lock_gate=gate,
        current_time="2026-05-11T00:25:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == EXECUTOR_PREVIEW_STATUS_BLOCKED
    assert CHECK_WRITE_LOCK_INTEGRITY in failed_checks
    assert "regenerate_write_lock_gate" in payload["required_actions"]


def test_mismatched_operation_patch_blocks_executor_preview():
    gate = _write_lock_gate()
    source_operation = (
        gate["locks"][0]["source_receipt"]["source_gate"]["source_dry_run"]["operations"][0]
    )
    source_operation["patch_digest"] = "b" * 64

    payload = build_evidence_repair_executor_preview(
        write_lock_gate=gate,
        current_time="2026-05-11T00:25:00+00:00",
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == EXECUTOR_PREVIEW_STATUS_BLOCKED
    assert CHECK_EXECUTION_OPERATIONS in failed_checks


def test_no_action_write_lock_gate_does_not_create_preview():
    payload = build_evidence_repair_executor_preview(
        write_lock_gate={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["executor_preview_available"] is False


def test_empty_executor_preview_is_read_only():
    payload = empty_evidence_repair_executor_preview().to_dict()

    assert payload["status"] == EXECUTOR_PREVIEW_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []
