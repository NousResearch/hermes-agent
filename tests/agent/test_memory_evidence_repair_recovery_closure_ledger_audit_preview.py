from copy import deepcopy

from agent.memory_evidence_repair_recovery_closure_ledger_audit_preview import (
    CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_REQUIREMENTS,
    CHECK_RECOVERY_CLOSURE_LEDGER_INTEGRITY,
    CHECK_RECOVERY_CLOSURE_LEDGER_SOURCE_CHAIN,
    RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY,
    RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED,
    RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY,
    build_evidence_repair_recovery_closure_ledger_audit_preview,
    empty_evidence_repair_recovery_closure_ledger_audit_preview,
)
from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    build_evidence_repair_recovery_closure_ledger_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_closure_ledger_preview import (
    _ready_closure_gate,
)


def _ready_ledger_preview():
    return build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=_ready_closure_gate()
    ).to_dict()


def test_ready_recovery_closure_ledger_creates_audit_preview():
    payload = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=_ready_ledger_preview()
    ).to_dict()

    preview = payload["previews"][0]
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_READY
    assert payload["summary"]["recovery_closure_ledger_audit_ready"] is True
    assert payload["summary"]["preview_count"] == 1
    assert payload["summary"]["audit_step_count"] == 6
    assert payload["summary"]["observed_pass_step_count"] == 6
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert preview["id"].startswith("recovery-closure-ledger-audit-")
    assert preview["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_PREVIEW_STATUS_READY
    assert preview["ledger_entry_id"].startswith("recovery-closure-ledger-")
    assert preview["closure_id"].startswith("recovery-closure-")
    assert preview["operation_ids"] == ["dryrun-op-123"]
    assert preview["would_verify_ledger_entry_integrity"] is True
    assert preview["would_verify_completion_audit_evidence"] is True


def test_blocked_recovery_closure_ledger_blocks_audit_preview():
    payload = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger={
            "status": "blocked",
            "blocking_reasons": ["upstream ledger blocked"],
            "required_actions": ["fix_upstream_ledger"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED
    assert payload["previews"] == []
    assert "fix_upstream_ledger" in payload["required_actions"]


def test_tampered_ledger_entry_digest_blocks_audit_preview():
    ledger = _ready_ledger_preview()
    ledger["entries"][0]["entry_digest"] = "bad"

    payload = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=ledger
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_CLOSURE_LEDGER_INTEGRITY in failed_checks
    assert "regenerate_recovery_closure_ledger_preview" in payload["required_actions"]


def test_tampered_source_completion_audit_blocks_source_chain():
    ledger = deepcopy(_ready_ledger_preview())
    ledger["entries"][0]["source_recovery_closure"][
        "source_recovery_completion_audit_preview"
    ]["observed_state_supplied"] = False

    payload = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=ledger
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_CLOSURE_LEDGER_SOURCE_CHAIN in failed_checks


def test_tampered_milestone_reference_blocks_audit_requirements():
    ledger = deepcopy(_ready_ledger_preview())
    ledger["entries"][0]["milestones"][0]["reference_id"] = "wrong-decision"

    payload = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=ledger
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_REQUIREMENTS in failed_checks


def test_no_action_recovery_closure_ledger_does_not_create_audit_preview():
    payload = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert payload["summary"]["recovery_closure_ledger_audit_preview_available"] is False


def test_empty_recovery_closure_ledger_audit_preview_is_read_only():
    payload = empty_evidence_repair_recovery_closure_ledger_audit_preview().to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []


def test_recovery_closure_ledger_audit_preview_id_is_stable():
    ledger = _ready_ledger_preview()

    first = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=ledger
    ).to_dict()
    second = build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=ledger
    ).to_dict()

    assert first["previews"][0]["id"] == second["previews"][0]["id"]
    assert first["previews"][0]["audit_digest"] == second["previews"][0]["audit_digest"]
