from copy import deepcopy

from agent.memory_evidence_repair_recovery_closure_governance_seal_audit_preview import (
    CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_REQUIREMENTS,
    CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_SOURCE_CHAIN,
    CHECK_RECOVERY_GOVERNANCE_SEAL_INTEGRITY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY,
    build_evidence_repair_recovery_closure_governance_seal_audit_preview,
    empty_evidence_repair_recovery_closure_governance_seal_audit_preview,
)
from agent.memory_evidence_repair_recovery_closure_governance_seal_preview import (
    build_evidence_repair_recovery_closure_governance_seal_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_closure_governance_seal_preview import (
    _ready_ledger_audit,
)


def _ready_governance_seal():
    return build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=_ready_ledger_audit()
    ).to_dict()


def test_ready_governance_seal_creates_audit_preview():
    payload = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=_ready_governance_seal()
    ).to_dict()

    preview = payload["previews"][0]
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_READY
    assert payload["summary"]["recovery_closure_governance_seal_audit_ready"] is True
    assert payload["summary"]["preview_count"] == 1
    assert payload["summary"]["audit_step_count"] == 6
    assert payload["summary"]["observed_pass_step_count"] == 6
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert preview["id"].startswith("recovery-closure-governance-seal-audit-")
    assert preview["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_PREVIEW_STATUS_READY
    assert preview["seal_id"].startswith("recovery-closure-governance-seal-")
    assert preview["operation_ids"] == ["dryrun-op-123"]
    assert preview["would_verify_seal_integrity"] is True
    assert preview["would_verify_governance_handoff"] is True
    assert preview["would_verify_source_audit_evidence"] is True
    assert preview["would_verify_read_only_constraints"] is True


def test_blocked_governance_seal_blocks_audit_preview():
    payload = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal={
            "status": "blocked",
            "blocking_reasons": ["governance seal blocked"],
            "required_actions": ["fix_governance_seal"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED
    assert payload["previews"] == []
    assert "fix_governance_seal" in payload["required_actions"]


def test_tampered_seal_digest_blocks_audit_preview():
    seal_report = _ready_governance_seal()
    seal_report["seals"][0]["seal_digest"] = "bad"

    payload = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=seal_report
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_INTEGRITY in failed_checks
    assert "regenerate_recovery_closure_governance_seal_preview" in payload["required_actions"]


def test_tampered_source_ledger_audit_blocks_source_chain():
    seal_report = deepcopy(_ready_governance_seal())
    seal_report["seals"][0]["source_recovery_closure_ledger_audit_preview"][
        "audit_digest"
    ] = "bad"

    payload = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=seal_report
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_SOURCE_CHAIN in failed_checks


def test_missing_governance_handoff_flag_blocks_audit_requirements():
    seal_report = deepcopy(_ready_governance_seal())
    seal_report["seals"][0]["would_enable_governed_handoff"] = False

    payload = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=seal_report
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_REQUIREMENTS in failed_checks


def test_no_action_governance_seal_does_not_create_audit_preview():
    payload = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["previews"] == []
    assert (
        payload["summary"]["recovery_closure_governance_seal_audit_preview_available"]
        is False
    )


def test_empty_governance_seal_audit_preview_is_read_only():
    payload = empty_evidence_repair_recovery_closure_governance_seal_audit_preview().to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_AUDIT_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["previews"] == []


def test_governance_seal_audit_preview_id_is_stable():
    seal_report = _ready_governance_seal()

    first = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=seal_report
    ).to_dict()
    second = build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=seal_report
    ).to_dict()

    assert first["previews"][0]["id"] == second["previews"][0]["id"]
    assert first["previews"][0]["audit_digest"] == second["previews"][0]["audit_digest"]
