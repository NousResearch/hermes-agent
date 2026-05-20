from copy import deepcopy

from agent.memory_evidence_repair_recovery_closure_finalization_readiness_preview import (
    CHECK_RECOVERY_FINALIZATION_REQUIREMENTS,
    CHECK_RECOVERY_FINALIZATION_SOURCE_CHAIN,
    CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_INTEGRITY,
    RECOVERY_CLOSURE_FINALIZATION_DRAFT_STATUS_READY,
    RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED,
    RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_FINALIZATION_STATUS_READY,
    build_evidence_repair_recovery_closure_finalization_readiness_preview,
    empty_evidence_repair_recovery_closure_finalization_readiness_preview,
)
from agent.memory_evidence_repair_recovery_closure_governance_seal_audit_preview import (
    build_evidence_repair_recovery_closure_governance_seal_audit_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_closure_governance_seal_audit_preview import (
    _ready_governance_seal,
)


def _ready_seal_audit():
    return build_evidence_repair_recovery_closure_governance_seal_audit_preview(
        recovery_closure_governance_seal=_ready_governance_seal()
    ).to_dict()


def test_ready_seal_audit_creates_finalization_readiness():
    payload = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit=_ready_seal_audit()
    ).to_dict()

    readiness = payload["readiness"][0]
    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_READY
    assert payload["summary"]["recovery_closure_finalization_ready"] is True
    assert payload["summary"]["readiness_count"] == 1
    assert payload["summary"]["would_allow_final_closure_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert readiness["id"].startswith("recovery-closure-finalization-readiness-")
    assert readiness["status"] == RECOVERY_CLOSURE_FINALIZATION_DRAFT_STATUS_READY
    assert readiness["seal_audit_preview_id"].startswith(
        "recovery-closure-governance-seal-audit-"
    )
    assert readiness["operation_ids"] == ["dryrun-op-123"]
    assert readiness["would_allow_final_closure"] is True
    assert readiness["would_preserve_full_governance_chain"] is True
    assert readiness["would_require_human_finalization"] is True
    assert readiness["would_keep_read_only_until_manual_commit"] is True


def test_blocked_seal_audit_blocks_finalization_readiness():
    payload = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit={
            "status": "blocked",
            "blocking_reasons": ["seal audit blocked"],
            "required_actions": ["fix_seal_audit"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED
    assert payload["readiness"] == []
    assert "fix_seal_audit" in payload["required_actions"]


def test_tampered_seal_audit_digest_blocks_finalization_readiness():
    seal_audit = _ready_seal_audit()
    seal_audit["previews"][0]["audit_digest"] = "bad"

    payload = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit=seal_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_AUDIT_INTEGRITY in failed_checks
    assert (
        "regenerate_recovery_closure_governance_seal_audit_preview"
        in payload["required_actions"]
    )


def test_tampered_source_seal_blocks_finalization_source_chain():
    seal_audit = deepcopy(_ready_seal_audit())
    seal_audit["previews"][0]["source_recovery_closure_governance_seal"][
        "seal_digest"
    ] = "bad"

    payload = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit=seal_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED
    assert CHECK_RECOVERY_FINALIZATION_SOURCE_CHAIN in failed_checks


def test_missing_governance_handoff_flag_blocks_finalization_requirements():
    seal_audit = deepcopy(_ready_seal_audit())
    seal_audit["previews"][0]["would_verify_governance_handoff"] = False

    payload = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit=seal_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_BLOCKED
    assert CHECK_RECOVERY_FINALIZATION_REQUIREMENTS in failed_checks


def test_no_action_seal_audit_does_not_create_finalization_readiness():
    payload = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED
    assert payload["readiness"] == []
    assert (
        payload["summary"]["recovery_closure_finalization_readiness_available"]
        is False
    )


def test_empty_finalization_readiness_preview_is_read_only():
    payload = (
        empty_evidence_repair_recovery_closure_finalization_readiness_preview()
        .to_dict()
    )

    assert payload["status"] == RECOVERY_CLOSURE_FINALIZATION_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["readiness"] == []


def test_finalization_readiness_id_is_stable():
    seal_audit = _ready_seal_audit()

    first = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit=seal_audit
    ).to_dict()
    second = build_evidence_repair_recovery_closure_finalization_readiness_preview(
        recovery_closure_governance_seal_audit=seal_audit
    ).to_dict()

    assert first["readiness"][0]["id"] == second["readiness"][0]["id"]
    assert (
        first["readiness"][0]["finalization_digest"]
        == second["readiness"][0]["finalization_digest"]
    )
