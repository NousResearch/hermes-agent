from copy import deepcopy

from agent.memory_evidence_repair_recovery_closure_governance_seal_preview import (
    CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_INTEGRITY,
    CHECK_RECOVERY_GOVERNANCE_SEAL_REQUIREMENTS,
    CHECK_RECOVERY_GOVERNANCE_SEAL_SOURCE_CHAIN,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY,
    RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE,
    build_evidence_repair_recovery_closure_governance_seal_preview,
    empty_evidence_repair_recovery_closure_governance_seal_preview,
)
from agent.memory_evidence_repair_recovery_closure_ledger_audit_preview import (
    build_evidence_repair_recovery_closure_ledger_audit_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_closure_ledger_audit_preview import (
    _ready_ledger_preview,
)


def _ready_ledger_audit():
    return build_evidence_repair_recovery_closure_ledger_audit_preview(
        recovery_closure_ledger=_ready_ledger_preview()
    ).to_dict()


def test_ready_recovery_closure_ledger_audit_creates_governance_seal():
    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=_ready_ledger_audit()
    ).to_dict()

    seal = payload["seals"][0]
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_READY
    assert payload["summary"]["recovery_closure_governance_seal_ready"] is True
    assert payload["summary"]["seal_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert seal["id"].startswith("recovery-closure-governance-seal-")
    assert seal["seal_type"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_TYPE
    assert seal["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_DRAFT_STATUS_READY
    assert seal["operation_ids"] == ["dryrun-op-123"]
    assert seal["would_freeze_recovery_closure"] is True
    assert seal["would_preserve_audit_evidence"] is True
    assert seal["would_enable_governed_handoff"] is True
    assert seal["would_block_unreviewed_mutation"] is True


def test_blocked_ledger_audit_blocks_governance_seal():
    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit={
            "status": "blocked",
            "blocking_reasons": ["ledger audit blocked"],
            "required_actions": ["fix_ledger_audit"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED
    assert payload["seals"] == []
    assert "fix_ledger_audit" in payload["required_actions"]


def test_tampered_ledger_audit_digest_blocks_governance_seal():
    ledger_audit = _ready_ledger_audit()
    ledger_audit["previews"][0]["audit_digest"] = "bad"

    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=ledger_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED
    assert CHECK_RECOVERY_CLOSURE_LEDGER_AUDIT_INTEGRITY in failed_checks
    assert "regenerate_recovery_closure_ledger_audit_preview" in payload["required_actions"]


def test_tampered_source_ledger_entry_blocks_governance_seal_source_chain():
    ledger_audit = deepcopy(_ready_ledger_audit())
    ledger_audit["previews"][0]["source_recovery_closure_ledger_entry"][
        "entry_digest"
    ] = "bad"

    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=ledger_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_SOURCE_CHAIN in failed_checks


def test_failed_ledger_audit_step_blocks_governance_seal_requirements():
    ledger_audit = deepcopy(_ready_ledger_audit())
    ledger_audit["previews"][0]["audit_steps"][0]["status"] = "fail"

    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=ledger_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_SOURCE_CHAIN in failed_checks


def test_missing_governance_flag_blocks_governance_seal_requirements():
    ledger_audit = deepcopy(_ready_ledger_audit())
    ledger_audit["previews"][0]["would_verify_lifecycle_milestones"] = False

    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=ledger_audit
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_BLOCKED
    assert CHECK_RECOVERY_GOVERNANCE_SEAL_REQUIREMENTS in failed_checks


def test_no_action_ledger_audit_does_not_create_governance_seal():
    payload = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED
    assert payload["seals"] == []
    assert payload["summary"]["recovery_closure_governance_seal_available"] is False


def test_empty_governance_seal_preview_is_read_only():
    payload = empty_evidence_repair_recovery_closure_governance_seal_preview().to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_GOVERNANCE_SEAL_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["seals"] == []


def test_governance_seal_id_is_stable():
    ledger_audit = _ready_ledger_audit()

    first = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=ledger_audit
    ).to_dict()
    second = build_evidence_repair_recovery_closure_governance_seal_preview(
        recovery_closure_ledger_audit=ledger_audit
    ).to_dict()

    assert first["seals"][0]["id"] == second["seals"][0]["id"]
    assert first["seals"][0]["seal_digest"] == second["seals"][0]["seal_digest"]
