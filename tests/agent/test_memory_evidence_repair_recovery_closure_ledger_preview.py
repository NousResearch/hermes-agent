from copy import deepcopy

from agent.memory_evidence_repair_recovery_closure_gate import (
    build_evidence_repair_recovery_closure_gate,
)
from agent.memory_evidence_repair_recovery_closure_ledger_preview import (
    CHECK_RECOVERY_CLOSURE_INTEGRITY,
    CHECK_RECOVERY_LIFECYCLE_CHAIN,
    RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY,
    RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE,
    RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED,
    RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED,
    RECOVERY_CLOSURE_LEDGER_STATUS_READY,
    build_evidence_repair_recovery_closure_ledger_preview,
    empty_evidence_repair_recovery_closure_ledger_preview,
)
from tests.agent.test_memory_evidence_repair_recovery_closure_gate import (
    _passing_completion_audit,
    _planned_completion_audit,
)


def _ready_closure_gate():
    return build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit=_passing_completion_audit()
    ).to_dict()


def test_ready_recovery_closure_creates_ledger_entry_preview():
    payload = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=_ready_closure_gate()
    ).to_dict()

    entry = payload["entries"][0]
    stages = {milestone["stage"] for milestone in entry["milestones"]}
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_STATUS_READY
    assert payload["summary"]["recovery_closure_ledger_ready"] is True
    assert payload["summary"]["entry_count"] == 1
    assert payload["summary"]["milestone_count"] == 8
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert entry["id"].startswith("recovery-closure-ledger-")
    assert entry["entry_type"] == RECOVERY_CLOSURE_LEDGER_ENTRY_TYPE
    assert entry["status"] == RECOVERY_CLOSURE_LEDGER_ENTRY_STATUS_READY
    assert entry["closure_id"].startswith("recovery-closure-")
    assert entry["operation_ids"] == ["dryrun-op-123"]
    assert entry["would_record_ledger_entry"] is True
    assert entry["would_preserve_recovery_evidence"] is True
    assert stages == {
        "recovery_decision",
        "recovery_execution_preview",
        "recovery_approval_token",
        "recovery_write_lock",
        "recovery_executor_preview",
        "recovery_completion_receipt",
        "recovery_completion_audit",
        "recovery_closure",
    }


def test_planned_recovery_closure_gate_blocks_ledger_preview():
    closure_gate = build_evidence_repair_recovery_closure_gate(
        recovery_completion_audit=_planned_completion_audit()
    ).to_dict()

    payload = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=closure_gate
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED
    assert payload["entries"] == []
    assert "provide_observed_recovery_completion_state" in payload["required_actions"]


def test_tampered_recovery_closure_digest_blocks_ledger_preview():
    closure_gate = _ready_closure_gate()
    closure_gate["closures"][0]["closure_digest"] = "bad"

    payload = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=closure_gate
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED
    assert CHECK_RECOVERY_CLOSURE_INTEGRITY in failed_checks
    assert "regenerate_recovery_closure_gate" in payload["required_actions"]


def test_unobserved_source_audit_blocks_lifecycle_chain():
    closure_gate = deepcopy(_ready_closure_gate())
    closure_gate["closures"][0]["source_recovery_completion_audit_preview"][
        "observed_state_supplied"
    ] = False

    payload = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=closure_gate
    ).to_dict()

    failed_checks = {check["id"] for check in payload["checks"] if check["status"] == "fail"}
    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_STATUS_BLOCKED
    assert CHECK_RECOVERY_LIFECYCLE_CHAIN in failed_checks
    assert "regenerate_recovery_closure_gate" in payload["required_actions"]


def test_no_action_recovery_closure_gate_does_not_create_ledger_entry():
    payload = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED
    assert payload["entries"] == []
    assert payload["summary"]["recovery_closure_ledger_preview_available"] is False


def test_empty_recovery_closure_ledger_preview_is_read_only():
    payload = empty_evidence_repair_recovery_closure_ledger_preview().to_dict()

    assert payload["status"] == RECOVERY_CLOSURE_LEDGER_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["entries"] == []


def test_recovery_closure_ledger_entry_id_is_stable():
    closure_gate = _ready_closure_gate()

    first = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=closure_gate
    ).to_dict()
    second = build_evidence_repair_recovery_closure_ledger_preview(
        recovery_closure_gate=closure_gate
    ).to_dict()

    assert first["entries"][0]["id"] == second["entries"][0]["id"]
    assert first["entries"][0]["entry_digest"] == second["entries"][0]["entry_digest"]
