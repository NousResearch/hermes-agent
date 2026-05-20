from agent.memory_evidence_repair_approval_token_gate import (
    build_evidence_repair_approval_token_gate,
)
from agent.memory_evidence_repair_commit_receipt import (
    RECEIPT_STATUS_BLOCKED,
    RECEIPT_STATUS_NO_ACTION_NEEDED,
    RECEIPT_STATUS_READY,
    RECEIPT_TYPE,
    build_evidence_repair_commit_receipt,
    empty_evidence_repair_commit_receipt,
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


def _verified_gate():
    token_report = _token_report()
    return build_evidence_repair_approval_token_gate(
        approval_token=token_report,
        confirmation_text=token_report["tokens"][0]["required_confirmation_text"],
        current_time="2026-05-11T00:10:00+00:00",
    ).to_dict()


def test_verified_gate_creates_read_only_commit_receipt():
    payload = build_evidence_repair_commit_receipt(
        token_gate=_verified_gate(),
        actor="han",
        commit_reason="manual memory evidence repair",
    ).to_dict()

    receipt = payload["receipts"][0]
    assert payload["status"] == RECEIPT_STATUS_READY
    assert payload["summary"]["commit_receipt_available"] is True
    assert payload["summary"]["would_mark_token_used_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert receipt["id"].startswith("commit-receipt-")
    assert receipt["receipt_type"] == RECEIPT_TYPE
    assert receipt["actor"] == "han"
    assert receipt["operation_ids"] == ["dryrun-op-123"]
    assert receipt["ledger_entry_ids"] == ["ledger-123"]
    assert receipt["candidate_ids"] == ["repair-123"]
    assert receipt["used_token_id"] in payload["used_token_ids"]


def test_blocked_gate_does_not_create_receipt():
    gate = _verified_gate()
    gate["status"] = "blocked"
    gate["blocking_reasons"] = ["Approval token has expired."]
    gate["required_actions"] = ["regenerate_human_approval_token"]

    payload = build_evidence_repair_commit_receipt(token_gate=gate).to_dict()

    assert payload["status"] == RECEIPT_STATUS_BLOCKED
    assert payload["receipts"] == []
    assert payload["blocking_reasons"] == ["Approval token has expired."]
    assert payload["required_actions"] == ["regenerate_human_approval_token"]


def test_existing_used_token_blocks_receipt():
    gate = _verified_gate()
    token_id = gate["token_id"]

    payload = build_evidence_repair_commit_receipt(
        token_gate=gate,
        existing_receipts={"used_token_ids": [token_id]},
    ).to_dict()

    assert payload["status"] == RECEIPT_STATUS_BLOCKED
    assert payload["receipts"] == []
    assert "Approval token is already present in the used-token ledger." in payload["blocking_reasons"]
    assert token_id in payload["used_token_ids"]


def test_receipt_id_and_digest_are_stable():
    gate = _verified_gate()
    first = build_evidence_repair_commit_receipt(
        token_gate=gate,
        actor="han",
        commit_reason="manual memory evidence repair",
    ).to_dict()
    second = build_evidence_repair_commit_receipt(
        token_gate=gate,
        actor="han",
        commit_reason="manual memory evidence repair",
    ).to_dict()

    assert first["receipts"][0]["id"] == second["receipts"][0]["id"]
    assert first["receipts"][0]["receipt_digest"] == second["receipts"][0]["receipt_digest"]


def test_no_action_gate_does_not_create_receipt():
    payload = build_evidence_repair_commit_receipt(
        token_gate={"status": "no_action_needed"}
    ).to_dict()

    assert payload["status"] == RECEIPT_STATUS_NO_ACTION_NEEDED
    assert payload["receipts"] == []
    assert payload["summary"]["commit_receipt_available"] is False


def test_empty_receipt_report_is_read_only():
    payload = empty_evidence_repair_commit_receipt().to_dict()

    assert payload["status"] == RECEIPT_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["receipts"] == []
