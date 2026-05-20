from agent.memory_evidence_repair_human_approval_token import (
    APPROVAL_TOKEN_SCOPE,
    APPROVAL_TOKEN_TYPE,
    TOKEN_STATUS_BLOCKED,
    TOKEN_STATUS_DRAFT_READY,
    TOKEN_STATUS_NO_ACTION_NEEDED,
    build_evidence_repair_human_approval_token,
    empty_evidence_repair_human_approval_token,
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


def test_ready_dry_run_creates_token_draft():
    payload = build_evidence_repair_human_approval_token(
        dry_run=_ready_dry_run(),
        approver="han",
        approval_reason="final manual approval",
        expires_in_minutes=45,
    ).to_dict()

    token = payload["tokens"][0]
    assert payload["status"] == TOKEN_STATUS_DRAFT_READY
    assert payload["summary"]["token_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert token["id"].startswith("approval-token-")
    assert token["token_type"] == APPROVAL_TOKEN_TYPE
    assert token["scope"] == APPROVAL_TOKEN_SCOPE
    assert token["approver"] == "han"
    assert token["expires_in_minutes"] == 45
    assert token["operation_ids"] == ["dryrun-op-123"]
    assert token["ledger_entry_ids"] == ["ledger-123"]
    assert token["candidate_ids"] == ["repair-123"]
    assert len(token["token_digest"]) == 64
    assert token["one_time_constraints"]["one_time_use"] is True
    assert token["required_confirmation_text"].startswith("APPROVE ")


def test_blocked_dry_run_does_not_create_token():
    dry_run = _ready_dry_run()
    dry_run["status"] = "blocked"
    dry_run["blocking_reasons"] = ["Pre-commit snapshots are still required."]
    dry_run["required_actions"] = ["capture_pre_commit_snapshots"]

    payload = build_evidence_repair_human_approval_token(dry_run=dry_run).to_dict()

    assert payload["status"] == TOKEN_STATUS_BLOCKED
    assert payload["tokens"] == []
    assert payload["blocking_reasons"] == ["Pre-commit snapshots are still required."]
    assert payload["required_actions"] == ["capture_pre_commit_snapshots"]


def test_no_action_dry_run_does_not_create_token():
    payload = build_evidence_repair_human_approval_token(
        dry_run={"status": "no_action_needed", "operations": []}
    ).to_dict()

    assert payload["status"] == TOKEN_STATUS_NO_ACTION_NEEDED
    assert payload["tokens"] == []
    assert payload["summary"]["no_action_count"] == 1


def test_token_id_and_digest_are_stable():
    first = build_evidence_repair_human_approval_token(
        dry_run=_ready_dry_run(),
        approver="han",
        approval_reason="final manual approval",
    ).to_dict()
    second = build_evidence_repair_human_approval_token(
        dry_run=_ready_dry_run(),
        approver="han",
        approval_reason="final manual approval",
    ).to_dict()

    assert first["tokens"][0]["id"] == second["tokens"][0]["id"]
    assert first["tokens"][0]["token_digest"] == second["tokens"][0]["token_digest"]


def test_empty_token_report_is_read_only():
    payload = empty_evidence_repair_human_approval_token().to_dict()

    assert payload["status"] == TOKEN_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["tokens"] == []
