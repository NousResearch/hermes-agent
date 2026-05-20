from agent.memory_evidence_repair_commit_ledger import (
    LEDGER_EVENT_TYPE,
    LEDGER_STATUS_DRAFT,
    OUTCOME_ALLOWED,
    OUTCOME_BLOCKED,
    OUTCOME_NEEDS_CONFIRMATION,
    build_evidence_repair_commit_ledger,
    empty_evidence_repair_commit_ledger,
)


def _allow_decision():
    return {
        "id": "gate-preview-123",
        "preview_id": "preview-123",
        "candidate_id": "repair-123",
        "decision": "allow_manual_commit",
        "provider": "builtin",
        "priority": "high",
        "repair_action": "attach_provenance",
        "reasons": [],
        "required_actions": [],
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_patch": {
            "source_url": "https://example.com/source",
            "observed_at": "2026-05-11T00:00:00Z",
            "verification_signal": "manual_verified",
        },
        "memory_patch_preview": {
            "operation": "merge_evidence_metadata",
            "set": {"evidence": {}},
        },
        "conflict_risk": "low",
    }


def test_ledger_entry_from_allow_decision():
    draft = build_evidence_repair_commit_ledger(
        decisions=[_allow_decision()],
        ledger_actor="tester",
        ledger_reason="audit before manual memory write",
    )

    payload = draft.to_dict()
    entry = payload["entries"][0]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["allow_count"] == 1
    assert entry["id"].startswith("ledger-")
    assert entry["event_type"] == LEDGER_EVENT_TYPE
    assert entry["status"] == LEDGER_STATUS_DRAFT
    assert entry["outcome"] == OUTCOME_ALLOWED
    assert entry["manual_commit_allowed"] is True
    assert entry["blocked"] is False
    assert entry["requires_followup"] is False
    assert entry["evidence_fields"] == [
        "source_url",
        "observed_at",
        "verification_signal",
    ]
    assert len(entry["patch_digest"]) == 64
    assert entry["metadata"]["ledger_actor"] == "tester"


def test_ledger_entry_from_blocked_decision_requires_followup():
    decision = _allow_decision()
    decision["decision"] = "block_commit"
    decision["reasons"] = ["missing_evidence"]
    decision["required_actions"] = ["complete_required_evidence"]

    payload = build_evidence_repair_commit_ledger(decisions=[decision]).to_dict()
    entry = payload["entries"][0]
    assert entry["outcome"] == OUTCOME_BLOCKED
    assert entry["manual_commit_allowed"] is False
    assert entry["blocked"] is True
    assert entry["requires_followup"] is True
    assert payload["summary"]["blocked_count"] == 1
    assert payload["summary"]["followup_count"] == 1


def test_ledger_entry_from_needs_confirmation_decision():
    decision = _allow_decision()
    decision["decision"] = "needs_user_confirmation"
    decision["reasons"] = ["missing_user_confirmation"]
    decision["required_actions"] = ["obtain_explicit_user_confirmation"]

    payload = build_evidence_repair_commit_ledger(decisions=[decision]).to_dict()
    entry = payload["entries"][0]
    assert entry["outcome"] == OUTCOME_NEEDS_CONFIRMATION
    assert entry["requires_followup"] is True
    assert payload["summary"]["by_outcome"] == {OUTCOME_NEEDS_CONFIRMATION: 1}


def test_ledger_entry_id_and_patch_digest_are_stable():
    first = build_evidence_repair_commit_ledger(decisions=[_allow_decision()]).to_dict()
    second = build_evidence_repair_commit_ledger(decisions=[_allow_decision()]).to_dict()

    assert first["entries"][0]["id"] == second["entries"][0]["id"]
    assert first["entries"][0]["patch_digest"] == second["entries"][0]["patch_digest"]


def test_empty_ledger_is_read_only():
    payload = empty_evidence_repair_commit_ledger().to_dict()

    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["entry_count"] == 0
    assert payload["entries"] == []
