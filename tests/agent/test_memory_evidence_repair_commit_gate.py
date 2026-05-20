from agent.memory_evidence_repair_commit_gate import (
    DECISION_ALLOW_MANUAL_COMMIT,
    DECISION_BLOCK_COMMIT,
    DECISION_NEEDS_USER_CONFIRMATION,
    REASON_MISSING_CONFIRMATION,
    REASON_MISSING_EVIDENCE,
    REASON_UNSAFE_OPERATION,
    build_evidence_repair_commit_gate,
    empty_evidence_repair_commit_gate,
)


def _ready_preview():
    return {
        "id": "preview-123",
        "candidate_id": "repair-123",
        "status": "ready_for_manual_apply",
        "provider": "builtin",
        "priority": "high",
        "repair_action": "attach_provenance",
        "operation": "merge_evidence_metadata",
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_patch": {
            "source_url": "https://example.com/source",
            "observed_at": "2026-05-11T00:00:00Z",
            "verification_signal": "manual_verified",
        },
        "memory_patch_preview": {
            "operation": "merge_evidence_metadata",
            "target": {"candidate_id": "repair-123", "provider": "builtin"},
            "set": {"evidence": {}},
        },
        "missing_evidence": [],
    }


def test_gate_allows_ready_preview_with_explicit_confirmation():
    report = build_evidence_repair_commit_gate(
        previews=[_ready_preview()],
        confirmed_preview_ids=["preview-123"],
    )

    payload = report.to_dict()
    decision = payload["decisions"][0]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["allow_count"] == 1
    assert decision["decision"] == DECISION_ALLOW_MANUAL_COMMIT
    assert decision["reasons"] == []
    assert decision["required_actions"] == []
    assert decision["preview_id"] == "preview-123"


def test_gate_needs_confirmation_for_ready_unconfirmed_preview():
    report = build_evidence_repair_commit_gate(previews=[_ready_preview()])

    payload = report.to_dict()
    decision = payload["decisions"][0]
    assert decision["decision"] == DECISION_NEEDS_USER_CONFIRMATION
    assert decision["reasons"] == [REASON_MISSING_CONFIRMATION]
    assert "obtain_explicit_user_confirmation" in decision["required_actions"]
    assert payload["summary"]["requires_user_confirmation"] is True


def test_gate_blocks_missing_evidence_even_when_confirmed():
    preview = _ready_preview()
    preview["missing_evidence"] = ["source_url"]

    report = build_evidence_repair_commit_gate(
        previews=[preview],
        confirmed_preview_ids=["preview-123"],
    )

    payload = report.to_dict()
    decision = payload["decisions"][0]
    assert decision["decision"] == DECISION_BLOCK_COMMIT
    assert REASON_MISSING_EVIDENCE in decision["reasons"]
    assert payload["summary"]["blocked_count"] == 1


def test_gate_blocks_unsafe_operation():
    preview = _ready_preview()
    preview["operation"] = "replace_memory_content"

    report = build_evidence_repair_commit_gate(
        previews=[preview],
        confirmed_preview_ids=["preview-123"],
    )

    decision = report.to_dict()["decisions"][0]
    assert decision["decision"] == DECISION_BLOCK_COMMIT
    assert REASON_UNSAFE_OPERATION in decision["reasons"]


def test_conflict_preview_requires_resolution():
    preview = _ready_preview()
    preview["repair_action"] = "review_conflict"

    report = build_evidence_repair_commit_gate(
        previews=[preview],
        confirmed_preview_ids=["preview-123"],
    )

    decision = report.to_dict()["decisions"][0]
    assert decision["decision"] == DECISION_BLOCK_COMMIT
    assert decision["conflict_risk"] == "high"
    assert "provide_conflict_resolution" in decision["required_actions"]


def test_empty_commit_gate_is_read_only():
    payload = empty_evidence_repair_commit_gate().to_dict()

    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["decision_count"] == 0
    assert payload["decisions"] == []
