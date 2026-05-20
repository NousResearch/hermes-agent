from agent.memory_evidence_repair_apply_preview import (
    PATCH_OPERATION,
    PREVIEW_STATUS_AWAITING_CONFIRMATION,
    PREVIEW_STATUS_BLOCKED_MISSING_EVIDENCE,
    PREVIEW_STATUS_READY_FOR_MANUAL_APPLY,
    build_evidence_repair_apply_preview,
    empty_evidence_repair_apply_preview,
)


def test_builds_ready_apply_preview_for_approved_candidate():
    report = build_evidence_repair_apply_preview(
        candidates=[
            {
                "id": "repair-123",
                "provider": "builtin",
                "priority": "high",
                "repair_action": "attach_provenance",
                "source": "policy_gate",
                "reason": "Missing explicit provenance.",
                "required_evidence": [
                    "source_url",
                    "observed_at",
                    "verification_signal",
                ],
                "proposed_evidence_patch": {
                    "source_url": "https://example.com/source",
                    "observed_at": "2026-05-11T00:00:00Z",
                    "verification_signal": "manual_verified",
                },
            }
        ],
        approved_candidate_ids=["repair-123"],
    )

    payload = report.to_dict()
    preview = payload["previews"][0]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["ready_count"] == 1
    assert preview["id"].startswith("preview-")
    assert preview["candidate_id"] == "repair-123"
    assert preview["status"] == PREVIEW_STATUS_READY_FOR_MANUAL_APPLY
    assert preview["operation"] == PATCH_OPERATION
    assert preview["would_update_fields"] == [
        "source_url",
        "observed_at",
        "verification_signal",
    ]
    assert preview["missing_evidence"] == []
    assert preview["memory_patch_preview"]["operation"] == PATCH_OPERATION


def test_preview_blocks_missing_evidence_until_completed():
    report = build_evidence_repair_apply_preview(
        candidates=[
            {
                "id": "repair-456",
                "provider": "graph",
                "priority": "medium",
                "repair_action": "refresh_observation",
                "required_evidence": ["observed_at", "freshness_check"],
                "proposed_evidence_patch": {
                    "observed_at": "<pending>",
                    "freshness_check": "confirmed_current",
                },
            }
        ],
        approved_candidate_ids=["repair-456"],
    )

    payload = report.to_dict()
    preview = payload["previews"][0]
    assert preview["status"] == PREVIEW_STATUS_BLOCKED_MISSING_EVIDENCE
    assert preview["missing_evidence"] == ["observed_at"]
    assert payload["summary"]["missing_evidence_count"] == 1
    assert payload["summary"]["ready_count"] == 0


def test_preview_awaits_confirmation_when_not_approved():
    report = build_evidence_repair_apply_preview(
        candidates=[
            {
                "id": "repair-789",
                "provider": "builtin",
                "repair_action": "attach_provenance",
                "required_evidence": ["source_url"],
                "proposed_evidence_patch": {"source_url": "https://example.com"},
            }
        ],
    )

    payload = report.to_dict()
    assert payload["previews"][0]["status"] == PREVIEW_STATUS_AWAITING_CONFIRMATION
    assert payload["summary"]["requires_user_confirmation"] is True


def test_proposed_evidence_can_fill_candidate_patch():
    report = build_evidence_repair_apply_preview(
        candidates=[
            {
                "id": "repair-abc",
                "provider": "builtin",
                "repair_action": "attach_provenance",
                "required_evidence": ["source_url", "observed_at"],
                "proposed_evidence_patch": {
                    "source_url": "<pending>",
                    "observed_at": "<pending>",
                },
            }
        ],
        approved_candidate_ids=["repair-abc"],
        proposed_evidence={
            "repair-abc": {
                "source_url": "https://example.com/source",
                "observed_at": "2026-05-11T00:00:00Z",
            }
        },
    )

    preview = report.to_dict()["previews"][0]
    assert preview["status"] == PREVIEW_STATUS_READY_FOR_MANUAL_APPLY
    assert preview["evidence_patch"] == {
        "source_url": "https://example.com/source",
        "observed_at": "2026-05-11T00:00:00Z",
    }


def test_empty_apply_preview_is_read_only():
    payload = empty_evidence_repair_apply_preview().to_dict()

    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["preview_count"] == 0
    assert payload["previews"] == []
