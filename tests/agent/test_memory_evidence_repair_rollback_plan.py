from agent.memory_evidence_repair_rollback_plan import (
    ROLLBACK_ACTION_NOOP,
    ROLLBACK_ACTION_RESTORE_EVIDENCE_METADATA,
    ROLLBACK_STATUS_NO_ACTION_NEEDED,
    ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK,
    ROLLBACK_STATUS_SNAPSHOT_REQUIRED,
    SNAPSHOT_REQUIRED_VALUE,
    build_evidence_repair_rollback_plan,
    empty_evidence_repair_rollback_plan,
)


def _allowed_entry():
    return {
        "id": "ledger-123",
        "sequence": 1,
        "outcome": "allowed_for_manual_commit",
        "decision_id": "gate-preview-123",
        "preview_id": "preview-123",
        "candidate_id": "repair-123",
        "provider": "builtin",
        "priority": "high",
        "repair_action": "attach_provenance",
        "evidence_fields": ["source_url", "observed_at", "verification_signal"],
        "patch_digest": "a" * 64,
        "manual_commit_allowed": True,
        "blocked": False,
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
    }


def test_ready_rollback_plan_from_allowed_entry_with_snapshot():
    plan = build_evidence_repair_rollback_plan(
        entries=[_allowed_entry()],
        pre_commit_snapshots={
            "ledger-123": {
                "evidence": {
                    "source_url": "https://old.example.com/source",
                    "observed_at": "2026-05-01T00:00:00Z",
                    "verification_signal": "previous_manual_verified",
                }
            }
        },
    )

    payload = plan.to_dict()
    step = payload["steps"][0]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["ready_count"] == 1
    assert step["id"].startswith("rollback-")
    assert step["status"] == ROLLBACK_STATUS_READY_FOR_MANUAL_ROLLBACK
    assert step["rollback_action"] == ROLLBACK_ACTION_RESTORE_EVIDENCE_METADATA
    assert step["inverse_patch_preview"]["set"]["evidence"] == {
        "source_url": "https://old.example.com/source",
        "observed_at": "2026-05-01T00:00:00Z",
        "verification_signal": "previous_manual_verified",
    }
    assert "manual_rollback_only" in step["required_preconditions"]


def test_allowed_entry_requires_snapshot_before_rollback_ready():
    plan = build_evidence_repair_rollback_plan(entries=[_allowed_entry()])

    payload = plan.to_dict()
    step = payload["steps"][0]
    assert step["status"] == ROLLBACK_STATUS_SNAPSHOT_REQUIRED
    assert step["inverse_patch_preview"]["set"]["evidence"] == SNAPSHOT_REQUIRED_VALUE
    assert "capture_pre_commit_snapshot_before_apply" in step["required_preconditions"]
    assert payload["summary"]["snapshot_required_count"] == 1
    assert payload["summary"]["requires_snapshot"] is True


def test_blocked_entry_has_no_rollback_action():
    entry = _allowed_entry()
    entry["outcome"] = "blocked"
    entry["manual_commit_allowed"] = False
    entry["blocked"] = True

    payload = build_evidence_repair_rollback_plan(entries=[entry]).to_dict()
    step = payload["steps"][0]
    assert step["status"] == ROLLBACK_STATUS_NO_ACTION_NEEDED
    assert step["rollback_action"] == ROLLBACK_ACTION_NOOP
    assert step["inverse_patch_preview"]["operation"] == ROLLBACK_ACTION_NOOP
    assert payload["summary"]["no_action_count"] == 1


def test_rollback_step_id_is_stable():
    first = build_evidence_repair_rollback_plan(entries=[_allowed_entry()]).to_dict()
    second = build_evidence_repair_rollback_plan(entries=[_allowed_entry()]).to_dict()

    assert first["steps"][0]["id"] == second["steps"][0]["id"]


def test_empty_rollback_plan_is_read_only():
    payload = empty_evidence_repair_rollback_plan().to_dict()

    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["rollback_step_count"] == 0
    assert payload["steps"] == []
