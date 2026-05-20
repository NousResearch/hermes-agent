from agent.memory_evidence_repair_snapshot_planner import (
    SNAPSHOT_ACTION_CAPTURE_EVIDENCE_METADATA,
    SNAPSHOT_ACTION_NOOP,
    SNAPSHOT_ACTION_VERIFY_EXISTING_SNAPSHOT,
    SNAPSHOT_STATUS_ALREADY_AVAILABLE,
    SNAPSHOT_STATUS_CAPTURE_REQUIRED,
    SNAPSHOT_STATUS_NO_ACTION_NEEDED,
    build_evidence_repair_snapshot_plan,
    empty_evidence_repair_snapshot_plan,
)


def _snapshot_required_step():
    return {
        "id": "rollback-123",
        "ledger_entry_id": "ledger-123",
        "sequence": 1,
        "status": "snapshot_required",
        "rollback_action": "restore_evidence_metadata",
        "provider": "builtin",
        "repair_action": "attach_provenance",
        "candidate_id": "repair-123",
        "preview_id": "preview-123",
        "patch_digest": "a" * 64,
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_fields": ["source_url", "observed_at", "verification_signal"],
    }


def test_snapshot_plan_requires_capture_for_snapshot_required_step():
    plan = build_evidence_repair_snapshot_plan(steps=[_snapshot_required_step()])

    payload = plan.to_dict()
    request = payload["requests"][0]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["capture_required_count"] == 1
    assert payload["summary"]["blocks_manual_commit"] is True
    assert request["id"].startswith("snapshot-")
    assert request["status"] == SNAPSHOT_STATUS_CAPTURE_REQUIRED
    assert request["snapshot_action"] == SNAPSHOT_ACTION_CAPTURE_EVIDENCE_METADATA
    assert request["blocks_manual_commit"] is True
    assert request["evidence_fields"] == [
        "source_url",
        "observed_at",
        "verification_signal",
    ]
    assert request["snapshot_payload_preview"]["capture"]["snapshot_key"] == "ledger-123"


def test_snapshot_plan_detects_existing_snapshot_from_ready_rollback_step():
    step = _snapshot_required_step()
    step["status"] = "ready_for_manual_rollback"

    payload = build_evidence_repair_snapshot_plan(steps=[step]).to_dict()
    request = payload["requests"][0]
    assert request["status"] == SNAPSHOT_STATUS_ALREADY_AVAILABLE
    assert request["snapshot_action"] == SNAPSHOT_ACTION_VERIFY_EXISTING_SNAPSHOT
    assert request["blocks_manual_commit"] is False
    assert payload["summary"]["already_available_count"] == 1


def test_snapshot_plan_noops_for_non_committed_entry():
    step = _snapshot_required_step()
    step["status"] = "no_action_needed"
    step["rollback_action"] = "no_rollback_required"

    payload = build_evidence_repair_snapshot_plan(steps=[step]).to_dict()
    request = payload["requests"][0]
    assert request["status"] == SNAPSHOT_STATUS_NO_ACTION_NEEDED
    assert request["snapshot_action"] == SNAPSHOT_ACTION_NOOP
    assert request["blocks_manual_commit"] is False
    assert payload["summary"]["no_action_count"] == 1


def test_snapshot_plan_can_build_from_ledger_and_existing_snapshot():
    entry = {
        "id": "ledger-123",
        "outcome": "allowed_for_manual_commit",
        "manual_commit_allowed": True,
        "provider": "builtin",
        "repair_action": "attach_provenance",
        "candidate_id": "repair-123",
        "preview_id": "preview-123",
        "patch_digest": "a" * 64,
        "target": {"candidate_id": "repair-123", "provider": "builtin"},
        "evidence_fields": ["source_url"],
    }

    payload = build_evidence_repair_snapshot_plan(
        entries=[entry],
        existing_snapshots={"ledger-123": {"evidence": {"source_url": "old"}}},
    ).to_dict()

    assert payload["requests"][0]["status"] == SNAPSHOT_STATUS_ALREADY_AVAILABLE
    assert payload["summary"]["already_available_count"] == 1


def test_snapshot_request_id_is_stable():
    first = build_evidence_repair_snapshot_plan(
        steps=[_snapshot_required_step()]
    ).to_dict()
    second = build_evidence_repair_snapshot_plan(
        steps=[_snapshot_required_step()]
    ).to_dict()

    assert first["requests"][0]["id"] == second["requests"][0]["id"]


def test_empty_snapshot_plan_is_read_only():
    payload = empty_evidence_repair_snapshot_plan().to_dict()

    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["snapshot_request_count"] == 0
    assert payload["requests"] == []
