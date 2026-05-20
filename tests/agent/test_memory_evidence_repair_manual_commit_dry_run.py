from agent.memory_evidence_repair_manual_commit_dry_run import (
    CHECK_STATUS_FAIL,
    CHECK_STATUS_PASS,
    DRY_RUN_STATUS_BLOCKED,
    DRY_RUN_STATUS_NO_ACTION_NEEDED,
    DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT,
    build_evidence_repair_manual_commit_dry_run,
    empty_evidence_repair_manual_commit_dry_run,
)


def _commit_gate_allow():
    return {
        "summary": {
            "decision_count": 1,
            "allow_count": 1,
            "blocked_count": 0,
            "needs_confirmation_count": 0,
        },
        "decisions": [{"decision": "allow_manual_commit"}],
    }


def _ledger_allow():
    return {
        "summary": {
            "entry_count": 1,
            "allow_count": 1,
            "blocked_count": 0,
            "followup_count": 0,
        },
        "entries": [
            {
                "id": "ledger-123",
                "manual_commit_allowed": True,
                "candidate_id": "repair-123",
                "preview_id": "preview-123",
                "provider": "builtin",
                "repair_action": "attach_provenance",
                "patch_digest": "a" * 64,
                "target": {"candidate_id": "repair-123"},
                "evidence_fields": [
                    "source_url",
                    "observed_at",
                    "verification_signal",
                ],
            }
        ],
    }


def _rollback_ready():
    return {
        "summary": {
            "rollback_step_count": 1,
            "ready_count": 1,
            "snapshot_required_count": 0,
        },
        "steps": [{"status": "ready_for_manual_rollback"}],
    }


def _snapshot_available():
    return {
        "summary": {
            "snapshot_request_count": 1,
            "capture_required_count": 0,
            "blocking_count": 0,
        },
        "requests": [{"status": "already_available"}],
    }


def test_ready_dry_run_allows_manual_commit_without_mutation():
    payload = build_evidence_repair_manual_commit_dry_run(
        snapshot_plan=_snapshot_available(),
        commit_gate=_commit_gate_allow(),
        ledger=_ledger_allow(),
        rollback_plan=_rollback_ready(),
    ).to_dict()

    assert payload["status"] == DRY_RUN_STATUS_READY_FOR_MANUAL_COMMIT
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["summary"]["manual_commit_allowed"] is True
    assert payload["summary"]["operation_count"] == 1
    assert payload["operations"][0]["id"].startswith("dryrun-op-")
    assert payload["blocking_reasons"] == []
    assert {check["status"] for check in payload["checks"]} == {CHECK_STATUS_PASS}


def test_dry_run_blocks_when_snapshot_capture_is_required():
    snapshot_plan = _snapshot_available()
    snapshot_plan["summary"]["capture_required_count"] = 1
    snapshot_plan["summary"]["blocking_count"] = 1

    payload = build_evidence_repair_manual_commit_dry_run(
        snapshot_plan=snapshot_plan,
        commit_gate=_commit_gate_allow(),
        ledger=_ledger_allow(),
        rollback_plan=_rollback_ready(),
    ).to_dict()

    assert payload["status"] == DRY_RUN_STATUS_BLOCKED
    assert payload["summary"]["has_blocks"] is True
    assert "Pre-commit snapshots are still required." in payload["blocking_reasons"]
    assert "capture_pre_commit_snapshots" in payload["required_actions"]


def test_dry_run_blocks_when_commit_gate_is_not_clean():
    commit_gate = _commit_gate_allow()
    commit_gate["summary"]["blocked_count"] = 1
    commit_gate["summary"]["allow_count"] = 0

    payload = build_evidence_repair_manual_commit_dry_run(
        snapshot_plan=_snapshot_available(),
        commit_gate=commit_gate,
        ledger=_ledger_allow(),
        rollback_plan=_rollback_ready(),
    ).to_dict()

    gate_check = next(check for check in payload["checks"] if check["id"] == "commit_gate")
    assert payload["status"] == DRY_RUN_STATUS_BLOCKED
    assert gate_check["status"] == CHECK_STATUS_FAIL
    assert "resolve_commit_gate_blocks" in gate_check["required_actions"]


def test_dry_run_has_no_action_without_operations():
    payload = build_evidence_repair_manual_commit_dry_run(
        snapshot_plan={},
        commit_gate={},
        ledger={},
        rollback_plan={},
    ).to_dict()

    assert payload["status"] == DRY_RUN_STATUS_NO_ACTION_NEEDED
    assert payload["summary"]["operation_count"] == 0
    assert payload["summary"]["manual_commit_allowed"] is False


def test_empty_dry_run_is_read_only():
    payload = empty_evidence_repair_manual_commit_dry_run().to_dict()

    assert payload["status"] == DRY_RUN_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["checks"] == []
    assert payload["operations"] == []
