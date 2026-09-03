"""Tests for Task 26A — read-only run-completion reconciliation inspection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from htr import approval_control, contracts, events, io, paths
from htr.action_plan import PlanningIntent
from htr.approval_control import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_CONSUMED,
    OUTCOME_SCHEMA_VERSION,
    OUTCOME_SCHEMA_VERSION_V2,
    _canonical_json,
    _compute_approval_digest,
    _compute_claim_digest,
    _compute_outcome_digest,
    claim_approval,
    issue_approval,
    record_use_outcome,
    revoke_approval,
)
from htr.contracts import run_completion_fingerprint, run_completion_record_json_path
from htr.events import EVENT_TYPE_MANUAL_RUN_COMPLETED
from htr.execution_lock import LOCKS_DIR_NAME, RunExecutionLockOccupiedError
from htr.ids import new_approval_id, new_event_id, new_run_id, new_task_id
from htr.invoke_run_completion import (
    PILOT_BOUND_API,
    REASON_POST_VERIFICATION_MISMATCH,
    REASON_VERIFIED_SUCCESS,
    invoke_approved_run_completion,
)
from htr.reconciliation_inspection import (
    INSPECTION_DIGEST_PROJECTION_VERSION,
    INSPECTION_SCHEMA_VERSION,
    compute_inspection_semantic_digest,
    inspect_run_completion_reconciliation,
)
from htr.state import (
    ReconciliationEvidenceIntegrityError,
    ReconciliationUnsupportedApprovalError,
    RunCompletionReconciliationInspection,
)

TASK16_PATH = Path(__file__).with_name("test_run_final_closure.py")


def _load_task16():
    spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16()


def _expires_in(hours: float = 1.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _run_ready_for_completion(tmp_path: Path) -> tuple[str, str, dict[str, Any]]:
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[task_id]
    )
    return run_id, task_id, completion


def _issue_completion_approval(
    tmp_path: Path,
    run_id: str,
    completion: dict[str, Any],
    *,
    event_id: str | None = None,
    approval_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    event_id = event_id or new_event_id()
    intent = PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs={"record": completion, "actor": "human", "event_id": event_id},
        htr_runs_root=str(tmp_path),
    )
    kwargs: dict[str, Any] = {
        "approver_id": "alice",
        "executor_id": "bob",
        "expires_at": _expires_in(),
        "base_dir": tmp_path,
    }
    if approval_id is not None:
        kwargs["approval_id"] = approval_id
    issue = issue_approval(run_id, intent, **kwargs)
    return issue, event_id


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_snapshot(root: Path, *, run_id: str | None = None) -> dict[str, Any]:
    if not root.exists():
        return {
            "root_exists": False,
            "files": {},
            "file_mtimes": {},
            "dir_mtimes": {},
            "locks_exists": False,
            "locks_files": {},
            "locks_file_mtimes": {},
            "locks_dir_mtimes": {},
            "control_exists": False,
            "control_files": {},
        }
    files: dict[str, str] = {}
    file_mtimes: dict[str, int] = {}
    dir_mtimes: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_file():
            files[rel] = _file_digest(path)
            file_mtimes[rel] = path.stat().st_mtime_ns
        elif path.is_dir():
            dir_mtimes[rel] = path.stat().st_mtime_ns
    dir_mtimes["."] = root.stat().st_mtime_ns
    locks_root = root / LOCKS_DIR_NAME
    locks_exists = locks_root.exists()
    locks_files: dict[str, str] = {}
    locks_file_mtimes: dict[str, int] = {}
    locks_dir_mtimes: dict[str, int] = {}
    if locks_exists:
        for path in sorted(locks_root.rglob("*")):
            rel = str(path.relative_to(root))
            if path.is_file():
                locks_files[rel] = _file_digest(path)
                locks_file_mtimes[rel] = path.stat().st_mtime_ns
            elif path.is_dir():
                locks_dir_mtimes[rel] = path.stat().st_mtime_ns
        locks_dir_mtimes[".execution_locks"] = locks_root.stat().st_mtime_ns
    control_root = root / ".control"
    control_exists = control_root.exists()
    control_files: dict[str, str] = {}
    if control_exists:
        for path in sorted(control_root.rglob("*")):
            if path.is_file():
                control_files[str(path.relative_to(root))] = _file_digest(path)
    return {
        "root_exists": True,
        "files": files,
        "file_mtimes": file_mtimes,
        "dir_mtimes": dir_mtimes,
        "locks_exists": locks_exists,
        "locks_files": locks_files,
        "locks_file_mtimes": locks_file_mtimes,
        "locks_dir_mtimes": locks_dir_mtimes,
        "control_exists": control_exists,
        "control_files": control_files,
    }


def _assert_readonly_snapshot(before: dict[str, Any], after: dict[str, Any]) -> None:
    assert after["root_exists"] == before["root_exists"]
    assert after["files"] == before["files"]
    assert after["file_mtimes"] == before["file_mtimes"]
    assert after["dir_mtimes"] == before["dir_mtimes"]
    assert after["locks_exists"] == before["locks_exists"]
    assert after["locks_files"] == before["locks_files"]
    assert after["locks_file_mtimes"] == before["locks_file_mtimes"]
    assert after["locks_dir_mtimes"] == before["locks_dir_mtimes"]
    assert after["control_files"] == before["control_files"]


def _inspect_with_zero_write_guard(
    approval_id: str,
    tmp_path: Path,
    *,
    run_id: str | None = None,
) -> RunCompletionReconciliationInspection:
    before = _full_snapshot(tmp_path, run_id=run_id)
    result = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
    after = _full_snapshot(tmp_path, run_id=run_id)
    _assert_readonly_snapshot(before, after)
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False
    return result


def _write_marker(tmp_path: Path, run_id: str, *, metadata: dict[str, Any] | None = None) -> Path:
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    marker_path = locks_root / f"{run_id}.marker"
    payload = metadata or {
        "schema_version": "1",
        "acquisition_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "hostname": "test-host",
        "run_id": run_id,
    }
    marker_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return marker_path


def _sample_v2_evidence(**overrides: Any) -> dict[str, Any]:
    base = {
        "reason_code": REASON_VERIFIED_SUCCESS,
        "error_classification": REASON_VERIFIED_SUCCESS,
        "bound_api": PILOT_BOUND_API,
        "event_id": new_event_id(),
        "pre_observation_digest": "sha256:" + "a" * 64,
        "post_observation_digest": "sha256:" + "b" * 64,
        "mutation_may_have_committed": True,
        "safe_to_retry": False,
        "verification_reason_codes": [],
        "observed_record_fingerprint": None,
        "observed_event_fingerprint": None,
    }
    base.update(overrides)
    return base


def _ambiguous_v2_evidence(event_id: str, **overrides: Any) -> dict[str, Any]:
    return _sample_v2_evidence(
        event_id=event_id,
        reason_code=REASON_POST_VERIFICATION_MISMATCH,
        error_classification=REASON_POST_VERIFICATION_MISMATCH,
        **overrides,
    )


# --- Control evidence ---


def test_control_issue_only(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "issue_only"
    assert result.overall_classification in {"indeterminate", "reconciliation_inspection_required"}


def test_control_revoked_before_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    revoke_approval(issue["approval_id"], "alice", "no longer needed", base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "revoked_before_claim"


def test_control_claim_without_outcome(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_approval(issue["approval_id"], claim_id="claim-1", claimant_id="bob", base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "claimed_without_outcome"
    assert result.reconciliation_case_required is True


def test_control_v1_consumed_outcome(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-v1", claimant_id="bob", base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_CONSUMED,
        base_dir=tmp_path,
    )
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "consumed_outcome"
    assert "outcome_v1_no_task25_diagnostics" in result.reason_codes


def test_control_v2_consumed_after_invoke(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(
        issue["approval_id"], claim_id="claim-happy", base_dir=tmp_path
    )
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "consumed_outcome"
    assert result.outcome_class == OUTCOME_CONSUMED
    assert result.overall_classification == "no_reconciliation_needed"


def test_control_v2_ambiguous_outcome(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-amb", claimant_id="bob", base_dir=tmp_path)
    evidence = _sample_v2_evidence(
        event_id=event_id,
        reason_code=REASON_POST_VERIFICATION_MISMATCH,
        error_classification=REASON_POST_VERIFICATION_MISMATCH,
    )
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_AMBIGUOUS,
        outcome_evidence=evidence,
        base_dir=tmp_path,
    )
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "ambiguous_outcome"


def test_control_malformed_issue_digest(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    bad = io.read_json(issue_path)
    bad["approval_digest"] = "sha256:" + "0" * 64
    issue_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "malformed_control_evidence"
    assert "issue_digest_mismatch" in result.reason_codes


def test_control_unsupported_bound_api(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    bad = io.read_json(issue_path)
    bad["bound_api"] = "review_run_manually"
    issue_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "unsupported_approval"
    assert result.overall_classification == "integrity_blocked"
    assert "unsupported_bound_api" in result.reason_codes
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False
    assert result.lifecycle_evidence_state == "no_lifecycle_evidence_observed"


def test_control_wrong_runs_root_digest(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    bad = io.read_json(issue_path)
    bad["htr_runs_root_path_digest"] = "sha256:" + "f" * 64
    issue_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "issue_runs_root_digest_mismatch" in result.reason_codes


# --- Marker evidence ---


def test_marker_absent_zero_write(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "absent"


def test_marker_valid_metadata(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_valid_metadata"


def test_marker_malformed_json(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    marker = _write_marker(tmp_path, run_id)
    marker.write_text("{not-json", encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_malformed_metadata"


def test_marker_wrong_run_id(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id, metadata={
        "schema_version": "1",
        "acquisition_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "hostname": "host",
        "run_id": new_run_id(),
    })
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_malformed_metadata"
    assert "marker_run_id_mismatch" in result.reason_codes


def test_marker_symlink_unsafe(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    real = locks_root / "real.marker"
    real.write_text("{}", encoding="utf-8")
    (locks_root / f"{run_id}.marker").symlink_to(real.name)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_unsafe_path"


def test_marker_missing_lock_directory_no_bootstrap(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    locks_root = tmp_path / LOCKS_DIR_NAME
    if locks_root.exists():
        import shutil

        shutil.rmtree(locks_root)
    before = _full_snapshot(tmp_path, run_id=run_id)
    result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    after = _full_snapshot(tmp_path, run_id=run_id)
    _assert_readonly_snapshot(before, after)
    assert result.marker_state == "absent"
    assert not locks_root.exists()


# --- Lifecycle evidence ---


def test_lifecycle_no_evidence(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.lifecycle_evidence_state == "no_lifecycle_evidence_observed"


def test_lifecycle_json_only(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    io.atomic_write_json(run_completion_record_json_path(run_id, tmp_path), completion)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.lifecycle_evidence_state == "completion_json_only"


def test_lifecycle_complete_after_invoke(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-complete", base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.lifecycle_evidence_state == "verified_completed"


def test_lifecycle_wrong_event_id(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    events.complete_run_manually(run_id, completion, actor="human", event_id=new_event_id(), base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "completion_event_missing" in result.reason_codes or "completion_event_count_mismatch" in result.reason_codes


# --- Cross-evidence classifications ---


def test_cross_consumed_complete_no_marker(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-cross", base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.overall_classification == "no_reconciliation_needed"
    assert result.marker_state == "absent"


def test_cross_consumed_complete_with_marker(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-residue", base_dir=tmp_path)
    _write_marker(tmp_path, run_id)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.overall_classification == "verified_completion_marker_residue"
    assert result.marker_disposition_allowed is False


def test_cross_ambiguous_json_only(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-partial", claimant_id="bob", base_dir=tmp_path)
    io.atomic_write_json(run_completion_record_json_path(run_id, tmp_path), completion)
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_AMBIGUOUS,
        outcome_evidence=_sample_v2_evidence(
            event_id=event_id,
            reason_code=REASON_POST_VERIFICATION_MISMATCH,
            error_classification=REASON_POST_VERIFICATION_MISMATCH,
        ),
        base_dir=tmp_path,
    )
    _write_marker(tmp_path, run_id)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.overall_classification == "partial_lifecycle_commit"


def test_cross_claim_no_outcome_no_lifecycle(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_approval(issue["approval_id"], claim_id="claim-open", claimant_id="bob", base_dir=tmp_path)
    _write_marker(tmp_path, run_id)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "claimed_without_outcome"
    assert result.lifecycle_evidence_state == "no_lifecycle_evidence_observed"
    assert result.overall_classification == "reconciliation_inspection_required"


# --- Safety ---


@pytest.mark.parametrize(
    "factory",
    [
        "issue_only",
        "claimed",
        "consumed",
        "marker",
    ],
)
def test_safety_never_authorizes_retry_or_disposition(tmp_path, factory: str):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    if factory == "claimed":
        claim_approval(issue["approval_id"], claim_id="c1", claimant_id="bob", base_dir=tmp_path)
    elif factory == "consumed":
        invoke_approved_run_completion(issue["approval_id"], claim_id="c2", base_dir=tmp_path)
    if factory == "marker":
        _write_marker(tmp_path, run_id)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False


def test_safety_no_invoke_side_effect(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    with patch("htr.events.complete_run_manually") as mocked:
        _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
        mocked.assert_not_called()


def test_safety_no_reconciliation_directory(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert not (tmp_path / ".control" / "reconciliation").exists()


# --- Digest / projection ---


def test_digest_excludes_observed_at(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    first = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    second = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    assert first.inspection_semantic_digest == second.inspection_semantic_digest
    assert first.observed_at != second.observed_at


def test_digest_canonical_order_independent():
    from htr.reconciliation_inspection import _inspection_digest_projection

    base = RunCompletionReconciliationInspection(
        inspection_schema_version=INSPECTION_SCHEMA_VERSION,
        inspection_projection_version=INSPECTION_DIGEST_PROJECTION_VERSION,
        approval_id="approval-test",
        approval_digest="sha256:" + "a" * 64,
        claim_id=None,
        claim_digest=None,
        outcome_class=None,
        outcome_digest=None,
        run_id="run-test",
        bound_api=PILOT_BOUND_API,
        event_id="event-test",
        htr_runs_root_path_digest="sha256:" + "b" * 64,
        approval_control_state="issue_only",
        marker_state="absent",
        lifecycle_evidence_state="no_lifecycle_evidence_observed",
        integrity_state="clean",
        overall_classification="indeterminate",
        reason_codes=("alpha", "beta"),
        observed_completion_record_fingerprint=None,
        observed_event_semantic_fingerprint=None,
        observed_manifest_status=None,
        current_observation_semantic_digest=None,
        source_observation_digest="sha256:" + "c" * 64,
        inspection_semantic_digest="sha256:" + "d" * 64,
        safe_to_retry=False,
        marker_disposition_allowed=False,
        reconciliation_case_required=False,
        recovery_protocol_required=False,
        observed_at="2026-07-22T10:00:00+00:00",
    )
    projection = _inspection_digest_projection(base)
    reordered = {k: projection[k] for k in reversed(list(projection))}
    digest_a = "sha256:" + hashlib.sha256(_canonical_json(projection).encode()).hexdigest()
    digest_b = "sha256:" + hashlib.sha256(_canonical_json(reordered).encode()).hexdigest()
    assert digest_a == digest_b
    assert compute_inspection_semantic_digest(base) == digest_a


GOLDEN_INSPECTION = RunCompletionReconciliationInspection(
    inspection_schema_version=INSPECTION_SCHEMA_VERSION,
    inspection_projection_version=INSPECTION_DIGEST_PROJECTION_VERSION,
    approval_id="approval-golden",
    approval_digest="sha256:" + "1" * 64,
    claim_id=None,
    claim_digest=None,
    outcome_class=None,
    outcome_digest=None,
    run_id="run-golden",
    bound_api=PILOT_BOUND_API,
    event_id="event-golden",
    htr_runs_root_path_digest="sha256:" + "2" * 64,
    approval_control_state="issue_only",
    marker_state="absent",
    lifecycle_evidence_state="no_lifecycle_evidence_observed",
    integrity_state="clean",
    overall_classification="indeterminate",
    reason_codes=("alpha",),
    observed_completion_record_fingerprint=None,
    observed_event_semantic_fingerprint=None,
    observed_manifest_status=None,
    current_observation_semantic_digest=None,
    source_observation_digest="sha256:" + "3" * 64,
    inspection_semantic_digest="",
    safe_to_retry=False,
    marker_disposition_allowed=False,
    reconciliation_case_required=False,
    recovery_protocol_required=False,
    observed_at="2026-07-22T10:00:00+00:00",
)


def test_golden_inspection_digest_bytes():
    digest = compute_inspection_semantic_digest(GOLDEN_INSPECTION)
    assert digest == "sha256:91b7b5b6f57d0b1192654240093c2f072b2e7ed134d7308cef3504b1a73a297f"


def test_result_is_frozen_dataclass(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    with pytest.raises(Exception):
        result.overall_classification = "mutated"  # type: ignore[misc]


def test_public_api_exports():
    import htr

    assert hasattr(htr, "inspect_run_completion_reconciliation")
    assert hasattr(htr, "RunCompletionReconciliationInspection")
    assert hasattr(htr, "ReconciliationUnsupportedApprovalError")

# --- Hardening: expanded control evidence ---


def test_control_v1_ambiguous_outcome(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-v1-amb", claimant_id="bob", base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_AMBIGUOUS,
        base_dir=tmp_path,
    )
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "ambiguous_outcome"
    assert "outcome_v1_no_task25_diagnostics" in result.reason_codes


def test_control_malformed_claim_digest(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-bad", claimant_id="bob", base_dir=tmp_path)
    claim_path = paths.approval_claim_path(issue["approval_id"], tmp_path)
    bad = io.read_json(claim_path)
    bad["claim_digest"] = "sha256:" + "0" * 64
    claim_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "claim_digest_mismatch" in result.reason_codes
    assert result.approval_control_state in {"malformed_control_evidence", "conflicting_control_evidence"}


def test_control_malformed_outcome_digest(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-od", claimant_id="bob", base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_AMBIGUOUS,
        outcome_evidence=_ambiguous_v2_evidence(event_id=event_id),
        base_dir=tmp_path,
    )
    outcome_path = paths.approval_outcome_path(issue["approval_id"], tmp_path)
    bad = io.read_json(outcome_path)
    bad["outcome_digest"] = "sha256:" + "0" * 64
    outcome_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "outcome_digest_mismatch" in result.reason_codes


def test_control_outcome_bound_to_other_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-real", claimant_id="bob", base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_AMBIGUOUS,
        outcome_evidence=_ambiguous_v2_evidence(event_id=event_id),
        base_dir=tmp_path,
    )
    outcome_path = paths.approval_outcome_path(issue["approval_id"], tmp_path)
    bad = io.read_json(outcome_path)
    bad["claim_id"] = "claim-other"
    outcome_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "outcome_claim_id_mismatch" in result.reason_codes


def test_control_claim_wrong_approval_digest(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-wad", claimant_id="bob", base_dir=tmp_path)
    claim_path = paths.approval_claim_path(issue["approval_id"], tmp_path)
    bad = io.read_json(claim_path)
    bad["approval_digest"] = "sha256:" + "0" * 64
    claim_path.write_text(json.dumps(bad), encoding="utf-8")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "claim_approval_digest_mismatch" in result.reason_codes


def test_result_exposes_inspection_schema_version(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    assert result.inspection_schema_version == INSPECTION_SCHEMA_VERSION


# --- Hardening: expanded marker evidence ---


def test_marker_missing_acquisition_id(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id, metadata={
        "schema_version": "1",
        "pid": os.getpid(),
        "hostname": "host",
        "run_id": run_id,
    })
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_malformed_metadata"
    assert "marker_missing_acquisition_id" in result.reason_codes


def test_marker_malformed_pid(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id, metadata={
        "schema_version": "1",
        "acquisition_id": str(uuid.uuid4()),
        "pid": "not-an-int",
        "hostname": "host",
        "run_id": run_id,
    })
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_malformed_metadata"
    assert "marker_malformed_pid" in result.reason_codes


def test_marker_non_regular_file(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    marker_dir = locks_root / f"{run_id}.marker"
    marker_dir.mkdir()
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_unsafe_path"


def test_marker_lock_directory_symlink(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    locks_root = tmp_path / LOCKS_DIR_NAME
    if locks_root.is_symlink():
        locks_root.unlink()
    elif locks_root.is_dir():
        import shutil

        shutil.rmtree(locks_root)
    real = tmp_path / "real_locks"
    real.mkdir()
    locks_root.symlink_to(real.name)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.marker_state == "present_unsafe_path"
    assert "lock_directory_symlink" in result.reason_codes


def test_marker_alias_paths_same_physical(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id)
    alias = tmp_path.resolve()
    result_canonical = inspect_run_completion_reconciliation(
        issue["approval_id"], base_dir=tmp_path
    )
    result_alias = inspect_run_completion_reconciliation(
        issue["approval_id"], base_dir=alias
    )
    assert result_canonical.marker_state == result_alias.marker_state == "present_valid_metadata"
    assert result_canonical.inspection_semantic_digest == result_alias.inspection_semantic_digest


def test_marker_replaced_during_read(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id)
    import htr.reconciliation_inspection as ri

    real_stat_entry = ri._stat_entry_identity
    calls = {"n": 0}

    def fake_stat_entry(dir_fd: int, name: str) -> tuple[int, int]:
        calls["n"] += 1
        if calls["n"] >= 2:
            return (999, 888)
        return real_stat_entry(dir_fd, name)

    with patch.object(ri, "_stat_entry_identity", side_effect=fake_stat_entry):
        result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    assert result.marker_state == "present_identity_mismatch"
    assert "marker_replaced_during_read" in result.reason_codes


# --- Hardening: expanded lifecycle evidence ---


def test_lifecycle_event_without_json(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    record_path = run_completion_record_json_path(run_id, tmp_path)
    if record_path.is_file():
        record_path.unlink()
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.lifecycle_evidence_state == "lifecycle_evidence_conflict"
    assert "completion_event_without_json" in result.reason_codes


def test_lifecycle_json_event_incomplete_manifest(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    manifest_path = paths.run_manifest_path(run_id, tmp_path)
    manifest = io.read_json(manifest_path)
    manifest["status"] = "active"
    io.atomic_write_json(manifest_path, manifest)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.lifecycle_evidence_state == "completion_json_and_event_manifest_incomplete"


def test_lifecycle_wrong_completion_record(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    wrong = dict(completion)
    wrong["reason"] = "tampered"
    io.atomic_write_json(run_completion_record_json_path(run_id, tmp_path), wrong)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "completion_record_semantic_mismatch" in result.reason_codes


def test_lifecycle_duplicate_completion_events(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    first = events.complete_run_manually(
        run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path
    )
    duplicate = dict(first)
    duplicate["event_id"] = new_event_id()
    events_path = paths.task_events_path(run_id, tmp_path)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "multiple_completion_events" in result.reason_codes


def test_lifecycle_wrong_actor(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    events.complete_run_manually(run_id, completion, actor="wrong-actor", event_id=event_id, base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert "event_actor_mismatch" in result.reason_codes


def test_cross_ambiguous_complete_lifecycle(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim = claim_approval(issue["approval_id"], claim_id="claim-ambig-full", claimant_id="bob", base_dir=tmp_path)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        claim["claim_id"],
        OUTCOME_AMBIGUOUS,
        outcome_evidence=_ambiguous_v2_evidence(event_id=event_id),
        base_dir=tmp_path,
    )
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.overall_classification == "completion_observed_outcome_missing"


def test_cross_consumed_lifecycle_contradiction(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-ok", base_dir=tmp_path)
    wrong = dict(completion)
    wrong["metadata"] = {"tampered": True}
    io.atomic_write_json(run_completion_record_json_path(run_id, tmp_path), wrong)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "consumed_outcome"
    assert "completion_record_semantic_mismatch" in result.reason_codes
    assert "consumed_outcome_current_evidence_mismatch" in result.reason_codes
    assert result.overall_classification == "control_lifecycle_evidence_conflict"
    assert result.overall_classification not in {"verified_completed", "no_reconciliation_needed", "partial_lifecycle_commit"}
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False


def test_cross_marker_without_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "issue_only"
    assert result.marker_state == "present_valid_metadata"


def test_cross_lifecycle_without_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    result = _inspect_with_zero_write_guard(issue["approval_id"], tmp_path, run_id=run_id)
    assert result.approval_control_state == "issue_only"
    assert result.lifecycle_evidence_state != "no_lifecycle_evidence_observed"


# --- Hardening: concurrency / path safety ---


def _assert_inspect_zero_write(approval_id: str, tmp_path: Path) -> RunCompletionReconciliationInspection:
    writes: list[str] = []
    real_open = os.open

    def track_open(path, flags, *args, **kwargs):
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_TRUNC):
            writes.append(f"open:{path}:{flags}")
        return real_open(path, flags, *args, **kwargs)

    with patch("htr.reconciliation_inspection.os.open", side_effect=track_open):
        result = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
    assert writes == []
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False
    return result


def test_control_evidence_replaced_during_read(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-race", base_dir=tmp_path)
    outcome_path = paths.approval_outcome_path(issue["approval_id"], tmp_path)
    original_outcome = io.read_json(outcome_path)

    inspect_gate = threading.Event()
    writer_done = threading.Event()
    import htr.reconciliation_inspection as recon_mod

    real_fn = recon_mod._lifecycle_verification_reasons

    def paused_lifecycle(*args, **kwargs):
        inspect_gate.set()
        assert writer_done.wait(timeout=5.0)
        return real_fn(*args, **kwargs)

    def replace_outcome() -> None:
        assert inspect_gate.wait(timeout=5.0)
        tampered = dict(original_outcome)
        tampered["outcome_evidence"] = {
            **(tampered.get("outcome_evidence") or {}),
            "reason_code": "tampered_during_read",
        }
        outcome_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
        writer_done.set()

    with patch.object(recon_mod, "_lifecycle_verification_reasons", side_effect=paused_lifecycle):
        writer = threading.Thread(target=replace_outcome)
        writer.start()
        writes: list[str] = []
        real_open = os.open
        inspect_tid = threading.get_ident()

        def track_open(path, flags, *args, **kwargs):
            if threading.get_ident() == inspect_tid and flags & (os.O_CREAT | os.O_WRONLY | os.O_TRUNC):
                writes.append(f"open:{path}:{flags}")
            return real_open(path, flags, *args, **kwargs)

        with patch("htr.reconciliation_inspection.os.open", side_effect=track_open):
            result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
        writer.join(timeout=10.0)
    assert writes == []
    assert "control_evidence_replaced_during_read" in result.reason_codes
    assert result.overall_classification == "control_lifecycle_evidence_conflict"
    assert result.overall_classification != "verified_completed"
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False


def test_concurrent_lifecycle_writer_during_inspect(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-writer", base_dir=tmp_path)
    record_path = run_completion_record_json_path(run_id, tmp_path)

    inspect_gate = threading.Event()
    writer_done = threading.Event()
    import htr.reconciliation_inspection as recon_mod

    real_fn = recon_mod._lifecycle_verification_reasons

    def paused_lifecycle(*args, **kwargs):
        inspect_gate.set()
        assert writer_done.wait(timeout=5.0)
        return real_fn(*args, **kwargs)

    def tamper_lifecycle() -> None:
        assert inspect_gate.wait(timeout=5.0)
        wrong = dict(completion)
        wrong["metadata"] = {"writer_tampered": True}
        io.atomic_write_json(record_path, wrong)
        writer_done.set()

    with patch.object(recon_mod, "_lifecycle_verification_reasons", side_effect=paused_lifecycle):
        writer = threading.Thread(target=tamper_lifecycle)
        writer.start()
        writes: list[str] = []
        real_open = os.open
        inspect_tid = threading.get_ident()

        def track_open(path, flags, *args, **kwargs):
            if threading.get_ident() == inspect_tid and flags & (os.O_CREAT | os.O_WRONLY | os.O_TRUNC):
                writes.append(f"open:{path}:{flags}")
            return real_open(path, flags, *args, **kwargs)

        with patch("htr.reconciliation_inspection.os.open", side_effect=track_open):
            result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
        writer.join(timeout=10.0)
    assert writes == []
    assert result.approval_control_state == "consumed_outcome"
    assert "consumed_outcome_current_evidence_mismatch" in result.reason_codes
    assert result.overall_classification == "control_lifecycle_evidence_conflict"
    assert result.overall_classification != "verified_completed"
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False


def test_chain_gap_with_untrustworthy_observation(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-gap", base_dir=tmp_path)

    import htr.reconciliation_inspection as recon_mod

    untrustworthy_snapshot = {
        "decision_support": {"snapshot_trustworthy": False},
        "integrity": {"status": "fail"},
    }

    real_build = recon_mod.build_run_snapshot

    def snapshot_with_gap(run_id_arg: str, *, base_dir=None):
        snap = real_build(run_id_arg, base_dir=base_dir)
        snap["decision_support"] = {"snapshot_trustworthy": False, "integrity_fully_clean": False}
        snap["integrity"] = {"status": "fail", "blocking_finding_codes": ["phase1_chain_gap"]}
        chain = snap.get("record_chain") or {}
        if isinstance(chain, dict):
            chain["has_gap"] = True
        snap["record_chain"] = chain
        return snap

    before = _full_snapshot(tmp_path, run_id=run_id)
    with patch.object(recon_mod, "build_run_snapshot", side_effect=snapshot_with_gap):
        result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    after = _full_snapshot(tmp_path, run_id=run_id)
    _assert_readonly_snapshot(before, after)
    assert "snapshot_not_trustworthy" in result.reason_codes or "current_snapshot_not_trustworthy" in result.reason_codes
    assert result.overall_classification in {
        "integrity_blocked",
        "control_lifecycle_evidence_conflict",
    }
    assert result.overall_classification != "verified_completed"
    assert result.overall_classification != "no_reconciliation_needed"
    assert result.safe_to_retry is False
    assert result.marker_disposition_allowed is False


def test_concurrent_thread_inspect_readonly(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    _write_marker(tmp_path, run_id)
    before = _full_snapshot(tmp_path, run_id=run_id)
    errors: list[BaseException] = []
    results: list[RunCompletionReconciliationInspection] = []

    def worker() -> None:
        try:
            results.append(
                inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    after = _full_snapshot(tmp_path, run_id=run_id)
    _assert_readonly_snapshot(before, after)
    assert not errors
    assert len(results) == 4
    for result in results:
        assert result.safe_to_retry is False
        assert result.marker_disposition_allowed is False


@pytest.mark.skipif(multiprocessing.get_start_method() != "fork", reason="fork test")
def test_fork_child_inspect_readonly(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    before = _full_snapshot(tmp_path, run_id=run_id)
    result_path = tmp_path.parent / f"fork_inspect_{uuid.uuid4().hex}.json"
    child = os.fork()
    if child == 0:
        try:
            result = inspect_run_completion_reconciliation(
                issue["approval_id"], base_dir=tmp_path
            )
            result_path.write_text(
                json.dumps({"ok": True, "classification": result.overall_classification}),
                encoding="utf-8",
            )
        except BaseException as exc:
            result_path.write_text(json.dumps({"ok": False, "error": str(exc)}), encoding="utf-8")
        os._exit(0)
    else:
        pid, status = os.waitpid(child, 0)
        assert pid == child
        assert os.WIFEXITED(status)
        assert os.WEXITSTATUS(status) == 0
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        after = _full_snapshot(tmp_path, run_id=run_id)
        _assert_readonly_snapshot(before, after)


def test_cross_project_isolation(tmp_path):
    root_a = tmp_path / "project_a"
    root_b = tmp_path / "project_b"
    run_id, _task_id, completion = _run_ready_for_completion(root_a)
    issue, _event_id = _issue_completion_approval(root_a, run_id, completion)
    _write_marker(root_b, run_id)
    result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=root_a)
    assert result.marker_state == "absent"


def test_inspect_zero_write_open_guard(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    writes: list[str] = []
    real_open = os.open

    def track_open(path, flags, *args, **kwargs):
        if flags & (os.O_CREAT | os.O_WRONLY | os.O_TRUNC):
            writes.append(f"open:{path}:{flags}")
        return real_open(path, flags, *args, **kwargs)

    with patch("htr.reconciliation_inspection.os.open", side_effect=track_open):
        inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    assert writes == []


def test_digest_schema_version_affects_digest():
    from htr.reconciliation_inspection import _inspection_digest_projection

    base = RunCompletionReconciliationInspection(
        inspection_schema_version=INSPECTION_SCHEMA_VERSION,
        inspection_projection_version=INSPECTION_DIGEST_PROJECTION_VERSION,
        approval_id="approval-test",
        approval_digest="sha256:" + "a" * 64,
        claim_id=None,
        claim_digest=None,
        outcome_class=None,
        outcome_digest=None,
        run_id="run-test",
        bound_api=PILOT_BOUND_API,
        event_id="event-test",
        htr_runs_root_path_digest="sha256:" + "b" * 64,
        approval_control_state="issue_only",
        marker_state="absent",
        lifecycle_evidence_state="no_lifecycle_evidence_observed",
        integrity_state="clean",
        overall_classification="indeterminate",
        reason_codes=("alpha",),
        observed_completion_record_fingerprint=None,
        observed_event_semantic_fingerprint=None,
        observed_manifest_status=None,
        current_observation_semantic_digest=None,
        source_observation_digest="sha256:" + "c" * 64,
        inspection_semantic_digest="sha256:" + "d" * 64,
        safe_to_retry=False,
        marker_disposition_allowed=False,
        reconciliation_case_required=False,
        recovery_protocol_required=False,
        observed_at="2026-07-22T10:00:00+00:00",
    )
    projection = _inspection_digest_projection(base)
    assert projection["inspection_schema_version"] == INSPECTION_SCHEMA_VERSION
    assert "inspection_schema_version" in projection
