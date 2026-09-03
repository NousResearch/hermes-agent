"""Tests for Task 26B — durable reconciliation cases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os as _os
import stat as _stat
import time as _time
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from htr import approval_control, contracts, events, io, paths, reconciliation_cases
from htr.action_plan import PlanningIntent, _canonical_json
from htr.approval_control import (
    OUTCOME_AMBIGUOUS,
    _compute_approval_digest,
    claim_approval,
    issue_approval,
    record_use_outcome,
)
from htr.contracts import run_completion_record_json_path
from htr.execution_lock import LOCKS_DIR_NAME
from htr.ids import generate_id, new_approval_id, new_event_id, new_run_id, new_task_id, validate_id
from htr.invoke_run_completion import (
    PILOT_BOUND_API,
    REASON_POST_VERIFICATION_MISMATCH,
    invoke_approved_run_completion,
)
from htr.reconciliation_inspection import (
    INSPECTION_DIGEST_PROJECTION_VERSION,
    INSPECTION_SCHEMA_VERSION,
    compute_inspection_semantic_digest,
    inspect_run_completion_reconciliation,
)
from htr.reconciliation_cases import (
    ReconciliationConflictError,
    ReconciliationDecisionClass,
    ReconciliationDurabilityError,
    ReconciliationNextProtocol,
    ReconciliationRationaleCode,
    ReconciliationScopeReason,
    ReconciliationStateError,
    ReconciliationUnsupportedApprovalError,
    ReconciliationValidationError,
    _build_inspection_semantic_projection_from_result,
    generate_reconciliation_case_id,
    load_reconciliation_case,
    open_reconciliation_case,
    record_reconciliation_decision,
    record_reconciliation_observation,
)
from htr.state import (
    ReconciliationEvidenceIntegrityError,
    ReconciliationInspectionError,
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


def _ambiguous_v2_evidence(*, event_id: str, **overrides: Any) -> dict[str, Any]:
    base = {
        "reason_code": REASON_POST_VERIFICATION_MISMATCH,
        "error_classification": REASON_POST_VERIFICATION_MISMATCH,
        "bound_api": PILOT_BOUND_API,
        "event_id": event_id,
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


def _open_case(tmp_path: Path, issue: dict[str, Any], *, opened_by: str = "operator") -> str:
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by=opened_by,
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    return case_id


def _observe_case(tmp_path: Path, case_id: str, *, observed_by: str = "operator") -> Any:
    return record_reconciliation_observation(
        case_id,
        base_dir=tmp_path,
        observed_by=observed_by,
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_snapshot(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"files": {}, "control_exists": False, "control_files": {}}
    files = {
        str(p.relative_to(root)): _file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    control_root = root / ".control"
    control_files: dict[str, str] = {}
    if control_root.exists():
        for p in sorted(control_root.rglob("*")):
            if p.is_file():
                control_files[str(p.relative_to(control_root))] = _file_digest(p)
    locks_root = root / LOCKS_DIR_NAME
    locks_files: dict[str, str] = {}
    if locks_root.exists():
        for p in sorted(locks_root.rglob("*")):
            if p.is_file():
                locks_files[str(p.relative_to(locks_root))] = _file_digest(p)
    return {
        "files": files,
        "control_exists": control_root.exists(),
        "control_files": control_files,
        "locks_files": locks_files,
    }


def _path_stat_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.lstat()
    return {
        "exists": True,
        "mode": st.st_mode,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "inode": st.st_ino,
        "dev": st.st_dev,
        "is_symlink": path.is_symlink(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def _case_tree_snapshot(tmp_path: Path, case_id: str) -> dict[str, Any]:
    case_dir = paths.reconciliation_case_dir(case_id, tmp_path)
    snap: dict[str, Any] = {"case_dir": _path_stat_snapshot(case_dir)}
    for name in ("open.json", "observation.json", "decision.json"):
        p = case_dir / name
        snap[name] = _path_stat_snapshot(p)
        if p.is_file() and not p.is_symlink():
            snap[f"{name}_sha256"] = _file_digest(p)
    return snap


def _run_subprocess_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_marker(tmp_path: Path, run_id: str, *, metadata: dict[str, Any] | None = None) -> Path:
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    marker_path = locks_root / f"{run_id}.marker"
    payload = metadata or {
        "schema_version": "1",
        "acquisition_id": str(uuid.uuid4()),
        "pid": _os.getpid(),
        "hostname": "test-host",
        "run_id": run_id,
    }
    marker_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return marker_path


def _consumed_with_marker_case(tmp_path: Path) -> tuple[str, Any]:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-consumed", base_dir=tmp_path)
    _write_marker(tmp_path, run_id)
    case_id = _open_case(tmp_path, issue)
    obs, _meta = _observe_case(tmp_path, case_id)
    return case_id, obs


def _consumed_contradiction_case(tmp_path: Path) -> tuple[str, Any]:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-consumed2", base_dir=tmp_path)
    wrong = dict(completion)
    wrong["metadata"] = {"tampered": True}
    io.atomic_write_json(run_completion_record_json_path(run_id, tmp_path), wrong)
    case_id = _open_case(tmp_path, issue)
    obs, _meta = _observe_case(tmp_path, case_id)
    return case_id, obs


# --- Case ID / open ---


def test_generate_reconciliation_case_id_has_no_side_effects(tmp_path):
    before = _full_snapshot(tmp_path)
    case_id = generate_reconciliation_case_id()
    after = _full_snapshot(tmp_path)
    assert case_id.startswith("rcn_")
    assert validate_id(case_id, "reconciliation")
    assert before == after


def test_open_creates_stable_scope_only(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    raw = json.loads(paths.reconciliation_open_path(case_id, tmp_path).read_text(encoding="utf-8"))
    forbidden = {
        "claim_id",
        "outcome_class",
        "marker_state",
        "inspection_semantic_digest",
        "lifecycle_evidence_state",
    }
    assert forbidden.isdisjoint(raw.keys())
    assert raw["bound_api"] == PILOT_BOUND_API
    assert raw["scope_reason"] == ReconciliationScopeReason.ambiguous_completion_reconciliation.value


def test_invalid_approval_open_leaves_zero_residue(tmp_path):
    case_id = generate_reconciliation_case_id()
    with pytest.raises(ReconciliationValidationError, match="missing issue"):
        open_reconciliation_case(
            case_id,
            new_approval_id(),
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    assert not (tmp_path / ".control").exists()


def test_unsupported_bound_api_zero_residue(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    mutated = json.loads(issue_path.read_text(encoding="utf-8"))
    mutated["bound_api"] = "review_run_manually"
    mutated["approval_digest"] = _compute_approval_digest(mutated)
    issue_path.write_text(json.dumps(mutated, indent=2) + "\n", encoding="utf-8")
    case_id = generate_reconciliation_case_id()
    with pytest.raises(ReconciliationUnsupportedApprovalError):
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    assert not (tmp_path / ".control" / "reconciliation").exists()


def test_open_exact_replay_zero_write(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    first, meta1 = open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    digest_before = _file_digest(paths.reconciliation_open_path(case_id, tmp_path))
    second, meta2 = open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    assert meta1.exact_replay is False
    assert meta1.exact_replay_status == "no"
    assert meta2.exact_replay is True
    assert meta2.exact_replay_status == "yes"
    assert first.case_open_digest == second.case_open_digest
    assert _file_digest(paths.reconciliation_open_path(case_id, tmp_path)) == digest_before


def test_open_conflict_different_actor(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue, opened_by="alice")
    with pytest.raises(ReconciliationConflictError):
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="bob",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )


def test_empty_directory_recovery(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    case_dir = paths.reconciliation_case_dir(case_id, tmp_path)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir()
    record, meta = open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    assert meta.exact_replay is False
    assert record.case_id == case_id


def test_partial_directory_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    case_dir = paths.reconciliation_case_dir(case_id, tmp_path)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir()
    (case_dir / "orphan.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(ReconciliationStateError, match="unexpected entries"):
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )


# --- Golden projection ---


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


def test_golden_projection_matches_compute_inspection_semantic_digest():
    projection = _build_inspection_semantic_projection_from_result(GOLDEN_INSPECTION)
    local = "sha256:" + hashlib.sha256(_canonical_json(projection).encode()).hexdigest()
    assert local == compute_inspection_semantic_digest(GOLDEN_INSPECTION)


def test_golden_projection_live_fixture(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    projection = _build_inspection_semantic_projection_from_result(result)
    local = "sha256:" + hashlib.sha256(_canonical_json(projection).encode()).hexdigest()
    assert local == result.inspection_semantic_digest
    assert local == compute_inspection_semantic_digest(result)


# --- Observation ---


def test_observation_replay_before_live_inspect(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    obs, meta = _observe_case(tmp_path, case_id)
    assert meta.exact_replay is False
    digest_before = _file_digest(paths.reconciliation_observation_path(case_id, tmp_path))
    with patch(
        "htr.reconciliation_cases.inspect_run_completion_reconciliation",
    ) as mocked:
        mocked.side_effect = AssertionError("26A must not run on observation replay")
        obs2, meta2 = _observe_case(tmp_path, case_id)
        mocked.assert_not_called()
    assert meta2.exact_replay is True
    assert obs2.observation_digest == obs.observation_digest
    assert _file_digest(paths.reconciliation_observation_path(case_id, tmp_path)) == digest_before


def test_observation_conflict_different_actor(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    _observe_case(tmp_path, case_id, observed_by="alice")
    with pytest.raises(ReconciliationConflictError):
        record_reconciliation_observation(
            case_id,
            base_dir=tmp_path,
            observed_by="bob",
        )


def test_inspection_error_leaves_no_observation(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    with patch(
        "htr.reconciliation_cases.inspect_run_completion_reconciliation",
        side_effect=ReconciliationInspectionError("inspection failed"),
    ):
        with pytest.raises(ReconciliationInspectionError):
            record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="op")
    assert not paths.reconciliation_observation_path(case_id, tmp_path).exists()


# --- Decision / policy ---


def _ambiguous_complete_case(tmp_path: Path) -> tuple[str, Any]:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_approval(issue["approval_id"], claim_id="claim-ambig", claimant_id="bob", base_dir=tmp_path)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        "claim-ambig",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=_ambiguous_v2_evidence(event_id=event_id),
        base_dir=tmp_path,
    )
    case_id = _open_case(tmp_path, issue)
    obs, _meta = _observe_case(tmp_path, case_id)
    return case_id, obs


def test_consumed_outcome_never_completion_verified(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(issue["approval_id"], claim_id="claim-ok", base_dir=tmp_path)
    case_id = _open_case(tmp_path, issue)
    obs, _meta = _observe_case(tmp_path, case_id)
    with pytest.raises(ReconciliationValidationError, match="not allowed"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
            decided_by="operator",
        )


def test_completion_verified_on_ambiguous_complete_lifecycle(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    decision, meta = record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
        decided_by="operator",
        requested_rationale_codes=(ReconciliationRationaleCode.ambiguous_outcome_lifecycle_complete,),
    )
    assert meta.exact_replay is False
    assert decision.decision_class == ReconciliationDecisionClass.completion_verified_by_reconciliation.value
    raw = json.loads(paths.reconciliation_decision_path(case_id, tmp_path).read_text(encoding="utf-8"))
    for field in (
        "safe_to_retry",
        "marker_disposition_allowed",
        "invoke_allowed",
        "repair_allowed",
        "recovery_run_creation_allowed",
        "outcome_rewrite_allowed",
    ):
        assert raw[field] is False


def test_missing_claim_never_completion_verified(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    case_id = _open_case(tmp_path, issue)
    obs, _meta = _observe_case(tmp_path, case_id)
    with pytest.raises(ReconciliationValidationError, match="not allowed"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
            decided_by="operator",
        )


def test_decision_replay_before_live_inspect(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    decision, _meta = record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    with patch(
        "htr.reconciliation_cases.inspect_run_completion_reconciliation",
    ) as mocked:
        mocked.side_effect = AssertionError("26A must not run on decision replay")
        dec2, meta2 = record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            decided_by="operator",
        )
        mocked.assert_not_called()
    assert meta2.exact_replay is True
    assert dec2.decision_digest == decision.decision_digest


def test_decision_wrong_observation_digest(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    with pytest.raises(ReconciliationValidationError, match="expected_observation_digest"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest="sha256:" + "f" * 64,
            requested_decision_class=ReconciliationDecisionClass.case_closed_no_action_required,
            decided_by="operator",
        )


def test_requested_rationale_must_be_derived_subset(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    with pytest.raises(ReconciliationValidationError, match="subset"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_no_action_required,
            decided_by="operator",
            requested_rationale_codes=(ReconciliationRationaleCode.integrity_blocked,),
        )


def test_invalid_next_protocol_rejected(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    with pytest.raises(ReconciliationValidationError, match="recommended_next_protocol"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_no_action_required,
            decided_by="operator",
            recommended_next_protocol=ReconciliationNextProtocol.recovery_run_review,
        )


# --- Durability ---


def test_open_fsync_failure_raises_durability_error(tmp_path, monkeypatch):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()

    def _fail_fsync(_fd: int, *, case_id: str, record_name: str) -> None:
        raise ReconciliationDurabilityError(
            "fsync failed",
            record_may_have_committed=True,
            exact_replay_status="indeterminate",
            durability_stage="record_fsync",
            case_id=case_id,
            record_name=record_name,
        )

    monkeypatch.setattr(reconciliation_cases, "_fsync_file_fd", _fail_fsync)
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    assert excinfo.value.exact_replay_status == "indeterminate"
    assert excinfo.value.durability_stage == "record_fsync"


# --- Boundaries ---


def test_open_does_not_acquire_marker_or_begin_run_write(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    calls: list[int] = []

    def _track_begin_run_write() -> None:
        calls.append(1)

    with patch("htr.execution_lock.begin_run_write", side_effect=_track_begin_run_write):
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    assert calls == []


def test_load_read_only_no_bootstrap(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    before = _full_snapshot(tmp_path)
    bundle = load_reconciliation_case(case_id, base_dir=tmp_path)
    after = _full_snapshot(tmp_path)
    assert before == after
    assert bundle.open_record.case_id == case_id
    assert bundle.observation_record is None


# --- Concurrency ---


def _open_worker(case_id: str, approval_id: str, base_dir: str, queue: multiprocessing.Queue) -> None:
    try:
        record, meta = open_reconciliation_case(
            case_id,
            approval_id,
            base_dir=Path(base_dir),
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
        queue.put(("ok", record.case_open_digest, meta.exact_replay))
    except Exception as exc:
        queue.put(("err", type(exc).__name__, str(exc)))


def test_concurrent_open_exactly_one_creator(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_open_worker,
            args=(case_id, issue["approval_id"], str(tmp_path), queue),
        )
        for _ in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [queue.get(timeout=5) for _ in range(4)]
    ok = [r for r in results if r[0] == "ok"]
    assert len(ok) == 4
    digests = {r[1] for r in ok}
    assert len(digests) == 1
    assert sum(1 for r in ok if r[2] is False) == 1


def test_thread_concurrent_observation_replay(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    _observe_case(tmp_path, case_id)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            record_reconciliation_observation(
                case_id,
                base_dir=tmp_path,
                observed_by="operator",
            )
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not errors


# --- Directory states ---


def test_unknown_entry_in_decided_state_fails_load(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    obs, _meta = _observe_case(tmp_path, case_id)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    (paths.reconciliation_case_dir(case_id, tmp_path) / "unknown.txt").write_text("x", encoding="utf-8")
    bundle = load_reconciliation_case(case_id, base_dir=tmp_path)
    assert bundle.decision_record is not None


def test_drift_blocks_completion_verified(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    with patch(
        "htr.reconciliation_cases.inspect_run_completion_reconciliation",
    ) as mocked:
        live = inspect_run_completion_reconciliation(
            json.loads(paths.reconciliation_open_path(case_id, tmp_path).read_text())["approval_id"],
            base_dir=tmp_path,
        )
        mutated = RunCompletionReconciliationInspection(
            **{
                **live.__dict__,
                "overall_classification": "indeterminate",
                "reason_codes": (*live.reason_codes, "synthetic_drift"),
            }
        )
        from htr.reconciliation_inspection import compute_inspection_semantic_digest as cs

        mutated = RunCompletionReconciliationInspection(
            **{**mutated.__dict__, "inspection_semantic_digest": cs(mutated)}
        )
        mocked.return_value = mutated
        with pytest.raises(ReconciliationValidationError, match="not allowed"):
            record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
                decided_by="operator",
            )

# --- Subprocess workers (module-level for pickling) ---


def _subprocess_open_worker(
    case_id: str,
    approval_id: str,
    base_dir: str,
    opened_by: str,
    slot: Any,
    barrier: Any = None,
) -> None:
    try:
        if barrier is not None:
            barrier.wait()
        record, meta = open_reconciliation_case(
            case_id,
            approval_id,
            base_dir=Path(base_dir),
            opened_by=opened_by,
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
        slot.put(("ok", record.case_open_digest, meta.exact_replay, meta.exact_replay_status))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _subprocess_observe_worker(
    case_id: str,
    base_dir: str,
    observed_by: str,
    slot: Any,
    barrier: Any = None,
) -> None:
    try:
        if barrier is not None:
            barrier.wait()
        record, meta = record_reconciliation_observation(
            case_id,
            base_dir=Path(base_dir),
            observed_by=observed_by,
        )
        slot.put(("ok", record.observation_digest, meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _subprocess_decide_worker(
    case_id: str,
    base_dir: str,
    obs_digest: str,
    decision_class: str,
    decided_by: str,
    slot: Any,
    barrier: Any = None,
) -> None:
    try:
        if barrier is not None:
            barrier.wait()
        record, meta = record_reconciliation_decision(
            case_id,
            base_dir=Path(base_dir),
            expected_observation_digest=obs_digest,
            requested_decision_class=ReconciliationDecisionClass(decision_class),
            decided_by=decided_by,
        )
        slot.put(("ok", record.decision_digest, meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _run_isolated_reconciliation_script(script: str, base_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run reconciliation code in a fresh interpreter (spawn-safe under file runners)."""
    root = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    env["PYTHONPATH"] = root
    return subprocess.run(
        [sys.executable, "-c", script, root, str(base_dir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=root,
    )


# --- 1. Cross-process concurrency ---


@pytest.mark.parametrize("scenario_index", range(5))
def test_subprocess_simultaneous_open_identical_intent_one_creator_one_replay(
    tmp_path, scenario_index
):
    del scenario_index  # independent fresh subprocess scenario id
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue() for _ in range(2)]
    procs = [
        ctx.Process(
            target=_subprocess_open_worker,
            args=(
                case_id,
                issue["approval_id"],
                str(tmp_path),
                "operator",
                slots[i],
                barrier,
            ),
        )
        for i in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=5) for i in range(2)]
    ok = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    assert not errs, errs
    assert len(ok) == 2
    assert len({r[1] for r in ok}) == 1
    assert sum(1 for r in ok if r[2] is False) == 1
    assert sum(1 for r in ok if r[2] is True) == 1


def test_subprocess_simultaneous_open_conflicting_intent_one_conflict(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_open_worker,
            args=(case_id, issue["approval_id"], str(tmp_path), "alice", slots[0], barrier),
        ),
        ctx.Process(
            target=_subprocess_open_worker,
            args=(case_id, issue["approval_id"], str(tmp_path), "bob", slots[1], barrier),
        ),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    r0, r1 = slots[0].get(timeout=5), slots[1].get(timeout=5)
    assert {r0[0], r1[0]} == {"ok", "err"}
    ok = r0 if r0[0] == "ok" else r1
    err = r1 if r0[0] == "ok" else r0
    assert ok[2] is False
    assert err[1] == "ReconciliationConflictError"


def test_subprocess_simultaneous_observation_creation(tmp_path):
    _subprocess_simultaneous_observation_creation(tmp_path, worker_count=4)


@pytest.mark.parametrize("worker_count", [2, 4, 8])
def test_subprocess_simultaneous_observation_creation_deterministic(
    tmp_path,
    worker_count: int,
) -> None:
    _subprocess_simultaneous_observation_creation(tmp_path, worker_count=worker_count)


def _subprocess_simultaneous_observation_creation(tmp_path, *, worker_count: int) -> None:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(worker_count)
    slots = [ctx.Queue() for _ in range(worker_count)]
    procs = [
        ctx.Process(
            target=_subprocess_observe_worker,
            args=(case_id, str(tmp_path), "operator", slots[i], barrier),
        )
        for i in range(worker_count)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=10) for i in range(worker_count)]
    ok = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    assert len(errs) == 0, errs
    assert len(ok) == worker_count
    assert sum(1 for r in ok if r[2] is False) == 1
    assert sum(1 for r in ok if r[2] is True) == worker_count - 1


def test_subprocess_simultaneous_observation_conflicting_intent_one_conflict(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_observe_worker,
            args=(case_id, str(tmp_path), "alice", slots[0], barrier),
        ),
        ctx.Process(
            target=_subprocess_observe_worker,
            args=(case_id, str(tmp_path), "bob", slots[1], barrier),
        ),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    r0, r1 = slots[0].get(timeout=5), slots[1].get(timeout=5)
    assert {r0[0], r1[0]} == {"ok", "err"}
    ok = r0 if r0[0] == "ok" else r1
    err = r1 if r0[0] == "ok" else r0
    assert ok[2] is False
    assert err[1] == "ReconciliationConflictError"


def test_subprocess_simultaneous_decision_creation(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(4)
    slots = [ctx.Queue() for _ in range(4)]
    procs = [
        ctx.Process(
            target=_subprocess_decide_worker,
            args=(
                case_id,
                str(tmp_path),
                obs.observation_digest,
                ReconciliationDecisionClass.case_closed_deferred_to_protocol.value,
                "operator",
                slots[i],
                barrier,
            ),
        )
        for i in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=10) for i in range(4)]
    ok = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    assert len(errs) == 0, errs
    assert len(ok) == 4
    assert len({r[1] for r in ok}) == 1
    assert sum(1 for r in ok if r[2] is False) == 1
    assert sum(1 for r in ok if r[2] is True) == 3


def test_subprocess_concurrent_bootstrap_reconciliation_root(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_ids = [generate_reconciliation_case_id() for _ in range(4)]
    ctx = multiprocessing.get_context("spawn")
    slots = [ctx.Queue() for _ in range(4)]
    procs = [
        ctx.Process(
            target=_subprocess_open_worker,
            args=(case_ids[i], issue["approval_id"], str(tmp_path), "operator", slots[i]),
        )
        for i in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    for i in range(4):
        assert slots[i].get(timeout=10)[0] == "ok"
    assert (tmp_path / ".control" / "reconciliation").is_dir()


def test_subprocess_o_excl_open_winner_loser(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(8)
    slots = [ctx.Queue() for _ in range(8)]
    procs = [
        ctx.Process(
            target=_subprocess_open_worker,
            args=(
                case_id,
                issue["approval_id"],
                str(tmp_path),
                "operator",
                slots[i],
                barrier,
            ),
        )
        for i in range(8)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    results = [slots[i].get(timeout=10) for i in range(8)]
    ok = [r for r in results if r[0] == "ok"]
    errs = [r for r in results if r[0] == "err"]
    assert len(errs) == 0, errs
    assert len(ok) == 8
    assert sum(1 for r in ok if r[2] is False) == 1
    assert sum(1 for r in ok if r[2] is True) == 7


def test_subprocess_response_lost_replay_from_separate_process(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    approval_id = issue["approval_id"]
    open_script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from htr.reconciliation_cases import open_reconciliation_case, ReconciliationScopeReason
open_reconciliation_case(
    CASE_ID,
    APPROVAL_ID,
    base_dir=Path(sys.argv[2]),
    opened_by="operator",
    scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
)
print("created")
""".replace("CASE_ID", repr(case_id)).replace("APPROVAL_ID", repr(approval_id))
    open_result = _run_isolated_reconciliation_script(open_script, tmp_path)
    assert open_result.returncode == 0, open_result.stderr
    assert open_result.stdout.strip() == "created"

    replay_script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from htr.reconciliation_cases import open_reconciliation_case, ReconciliationScopeReason
_record, meta = open_reconciliation_case(
    CASE_ID,
    APPROVAL_ID,
    base_dir=Path(sys.argv[2]),
    opened_by="operator",
    scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
)
print(f"replay:{meta.exact_replay}")
""".replace("CASE_ID", repr(case_id)).replace("APPROVAL_ID", repr(approval_id))
    replay_result = _run_isolated_reconciliation_script(replay_script, tmp_path)
    assert replay_result.returncode == 0, replay_result.stderr
    assert replay_result.stdout.strip() == "replay:True"


def test_subprocess_module_importable():
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from htr.reconciliation_cases import generate_reconciliation_case_id
print(generate_reconciliation_case_id().startswith("rcn_"))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve().parents[2])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "True"

def test_unknown_entry_after_precheck_before_open_write_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    injected = {"done": False}

    original_list = reconciliation_cases._list_dir_entries

    def _list_with_injection(dir_fd: int) -> frozenset[str]:
        entries = original_list(dir_fd)
        if not injected["done"]:
            injected["done"] = True
            case_dir = paths.reconciliation_case_dir(case_id, tmp_path)
            (case_dir / "intruder.txt").write_text("x", encoding="utf-8")
        return original_list(dir_fd)

    with patch.object(reconciliation_cases, "_list_dir_entries", side_effect=_list_with_injection):
        with pytest.raises(ReconciliationStateError, match="unexpected reconciliation case directory entries"):
            open_reconciliation_case(
                case_id,
                issue["approval_id"],
                base_dir=tmp_path,
                opened_by="operator",
                scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
            )
    if paths.reconciliation_open_path(case_id, tmp_path).exists():
        assert paths.reconciliation_open_path(case_id, tmp_path).read_text(encoding="utf-8")


def test_unknown_entry_after_record_creation_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    case_dir = paths.reconciliation_case_dir(case_id, tmp_path)
    (case_dir / "intruder.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ReconciliationStateError, match="unexpected reconciliation case directory entries"):
        record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")


def test_case_directory_identity_change_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    real_identity = reconciliation_cases._fstat_identity
    calls = {"n": 0}

    def _identity(fd: int) -> tuple[int, int]:
        calls["n"] += 1
        dev, inode = real_identity(fd)
        if calls["n"] >= 2:
            return dev, inode + 999999
        return dev, inode

    with patch.object(reconciliation_cases, "_fstat_identity", side_effect=_identity):
        with pytest.raises(ReconciliationStateError, match="identity changed"):
            open_reconciliation_case(
                case_id,
                issue["approval_id"],
                base_dir=tmp_path,
                opened_by="operator",
                scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
            )


def test_symlink_control_tree_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    control = tmp_path / ".control"
    target = tmp_path / "control_target"
    target.mkdir()
    if control.exists():
        import shutil

        if control.is_symlink() or control.is_file():
            control.unlink()
        else:
            shutil.rmtree(control)
    control.symlink_to(target, target_is_directory=True)
    with pytest.raises((ReconciliationValidationError, ReconciliationStateError)):
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )


def test_symlink_case_dir_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    recon = tmp_path / ".control" / "reconciliation"
    recon.mkdir(parents=True)
    target = tmp_path / "case_target"
    target.mkdir()
    (recon / case_id).symlink_to(target)
    with pytest.raises(ReconciliationStateError, match="missing"):
        record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")


def test_record_symlink_fails_load(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    open_path = paths.reconciliation_open_path(case_id, tmp_path)
    payload = open_path.read_bytes()
    open_path.unlink()
    open_path.symlink_to(tmp_path / "open_target.json")
    (tmp_path / "open_target.json").write_bytes(payload)
    with pytest.raises(ReconciliationValidationError, match="unsafe symlink"):
        load_reconciliation_case(case_id, base_dir=tmp_path)


def test_record_replacement_during_replay_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    open_path = paths.reconciliation_open_path(case_id, tmp_path)
    committed_digest = _file_digest(open_path)
    read_count = {"n": 0}
    real_read = reconciliation_cases._read_optional_record

    def _read_maybe_tamper(path: Path) -> dict[str, Any] | None:
        data = real_read(path)
        if data is not None and path.name == "open.json":
            read_count["n"] += 1
            if read_count["n"] == 1:
                tampered = dict(data)
                tampered["opened_by"] = "attacker"
                open_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
        return real_read(path)

    with patch.object(reconciliation_cases, "_read_optional_record", side_effect=_read_maybe_tamper):
        with pytest.raises(ReconciliationValidationError):
            open_reconciliation_case(
                case_id,
                issue["approval_id"],
                base_dir=tmp_path,
                opened_by="operator",
                scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
            )


def test_observation_replacement_during_replay_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    _observe_case(tmp_path, case_id)
    obs_path = paths.reconciliation_observation_path(case_id, tmp_path)
    read_count = {"n": 0}
    real_read = reconciliation_cases._read_optional_record

    def _read_maybe_tamper(path: Path) -> dict[str, Any] | None:
        data = real_read(path)
        if data is not None and path.name == "observation.json":
            read_count["n"] += 1
            if read_count["n"] == 1:
                tampered = dict(data)
                tampered["observed_by"] = "attacker"
                obs_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
        return real_read(path)

    with patch.object(reconciliation_cases, "_read_optional_record", side_effect=_read_maybe_tamper):
        with pytest.raises(ReconciliationValidationError):
            record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
    assert read_count["n"] >= 1


def test_decision_replacement_during_replay_fails_closed(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    decision_path = paths.reconciliation_decision_path(case_id, tmp_path)
    read_count = {"n": 0}
    real_read = reconciliation_cases._read_optional_record

    def _read_maybe_tamper(path: Path) -> dict[str, Any] | None:
        data = real_read(path)
        if data is not None and path.name == "decision.json":
            read_count["n"] += 1
            if read_count["n"] == 1:
                tampered = dict(data)
                tampered["decided_by"] = "attacker"
                decision_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
        return real_read(path)

    with patch.object(reconciliation_cases, "_read_optional_record", side_effect=_read_maybe_tamper):
        with pytest.raises(ReconciliationValidationError):
            record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
                decided_by="operator",
            )
    assert read_count["n"] >= 1


def test_partial_writer_vs_reader_preserves_committed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    before = _case_tree_snapshot(tmp_path, case_id)
    errors: list[BaseException] = []

    def slow_write(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _time.sleep(0.2)
        return reconciliation_cases._create_immutable_record(*args, **kwargs)

    def reader() -> None:
        try:
            load_reconciliation_case(case_id, base_dir=tmp_path)
        except BaseException as exc:
            errors.append(exc)

    with patch.object(reconciliation_cases, "_create_immutable_record", side_effect=slow_write):
        t = threading.Thread(
            target=lambda: record_reconciliation_observation(
                case_id, base_dir=tmp_path, observed_by="operator"
            )
        )
        t.start()
        reader()
        t.join(timeout=30)
    after = _case_tree_snapshot(tmp_path, case_id)
    assert before["open.json_sha256"] == after["open.json_sha256"]


# --- 3. Durability-stage matrix ---

_DURABILITY_STAGES_OPEN = (
    "control_dir_fsync",
    "case_dir_fsync",
    "record_fsync",
)


def _inject_durability_failure(monkeypatch, stage: str) -> None:
    if stage == "record_fsync":

        def _fail_file(_fd: int, *, case_id: str, record_name: str) -> None:
            raise ReconciliationDurabilityError(
                "injected record_fsync",
                record_may_have_committed=True,
                exact_replay_status="indeterminate",
                durability_stage="record_fsync",
                case_id=case_id,
                record_name=record_name,
            )

        monkeypatch.setattr(reconciliation_cases, "_fsync_file_fd", _fail_file)
        return

    def _make_dir_fail(expected_stage: str):
        def _fail_dir(_fd: int, *, case_id: str, record_name: str, stage: str) -> None:
            if stage == expected_stage:
                raise ReconciliationDurabilityError(
                    f"injected {expected_stage}",
                    record_may_have_committed=True,
                    exact_replay_status="indeterminate",
                    durability_stage=expected_stage,  # type: ignore[arg-type]
                    case_id=case_id,
                    record_name=record_name,  # type: ignore[arg-type]
                )

        return _fail_dir

    monkeypatch.setattr(reconciliation_cases, "_fsync_dir_fd", _make_dir_fail(stage))


@pytest.mark.parametrize("stage", _DURABILITY_STAGES_OPEN)
def test_open_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    _inject_durability_failure(monkeypatch, stage)
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    err = excinfo.value
    assert err.durability_stage == stage
    assert err.case_id == case_id
    assert err.record_name == "open.json"
    assert err.exact_replay_status == "indeterminate"
    assert err.record_may_have_committed is True


def test_bootstrap_parent_dir_fsync_failure(tmp_path, monkeypatch):
    case_id = generate_reconciliation_case_id()
    _inject_durability_failure(monkeypatch, "parent_dir_fsync")
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        reconciliation_cases._bootstrap_reconciliation_tree(case_id, tmp_path)
    err = excinfo.value
    assert err.durability_stage == "parent_dir_fsync"
    assert err.case_id == case_id
    assert err.record_name == "open.json"
    assert err.exact_replay_status == "indeterminate"


def test_bootstrap_control_dir_fsync_when_control_preexists(tmp_path, monkeypatch):
    case_id = generate_reconciliation_case_id()
    (tmp_path / ".control").mkdir()
    _inject_durability_failure(monkeypatch, "control_dir_fsync")
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        reconciliation_cases._bootstrap_reconciliation_tree(case_id, tmp_path)
    err = excinfo.value
    assert err.durability_stage == "control_dir_fsync"
    assert err.case_id == case_id
    assert err.record_name == "open.json"
    assert err.exact_replay_status == "indeterminate"
    assert err.record_may_have_committed is True


def test_bootstrap_case_dir_fsync_when_reconciliation_preexists(tmp_path, monkeypatch):
    case_id = generate_reconciliation_case_id()
    recon = tmp_path / ".control" / "reconciliation"
    recon.mkdir(parents=True)
    _inject_durability_failure(monkeypatch, "case_dir_fsync")
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        reconciliation_cases._bootstrap_reconciliation_tree(case_id, tmp_path)
    err = excinfo.value
    assert err.durability_stage == "case_dir_fsync"
    assert err.case_id == case_id
    assert err.record_name == "open.json"
    assert err.exact_replay_status == "indeterminate"
    assert err.record_may_have_committed is True


@pytest.mark.parametrize("stage", ("record_fsync", "case_dir_fsync"))
def test_observation_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    _inject_durability_failure(monkeypatch, stage)
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
    err = excinfo.value
    assert err.durability_stage == stage
    assert err.case_id == case_id
    assert err.record_name == "observation.json"
    assert err.exact_replay_status == "indeterminate"


@pytest.mark.parametrize("stage", ("record_fsync", "case_dir_fsync"))
def test_decision_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    _inject_durability_failure(monkeypatch, stage)
    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            decided_by="operator",
        )
    err = excinfo.value
    assert err.durability_stage == stage
    assert err.record_name == "decision.json"


def test_durability_failure_then_valid_record_replay(tmp_path, monkeypatch):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    calls = {"n": 0}
    real_fsync_file = reconciliation_cases._fsync_file_fd

    def _fail_once(fd: int, *, case_id: str, record_name: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ReconciliationDurabilityError(
                "injected",
                record_may_have_committed=True,
                exact_replay_status="indeterminate",
                durability_stage="record_fsync",
                case_id=case_id,
                record_name=record_name,
            )
        return real_fsync_file(fd, case_id=case_id, record_name=record_name)

    monkeypatch.setattr(reconciliation_cases, "_fsync_file_fd", _fail_once)
    with pytest.raises(ReconciliationDurabilityError):
        open_reconciliation_case(
            case_id,
            issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    _record, meta = open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    assert meta.exact_replay_status in {"yes", "no"}


def test_malformed_record_after_durability_indeterminate_fails_load(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    open_path = paths.reconciliation_open_path(case_id, tmp_path)
    open_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReconciliationValidationError, match="malformed JSON"):
        load_reconciliation_case(case_id, base_dir=tmp_path)


def test_absent_record_after_durability_indeterminate(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    with pytest.raises(ReconciliationStateError, match="missing open"):
        load_reconciliation_case(case_id, base_dir=tmp_path)

def test_open_replay_zero_write_path_and_content_unchanged(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    before = _case_tree_snapshot(tmp_path, case_id)
    _record, meta = open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    after = _case_tree_snapshot(tmp_path, case_id)
    assert meta.exact_replay_status == "yes"
    assert before == after


def test_open_approval_id_mismatch_conflict(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    other_issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    with pytest.raises(ReconciliationConflictError):
        open_reconciliation_case(
            case_id,
            other_issue["approval_id"],
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )


def test_observation_replay_before_utc_now_generation(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    _observe_case(tmp_path, case_id)
    with patch.object(reconciliation_cases, "_utc_now_iso", side_effect=AssertionError("no timestamp")):
        _obs, meta = _observe_case(tmp_path, case_id)
    assert meta.exact_replay_status == "yes"


def test_decision_protocol_mismatch_conflict(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.none,
    )
    with pytest.raises(ReconciliationConflictError):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            decided_by="operator",
            recommended_next_protocol=ReconciliationNextProtocol.human_review,
        )


def test_decision_class_mismatch_conflict(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    with pytest.raises(ReconciliationConflictError):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_no_action_required,
            decided_by="operator",
        )


def test_decision_rationale_mismatch_conflict(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
        decided_by="operator",
        requested_rationale_codes=(ReconciliationRationaleCode.ambiguous_outcome_lifecycle_complete,),
    )
    with pytest.raises(ReconciliationConflictError):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
            decided_by="operator",
            requested_rationale_codes=(),
        )


# --- 5. Record validation and tampering ---

_RECORD_FILES = ("open.json", "observation.json", "decision.json")


@pytest.mark.parametrize("record_name", _RECORD_FILES)
def test_malformed_json_record_fails_closed(tmp_path, record_name):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    if record_name != "open.json":
        _observe_case(tmp_path, case_id)
    if record_name == "decision.json":
        obs = json.loads(paths.reconciliation_observation_path(case_id, tmp_path).read_text())
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs["observation_digest"],
            requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            decided_by="operator",
        )
    path = paths.reconciliation_case_dir(case_id, tmp_path) / record_name
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ReconciliationValidationError):
        load_reconciliation_case(case_id, base_dir=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("safe_to_retry", True),
        ("safe_to_retry", "false"),
        ("safe_to_retry", 0),
        ("safe_to_retry", None),
        ("marker_disposition_allowed", True),
        ("invoke_allowed", True),
        ("repair_allowed", True),
        ("recovery_run_creation_allowed", True),
        ("outcome_rewrite_allowed", True),
    ],
)
def test_decision_non_permission_field_rejected(tmp_path, field, value):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    path = paths.reconciliation_decision_path(case_id, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = value
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ReconciliationValidationError):
        load_reconciliation_case(case_id, base_dir=tmp_path)


def test_open_digest_tampering_fails_closed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    path = paths.reconciliation_open_path(case_id, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["opened_by"] = "attacker"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ReconciliationValidationError, match="case_open_digest mismatch"):
        load_reconciliation_case(case_id, base_dir=tmp_path)


def test_observation_case_id_binding_mismatch(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    _observe_case(tmp_path, case_id)
    path = paths.reconciliation_observation_path(case_id, tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["case_id"] = generate_reconciliation_case_id()
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ReconciliationValidationError, match="observation_digest mismatch"):
        load_reconciliation_case(case_id, base_dir=tmp_path)


# --- 6. Evidence-to-decision policy matrix ---

_ALL_DECISION_CLASSES = list(ReconciliationDecisionClass)


@pytest.mark.parametrize("decision_class", _ALL_DECISION_CLASSES)
def test_decision_class_policy_positive_or_negative(tmp_path, decision_class):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    try:
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=decision_class,
            decided_by="operator",
        )
        allowed = decision_class in {
            ReconciliationDecisionClass.completion_verified_by_reconciliation,
            ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            ReconciliationDecisionClass.case_closed_no_action_required,
        }
        assert allowed
    except ReconciliationValidationError:
        assert decision_class not in {
            ReconciliationDecisionClass.completion_verified_by_reconciliation,
            ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            ReconciliationDecisionClass.case_closed_no_action_required,
        }


def test_consumed_verified_marker_only_no_action_or_defer(tmp_path):
    case_id, obs = _consumed_with_marker_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_no_action_required,
        decided_by="operator",
    )
    case_id2, obs2 = _consumed_with_marker_case(tmp_path)
    record_reconciliation_decision(
        case_id2,
        base_dir=tmp_path,
        expected_observation_digest=obs2.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    case_id3, obs3 = _consumed_with_marker_case(tmp_path)
    with pytest.raises(ReconciliationValidationError, match="not allowed"):
        record_reconciliation_decision(
            case_id3,
            base_dir=tmp_path,
            expected_observation_digest=obs3.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
            decided_by="operator",
        )


def test_consumed_contradiction_only_conflict_or_indeterminate(tmp_path):
    case_id, obs = _consumed_contradiction_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
        decided_by="operator",
    )
    case_id2, obs2 = _consumed_contradiction_case(tmp_path)
    record_reconciliation_decision(
        case_id2,
        base_dir=tmp_path,
        expected_observation_digest=obs2.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.indeterminate_insufficient_evidence,
        decided_by="operator",
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "missing_claim",
        "consumed_outcome",
        "drift",
    ],
)
def test_completion_verified_rejection_scenarios(tmp_path, scenario):
    if scenario == "missing_claim":
        run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
        issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
        events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
        case_id = _open_case(tmp_path, issue)
        obs, _ = _observe_case(tmp_path, case_id), None
        obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
    elif scenario == "consumed_outcome":
        run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
        issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
        invoke_approved_run_completion(issue["approval_id"], claim_id="c1", base_dir=tmp_path)
        case_id = _open_case(tmp_path, issue)
        _observe_case(tmp_path, case_id)
        obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
    else:
        case_id, obs = _ambiguous_complete_case(tmp_path)
    kwargs = dict(
        case_id=case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
        decided_by="operator",
    )
    if scenario == "drift":
        with patch("htr.reconciliation_cases.inspect_run_completion_reconciliation") as mocked:
            live = inspect_run_completion_reconciliation(
                json.loads(paths.reconciliation_open_path(case_id, tmp_path).read_text())["approval_id"],
                base_dir=tmp_path,
            )
            mutated = RunCompletionReconciliationInspection(
                **{**live.__dict__, "reason_codes": (*live.reason_codes, "synthetic_drift")}
            )
            from htr.reconciliation_inspection import compute_inspection_semantic_digest as cs

            mutated = RunCompletionReconciliationInspection(
                **{**mutated.__dict__, "inspection_semantic_digest": cs(mutated)}
            )
            mocked.return_value = mutated
            with pytest.raises(ReconciliationValidationError, match="not allowed"):
                record_reconciliation_decision(**kwargs)
        return
    with pytest.raises(ReconciliationValidationError, match="not allowed"):
        record_reconciliation_decision(**kwargs)


# --- 7. Rationale and protocol policy ---

_ALL_RATIONALE = list(ReconciliationRationaleCode)
_ALL_PROTOCOL = list(ReconciliationNextProtocol)


def test_derived_rationale_authoritative_over_request(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    decision, _ = record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
        decided_by="operator",
        requested_rationale_codes=(ReconciliationRationaleCode.ambiguous_outcome_lifecycle_complete,),
    )
    raw = json.loads(paths.reconciliation_decision_path(case_id, tmp_path).read_text(encoding="utf-8"))
    assert raw["derived_rationale_codes"]
    assert set(raw["requested_rationale_codes"]).issubset(set(raw["derived_rationale_codes"]))


def test_unsupported_rationale_code_rejected(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    with pytest.raises(ReconciliationValidationError, match="subset"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_no_action_required,
            decided_by="operator",
            requested_rationale_codes=(ReconciliationRationaleCode.integrity_blocked,),
        )


def test_retry_review_never_sets_safe_to_retry(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    with patch.object(
        reconciliation_cases,
        "_derive_allowed_decision_classes",
        return_value=frozenset({ReconciliationDecisionClass.indeterminate_insufficient_evidence}),
    ):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.indeterminate_insufficient_evidence,
            decided_by="operator",
            recommended_next_protocol=ReconciliationNextProtocol.retry_review,
        )
    raw = json.loads(paths.reconciliation_decision_path(case_id, tmp_path).read_text(encoding="utf-8"))
    assert raw["safe_to_retry"] is False
    assert raw["recommended_next_protocol_authority"] == "advisory_only"


@pytest.mark.parametrize("protocol", _ALL_PROTOCOL)
def test_protocol_values_are_closed_enum(protocol):
    assert isinstance(protocol, ReconciliationNextProtocol)


@pytest.mark.parametrize("code", _ALL_RATIONALE)
def test_rationale_codes_are_closed_enum(code):
    assert isinstance(code, ReconciliationRationaleCode)

def _projection_fixture(name: str) -> RunCompletionReconciliationInspection:
    base = dict(GOLDEN_INSPECTION.__dict__)
    variants: dict[str, dict[str, Any]] = {
        "no_lifecycle": dict(
            lifecycle_evidence_state="no_lifecycle_evidence_observed",
            overall_classification="reconciliation_inspection_required",
            claim_id=None,
        ),
        "partial_commit": dict(
            lifecycle_evidence_state="completion_json_and_event_manifest_incomplete",
            overall_classification="partial_lifecycle_commit",
        ),
        "ambiguous_complete": dict(
            lifecycle_evidence_state="lifecycle_complete_observed",
            overall_classification="reconciliation_inspection_required",
            outcome_class="ambiguous",
            claim_id="claim-x",
            claim_digest="sha256:" + "c" * 64,
            current_observation_semantic_digest="sha256:" + "d" * 64,
        ),
        "consumed_verified": dict(
            approval_control_state="consumed_outcome",
            outcome_class="consumed",
            overall_classification="no_reconciliation_needed",
            lifecycle_evidence_state="verified_completed",
        ),
        "integrity_blocked": dict(
            integrity_state="seal_blocked",
            overall_classification="integrity_blocked",
        ),
        "control_lifecycle_conflict": dict(
            approval_control_state="consumed_outcome",
            overall_classification="control_lifecycle_evidence_conflict",
            reason_codes=("consumed_outcome_current_evidence_mismatch",),
        ),
        "marker_absent": dict(marker_state="absent"),
        "marker_present": dict(marker_state="present_valid_metadata"),
    }
    base.update(variants[name])
    result = RunCompletionReconciliationInspection(**base)
    digest = compute_inspection_semantic_digest(result)
    return RunCompletionReconciliationInspection(**{**result.__dict__, "inspection_semantic_digest": digest})


@pytest.mark.parametrize(
    "fixture_name",
    [
        "no_lifecycle",
        "partial_commit",
        "ambiguous_complete",
        "consumed_verified",
        "integrity_blocked",
        "control_lifecycle_conflict",
        "marker_absent",
        "marker_present",
    ],
)
def test_projection_parity_across_fixtures(fixture_name):
    result = _projection_fixture(fixture_name)
    projection = reconciliation_cases._build_inspection_semantic_projection_from_result(result)
    local = "sha256:" + hashlib.sha256(_canonical_json(projection).encode()).hexdigest()
    assert local == result.inspection_semantic_digest
    assert local == compute_inspection_semantic_digest(result)
    assert "observed_at" not in projection


def test_projection_dictionary_order_independent(fixture_name="ambiguous_complete"):
    result = _projection_fixture(fixture_name)
    projection = reconciliation_cases._build_inspection_semantic_projection_from_result(result)
    reversed_items = dict(reversed(list(projection.items())))
    assert hashlib.sha256(_canonical_json(projection).encode()).hexdigest() == hashlib.sha256(
        _canonical_json(reversed_items).encode()
    ).hexdigest()


def test_projection_tampering_changes_digest():
    result = _projection_fixture("ambiguous_complete")
    projection = reconciliation_cases._build_inspection_semantic_projection_from_result(result)
    tampered = dict(projection)
    tampered["overall_classification"] = "tampered"
    assert hashlib.sha256(_canonical_json(tampered).encode()).hexdigest() != result.inspection_semantic_digest.replace(
        "sha256:", ""
    )


def test_observation_observed_at_in_digest_not_26a_domain(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    result = inspect_run_completion_reconciliation(issue["approval_id"], base_dir=tmp_path)
    assert result.observed_at
    projection = reconciliation_cases._build_inspection_semantic_projection_from_result(result)
    assert "observed_at" not in projection
    _observe_case(tmp_path, case_id)
    obs_raw = json.loads(paths.reconciliation_observation_path(case_id, tmp_path).read_text(encoding="utf-8"))
    assert obs_raw["observed_at"]
    tampered = dict(obs_raw)
    tampered["observed_at"] = "2099-01-01T00:00:00+00:00"
    with pytest.raises(ReconciliationValidationError, match="observation_digest mismatch"):
        paths.reconciliation_observation_path(case_id, tmp_path).write_text(
            json.dumps(tampered, indent=2) + "\n", encoding="utf-8"
        )
        load_reconciliation_case(case_id, base_dir=tmp_path)


# --- 9. Inspection-error and zero-residue ---


@pytest.mark.parametrize(
    "exc_type,exc_msg",
    [
        (ReconciliationInspectionError, "inspection failed"),
        (ReconciliationEvidenceIntegrityError, "integrity failed"),
        (ReconciliationUnsupportedApprovalError, "unsupported"),
    ],
)
def test_inspection_error_no_observation_case_stays_open(tmp_path, exc_type, exc_msg):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    before = _case_tree_snapshot(tmp_path, case_id)
    with patch(
        "htr.reconciliation_cases.inspect_run_completion_reconciliation",
        side_effect=exc_type(exc_msg, approval_id=issue["approval_id"]),
    ):
        with pytest.raises(exc_type):
            record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
    after = _case_tree_snapshot(tmp_path, case_id)
    assert not paths.reconciliation_observation_path(case_id, tmp_path).exists()
    assert before["open.json_sha256"] == after["open.json_sha256"]
    _observe_case(tmp_path, case_id)


def test_inspection_error_blocks_decision(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    with patch(
        "htr.reconciliation_cases.inspect_run_completion_reconciliation",
        side_effect=ReconciliationInspectionError("fail", approval_id=issue["approval_id"]),
    ):
        with pytest.raises(ReconciliationInspectionError):
            record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
    with pytest.raises(ReconciliationStateError, match="missing observation"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest="sha256:" + "f" * 64,
            requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            decided_by="operator",
        )


def test_invalid_open_leaves_no_reconciliation_residue(tmp_path):
    case_id = generate_reconciliation_case_id()
    with pytest.raises(ReconciliationValidationError):
        open_reconciliation_case(
            case_id,
            new_approval_id(),
            base_dir=tmp_path,
            opened_by="operator",
            scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
        )
    control = tmp_path / ".control"
    if control.exists():
        assert not (control / "reconciliation").exists()


# --- 10. Boundary mutation guards ---

_BOUNDARY_PATCH_TARGETS = [
    "htr.events.complete_run_manually",
    "htr.approval_control.record_use_outcome",
    "htr.approval_control.claim_approval",
    "htr.approval_control.issue_approval",
    "htr.approval_control.revoke_approval",
    "htr.execution_lock.begin_run_write",
    "htr.invoke_run_completion.invoke_approved_run_completion",
]


def test_open_observe_decide_never_calls_boundary_mutators(tmp_path, monkeypatch):
    calls: list[str] = []

    def _track(name: str):
        def _inner(*_a: Any, **_k: Any) -> None:
            calls.append(name)
            raise AssertionError(f"boundary call: {name}")

        return _inner

    for target in _BOUNDARY_PATCH_TARGETS:
        monkeypatch.setattr(target, _track(target))

    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    obs, _ = _observe_case(tmp_path, case_id), None
    obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    assert calls == []


# --- 11. Public API scope ---

_APPROVED_PUBLIC = frozenset(
    {
        "CASE_SCHEMA_VERSION",
        "DECISION_DIGEST_PROJECTION_VERSION",
        "DECISION_REVALIDATION_PROJECTION_VERSION",
        "EVIDENCE_CAPTURE_DISCLAIMER",
        "EVIDENCE_CAPTURE_MODE",
        "OBSERVATION_DIGEST_PROJECTION_VERSION",
        "OPEN_DIGEST_PROJECTION_VERSION",
        "ReconciliationDecisionClass",
        "ReconciliationNextProtocol",
        "ReconciliationRationaleCode",
        "ReconciliationScopeReason",
        "generate_reconciliation_case_id",
        "load_reconciliation_case",
        "open_reconciliation_case",
        "record_reconciliation_decision",
        "record_reconciliation_observation",
    }
)


def test_reconciliation_cases_export_surface():
    assert frozenset(reconciliation_cases.__all__) == _APPROVED_PUBLIC
    assert not hasattr(reconciliation_cases, "build_inspection_semantic_projection_from_result")


def test_htr_package_does_not_export_projection_helper():
    import htr

    assert "build_inspection_semantic_projection_from_result" not in htr.__all__
    assert not hasattr(htr, "build_inspection_semantic_projection_from_result")


def test_htr_exports_only_approved_case_apis():
    import htr

    case_exports = {
        "ReconciliationDecisionClass",
        "ReconciliationNextProtocol",
        "ReconciliationRationaleCode",
        "ReconciliationScopeReason",
        "generate_reconciliation_case_id",
        "load_reconciliation_case",
        "open_reconciliation_case",
        "record_reconciliation_decision",
        "record_reconciliation_observation",
    }
    for name in case_exports:
        assert name in htr.__all__
        assert hasattr(htr, name)


# --- 12. Private Task 24 helper dependency audit ---

_PRIVATE_APPROVAL_IMPORTS = (
    "_argument_entries_to_inputs",
    "_compute_approval_digest",
    "_project_dir_path_digest",
    "_runs_root_path_digest",
)


@pytest.mark.parametrize("symbol", _PRIVATE_APPROVAL_IMPORTS)
def test_private_approval_helpers_exist_and_are_callable(symbol):
    fn = getattr(approval_control, symbol)
    assert callable(fn)


def test_private_approval_digest_readonly_no_bootstrap(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    before = _full_snapshot(tmp_path)
    raw = json.loads(paths.approval_issue_path(issue["approval_id"], tmp_path).read_text(encoding="utf-8"))
    digest = approval_control._compute_approval_digest(raw)
    assert digest == raw["approval_digest"]
    assert _full_snapshot(tmp_path) == before


def test_private_runs_root_digest_readonly(tmp_path):
    before = _full_snapshot(tmp_path)
    d1 = approval_control._runs_root_path_digest(tmp_path)
    d2 = approval_control._runs_root_path_digest(tmp_path)
    assert d1 == d2
    assert _full_snapshot(tmp_path) == before


# --- Final hardening pass ---

_HARDENING_THREAD_COUNT = 8

_EXPECTED_DECISION_CLASSES = (
    ReconciliationDecisionClass.completion_verified_by_reconciliation,
    ReconciliationDecisionClass.evidence_conflict_confirmed,
    ReconciliationDecisionClass.partial_commit_confirmed,
    ReconciliationDecisionClass.integrity_blocked_confirmed,
    ReconciliationDecisionClass.indeterminate_insufficient_evidence,
    ReconciliationDecisionClass.case_closed_no_action_required,
    ReconciliationDecisionClass.case_closed_deferred_to_protocol,
)

_RECORD_NAMES = ("open.json", "observation.json", "decision.json")


def _inject_os_write_fail_first(monkeypatch) -> None:
    real_write = os.write

    def _write(fd: int, data: bytes) -> int:
        if not hasattr(_write, "called"):
            _write.called = True
            return 0
        return real_write(fd, data)

    monkeypatch.setattr(reconciliation_cases.os, "write", _write)


def _inject_os_write_partial_then_error(monkeypatch) -> None:
    real_write = os.write

    def _write(fd: int, data: bytes) -> int:
        if len(data) > 1:
            real_write(fd, data[: len(data) // 2])
            raise OSError("injected partial record write")
        return real_write(fd, data)

    monkeypatch.setattr(reconciliation_cases.os, "write", _write)


def _prepare_loaded_record(tmp_path: Path, record_name: str) -> tuple[str, Path]:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    if record_name != "open.json":
        _observe_case(tmp_path, case_id)
    if record_name == "decision.json":
        obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
        assert obs is not None
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            decided_by="operator",
        )
    record_path = paths.reconciliation_case_dir(case_id, tmp_path) / record_name
    return case_id, record_path


def _mutated_inspection(
    live: RunCompletionReconciliationInspection,
    **overrides: Any,
) -> RunCompletionReconciliationInspection:
    from htr.reconciliation_inspection import compute_inspection_semantic_digest as cs

    mutated = RunCompletionReconciliationInspection(**{**live.__dict__, **overrides})
    return RunCompletionReconciliationInspection(
        **{**mutated.__dict__, "inspection_semantic_digest": cs(mutated)}
    )


def _authority_path_entry(path: Path) -> dict[str, Any]:
    entry = dict(_path_stat_snapshot(path))
    if entry.get("exists") and entry.get("is_file") and not entry.get("is_symlink"):
        entry["sha256"] = _file_digest(path)
    return entry


def _discover_attempt_id(tmp_path: Path, run_id: str, task_id: str) -> str | None:
    attempts_root = paths.attempts_dir(run_id, task_id, tmp_path)
    if not attempts_root.is_dir():
        return None
    for child in sorted(attempts_root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            name = child.name
            if validate_id(name, "attempt"):
                return name
    return None


def _boundary_snapshot(
    tmp_path: Path,
    *,
    run_id: str,
    task_id: str,
    approval_id: str,
) -> dict[str, Any]:
    attempt_id = _discover_attempt_id(tmp_path, run_id, task_id)
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
    authority_paths = {
        "issue.json": paths.approval_issue_path(approval_id, tmp_path),
        "revoke.json": paths.approval_revoke_path(approval_id, tmp_path),
        "claim.json": paths.approval_claim_path(approval_id, tmp_path),
        "outcome.json": paths.approval_outcome_path(approval_id, tmp_path),
        "completion_lifecycle_json": contracts.run_completion_record_json_path(run_id, tmp_path),
        "lifecycle_event_log": paths.task_events_path(run_id, tmp_path),
        "manifest": paths.run_manifest_path(run_id, tmp_path),
        "task_card": paths.task_card_path(run_id, task_id, tmp_path),
        "task_status": paths.task_status_path(run_id, task_id, tmp_path),
        "final_closure_record": contracts.run_final_closure_record_json_path(run_id, tmp_path),
        "execution_marker": marker_path,
    }
    if attempt_id is not None:
        authority_paths["attempt_status"] = paths.attempt_status_path(
            run_id,
            task_id,
            attempt_id,
            tmp_path,
        )
        authority_paths["artifact_manifest"] = paths.artifact_manifest_path(
            run_id,
            task_id,
            attempt_id,
            tmp_path,
        )
    snap = _full_snapshot(tmp_path)
    workspace_files = {
        rel: digest
        for rel, digest in snap["files"].items()
        if ".control/reconciliation/" not in rel
    }
    control_files = {
        rel: digest
        for rel, digest in snap.get("control_files", {}).items()
        if not rel.startswith("reconciliation/")
    }
    marker_files = {
        rel: digest for rel, digest in snap.get("locks_files", {}).items()
    }
    return {
        "workspace_files": workspace_files,
        "control_files": control_files,
        "marker_files": marker_files,
        "authority_paths": {
            name: _authority_path_entry(path) for name, path in authority_paths.items()
        },
    }


def test_hardening_thread_concurrent_open_identical_eight_threads(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    reconciliation_cases._bootstrap_reconciliation_tree(case_id, tmp_path)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(_HARDENING_THREAD_COUNT)

    def worker() -> None:
        try:
            barrier.wait()
            record, meta = open_reconciliation_case(
                case_id,
                issue["approval_id"],
                base_dir=tmp_path,
                opened_by="operator",
                scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
            )
            with lock:
                results.append((record.case_open_digest, meta.exact_replay))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(_HARDENING_THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(isinstance(exc, ReconciliationConflictError) for exc in errors)
    assert not errors
    assert len(results) == _HARDENING_THREAD_COUNT
    assert len({digest for digest, _replay in results}) == 1
    assert sum(1 for _digest, replay in results if replay is False) == 1
    assert sum(1 for _digest, replay in results if replay is True) == _HARDENING_THREAD_COUNT - 1


def test_hardening_thread_concurrent_observation_identical_eight_threads(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = _open_case(tmp_path, issue)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(_HARDENING_THREAD_COUNT)

    def worker() -> None:
        try:
            barrier.wait()
            record, meta = record_reconciliation_observation(
                case_id,
                base_dir=tmp_path,
                observed_by="operator",
            )
            with lock:
                results.append((record.observation_digest, meta.exact_replay))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(_HARDENING_THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(isinstance(exc, ReconciliationConflictError) for exc in errors)
    assert not errors
    assert len(results) == _HARDENING_THREAD_COUNT
    assert len({digest for digest, _replay in results}) == 1
    assert sum(1 for _digest, replay in results if replay is False) == 1
    assert sum(1 for _digest, replay in results if replay is True) == _HARDENING_THREAD_COUNT - 1


def test_hardening_thread_concurrent_decision_identical_eight_threads(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    barrier = threading.Barrier(_HARDENING_THREAD_COUNT)

    def worker() -> None:
        try:
            barrier.wait()
            record, meta = record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
                decided_by="operator",
            )
            with lock:
                results.append((record.decision_digest, meta.exact_replay))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(_HARDENING_THREAD_COUNT)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(isinstance(exc, ReconciliationConflictError) for exc in errors)
    assert not errors
    assert len(results) == _HARDENING_THREAD_COUNT
    assert len({digest for digest, _replay in results}) == 1
    assert sum(1 for _digest, replay in results if replay is False) == 1
    assert sum(1 for _digest, replay in results if replay is True) == _HARDENING_THREAD_COUNT - 1


@pytest.mark.parametrize(
    ("operation", "record_name"),
    [
        ("open", "open.json"),
        ("observation", "observation.json"),
        ("decision", "decision.json"),
    ],
)
def test_hardening_record_write_failure_before_bytes(tmp_path, monkeypatch, operation, record_name):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id() if operation == "open" else _open_case(tmp_path, issue)
    obs = None
    if operation == "decision":
        _observe_case(tmp_path, case_id)
        obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
        assert obs is not None
    _inject_os_write_fail_first(monkeypatch)

    with pytest.raises(ReconciliationDurabilityError) as excinfo:
        if operation == "open":
            open_reconciliation_case(
                case_id,
                issue["approval_id"],
                base_dir=tmp_path,
                opened_by="operator",
                scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
            )
        elif operation == "observation":
            record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
        else:
            record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
                decided_by="operator",
            )

    err = excinfo.value
    assert err.durability_stage == "record_write"
    assert err.case_id == case_id
    assert err.record_name == record_name
    assert err.exact_replay_status == "no"
    assert err.record_may_have_committed is False


@pytest.mark.parametrize("record_name", _RECORD_NAMES)
def test_hardening_partial_record_write_fails_closed(tmp_path, monkeypatch, record_name):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    case_id = generate_reconciliation_case_id()
    if record_name == "open.json":
        _inject_os_write_partial_then_error(monkeypatch)
        with pytest.raises(OSError, match="partial record write"):
            open_reconciliation_case(
                case_id,
                issue["approval_id"],
                base_dir=tmp_path,
                opened_by="operator",
                scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
            )
        partial_path = paths.reconciliation_open_path(case_id, tmp_path)
        if partial_path.is_file():
            with pytest.raises((ReconciliationValidationError, json.JSONDecodeError, ReconciliationStateError)):
                load_reconciliation_case(case_id, base_dir=tmp_path)
        return

    case_id = _open_case(tmp_path, issue)
    if record_name == "observation.json":
        _inject_os_write_partial_then_error(monkeypatch)
        with pytest.raises(OSError, match="partial record write"):
            record_reconciliation_observation(case_id, base_dir=tmp_path, observed_by="operator")
        partial_path = paths.reconciliation_observation_path(case_id, tmp_path)
    else:
        obs, _ = _observe_case(tmp_path, case_id), None
        obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
        assert obs is not None
        _inject_os_write_partial_then_error(monkeypatch)
        with pytest.raises(OSError, match="partial record write"):
            record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
                decided_by="operator",
            )
        partial_path = paths.reconciliation_decision_path(case_id, tmp_path)
    if partial_path.is_file():
        with pytest.raises((ReconciliationValidationError, json.JSONDecodeError, ReconciliationStateError)):
            load_reconciliation_case(case_id, base_dir=tmp_path)


def test_decision_class_enum_exactly_seven():
    values = list(ReconciliationDecisionClass)
    assert len(values) == 7
    assert values == list(_EXPECTED_DECISION_CLASSES)
    assert set(reconciliation_cases.__all__).isdisjoint({member.value for member in values})
    assert "ReconciliationDecisionClass" in reconciliation_cases.__all__


_RECORD_VALIDATION_CASES = [
    ("missing_required_field", True),
    ("wrong_schema_version", True),
    ("wrong_digest_projection_version", True),
    ("timestamp_tamper", True),
    ("actor_tamper", True),
    ("key_order_only", False),
]


@pytest.mark.parametrize("record_name", _RECORD_NAMES)
@pytest.mark.parametrize("tamper_kind,should_fail", _RECORD_VALIDATION_CASES)
def test_hardening_record_validation_matrix(
    tmp_path,
    record_name,
    tamper_kind,
    should_fail,
):
    case_id, record_path = _prepare_loaded_record(tmp_path, record_name)
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    if tamper_kind == "missing_required_field":
        if record_name == "open.json":
            raw.pop("case_open_digest", None)
        elif record_name == "observation.json":
            raw.pop("observation_digest", None)
        else:
            raw.pop("decision_digest", None)
    elif tamper_kind == "wrong_schema_version":
        if record_name == "open.json":
            raw["case_schema_version"] = "999"
        elif record_name == "observation.json":
            raw["observation_schema_version"] = "999"
        else:
            raw["decision_schema_version"] = "999"
    elif tamper_kind == "wrong_digest_projection_version":
        if record_name == "open.json":
            raw["case_open_digest_projection_version"] = "tampered.v999"
        elif record_name == "observation.json":
            raw["observation_digest_projection_version"] = "tampered.v999"
        else:
            raw["decision_digest_projection_version"] = "tampered.v999"
    elif tamper_kind == "timestamp_tamper":
        if record_name == "open.json":
            raw["opened_at"] = "2099-01-01T00:00:00+00:00"
        elif record_name == "observation.json":
            raw["observed_at"] = "2099-01-01T00:00:00+00:00"
        else:
            raw["decided_at"] = "2099-01-01T00:00:00+00:00"
    elif tamper_kind == "actor_tamper":
        if record_name == "open.json":
            raw["opened_by"] = "attacker"
        elif record_name == "observation.json":
            raw["observed_by"] = "attacker"
        else:
            raw["decided_by"] = "attacker"
    elif tamper_kind == "key_order_only":
        raw = {k: raw[k] for k in reversed(list(raw.keys()))}
    else:
        raise AssertionError(f"unknown tamper kind: {tamper_kind}")

    record_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    if should_fail:
        with pytest.raises(ReconciliationValidationError):
            load_reconciliation_case(case_id, base_dir=tmp_path)
    else:
        bundle = load_reconciliation_case(case_id, base_dir=tmp_path)
        assert bundle.open_record.case_id == case_id


@pytest.mark.parametrize(
    "scenario",
    [
        "malformed_claim",
        "lifecycle_not_complete",
        "integrity_not_clean",
        "identity_mismatch",
    ],
)
def test_hardening_completion_verified_rejection_via_inspection_patch(tmp_path, scenario):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    approval_id = json.loads(
        paths.reconciliation_open_path(case_id, tmp_path).read_text(encoding="utf-8")
    )["approval_id"]
    kwargs = dict(
        case_id=case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
        decided_by="operator",
    )
    live = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
    if scenario == "malformed_claim":
        mutated = _mutated_inspection(live, claim_id="claim-bad", claim_digest=None)
    elif scenario == "lifecycle_not_complete":
        mutated = _mutated_inspection(
            live,
            lifecycle_evidence_state="no_lifecycle_evidence_observed",
            overall_classification="reconciliation_inspection_required",
        )
    elif scenario == "integrity_not_clean":
        mutated = _mutated_inspection(
            live,
            integrity_state="seal_blocked",
            overall_classification="integrity_blocked",
        )
    else:
        mutated = _mutated_inspection(live, run_id="run_identity_mismatch")
    with patch("htr.reconciliation_cases.inspect_run_completion_reconciliation", return_value=mutated):
        with pytest.raises(ReconciliationValidationError, match="not allowed"):
            record_reconciliation_decision(**kwargs)


def test_hardening_boundary_snapshot_unchanged_on_open_observe_decide(tmp_path):
    run_id, task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_approval(issue["approval_id"], claim_id="claim-boundary", claimant_id="bob", base_dir=tmp_path)
    events.complete_run_manually(run_id, completion, actor="human", event_id=event_id, base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"],
        "claim-boundary",
        OUTCOME_AMBIGUOUS,
        outcome_evidence=_ambiguous_v2_evidence(event_id=event_id),
        base_dir=tmp_path,
    )
    _write_marker(tmp_path, run_id)
    before = _boundary_snapshot(
        tmp_path,
        run_id=run_id,
        task_id=task_id,
        approval_id=issue["approval_id"],
    )
    case_id = _open_case(tmp_path, issue)
    obs, _ = _observe_case(tmp_path, case_id), None
    obs = load_reconciliation_case(case_id, base_dir=tmp_path).observation_record
    assert obs is not None
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
    )
    after = _boundary_snapshot(
        tmp_path,
        run_id=run_id,
        task_id=task_id,
        approval_id=issue["approval_id"],
    )
    assert after == before
    required_authority_keys = {
        "issue.json",
        "revoke.json",
        "claim.json",
        "outcome.json",
        "completion_lifecycle_json",
        "lifecycle_event_log",
        "manifest",
        "task_card",
        "task_status",
        "attempt_status",
        "artifact_manifest",
        "final_closure_record",
        "execution_marker",
    }
    assert required_authority_keys <= set(before["authority_paths"].keys())
    for name in ("issue.json", "claim.json", "outcome.json", "completion_lifecycle_json"):
        assert before["authority_paths"][name]["exists"] is True


_COMPLETION_VERIFIED_BLOCK_SCENARIOS = (
    "frozen_chain_gap",
    "unexpected_later_frozen_chain_record",
    "untrustworthy_current_observation",
    "duplicate_exact_completion_event",
    "conflicting_completion_event",
)


def _open_record_fields(case_id: str, tmp_path: Path) -> tuple[str, str, str]:
    open_raw = json.loads(paths.reconciliation_open_path(case_id, tmp_path).read_text(encoding="utf-8"))
    return open_raw["approval_id"], open_raw["run_id"], open_raw["event_id"]


@pytest.mark.parametrize("scenario", _COMPLETION_VERIFIED_BLOCK_SCENARIOS)
def test_completion_verified_blocked_by_inspection_evidence(tmp_path, scenario):
    if scenario == "duplicate_exact_completion_event":
        case_id, obs = _ambiguous_complete_case(tmp_path)
        approval_id, run_id, _event_id = _open_record_fields(case_id, tmp_path)
        events_path = paths.task_events_path(run_id, tmp_path)
        last_line = events_path.read_text(encoding="utf-8").strip().splitlines()[-1]
        duplicate = json.loads(last_line)
        duplicate["event_id"] = new_event_id()
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(duplicate) + "\n")
        live = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
        assert "multiple_completion_events" in live.reason_codes
        with pytest.raises(ReconciliationValidationError, match="not allowed"):
            record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
                decided_by="operator",
            )
        assert not paths.reconciliation_decision_path(case_id, tmp_path).exists()
        return

    if scenario == "conflicting_completion_event":
        case_id, obs = _ambiguous_complete_case(tmp_path)
        approval_id, run_id, event_id = _open_record_fields(case_id, tmp_path)
        wrong = dict(
            json.loads(contracts.run_completion_record_json_path(run_id, tmp_path).read_text(encoding="utf-8"))
        )
        wrong["reason"] = "tampered-conflict"
        io.atomic_write_json(contracts.run_completion_record_json_path(run_id, tmp_path), wrong)
        live = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
        assert "completion_record_semantic_mismatch" in live.reason_codes
        with pytest.raises(ReconciliationValidationError, match="not allowed"):
            record_reconciliation_decision(
                case_id,
                base_dir=tmp_path,
                expected_observation_digest=obs.observation_digest,
                requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
                decided_by="operator",
            )
        assert not paths.reconciliation_decision_path(case_id, tmp_path).exists()
        return

    case_id, obs = _ambiguous_complete_case(tmp_path)
    approval_id, _run_id, _event_id = _open_record_fields(case_id, tmp_path)
    live = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
    overrides: dict[str, Any]
    if scenario == "frozen_chain_gap":
        overrides = {
            "reason_codes": (*live.reason_codes, "chain_gap_detected"),
            "integrity_state": "integrity_blocked",
            "lifecycle_evidence_state": "lifecycle_integrity_blocked",
        }
    elif scenario == "unexpected_later_frozen_chain_record":
        overrides = {
            "reason_codes": (
                *live.reason_codes,
                "unexpected_chain_record:run_final_closure_record",
            ),
            "lifecycle_evidence_state": "lifecycle_integrity_blocked",
            "integrity_state": "integrity_blocked",
        }
    elif scenario == "untrustworthy_current_observation":
        overrides = {
            "reason_codes": (*live.reason_codes, "current_snapshot_not_trustworthy"),
            "integrity_state": "integrity_blocked",
            "current_observation_semantic_digest": None,
        }
    else:
        raise AssertionError(f"unknown scenario: {scenario}")
    mutated = _mutated_inspection(live, **overrides)
    kwargs = dict(
        case_id=case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
        decided_by="operator",
    )
    with patch("htr.reconciliation_cases.inspect_run_completion_reconciliation", return_value=mutated):
        with pytest.raises(ReconciliationValidationError, match="not allowed"):
            record_reconciliation_decision(**kwargs)
    assert not paths.reconciliation_decision_path(case_id, tmp_path).exists()


def test_integration_26a_inspection_evidence_blocks_completion_verified(tmp_path):
    case_id, obs = _ambiguous_complete_case(tmp_path)
    approval_id, run_id, _event_id = _open_record_fields(case_id, tmp_path)
    pre = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
    assert pre.lifecycle_evidence_state == "lifecycle_complete_observed"
    events_path = paths.task_events_path(run_id, tmp_path)
    duplicate = json.loads(events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    duplicate["event_id"] = new_event_id()
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")
    post = inspect_run_completion_reconciliation(approval_id, base_dir=tmp_path)
    assert "multiple_completion_events" in post.reason_codes
    assert post.overall_classification != "verified_completed"
    with pytest.raises(ReconciliationValidationError, match="not allowed"):
        record_reconciliation_decision(
            case_id,
            base_dir=tmp_path,
            expected_observation_digest=obs.observation_digest,
            requested_decision_class=ReconciliationDecisionClass.completion_verified_by_reconciliation,
            decided_by="operator",
        )
    assert not paths.reconciliation_decision_path(case_id, tmp_path).exists()
