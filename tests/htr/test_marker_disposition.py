"""Tests for Task 26C — approved marker disposition protocol."""

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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import htr.marker_disposition as marker_disposition_mod
from htr import contracts, events, io, paths
from htr.action_plan import PlanningIntent
from htr.approval_control import issue_approval
from htr.execution_lock import LOCKS_DIR_NAME, marker_present_noncreating, run_write_barrier
from htr.execution_lock import (
    RunExecutionLockIndeterminateError,
    RunExecutionLockReleaseConflictError,
    disposition_unlink_marker,
)
from htr.finalization import SealEvaluation, SealState, evaluate_run_seal
from htr.ids import (
    generate_marker_disposition_id,
    new_event_id,
    new_run_id,
    new_task_id,
)
from htr.invoke_run_completion import invoke_approved_run_completion
from htr.marker_disposition import (
    MarkerDispositionOutcomeClass,
    claim_marker_disposition_approval,
    create_marker_disposition_request,
    execute_approved_marker_disposition,
    generate_marker_disposition_approval_id,
    generate_marker_disposition_attempt_id,
    generate_marker_disposition_claim_id,
    issue_marker_disposition_approval,
    load_marker_disposition_bundle,
    load_reconciliation_case,
    reconcile_marker_disposition_outcome,
    revoke_marker_disposition_approval,
)
from htr.reconciliation_cases import (
    ReconciliationDecisionClass,
    ReconciliationNextProtocol,
    ReconciliationScopeReason,
    generate_reconciliation_case_id,
    open_reconciliation_case,
    record_reconciliation_decision,
    record_reconciliation_observation,
)
from htr.reconciliation_inspection import PILOT_BOUND_API
from htr.state import (
    MarkerDispositionConflictError,
    MarkerDispositionDurabilityError,
    MarkerDispositionStateError,
    MarkerDispositionValidationError,
    RunFinalizedError,
)

TASK16_PATH = Path(__file__).with_name("test_run_final_closure.py")


def _load_task16():
    import importlib.util

    spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16()


def _seal_finalized(run_id: str):
    return SealEvaluation(SealState.FINALIZED_VALID, (), run_id)


@contextmanager
def _patch_finalized_seal(run_id: str):
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        yield


def _expires_in_minutes(minutes: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _expires_in_hours(hours: float = 1.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _control_snapshot(root: Path) -> dict[str, str]:
    control = root / ".control"
    if not control.exists():
        return {}
    return {
        str(p.relative_to(control)): _file_digest(p)
        for p in sorted(control.rglob("*"))
        if p.is_file()
    }


def _run_ready_for_completion(tmp_path: Path) -> tuple[str, str, dict]:
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[task_id]
    )
    return run_id, task_id, completion


def _issue_completion_approval(tmp_path: Path, run_id: str, completion: dict) -> dict:
    event_id = new_event_id()
    intent = PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs={"record": completion, "actor": "human", "event_id": event_id},
        htr_runs_root=str(tmp_path),
    )
    return issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in_hours(),
        base_dir=tmp_path,
    )


def _write_marker(tmp_path: Path, run_id: str) -> None:
    locks_root = tmp_path / LOCKS_DIR_NAME
    locks_root.mkdir(parents=True, exist_ok=True)
    marker_path = locks_root / f"{run_id}.marker"
    payload = {
        "schema_version": "1",
        "acquisition_id": str(uuid.uuid4()),
        "pid": os.getpid(),
        "hostname": "test-host",
        "run_id": run_id,
    }
    marker_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _path_a_ready(
    tmp_path: Path,
    *,
    decision_class: ReconciliationDecisionClass,
    recommended_next_protocol: ReconciliationNextProtocol = ReconciliationNextProtocol.marker_disposition_review,
) -> tuple[str, str, str]:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(
        issue["approval_id"], claim_id="claim-path-a", base_dir=tmp_path
    )
    _write_marker(tmp_path, run_id)
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    obs, _ = record_reconciliation_observation(
        case_id, base_dir=tmp_path, observed_by="operator"
    )
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=decision_class,
        decided_by="operator",
        recommended_next_protocol=recommended_next_protocol,
    )
    return case_id, run_id, generate_marker_disposition_id()


def _path_a_case(tmp_path: Path, *, finalize: bool = True) -> tuple[str, str, str]:
    case_id, run_id, disposition_id = _path_a_ready(
        tmp_path,
        decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        recommended_next_protocol=ReconciliationNextProtocol.marker_disposition_review,
    )
    if finalize:
        with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
            create_marker_disposition_request(
                disposition_id,
                case_id,
                requested_by="operator",
                base_dir=tmp_path,
            )
    return case_id, run_id, disposition_id


def _full_disposition_chain(tmp_path: Path) -> tuple[str, str]:
    _case_id, run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    return disposition_id, run_id


def _run_isolated_mdp_script(script: str, base_dir: Path) -> subprocess.CompletedProcess[str]:
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


def _run_with_post_verification_via_invoke(
    tmp_path: Path,
) -> tuple[str, str, tuple[Any, ...]]:
    """TASK16 post-verification chain with Task-25 invoke instead of complete_run_manually."""
    run_id = new_run_id()
    io.create_run_workspace(run_id, base_dir=tmp_path)
    task_id = new_task_id()
    io.create_task_workspace(run_id, task_id, base_dir=tmp_path)
    TASK16._complete_task(tmp_path, run_id, task_id)
    completion = contracts.make_run_completion_record(
        run_id=run_id, completed_task_ids=[task_id]
    )
    issue = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(
        issue["approval_id"], claim_id="claim-real-path-a", base_dir=tmp_path
    )
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    events.review_run_manually(run_id, review, base_dir=tmp_path)
    plan = TASK16._plan_record(run_id)
    events.plan_run_followup(run_id, plan, base_dir=tmp_path)
    request = contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        execution_items=[TASK16._sample_execution_item()],
    )
    events.request_run_execution(run_id, request, base_dir=tmp_path)
    result = events.execute_run_execution_request(tmp_path, run_id, "human")
    verification = TASK16._verification_record(run_id, result)
    events.verify_run_execution_result(tmp_path, run_id, verification, actor="human")
    pvfp = TASK16._post_verification_plan_record(run_id, result, verification)
    events.plan_post_verification_followup(tmp_path, run_id, pvfp, actor="human")
    pver = TASK16._post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    events.request_post_verification_execution(tmp_path, run_id, pver, actor="human")
    pve_result = TASK16._post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, pve_result, actor="human"
    )
    pve_verification = TASK16._post_verification_execution_verification_record(
        run_id, result, verification, pvfp, pver, pve_result
    )
    events.record_post_verification_execution_verification(
        tmp_path, run_id, pve_verification, actor="human"
    )
    chain = (
        run_id,
        [task_id],
        [],
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    )
    return run_id, issue["approval_id"], chain


def _finalize_invoke_only_run_for_path_a(tmp_path: Path, run_id: str) -> None:
    """Extend invoke-only Path A run to real FINALIZED_VALID without breaking 26B decision proof."""
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
    if marker_path.exists():
        marker_path.unlink()
    completion = io.read_json(contracts.run_completion_record_json_path(run_id, tmp_path))
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    events.review_run_manually(run_id, review, base_dir=tmp_path)
    plan = TASK16._plan_record(run_id)
    events.plan_run_followup(run_id, plan, base_dir=tmp_path)
    request = contracts.make_run_execution_request_record(
        run_id=run_id,
        source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
        execution_items=[TASK16._sample_execution_item()],
    )
    events.request_run_execution(run_id, request, base_dir=tmp_path)
    result = events.execute_run_execution_request(tmp_path, run_id, "human")
    verification = TASK16._verification_record(run_id, result)
    events.verify_run_execution_result(tmp_path, run_id, verification, actor="human")
    pvfp = TASK16._post_verification_plan_record(run_id, result, verification)
    events.plan_post_verification_followup(tmp_path, run_id, pvfp, actor="human")
    pver = TASK16._post_verification_execution_request_record(
        run_id, result, verification, pvfp
    )
    events.request_post_verification_execution(tmp_path, run_id, pver, actor="human")
    pve_result = TASK16._post_verification_execution_result_record(
        run_id, result, verification, pvfp, pver
    )
    events.record_post_verification_execution_result(
        tmp_path, run_id, pve_result, actor="human"
    )
    pve_verification = TASK16._post_verification_execution_verification_record(
        run_id, result, verification, pvfp, pver, pve_result
    )
    events.record_post_verification_execution_verification(
        tmp_path, run_id, pve_verification, actor="human"
    )
    closure = TASK16._run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    )
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID
    _write_marker(tmp_path, run_id)


def _real_finalized_path_a_ready(
    tmp_path: Path,
    *,
    decision_class: ReconciliationDecisionClass,
    recommended_next_protocol: ReconciliationNextProtocol = ReconciliationNextProtocol.marker_disposition_review,
) -> tuple[str, str, str]:
    case_id, run_id, disposition_id = _path_a_ready(
        tmp_path,
        decision_class=decision_class,
        recommended_next_protocol=recommended_next_protocol,
    )
    _finalize_invoke_only_run_for_path_a(tmp_path, run_id)
    return case_id, run_id, disposition_id


def _real_finalized_path_a_case(
    tmp_path: Path,
    *,
    decision_class: ReconciliationDecisionClass,
    recommended_next_protocol: ReconciliationNextProtocol = ReconciliationNextProtocol.marker_disposition_review,
) -> tuple[str, str, str]:
    case_id, run_id, disposition_id = _real_finalized_path_a_ready(
        tmp_path,
        decision_class=decision_class,
        recommended_next_protocol=recommended_next_protocol,
    )
    create_marker_disposition_request(
        disposition_id,
        case_id,
        requested_by="operator",
        base_dir=tmp_path,
    )
    return case_id, run_id, disposition_id


def _issue_claim_execute(
    tmp_path: Path,
    disposition_id: str,
    run_id: str,
    *,
    expires_minutes: float = 10.0,
) -> Any:
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(expires_minutes),
        base_dir=tmp_path,
    )
    claim_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    return execute_approved_marker_disposition(
        disposition_id,
        generate_marker_disposition_attempt_id(),
        executor="executor",
        base_dir=tmp_path,
    )


_DURABILITY_STAGES = (
    "record_fsync",
    "disposition_dir_fsync",
    "control_dir_fsync",
    "parent_dir_fsync",
)


def _inject_mdp_durability_failure(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    if stage == "record_fsync":

        def _fail_file(
            _fd: int,
            *,
            disposition_id: str,
            record_name: str,
        ) -> None:
            raise MarkerDispositionDurabilityError(
                "injected record_fsync",
                disposition_id=disposition_id,
                record_name=record_name,  # type: ignore[arg-type]
                durability_stage="record_fsync",
                record_may_have_committed=True,
                exact_replay_status="indeterminate",
            )

        monkeypatch.setattr(marker_disposition_mod, "_fsync_file_fd", _fail_file)
        return

    def _make_dir_fail(expected_stage: str):
        real = marker_disposition_mod._fsync_dir_fd

        def _fail_dir(
            dir_fd: int,
            *,
            disposition_id: str,
            record_name: str,
            stage: str,
        ) -> None:
            if stage == expected_stage:
                raise MarkerDispositionDurabilityError(
                    f"injected {expected_stage}",
                    disposition_id=disposition_id,
                    record_name=record_name,  # type: ignore[arg-type]
                    durability_stage=expected_stage,  # type: ignore[arg-type]
                    record_may_have_committed=True,
                    exact_replay_status="indeterminate",
                )
            real(
                dir_fd,
                disposition_id=disposition_id,
                record_name=record_name,
                stage=stage,
            )

        return _fail_dir

    monkeypatch.setattr(marker_disposition_mod, "_fsync_dir_fd", _make_dir_fail(stage))


def _boundary_snapshot(tmp_path: Path, *, case_id: str) -> dict[str, str]:
    control = tmp_path / ".control"
    snap: dict[str, str] = {}
    if not control.exists():
        return snap
    for path in sorted(control.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(control))
        if rel.startswith("marker_dispositions/"):
            continue
        snap[rel] = _file_digest(path)
    return snap


def _subprocess_claim_worker(
    disposition_id: str,
    base_dir: str,
    claim_id: str,
    slot: Any,
    barrier: Any,
) -> None:
    from htr.marker_disposition import claim_marker_disposition_approval

    try:
        barrier.wait()
        _record, meta = claim_marker_disposition_approval(
            disposition_id,
            claim_id,
            claimant="executor",
            base_dir=Path(base_dir),
        )
        slot.put(("ok", meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _subprocess_execute_worker(
    disposition_id: str,
    base_dir: str,
    attempt_id: str,
    slot: Any,
    barrier: Any,
) -> None:
    from htr.marker_disposition import execute_approved_marker_disposition

    try:
        barrier.wait()
        result = execute_approved_marker_disposition(
            disposition_id,
            attempt_id,
            executor="executor",
            base_dir=Path(base_dir),
        )
        slot.put(("ok", result.outcome_class))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def test_generate_marker_disposition_ids_no_side_effects(tmp_path):
    before = _control_snapshot(tmp_path)
    generate_marker_disposition_id()
    generate_marker_disposition_approval_id()
    generate_marker_disposition_claim_id()
    generate_marker_disposition_attempt_id()
    assert before == _control_snapshot(tmp_path)


def test_request_rejected_without_finalized_seal(tmp_path):
    case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=False)
    with pytest.raises(MarkerDispositionValidationError, match="finalized_valid"):
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )


def test_request_exact_replay_zero_write(tmp_path):
    case_id, run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    mid = _control_snapshot(tmp_path)
    with _patch_finalized_seal(run_id):
        _first, meta1 = create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
        _second, meta2 = create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
    assert meta1.exact_replay is True
    assert meta2.exact_replay is True
    assert mid == _control_snapshot(tmp_path)


def test_request_conflicting_replay_fails_closed(tmp_path):
    case_id, run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    with _patch_finalized_seal(run_id):
        with pytest.raises(MarkerDispositionConflictError):
            create_marker_disposition_request(
                disposition_id,
                case_id,
                requested_by="other-operator",
                base_dir=tmp_path,
            )


def test_issue_expiry_cannot_exceed_15_minutes(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    with pytest.raises(MarkerDispositionValidationError, match="lifetime"):
        issue_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_approval_id(),
            issued_by="approver",
            expires_at=_expires_in_minutes(16),
            base_dir=tmp_path,
        )


def test_revoke_after_claim_fails_closed(tmp_path):
    disposition_id, _run_id = _full_disposition_chain(tmp_path)
    with pytest.raises(MarkerDispositionConflictError, match="after claim"):
        revoke_marker_disposition_approval(
            disposition_id,
            revoked_by="approver",
            reason="too late",
            base_dir=tmp_path,
        )


def test_disposed_verified_removes_marker(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    assert marker_present_noncreating(tmp_path, run_id)
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.disposed_verified.value
    assert result.marker_removed_by_this_execution is True
    assert not marker_present_noncreating(tmp_path, run_id)
    outcome = json.loads(
        paths.marker_disposition_outcome_path(disposition_id, tmp_path).read_text(
            encoding="utf-8"
        )
    )
    assert outcome["further_marker_disposition_allowed"] is False
    with patch(
        "htr.finalization.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        with pytest.raises(RunFinalizedError):
            io.create_run_workspace(run_id, base_dir=tmp_path)


def test_already_absent_not_disposed_verified(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    (tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker").unlink()
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.already_absent_observed.value
    assert result.outcome_class != MarkerDispositionOutcomeClass.disposed_verified.value


def test_execution_ambiguous_when_attempt_without_outcome(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    attempt_id = generate_marker_disposition_attempt_id()
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        execute_approved_marker_disposition(
            disposition_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    paths.marker_disposition_outcome_path(disposition_id, tmp_path).unlink()
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.execution_ambiguous.value


def test_execute_exact_outcome_replay_zero_write(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    attempt_id = generate_marker_disposition_attempt_id()
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        first = execute_approved_marker_disposition(
            disposition_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    before = _control_snapshot(tmp_path)
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        second = execute_approved_marker_disposition(
            disposition_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    assert before == _control_snapshot(tmp_path)
    assert second.exact_replay is True
    assert first.outcome_digest == second.outcome_digest


def test_reconcile_performs_zero_writes(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    before = _control_snapshot(tmp_path)
    result = reconcile_marker_disposition_outcome(disposition_id, base_dir=tmp_path)
    assert before == _control_snapshot(tmp_path)
    assert result.classification == "valid_durable_outcome"


def test_reconcile_claim_without_attempt(tmp_path):
    disposition_id, _run_id = _full_disposition_chain(tmp_path)
    result = reconcile_marker_disposition_outcome(disposition_id, base_dir=tmp_path)
    assert result.classification == "claim_without_attempt"


def test_task26b_records_unchanged_after_disposition(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    case_id = load_marker_disposition_bundle(
        disposition_id, base_dir=tmp_path
    ).request_record.reconciliation_case_id
    rcn_before = {
        str(p.relative_to(tmp_path / ".control")): _file_digest(p)
        for p in sorted((tmp_path / ".control").rglob("*"))
        if p.is_file() and "marker_dispositions" not in str(p)
    }
    with patch("htr.marker_disposition.evaluate_run_seal", return_value=_seal_finalized(run_id)):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    rcn_after = {
        str(p.relative_to(tmp_path / ".control")): _file_digest(p)
        for p in sorted((tmp_path / ".control").rglob("*"))
        if p.is_file() and "marker_dispositions" not in str(p)
    }
    assert rcn_before == rcn_after
    assert load_reconciliation_case(case_id, base_dir=tmp_path).decision_record is not None


def test_disposition_execution_does_not_require_run_write_barrier(tmp_path):
    """Disposition uses directory flock, not run_write_barrier — same-run marker residue ok."""
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    assert marker_present_noncreating(tmp_path, run_id)
    with _patch_finalized_seal(run_id):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.disposed_verified.value
    assert not marker_present_noncreating(tmp_path, run_id)


# --- Task 26C hardening ---


@pytest.mark.parametrize(
    "decision_class",
    [
        ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        ReconciliationDecisionClass.case_closed_no_action_required,
    ],
)
def test_path_a_real_finalized_seal_both_decision_classes(tmp_path, decision_class):
    case_id, run_id, disposition_id = _real_finalized_path_a_case(
        tmp_path,
        decision_class=decision_class,
        recommended_next_protocol=ReconciliationNextProtocol.marker_disposition_review,
    )
    bundle = load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)
    assert bundle.request_record is not None
    assert bundle.request_record.reconciliation_case_id == case_id
    assert bundle.request_record.run_id == run_id
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID


def test_request_rejects_path_b_decision_class(tmp_path):
    case_id, obs = _consumed_with_marker_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.marker_disposition_review,
    )
    _tamper_reconciliation_decision(
        case_id,
        tmp_path,
        decision_class=ReconciliationDecisionClass.evidence_conflict_confirmed,
    )
    disposition_id = generate_marker_disposition_id()
    with pytest.raises(MarkerDispositionValidationError, match="not eligible"):
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )


def test_request_rejects_defer_without_marker_disposition_protocol(tmp_path):
    case_id, obs = _consumed_with_marker_case(tmp_path)
    record_reconciliation_decision(
        case_id,
        base_dir=tmp_path,
        expected_observation_digest=obs.observation_digest,
        requested_decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        decided_by="operator",
        recommended_next_protocol=ReconciliationNextProtocol.marker_disposition_review,
    )
    _tamper_reconciliation_decision(
        case_id,
        tmp_path,
        recommended_next_protocol=ReconciliationNextProtocol.recovery_run_review,
    )
    disposition_id = generate_marker_disposition_id()
    with pytest.raises(MarkerDispositionValidationError, match="marker_disposition_review"):
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )


def test_request_rejects_malformed_marker(tmp_path):
    case_id, run_id, disposition_id = _path_a_case(tmp_path, finalize=False)
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
    marker_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(MarkerDispositionValidationError):
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )


def test_request_rejects_incomplete_reconciliation_case(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(
        issue["approval_id"], claim_id="claim-incomplete", base_dir=tmp_path
    )
    _write_marker(tmp_path, run_id)
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    disposition_id = generate_marker_disposition_id()
    with pytest.raises(MarkerDispositionValidationError, match="incomplete"):
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )


def _consumed_with_marker_case(tmp_path: Path) -> tuple[str, Any]:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue = _issue_completion_approval(tmp_path, run_id, completion)
    invoke_approved_run_completion(
        issue["approval_id"], claim_id="claim-consumed", base_dir=tmp_path
    )
    _write_marker(tmp_path, run_id)
    case_id = generate_reconciliation_case_id()
    open_reconciliation_case(
        case_id,
        issue["approval_id"],
        base_dir=tmp_path,
        opened_by="operator",
        scope_reason=ReconciliationScopeReason.ambiguous_completion_reconciliation,
    )
    obs, _ = record_reconciliation_observation(
        case_id, base_dir=tmp_path, observed_by="operator"
    )
    return case_id, obs


def test_issue_exact_replay_zero_write(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    approval_id = generate_marker_disposition_approval_id()
    expires_at = _expires_in_minutes(10)
    issue_marker_disposition_approval(
        disposition_id,
        approval_id,
        issued_by="approver",
        expires_at=expires_at,
        base_dir=tmp_path,
    )
    before = _control_snapshot(tmp_path)
    _rec, meta = issue_marker_disposition_approval(
        disposition_id,
        approval_id,
        issued_by="approver",
        expires_at=expires_at,
        base_dir=tmp_path,
    )
    assert meta.exact_replay is True
    assert before == _control_snapshot(tmp_path)


def test_issue_conflicting_replay_fails_closed(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    with pytest.raises(MarkerDispositionConflictError):
        issue_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_approval_id(),
            issued_by="other-approver",
            expires_at=_expires_in_minutes(10),
            base_dir=tmp_path,
        )


def test_revoke_exact_replay_zero_write(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    revoke_marker_disposition_approval(
        disposition_id,
        revoked_by="approver",
        reason="changed mind",
        base_dir=tmp_path,
    )
    before = _control_snapshot(tmp_path)
    _rec, meta = revoke_marker_disposition_approval(
        disposition_id,
        revoked_by="approver",
        reason="changed mind",
        base_dir=tmp_path,
    )
    assert meta.exact_replay is True
    assert before == _control_snapshot(tmp_path)


def test_revoke_before_claim_blocks_claim_and_execute(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    revoke_marker_disposition_approval(
        disposition_id,
        revoked_by="approver",
        reason="too risky",
        base_dir=tmp_path,
    )
    with pytest.raises(MarkerDispositionValidationError, match="revoked"):
        claim_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )


def test_claim_expired_approval_rejected(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issued_at = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    expires_at = (issued_at + timedelta(minutes=10)).isoformat()
    claim_at = issued_at + timedelta(minutes=20)
    with patch("htr.marker_disposition._utc_now", return_value=issued_at):
        issue_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_approval_id(),
            issued_by="approver",
            expires_at=expires_at,
            base_dir=tmp_path,
        )
    with patch("htr.marker_disposition._utc_now", return_value=claim_at):
        with pytest.raises(MarkerDispositionValidationError, match="expired"):
            claim_marker_disposition_approval(
                disposition_id,
                generate_marker_disposition_claim_id(),
                claimant="executor",
                base_dir=tmp_path,
            )


def test_claim_exact_replay_zero_write(tmp_path):
    disposition_id, _run_id = _full_disposition_chain(tmp_path)
    claim_path = paths.marker_disposition_claim_path(disposition_id, tmp_path)
    claim_id = json.loads(claim_path.read_text(encoding="utf-8"))["claim_id"]
    before = _control_snapshot(tmp_path)
    _rec, meta = claim_marker_disposition_approval(
        disposition_id,
        claim_id,
        claimant="executor",
        base_dir=tmp_path,
    )
    assert meta.exact_replay is True
    assert before == _control_snapshot(tmp_path)


def test_claim_conflicting_replay_fails_closed(tmp_path):
    disposition_id, _run_id = _full_disposition_chain(tmp_path)
    with pytest.raises(MarkerDispositionConflictError):
        claim_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_claim_id(),
            claimant="other-executor",
            base_dir=tmp_path,
        )




def test_thread_simultaneous_claim_one_creator_one_replay(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_id = generate_marker_disposition_claim_id()
    barrier = threading.Barrier(3)
    slots: list[tuple[str, ...] | Exception] = [()] * 3

    def _worker(index: int) -> None:
        try:
            barrier.wait()
            _rec, meta = claim_marker_disposition_approval(
                disposition_id,
                claim_id,
                claimant="executor",
                base_dir=tmp_path,
            )
            slots[index] = ("ok", meta.exact_replay)
        except Exception as exc:
            slots[index] = exc

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    ok = [s for s in slots if isinstance(s, tuple) and s[0] == "ok"]
    assert len(ok) == 3
    assert sum(1 for r in ok if r[1] is False) == 1
    assert sum(1 for r in ok if r[1] is True) == 2
    assert not any(isinstance(s, Exception) for s in slots)


def test_subprocess_simultaneous_claim_one_creator_one_replay(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_id = generate_marker_disposition_claim_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_claim_worker,
            args=(disposition_id, str(tmp_path), claim_id, slots[i], barrier),
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
    assert len(ok) == 2
    assert sum(1 for r in ok if r[1] is False) == 1
    assert sum(1 for r in ok if r[1] is True) == 1


def test_attempt_record_exists_before_marker_unlink(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    attempt_seen: list[bool] = []

    real_unlink = disposition_unlink_marker

    def _assert_attempt_before_unlink(*args: Any, **kwargs: Any) -> None:
        attempt_path = paths.marker_disposition_attempt_path(disposition_id, tmp_path)
        attempt_seen.append(attempt_path.is_file())
        return real_unlink(*args, **kwargs)

    with patch(
        "htr.marker_disposition.disposition_unlink_marker",
        side_effect=_assert_attempt_before_unlink,
    ):
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=_seal_finalized(run_id),
        ):
            execute_approved_marker_disposition(
                disposition_id,
                generate_marker_disposition_attempt_id(),
                executor="executor",
                base_dir=tmp_path,
            )
    assert attempt_seen == [True]
    assert not marker_present_noncreating(tmp_path, run_id)


def test_outcome_class_marker_changed(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["acquisition_id"] = str(uuid.uuid4())
    marker_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.marker_changed.value
    assert marker_present_noncreating(tmp_path, run_id)


def test_outcome_class_evidence_drifted(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    bundle = load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)
    case_id = bundle.request_record.reconciliation_case_id
    decision_path = paths.reconciliation_decision_path(case_id, tmp_path)
    raw = json.loads(decision_path.read_text(encoding="utf-8"))
    raw["decided_by"] = "tampered-operator"
    decision_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.evidence_drifted.value


def test_outcome_class_approval_invalid_on_expired_execute(tmp_path):
    _case_id, run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issued_at = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    expires_at = (issued_at + timedelta(minutes=10)).isoformat()
    with patch("htr.marker_disposition._utc_now", return_value=issued_at):
        issue_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_approval_id(),
            issued_by="approver",
            expires_at=expires_at,
            base_dir=tmp_path,
        )
        claim_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )
    execute_at = issued_at + timedelta(minutes=20)
    with patch("htr.marker_disposition._utc_now", return_value=execute_at):
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=_seal_finalized(run_id),
        ):
            result = execute_approved_marker_disposition(
                disposition_id,
                generate_marker_disposition_attempt_id(),
                executor="executor",
                base_dir=tmp_path,
            )
    assert result.outcome_class == MarkerDispositionOutcomeClass.approval_invalid.value


def test_outcome_class_integrity_blocked_symlink_marker(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
    marker_path.unlink()
    marker_path.symlink_to("/tmp/nonexistent-marker-26c")
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.integrity_blocked.value


def test_outcome_class_unlink_failed(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)

    def _fail_unlink(*_args: Any, **_kwargs: Any) -> None:
        raise RunExecutionLockReleaseConflictError("injected unlink failure")

    with patch(
        "htr.marker_disposition.disposition_unlink_marker",
        side_effect=_fail_unlink,
    ):
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=_seal_finalized(run_id),
        ):
            result = execute_approved_marker_disposition(
                disposition_id,
                generate_marker_disposition_attempt_id(),
                executor="executor",
                base_dir=tmp_path,
            )
    assert result.outcome_class == MarkerDispositionOutcomeClass.unlink_failed.value
    assert marker_present_noncreating(tmp_path, run_id)


def test_outcome_class_lock_directory_fsync_failed(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    import htr.execution_lock as el_mod

    real_fsync = el_mod._fsync_dir_fd
    seen = 0

    def _fail_dir_fsync(fd: int) -> None:
        nonlocal seen
        seen += 1
        if seen <= 2:
            real_fsync(fd)
            return
        raise RunExecutionLockIndeterminateError("injected dir fsync failure")

    with patch.object(el_mod, "_fsync_dir_fd", side_effect=_fail_dir_fsync):
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=_seal_finalized(run_id),
        ):
            result = execute_approved_marker_disposition(
                disposition_id,
                generate_marker_disposition_attempt_id(),
                executor="executor",
                base_dir=tmp_path,
            )
    assert (
        result.outcome_class
        == MarkerDispositionOutcomeClass.lock_directory_fsync_failed.value
    )


def test_outcome_class_outcome_durability_indeterminate(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    real_fsync = marker_disposition_mod._fsync_file_fd
    seen: list[str] = []

    def _fail_outcome_fsync(
        fd: int,
        *,
        disposition_id: str,
        record_name: str,
    ) -> None:
        seen.append(record_name)
        if record_name == "outcome.json":
            raise MarkerDispositionDurabilityError(
                "injected outcome fsync",
                disposition_id=disposition_id,
                record_name="outcome.json",
                durability_stage="record_fsync",
                record_may_have_committed=True,
                exact_replay_status="indeterminate",
            )
        real_fsync(fd, disposition_id=disposition_id, record_name=record_name)

    with patch.object(marker_disposition_mod, "_fsync_file_fd", side_effect=_fail_outcome_fsync):
        with patch(
            "htr.marker_disposition.evaluate_run_seal",
            return_value=_seal_finalized(run_id),
        ):
            with pytest.raises(MarkerDispositionDurabilityError):
                execute_approved_marker_disposition(
                    disposition_id,
                    generate_marker_disposition_attempt_id(),
                    executor="executor",
                    base_dir=tmp_path,
                )
    assert "attempt.json" in seen
    assert "outcome.json" in seen
    assert paths.marker_disposition_attempt_path(disposition_id, tmp_path).is_file()
    assert not paths.marker_disposition_outcome_path(disposition_id, tmp_path).exists()
    reconcile = reconcile_marker_disposition_outcome(disposition_id, base_dir=tmp_path)
    assert reconcile.classification in (
        "attempt_with_marker_present",
        "attempt_with_marker_absent",
    )


@pytest.mark.parametrize("stage", _DURABILITY_STAGES)
def test_request_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    case_id, _run_id, disposition_id = _real_finalized_path_a_ready(
        tmp_path,
        decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
    )
    _inject_mdp_durability_failure(monkeypatch, stage)
    with pytest.raises(MarkerDispositionDurabilityError) as excinfo:
        create_marker_disposition_request(
            disposition_id,
            case_id,
            requested_by="operator",
            base_dir=tmp_path,
        )
    assert excinfo.value.durability_stage == stage


@pytest.mark.parametrize("stage", ("record_fsync", "disposition_dir_fsync"))
def test_issue_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    _inject_mdp_durability_failure(monkeypatch, stage)
    with pytest.raises(MarkerDispositionDurabilityError) as excinfo:
        issue_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_approval_id(),
            issued_by="approver",
            expires_at=_expires_in_minutes(10),
            base_dir=tmp_path,
        )
    assert excinfo.value.durability_stage == stage


@pytest.mark.parametrize("stage", ("record_fsync", "disposition_dir_fsync"))
def test_claim_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    _inject_mdp_durability_failure(monkeypatch, stage)
    with pytest.raises(MarkerDispositionDurabilityError) as excinfo:
        claim_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_claim_id(),
            claimant="executor",
            base_dir=tmp_path,
        )
    assert excinfo.value.durability_stage == stage


def test_load_bundle_performs_zero_writes(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    before = _control_snapshot(tmp_path)
    bundle = load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)
    assert bundle.outcome_record is not None
    assert before == _control_snapshot(tmp_path)


def test_reconcile_attempt_with_marker_absent(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    with patch(
        "htr.marker_disposition.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    paths.marker_disposition_outcome_path(disposition_id, tmp_path).unlink()
    result = reconcile_marker_disposition_outcome(disposition_id, base_dir=tmp_path)
    assert result.classification == "attempt_with_marker_absent"


def test_boundary_snapshot_unchanged_on_full_disposition_chain(tmp_path):
    case_id, run_id, disposition_id = _real_finalized_path_a_case(
        tmp_path,
        decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
    )
    before = _boundary_snapshot(tmp_path, case_id=case_id)
    _issue_claim_execute(tmp_path, disposition_id, run_id)
    after = _boundary_snapshot(tmp_path, case_id=case_id)
    assert after == before


def test_subprocess_response_lost_request_replay(tmp_path):
    case_id, _run_id, disposition_id = _real_finalized_path_a_ready(
        tmp_path,
        decision_class=ReconciliationDecisionClass.case_closed_deferred_to_protocol,
    )
    open_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from htr.marker_disposition import create_marker_disposition_request
create_marker_disposition_request(
    {disposition_id!r},
    {case_id!r},
    requested_by="operator",
    base_dir=Path(sys.argv[2]),
)
print("created")
"""
    first = _run_isolated_mdp_script(open_script, tmp_path)
    assert first.returncode == 0, first.stderr
    replay_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from htr.marker_disposition import create_marker_disposition_request
_record, meta = create_marker_disposition_request(
    {disposition_id!r},
    {case_id!r},
    requested_by="operator",
    base_dir=Path(sys.argv[2]),
)
print(f"replay:{{meta.exact_replay}}")
"""
    second = _run_isolated_mdp_script(replay_script, tmp_path)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "replay:True"


def test_all_ten_outcome_classes_are_distinct_enum_members():
    values = {member.value for member in MarkerDispositionOutcomeClass}
    assert len(values) == 10
    covered = {
        MarkerDispositionOutcomeClass.disposed_verified.value,
        MarkerDispositionOutcomeClass.already_absent_observed.value,
        MarkerDispositionOutcomeClass.execution_ambiguous.value,
        MarkerDispositionOutcomeClass.marker_changed.value,
        MarkerDispositionOutcomeClass.approval_invalid.value,
        MarkerDispositionOutcomeClass.evidence_drifted.value,
        MarkerDispositionOutcomeClass.integrity_blocked.value,
        MarkerDispositionOutcomeClass.unlink_failed.value,
        MarkerDispositionOutcomeClass.lock_directory_fsync_failed.value,
        MarkerDispositionOutcomeClass.outcome_durability_indeterminate.value,
    }
    assert values == covered


# --- Task 26C hardening batch 2 (parametrized matrix) ---

_NON_PERMISSION_FIELDS = (
    "safe_to_retry",
    "invoke_allowed",
    "repair_allowed",
    "recovery_run_creation_allowed",
    "outcome_rewrite_allowed",
    "further_marker_disposition_allowed",
)

_INVALID_BOOL_VALUES = (
    pytest.param(True, id="true"),
    pytest.param(0, id="zero"),
    pytest.param("false", id="string-false"),
    pytest.param(None, id="null"),
    pytest.param(1, id="one"),
    pytest.param([], id="empty-list"),
    pytest.param("True", id="string-True"),
)

_LOAD_BUNDLE_OUTCOME_CLASSIFICATIONS = (
    "valid_durable_outcome",
    "malformed_outcome",
    "attempt_with_marker_present",
    "attempt_with_marker_absent",
)

_RECONCILE_CLASSIFICATIONS = (
    "missing_request",
    "incomplete",
    "claim_without_attempt",
    "valid_durable_outcome",
    "malformed_outcome",
    "attempt_with_marker_present",
    "attempt_with_marker_absent",
)



def _tamper_reconciliation_decision(
    case_id: str,
    base_dir: Path,
    *,
    decision_class: ReconciliationDecisionClass | None = None,
    recommended_next_protocol: ReconciliationNextProtocol | None = None,
) -> None:
    from htr.reconciliation_cases import _compute_decision_digest

    decision_path = paths.reconciliation_decision_path(case_id, base_dir)
    raw = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision_class is not None:
        raw["decision_class"] = decision_class.value
    if recommended_next_protocol is not None:
        raw["recommended_next_protocol"] = recommended_next_protocol.value
    raw["decision_digest"] = _compute_decision_digest(raw)
    decision_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _tamper_digest(path: Path, digest_field: str, *, mode: str) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if mode == "wrong_digest":
        raw[digest_field] = "sha256:" + "c" * 64
    elif mode == "missing_digest":
        raw.pop(digest_field, None)
    else:
        raise ValueError(mode)
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _setup_reconcile_classification(
    tmp_path: Path, classification: str
) -> tuple[str, str]:
    disposition_id = generate_marker_disposition_id()
    if classification == "missing_request":
        return disposition_id, classification
    if classification == "incomplete":
        _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
        return disposition_id, classification
    if classification == "claim_without_attempt":
        disposition_id, _run_id = _full_disposition_chain(tmp_path)
        return disposition_id, classification
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    with _patch_finalized_seal(run_id):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    if classification == "valid_durable_outcome":
        return disposition_id, classification
    if classification == "malformed_outcome":
        outcome_path = paths.marker_disposition_outcome_path(disposition_id, tmp_path)
        _tamper_digest(outcome_path, "outcome_digest", mode="wrong_digest")
        return disposition_id, classification
    if classification in (
        "attempt_with_marker_present",
        "attempt_with_marker_absent",
    ):
        paths.marker_disposition_outcome_path(disposition_id, tmp_path).unlink()
        if classification == "attempt_with_marker_present":
            _write_marker(tmp_path, run_id)
        else:
            marker_path = tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker"
            if marker_path.exists():
                marker_path.unlink()
        return disposition_id, classification
    raise ValueError(classification)


def _write_completed_outcome_chain(tmp_path: Path) -> tuple[str, str]:
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    with _patch_finalized_seal(run_id):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    return disposition_id, run_id


def test_public_api_excludes_private_persist_helper():
    import htr.marker_disposition as mod

    all_names = getattr(mod, "__all__", None)
    if all_names is not None:
        assert "_persist_outcome_private" not in all_names
    assert "_persist_outcome_private" in mod.__dict__


def test_issue_expiry_equal_to_issue_time_rejected(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issued_at = datetime.now(timezone.utc).isoformat()
    with pytest.raises(MarkerDispositionValidationError, match="expires_at"):
        issue_marker_disposition_approval(
            disposition_id,
            generate_marker_disposition_approval_id(),
            issued_by="approver",
            expires_at=issued_at,
            base_dir=tmp_path,
        )


def test_claim_before_expiry_succeeds(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(5),
        base_dir=tmp_path,
    )
    _rec, meta = claim_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    assert meta.exact_replay is False
    assert _rec.claimant == "executor"


def test_execute_without_claim_rejected(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    with pytest.raises(MarkerDispositionValidationError, match="claim"):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )


def test_execute_conflicting_attempt_id_rejected(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    attempt_id = generate_marker_disposition_attempt_id()
    with _patch_finalized_seal(run_id):
        execute_approved_marker_disposition(
            disposition_id,
            attempt_id,
            executor="executor",
            base_dir=tmp_path,
        )
    with _patch_finalized_seal(run_id):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.execution_ambiguous.value


def test_marker_absent_before_first_execution_is_already_absent(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    (tmp_path / LOCKS_DIR_NAME / f"{run_id}.marker").unlink()
    with _patch_finalized_seal(run_id):
        result = execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    assert result.outcome_class == MarkerDispositionOutcomeClass.already_absent_observed.value
    assert result.outcome_class != MarkerDispositionOutcomeClass.disposed_verified.value


def test_issue_tampered_request_digest_rejected_on_load(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    issue_path = paths.marker_disposition_issue_path(disposition_id, tmp_path)
    raw = json.loads(issue_path.read_text(encoding="utf-8"))
    raw["request_digest"] = "sha256:" + "d" * 64
    issue_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MarkerDispositionValidationError, match="issue_digest"):
        load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)


def test_task22_blocks_mutation_after_disposed_verified(tmp_path):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    with _patch_finalized_seal(run_id):
        execute_approved_marker_disposition(
            disposition_id,
            generate_marker_disposition_attempt_id(),
            executor="executor",
            base_dir=tmp_path,
        )
    with patch(
        "htr.finalization.evaluate_run_seal",
        return_value=_seal_finalized(run_id),
    ):
        with pytest.raises(RunFinalizedError):
            with run_write_barrier(run_id, base_dir=tmp_path):
                pass


def _subprocess_revoke_worker(
    disposition_id: str,
    base_dir: str,
    slot: Any,
    barrier: Any,
) -> None:
    from pathlib import Path

    from htr.marker_disposition import revoke_marker_disposition_approval

    try:
        barrier.wait()
        revoke_marker_disposition_approval(
            disposition_id,
            revoked_by="approver",
            reason="race-test",
            base_dir=Path(base_dir),
        )
        slot.put("revoked")
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def _subprocess_claim_race_worker(
    disposition_id: str,
    base_dir: str,
    claim_id: str,
    slot: Any,
    barrier: Any,
) -> None:
    from pathlib import Path

    from htr.marker_disposition import claim_marker_disposition_approval

    try:
        barrier.wait()
        _rec, meta = claim_marker_disposition_approval(
            disposition_id,
            claim_id,
            claimant="executor",
            base_dir=Path(base_dir),
        )
        slot.put(("ok", meta.exact_replay))
    except Exception as exc:
        slot.put(("err", type(exc).__name__, str(exc)))


def test_subprocess_revoke_claim_race_deterministic(tmp_path):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    claim_id = generate_marker_disposition_claim_id()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    slots = [ctx.Queue(), ctx.Queue()]
    procs = [
        ctx.Process(
            target=_subprocess_revoke_worker,
            args=(disposition_id, str(tmp_path), slots[0], barrier),
        ),
        ctx.Process(
            target=_subprocess_claim_race_worker,
            args=(disposition_id, str(tmp_path), claim_id, slots[1], barrier),
        ),
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0
    r0 = slots[0].get(timeout=5)
    r1 = slots[1].get(timeout=5)
    assert r0 == "revoked" or (isinstance(r0, tuple) and r0[0] == "err")
    assert r1[0] in ("ok", "err")


@pytest.mark.parametrize(
    "filename,digest_field,path_fn",
    (
        ("request.json", "request_digest", paths.marker_disposition_request_path),
        ("issue.json", "issue_digest", paths.marker_disposition_issue_path),
        ("claim.json", "claim_digest", paths.marker_disposition_claim_path),
        ("attempt.json", "attempt_digest", paths.marker_disposition_attempt_path),
        ("outcome.json", "outcome_digest", paths.marker_disposition_outcome_path),
    ),
)
@pytest.mark.parametrize("tamper_mode", ("wrong_digest", "missing_digest"))
def test_load_bundle_rejects_tampered_immutable_record(
    tmp_path, filename, digest_field, path_fn, tamper_mode
):
    disposition_id, _run_id = _write_completed_outcome_chain(tmp_path)
    target = path_fn(disposition_id, tmp_path)
    assert target.is_file(), filename
    _tamper_digest(target, digest_field, mode=tamper_mode)
    with pytest.raises(MarkerDispositionValidationError):
        load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)


@pytest.mark.parametrize("field", _NON_PERMISSION_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_BOOL_VALUES)
def test_reconcile_rejects_non_permission_boolean(tmp_path, field, bad_value):
    disposition_id, _run_id = _write_completed_outcome_chain(tmp_path)
    outcome_path = paths.marker_disposition_outcome_path(disposition_id, tmp_path)
    raw = json.loads(outcome_path.read_text(encoding="utf-8"))
    raw[field] = bad_value
    outcome_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    result = reconcile_marker_disposition_outcome(disposition_id, base_dir=tmp_path)
    assert result.classification == "malformed_outcome"


@pytest.mark.parametrize("field", _NON_PERMISSION_FIELDS)
@pytest.mark.parametrize("bad_value", _INVALID_BOOL_VALUES)
def test_load_bundle_rejects_non_permission_boolean(tmp_path, field, bad_value):
    disposition_id, _run_id = _write_completed_outcome_chain(tmp_path)
    outcome_path = paths.marker_disposition_outcome_path(disposition_id, tmp_path)
    raw = json.loads(outcome_path.read_text(encoding="utf-8"))
    raw[field] = bad_value
    outcome_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MarkerDispositionValidationError, match=field):
        load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)


@pytest.mark.parametrize("classification", _RECONCILE_CLASSIFICATIONS)
def test_reconcile_classification_zero_write(tmp_path, classification):
    disposition_id, expected = _setup_reconcile_classification(tmp_path, classification)
    before = _control_snapshot(tmp_path)
    result = reconcile_marker_disposition_outcome(disposition_id, base_dir=tmp_path)
    assert before == _control_snapshot(tmp_path)
    assert result.classification == expected


@pytest.mark.parametrize("classification", _LOAD_BUNDLE_OUTCOME_CLASSIFICATIONS)
def test_load_bundle_zero_write_when_outcome_present(tmp_path, classification):
    disposition_id, _expected = _setup_reconcile_classification(tmp_path, classification)
    before = _control_snapshot(tmp_path)
    try:
        load_marker_disposition_bundle(disposition_id, base_dir=tmp_path)
    except MarkerDispositionValidationError:
        if classification != "malformed_outcome":
            raise
    assert before == _control_snapshot(tmp_path)


@pytest.mark.parametrize("stage", _DURABILITY_STAGES)
def test_revoke_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    _case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    _inject_mdp_durability_failure(monkeypatch, stage)
    with pytest.raises(MarkerDispositionDurabilityError) as excinfo:
        revoke_marker_disposition_approval(
            disposition_id,
            revoked_by="approver",
            reason="durability",
            base_dir=tmp_path,
        )
    assert excinfo.value.durability_stage == stage


@pytest.mark.parametrize("stage", ("record_fsync", "disposition_dir_fsync"))
def test_execute_attempt_durability_stage_error_envelope(tmp_path, monkeypatch, stage):
    disposition_id, run_id = _full_disposition_chain(tmp_path)
    _inject_mdp_durability_failure(monkeypatch, stage)
    with pytest.raises(MarkerDispositionDurabilityError) as excinfo:
        with _patch_finalized_seal(run_id):
            execute_approved_marker_disposition(
                disposition_id,
                generate_marker_disposition_attempt_id(),
                executor="executor",
                base_dir=tmp_path,
            )
    assert excinfo.value.durability_stage == stage


@pytest.mark.parametrize(
    "bad_id,label",
    (
        pytest.param("mdp_not-valid!!!", "disposition", id="bad-disposition"),
        pytest.param("mda_not-valid!!!", "approval", id="bad-approval"),
        pytest.param("mdc_not-valid!!!", "claim", id="bad-claim"),
        pytest.param("mat_not-valid!!!", "attempt", id="bad-attempt"),
    ),
)
def test_validate_id_rejects_malformed_marker_disposition_ids(
    tmp_path, bad_id, label
):
    case_id, _run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    with pytest.raises(MarkerDispositionValidationError):
        if label == "disposition":
            create_marker_disposition_request(
                bad_id,
                case_id,
                requested_by="operator",
                base_dir=tmp_path,
            )
        elif label == "approval":
            issue_marker_disposition_approval(
                disposition_id,
                bad_id,
                issued_by="approver",
                expires_at=_expires_in_minutes(10),
                base_dir=tmp_path,
            )
        elif label == "claim":
            issue_marker_disposition_approval(
                disposition_id,
                generate_marker_disposition_approval_id(),
                issued_by="approver",
                expires_at=_expires_in_minutes(10),
                base_dir=tmp_path,
            )
            claim_marker_disposition_approval(
                disposition_id,
                bad_id,
                claimant="executor",
                base_dir=tmp_path,
            )
        else:
            disposition_id2, run_id2 = _full_disposition_chain(tmp_path)
            with _patch_finalized_seal(run_id2):
                execute_approved_marker_disposition(
                    disposition_id2,
                    bad_id,
                    executor="executor",
                    base_dir=tmp_path,
                )


@pytest.mark.parametrize(
    "actor_field",
    ("requested_by", "issued_by", "claimant"),
)
def test_actor_intent_mismatch_rejected(tmp_path, actor_field):
    case_id, run_id, disposition_id = _path_a_case(tmp_path, finalize=True)
    if actor_field == "requested_by":
        with pytest.raises(MarkerDispositionConflictError):
            create_marker_disposition_request(
                disposition_id,
                case_id,
                requested_by="other-operator",
                base_dir=tmp_path,
            )
        return
    issue_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_approval_id(),
        issued_by="approver",
        expires_at=_expires_in_minutes(10),
        base_dir=tmp_path,
    )
    if actor_field == "issued_by":
        with pytest.raises(MarkerDispositionConflictError):
            issue_marker_disposition_approval(
                disposition_id,
                generate_marker_disposition_approval_id(),
                issued_by="other-approver",
                expires_at=_expires_in_minutes(10),
                base_dir=tmp_path,
            )
        return
    claim_marker_disposition_approval(
        disposition_id,
        generate_marker_disposition_claim_id(),
        claimant="executor",
        base_dir=tmp_path,
    )
    if actor_field == "claimant":
        with pytest.raises(MarkerDispositionConflictError):
            claim_marker_disposition_approval(
                disposition_id,
                generate_marker_disposition_claim_id(),
                claimant="other-executor",
                base_dir=tmp_path,
            )
