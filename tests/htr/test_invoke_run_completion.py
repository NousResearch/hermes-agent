"""Tests for Task 25 — human-gated single-API run-completion invoke pilot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from htr import approval_control, contracts, events, execution_lock as _el, io, paths
from htr.action_plan import PlanningIntent, _CATALOG
from htr.approval_control import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_CONSUMED,
    claim_approval,
    get_approval,
    issue_approval,
    record_use_outcome,
    revoke_approval,
)
from htr.approval_control import OUTCOME_SCHEMA_VERSION_V2
from htr.contracts import run_completion_fingerprint, run_completion_record_json_path
from htr.events import EVENT_TYPE_MANUAL_RUN_COMPLETED
from htr.execution_lock import (
    LOCKS_DIR_NAME,
    RunExecutionLockDurabilityError,
    RunExecutionLockIndeterminateError,
    RunExecutionLockOccupiedError,
    marker_present_noncreating,
)
from htr.finalization import SealEvaluation, SealState, evaluate_run_seal
from htr.ids import new_approval_id, new_event_id, new_run_id, new_task_id
from htr.invoke_run_completion import (
    PILOT_BOUND_API,
    REASON_CLAIMED_INVOKE_NOT_STARTED,
    REASON_INVOKE_RAISED_COMMIT_UNKNOWN,
    REASON_POST_VERIFICATION_MISMATCH,
    REASON_VERIFIED_SUCCESS,
    invoke_approved_run_completion,
)
from htr.observe import ObserveInvocationError
from htr.state import (
    ApprovalValidationError,
    InvokeAmbiguousOutcomeError,
    InvokeCleanupDurabilityError,
    InvokeOutcomePersistenceError,
    InvokeStaleApprovalError,
    RUN_COMPLETED,
)

TASK16_PATH = Path(__file__).with_name("test_run_final_closure.py")


def _load_task16():
    spec = importlib.util.spec_from_file_location("task16_helpers", TASK16_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16()


def _run_with_completion_only(tmp_path: Path) -> str:
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    events.complete_run_manually(run_id, completion, base_dir=tmp_path)
    return run_id


def _review_intent(run_id: str, tmp_path: Path, event_id: str) -> PlanningIntent:
    review = contracts.make_run_review_record(
        run_id=run_id, decision=contracts.RUN_REVIEW_NEEDS_FOLLOWUP
    )
    return PlanningIntent(
        requested_action="review_run_manually",
        action_inputs={"record": review, "actor": "human", "event_id": event_id},
        htr_runs_root=str(tmp_path),
    )


def _lifecycle_api_run_state(api: str, tmp_path: Path) -> tuple[str, PlanningIntent]:
    event_id = new_event_id()
    if api == "complete_run_manually":
        run_id, task_id, completion = _run_ready_for_completion(tmp_path)
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"record": completion, "actor": "human", "event_id": event_id},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "review_run_manually":
        run_id = _run_with_completion_only(tmp_path)
        return run_id, _review_intent(run_id, tmp_path, event_id)
    if api == "plan_run_followup":
        run_id = TASK16._run_with_reviewed_run(tmp_path)[0]
        record = TASK16._plan_record(run_id)
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"record": record, "actor": "human", "event_id": event_id},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "request_run_execution":
        chain = TASK16._run_with_planned_run(tmp_path)
        run_id = chain[0]
        plan = chain[5]
        record = contracts.make_run_execution_request_record(
            run_id=run_id,
            source_followup_plan_fingerprint=contracts.run_followup_plan_fingerprint(plan),
            execution_items=[TASK16._sample_execution_item()],
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"record": record, "actor": "human", "event_id": event_id},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "execute_run_execution_request":
        chain = TASK16._run_with_execution_request(tmp_path)
        run_id = chain[0]
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={"executor": "bob", "event_id": event_id, "project_dir": str(tmp_path)},
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "verify_run_execution_result":
        chain = TASK16._run_with_execution_result(tmp_path)
        run_id = chain[0]
        record = TASK16._verification_record(run_id, chain[7])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "plan_post_verification_followup":
        chain = TASK16._run_with_verified_execution(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_plan_record(run_id, chain[7], chain[8])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "request_post_verification_execution":
        chain = TASK16._run_with_post_verification_plan(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_execution_request_record(
            run_id, chain[7], chain[8], chain[9]
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "record_post_verification_execution_result":
        chain = TASK16._run_with_post_verification_execution_request(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_execution_result_record(
            run_id, chain[7], chain[8], chain[9], chain[10]
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "record_post_verification_execution_verification":
        chain = TASK16._run_with_post_verification_execution_result(tmp_path)
        run_id = chain[0]
        record = TASK16._post_verification_execution_verification_record(
            run_id, chain[7], chain[8], chain[9], chain[10], chain[11]
        )
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": record,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    if api == "record_run_final_closure":
        chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
        run_id = chain[0]
        closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
        intent = PlanningIntent(
            requested_action=api,
            action_inputs={
                "record": closure,
                "actor": "human",
                "event_id": event_id,
                "project_dir": str(tmp_path),
            },
            htr_runs_root=str(tmp_path),
        )
        return run_id, intent
    raise AssertionError(f"unhandled api {api!r}")


def _expires_in(hours: float = 1.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _expires_past() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()


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


def _completion_intent(
    run_id: str,
    completion: dict[str, Any],
    tmp_path: Path,
    event_id: str,
) -> PlanningIntent:
    return PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs={"record": completion, "actor": "human", "event_id": event_id},
        htr_runs_root=str(tmp_path),
    )


def _issue_completion_approval(
    tmp_path: Path,
    run_id: str,
    completion: dict[str, Any],
    *,
    event_id: str | None = None,
    executor_id: str = "bob",
    expires_at: str | None = None,
    approval_id: str | None = None,
    checkpoint: str | None = None,
) -> tuple[dict[str, Any], str]:
    event_id = event_id or new_event_id()
    intent = _completion_intent(run_id, completion, tmp_path, event_id)
    if checkpoint is not None:
        intent = PlanningIntent(
            requested_action=PILOT_BOUND_API,
            action_inputs={"record": completion, "actor": "human", "event_id": event_id},
            project_repository_checkpoint=checkpoint,
            htr_runs_root=str(tmp_path),
        )
    kwargs: dict[str, Any] = {
        "approver_id": "alice",
        "executor_id": executor_id,
        "expires_at": expires_at or _expires_in(),
        "base_dir": tmp_path,
    }
    if approval_id is not None:
        kwargs["approval_id"] = approval_id
    issue = issue_approval(run_id, intent, **kwargs)
    return issue, event_id


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


def test_happy_path_exact_completion_and_v2_consumed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_id = "claim-task25-happy"

    with patch(
        "htr.invoke_run_completion.complete_run_manually",
        wraps=events.complete_run_manually,
    ) as mocked:
        result = invoke_approved_run_completion(
            issue["approval_id"],
            claim_id=claim_id,
            base_dir=tmp_path,
        )
        mocked.assert_called_once_with(
            run_id,
            completion,
            actor="human",
            event_id=event_id,
            base_dir=tmp_path,
        )

    assert result.approval_id == issue["approval_id"]
    assert result.claim_id == claim_id
    assert result.run_id == run_id
    assert result.event_id == event_id
    assert result.completion_record_fingerprint == run_completion_fingerprint(completion)
    assert result.pre_observation_digest == issue["source_observation_digest"]
    assert result.post_observation_digest.startswith("sha256:")
    assert result.outcome_digest.startswith("sha256:")

    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["claim"]["claim_id"] == claim_id
    assert bundle["claim"]["claimant_id"] == "bob"
    outcome = bundle["outcome"]
    assert outcome["outcome_schema_version"] == OUTCOME_SCHEMA_VERSION_V2
    assert outcome["outcome_class"] == OUTCOME_CONSUMED
    assert outcome["outcome_digest"].startswith("sha256:")
    assert outcome["outcome_evidence"]["reason_code"] == REASON_VERIFIED_SUCCESS
    assert outcome["outcome_evidence"]["safe_to_retry"] is False

    assert io.read_json(paths.run_manifest_path(run_id, tmp_path))["status"] == RUN_COMPLETED
    assert run_completion_record_json_path(run_id, tmp_path).is_file()
    completion_events = [
        ev
        for ev in events.read_task_events(run_id, base_dir=tmp_path)
        if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
    ]
    assert len(completion_events) == 1
    assert completion_events[0]["event_id"] == event_id
    assert not marker_present_noncreating(tmp_path, run_id)


@pytest.mark.parametrize(
    "api",
    sorted(api for api in _CATALOG if api != PILOT_BOUND_API),
)
def test_non_completion_approvals_rejected_before_claim(tmp_path, api: str):
    run_id, intent = _lifecycle_api_run_state(api, tmp_path)
    issue = issue_approval(
        run_id,
        intent,
        approver_id="alice",
        executor_id="bob",
        expires_at=_expires_in(),
        base_dir=tmp_path,
    )
    with patch.object(events, "complete_run_manually") as mocked:
        with pytest.raises(InvokeStaleApprovalError, match="bound_api"):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-allowlist",
                base_dir=tmp_path,
            )
        mocked.assert_not_called()
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["claim"] is None
    assert bundle["outcome"] is None


def test_rejects_pre_existing_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_approval(issue["approval_id"], "claim-pre", "bob", base_dir=tmp_path)
    with pytest.raises(InvokeStaleApprovalError, match="already claimed"):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-new",
            base_dir=tmp_path,
        )


def test_rejects_pre_existing_outcome(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    claim_approval(issue["approval_id"], "claim-out", "bob", base_dir=tmp_path)
    record_use_outcome(
        issue["approval_id"], "claim-out", OUTCOME_CONSUMED, base_dir=tmp_path
    )
    with pytest.raises(InvokeStaleApprovalError, match="already claimed"):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-retry",
            base_dir=tmp_path,
        )


def test_rejects_non_null_project_repository_checkpoint(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(
        tmp_path, run_id, completion, checkpoint="cp-live"
    )
    with pytest.raises(InvokeStaleApprovalError, match="project_repository_checkpoint"):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-cp",
            base_dir=tmp_path,
        )


def test_rejects_expired_approval(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    expired_now = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    with patch(
        "htr.invoke_run_completion._utc_now",
        return_value=approval_control._parse_utc_iso(expired_now),
    ):
        with pytest.raises(InvokeStaleApprovalError, match="expired"):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-exp",
                base_dir=tmp_path,
            )


def test_rejects_revoked_approval(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    revoke_approval(issue["approval_id"], "alice", "cancel", base_dir=tmp_path)
    with pytest.raises(InvokeStaleApprovalError, match="revoked"):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-rev",
            base_dir=tmp_path,
        )


def test_rejects_pre_existing_completion_record(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _event_id = _issue_completion_approval(tmp_path, run_id, completion)
    from unittest.mock import MagicMock

    mock_path = MagicMock()
    mock_path.exists.return_value = True

    with patch(
        "htr.invoke_run_completion.run_completion_record_json_path",
        return_value=mock_path,
    ):
        with pytest.raises(InvokeStaleApprovalError, match="run_completion_record"):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-dup-rec",
                base_dir=tmp_path,
            )


def test_rejects_pre_existing_event_id(tmp_path):
    run_id, task_id, completion = _run_ready_for_completion(tmp_path)
    event_id = new_event_id()
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion, event_id=event_id)
    events.append_task_event(
        run_id,
        events.make_event(
            event_type="manual_task_status_changed",
            run_id=run_id,
            task_id=task_id,
            event_id=event_id,
            actor="human",
            previous_status="completed",
            new_status="completed",
            payload={},
        ),
        base_dir=tmp_path,
    )
    with pytest.raises(InvokeStaleApprovalError, match="event_id"):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-ev",
            base_dir=tmp_path,
        )


def test_rejects_finalized_run(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    run_id = chain[0]
    closure = TASK16._run_final_closure_record(run_id, *chain[3:13])
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    assert evaluate_run_seal(run_id, tmp_path).state == SealState.FINALIZED_VALID
    run_id2, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id2, completion)
    with patch(
        "htr.invoke_run_completion.evaluate_run_seal",
        return_value=SealEvaluation(SealState.FINALIZED_VALID, (), run_id2),
    ):
        with pytest.raises(InvokeStaleApprovalError, match="finalized"):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-fin",
                base_dir=tmp_path,
            )


def test_rejects_invalid_completed_task_reference(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    bogus = dict(completion)
    bogus["completed_task_ids"] = [new_task_id()]
    issue, _ = _issue_completion_approval(tmp_path, run_id, bogus)
    with pytest.raises(InvokeStaleApprovalError, match="task"):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-bogus-task",
            base_dir=tmp_path,
        )


def test_rejects_existing_marker_before_session(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    lock_root = tmp_path / LOCKS_DIR_NAME
    lock_root.mkdir(parents=True, exist_ok=True)
    (lock_root / f"{run_id}.marker").write_text("{}", encoding="utf-8")
    with pytest.raises(RunExecutionLockOccupiedError):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-occ",
            base_dir=tmp_path,
        )


def test_concurrent_invoke_exactly_one_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    barrier = threading.Barrier(2)
    results: list[Any] = []

    def worker(claim_id: str) -> None:
        barrier.wait(timeout=5)
        try:
            results.append(
                (
                    "ok",
                    invoke_approved_run_completion(
                        issue["approval_id"],
                        claim_id=claim_id,
                        base_dir=tmp_path,
                    ),
                )
            )
        except Exception as exc:
            results.append(("err", type(exc).__name__, str(exc)))

    t1 = threading.Thread(target=worker, args=("claim-a",))
    t2 = threading.Thread(target=worker, args=("claim-b",))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    ok = [r for r in results if r[0] == "ok"]
    err = [r for r in results if r[0] == "err"]
    assert len(ok) == 1
    assert len(err) == 1
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["claim"] is not None
    assert bundle["outcome"]["outcome_class"] == OUTCOME_CONSUMED


def test_claimed_invoke_not_started_records_ambiguous_and_cleans_marker(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    original_claim = approval_control._claim_approval_during_session

    def claim_then_fail(*args, **kwargs):
        original_claim(*args, **kwargs)
        raise RuntimeError("invoke never started")

    with patch(
        "htr.invoke_run_completion._claim_approval_during_session",
        side_effect=claim_then_fail,
    ):
        with pytest.raises(InvokeAmbiguousOutcomeError) as excinfo:
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-not-started",
                base_dir=tmp_path,
            )
    err = excinfo.value
    assert err.reason_code == REASON_CLAIMED_INVOKE_NOT_STARTED
    assert err.mutation_may_have_committed is False
    assert err.safe_to_retry is False
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["outcome"]["outcome_class"] == OUTCOME_AMBIGUOUS
    assert bundle["outcome"]["outcome_evidence"]["reason_code"] == REASON_CLAIMED_INVOKE_NOT_STARTED
    assert not marker_present_noncreating(tmp_path, run_id)


def test_invoke_attempted_ambiguity_preserves_marker(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    from htr.observe import build_run_snapshot as real_observe

    calls = {"n": 0}

    def observe(run_id_arg, *, base_dir=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            return real_observe(run_id_arg, base_dir=base_dir)
        raise ObserveInvocationError("post observe failed")

    with patch("htr.invoke_run_completion.build_run_snapshot", side_effect=observe):
        with pytest.raises(InvokeAmbiguousOutcomeError) as excinfo:
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-post-obs",
                base_dir=tmp_path,
            )
    assert excinfo.value.mutation_may_have_committed is True
    assert marker_present_noncreating(tmp_path, run_id)
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["outcome"]["outcome_class"] == OUTCOME_AMBIGUOUS


def test_post_verification_mismatch_records_ambiguous(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)

    real = events.complete_run_manually

    def tampered_complete(*args, **kwargs):
        event = real(*args, **kwargs)
        tampered = dict(event)
        tampered["actor"] = "wrong-actor"
        return tampered

    with patch("htr.invoke_run_completion.complete_run_manually", side_effect=tampered_complete):
        with pytest.raises(InvokeAmbiguousOutcomeError) as excinfo:
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-verify-mismatch",
                base_dir=tmp_path,
            )
    assert excinfo.value.reason_code == REASON_POST_VERIFICATION_MISMATCH
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["outcome"]["outcome_evidence"]["verification_reason_codes"]


def test_outcome_persistence_failure_preserves_marker(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)

    with patch("htr.invoke_run_completion._record_use_outcome_during_session", side_effect=RuntimeError("fsync failed")):
        with pytest.raises(InvokeOutcomePersistenceError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-out-fail",
                base_dir=tmp_path,
            )
    assert marker_present_noncreating(tmp_path, run_id)
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["claim"] is not None
    assert bundle["outcome"] is None


def test_cleanup_failure_after_consumed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)

    with patch(
        "htr.execution_lock._release_marker_success",
        side_effect=RunExecutionLockDurabilityError("cleanup fsync failed", run_id=run_id),
    ):
        with pytest.raises(InvokeCleanupDurabilityError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-cleanup-fail",
                base_dir=tmp_path,
            )
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["outcome"]["outcome_class"] == OUTCOME_CONSUMED


def test_stale_rejection_after_preliminary_before_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)

    with patch(
        "htr.invoke_run_completion._post_marker_validate",
        side_effect=InvokeStaleApprovalError("approval revoked", approval_id=issue["approval_id"]),
    ):
        with pytest.raises(InvokeStaleApprovalError, match="revoked"):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-race-revoke",
                base_dir=tmp_path,
            )
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["claim"] is None


def test_v2_outcome_digest_binds_evidence(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    result = invoke_approved_run_completion(
        issue["approval_id"],
        claim_id="claim-digest",
        base_dir=tmp_path,
    )
    outcome = get_approval(issue["approval_id"], base_dir=tmp_path)["outcome"]
    path = paths.approval_outcome_path(issue["approval_id"], tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    body = dict(raw)
    body.pop("outcome_digest", None)
    projection = {
        "outcome_digest_projection_version": "htr.approval.outcome.digest.v2",
        "outcome_schema_version": "2",
        "approval_id": body["approval_id"],
        "approval_digest": body["approval_digest"],
        "claim_id": body["claim_id"],
        "claim_digest": body["claim_digest"],
        "outcome_class": body["outcome_class"],
        "recorded_at": body["recorded_at"],
        "outcome_evidence": body["outcome_evidence"],
    }
    from htr.action_plan import _sha256_digest

    expected = _sha256_digest(projection)
    assert outcome["outcome_digest"] == expected
    assert result.outcome_digest == expected


# --- Task 25 hardening ---


def _snapshot_invoke_state(
    tmp_path: Path,
    run_id: str,
    approval_id: str,
    *,
    complete_run_manually_call_count: int | None = None,
) -> dict[str, Any]:
    bundle = get_approval(approval_id, base_dir=tmp_path)
    events_path = paths.task_events_path(run_id, tmp_path)
    completion_events = 0
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED:
                completion_events += 1
    manifest_status = None
    manifest_path = paths.run_manifest_path(run_id, tmp_path)
    if manifest_path.is_file():
        manifest_status = io.read_json(manifest_path).get("status")
    snap: dict[str, Any] = {
        "issue_present": paths.approval_issue_path(approval_id, tmp_path).is_file(),
        "claim": bundle.get("claim"),
        "outcome": bundle.get("outcome"),
        "completion_json_exists": run_completion_record_json_path(run_id, tmp_path).is_file(),
        "completion_event_count": completion_events,
        "manifest_status": manifest_status,
        "marker_present": marker_present_noncreating(tmp_path, run_id),
    }
    if complete_run_manually_call_count is not None:
        snap["complete_run_manually_call_count"] = complete_run_manually_call_count
    return snap




def _outcome_not_consumed(snap: dict[str, Any]) -> bool:
    outcome = snap.get("outcome")
    if outcome is None:
        return True
    return outcome.get("outcome_class") != OUTCOME_CONSUMED


def _assert_fsync_failure_snapshot(
    snap: dict[str, Any],
    *,
    expect_claim: bool,
    expect_completion_json: bool | None = None,
) -> None:
    assert snap["marker_present"] is True
    assert _outcome_not_consumed(snap)
    if expect_claim:
        assert snap["claim"] is not None
    if expect_completion_json is not None:
        assert snap["completion_json_exists"] is expect_completion_json

def _patch_outcome_only_fsync(*, target: str):
    """Patch approval-control fsync to fail only while writing outcome.json."""
    assert target in {"file", "dir"}
    state = {"active": False}
    real_create = approval_control._create_immutable_record

    def track_create(approval_id, filename, record, *, digest_field, base_dir):
        if filename == "outcome.json":
            state["active"] = True
        try:
            return real_create(
                approval_id, filename, record, digest_field=digest_field, base_dir=base_dir
            )
        finally:
            if filename == "outcome.json":
                state["active"] = False

    if target == "file":
        real_file = approval_control._fsync_file_fd

        def selective_file(file_fd: int) -> None:
            if state["active"]:
                raise RunExecutionLockIndeterminateError("outcome file fsync failed")
            return real_file(file_fd)

        return (
            patch("htr.approval_control._create_immutable_record", side_effect=track_create),
            patch("htr.approval_control._fsync_file_fd", side_effect=selective_file),
        )

    real_dir = approval_control._fsync_dir_fd

    def selective_dir(dir_fd: int) -> None:
        if state["active"]:
            raise RunExecutionLockIndeterminateError("outcome directory fsync failed")
        return real_dir(dir_fd)

    return (
        patch("htr.approval_control._create_immutable_record", side_effect=track_create),
        patch("htr.approval_control._fsync_dir_fd", side_effect=selective_dir),
    )


def _lock_dir_fsync_failure_after_unlink():
    real_dir = _el._fsync_dir_fd
    state = {"unlink_seen": False}
    original_unlink = os.unlink

    def tracking_unlink(name, *, dir_fd=None):
        state["unlink_seen"] = True
        return original_unlink(name, dir_fd=dir_fd)

    def selective_dir(dir_fd: int) -> None:
        if state["unlink_seen"]:
            raise RunExecutionLockDurabilityError("lock dir fsync failed", run_id="ignored")
        return real_dir(dir_fd)

    return (
        patch.object(os, "unlink", side_effect=tracking_unlink),
        patch.object(_el, "_fsync_dir_fd", side_effect=selective_dir),
    )


def _subprocess_invoke_crash_before_claim_worker(
    runs_root: str,
    approval_id: str,
    claim_id: str,
) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from htr.invoke_run_completion import invoke_approved_run_completion

    base = Path(runs_root)

    def crash_before_claim(*_args, **_kwargs):
        os._exit(71)

    with patch(
        "htr.invoke_run_completion._claim_approval_during_session",
        side_effect=crash_before_claim,
    ):
        try:
            invoke_approved_run_completion(
                approval_id,
                claim_id=claim_id,
                base_dir=base,
            )
        except BaseException:
            pass


def test_subprocess_invoke_crash_after_marker_before_claim(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_subprocess_invoke_crash_before_claim_worker,
        args=(str(tmp_path), issue["approval_id"], "claim-crash-71"),
    )
    proc.start()
    proc.join(timeout=20)
    assert proc.exitcode == 71
    snap = _snapshot_invoke_state(
        tmp_path,
        run_id,
        issue["approval_id"],
        complete_run_manually_call_count=0,
    )
    assert snap["marker_present"] is True
    assert snap["claim"] is None
    assert snap["outcome"] is None
    assert snap["completion_json_exists"] is False
    assert snap["completion_event_count"] == 0
    assert snap["manifest_status"] != RUN_COMPLETED
    assert snap["complete_run_manually_call_count"] == 0
    with patch.object(events, "complete_run_manually") as mocked:
        with pytest.raises(RunExecutionLockOccupiedError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-after-crash",
                base_dir=tmp_path,
            )
        mocked.assert_not_called()


def test_ambiguous_outcome_file_fsync_failure_claimed_invoke_not_started(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    original_claim = approval_control._claim_approval_during_session

    def claim_then_fail(*args, **kwargs):
        original_claim(*args, **kwargs)
        raise RuntimeError("invoke never started")

    file_patch, create_patch = _patch_outcome_only_fsync(target="file")
    with ExitStack() as stack:
        stack.enter_context(file_patch)
        stack.enter_context(create_patch)
        stack.enter_context(
            patch(
                "htr.invoke_run_completion._claim_approval_during_session",
                side_effect=claim_then_fail,
            )
        )
        with pytest.raises(InvokeOutcomePersistenceError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-fs-file-not-started",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    _assert_fsync_failure_snapshot(snap, expect_claim=True)


def test_ambiguous_outcome_file_fsync_failure_post_verification_mismatch(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    real = events.complete_run_manually

    def tampered_complete(*args, **kwargs):
        event = real(*args, **kwargs)
        return dict(event) | {"actor": "wrong-actor"}

    file_patch, create_patch = _patch_outcome_only_fsync(target="file")
    with ExitStack() as stack:
        stack.enter_context(file_patch)
        stack.enter_context(create_patch)
        stack.enter_context(
            patch(
                "htr.invoke_run_completion.complete_run_manually",
                side_effect=tampered_complete,
            )
        )
        with pytest.raises(InvokeOutcomePersistenceError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-fs-file-verify",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    _assert_fsync_failure_snapshot(snap, expect_claim=True, expect_completion_json=True)


@pytest.mark.parametrize(
    "case_id",
    [
        pytest.param("claimed_invoke_not_started", id="claimed_invoke_not_started"),
        pytest.param(
            "invoke_attempted_post_verify_mismatch",
            id="invoke_attempted_post_verify_mismatch",
        ),
    ],
)
def test_ambiguous_outcome_dir_fsync_failure(tmp_path, case_id):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    dir_patch, create_patch = _patch_outcome_only_fsync(target="dir")
    with ExitStack() as stack:
        stack.enter_context(dir_patch)
        stack.enter_context(create_patch)
        if case_id == "claimed_invoke_not_started":
            original_claim = approval_control._claim_approval_during_session

            def claim_then_fail(*args, **kwargs):
                original_claim(*args, **kwargs)
                raise RuntimeError("invoke never started")

            stack.enter_context(
                patch(
                    "htr.invoke_run_completion._claim_approval_during_session",
                    side_effect=claim_then_fail,
                )
            )
        else:
            real = events.complete_run_manually

            def tampered_complete(*args, **kwargs):
                event = real(*args, **kwargs)
                return dict(event) | {"actor": "wrong-actor"}

            stack.enter_context(
                patch(
                    "htr.invoke_run_completion.complete_run_manually",
                    side_effect=tampered_complete,
                )
            )
        with pytest.raises(InvokeOutcomePersistenceError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id=f"claim-dir-fsync-{case_id}",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    _assert_fsync_failure_snapshot(snap, expect_claim=True)


def test_consumed_outcome_file_fsync_failure_after_success(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    file_patch, create_patch = _patch_outcome_only_fsync(target="file")
    with ExitStack() as stack:
        stack.enter_context(file_patch)
        stack.enter_context(create_patch)
        with pytest.raises(InvokeOutcomePersistenceError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-consumed-file-fsync",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    assert snap["marker_present"] is True
    assert snap["claim"] is not None
    assert snap["manifest_status"] == RUN_COMPLETED
    assert snap["completion_json_exists"] is True
    assert snap["completion_event_count"] == 1


def test_consumed_outcome_dir_fsync_failure_after_success(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    dir_patch, create_patch = _patch_outcome_only_fsync(target="dir")
    with ExitStack() as stack:
        stack.enter_context(dir_patch)
        stack.enter_context(create_patch)
        with pytest.raises(InvokeOutcomePersistenceError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-consumed-dir-fsync",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    _assert_fsync_failure_snapshot(snap, expect_claim=True, expect_completion_json=True)
    assert snap["manifest_status"] == RUN_COMPLETED


def test_invoke_cleanup_unlink_failure_after_consumed(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    with patch.object(
        _el,
        "_release_marker_success",
        side_effect=RunExecutionLockDurabilityError("unlink failed", run_id=run_id),
    ):
        with pytest.raises(InvokeCleanupDurabilityError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-cleanup-unlink",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    assert snap["outcome"]["outcome_class"] == OUTCOME_CONSUMED
    assert snap["marker_present"] is True


def test_invoke_lock_dir_fsync_failure_after_marker_unlink(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    unlink_patch, dir_patch = _lock_dir_fsync_failure_after_unlink()
    with ExitStack() as stack:
        stack.enter_context(unlink_patch)
        stack.enter_context(dir_patch)
        with pytest.raises(InvokeCleanupDurabilityError):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-lock-dir-fsync",
                base_dir=tmp_path,
            )
    snap = _snapshot_invoke_state(tmp_path, run_id, issue["approval_id"])
    assert snap["outcome"]["outcome_class"] == OUTCOME_CONSUMED
    assert snap["marker_present"] is False


POST_VERIFY_MISMATCH_EXPECTED_CODE: dict[str, str] = {
    "post_snapshot_not_trustworthy": "post_snapshot_not_trustworthy",
    "completion_record_missing": "completion_record_missing",
    "completion_record_schema_invalid": "completion_record_schema_invalid",
    "completion_record_wrong_run_id": "completion_record_wrong_run_id",
    "completion_record_wrong_completed_task_ids": "completion_record_wrong_completed_task_ids",
    "completion_record_wrong_reason": "completion_record_wrong_reason",
    "completion_record_null_reason_to_value": "completion_record_null_reason_to_value",
    "completion_record_value_reason_to_null": "completion_record_value_reason_to_null",
    "completion_record_wrong_metadata": "completion_record_wrong_metadata",
    "completion_record_wrong_created_at": "completion_record_wrong_created_at",
    "completion_record_fingerprint_mismatch": "completion_record_fingerprint_mismatch",
    "completion_event_missing": "completion_event_count_mismatch",
    "completion_event_count_mismatch": "completion_event_count_mismatch",
    "multiple_completion_events": "multiple_completion_events",
    "event_wrong_event_id": "completion_event_count_mismatch",
    "event_wrong_payload_completed_task_ids": "event_completed_task_ids_mismatch",
    "json_without_matching_event": "completion_event_count_mismatch",
    "event_without_matching_json": "completion_record_missing",
    "event_run_id_mismatch": "event_run_id_mismatch",
    "event_has_task_id": "event_has_task_id",
    "event_actor_mismatch": "event_actor_mismatch",
    "event_previous_status_mismatch": "event_previous_status_mismatch",
    "event_new_status_mismatch": "event_new_status_mismatch",
    "event_completed_task_ids_mismatch": "event_completed_task_ids_mismatch",
    "event_record_fingerprint_mismatch": "event_record_fingerprint_mismatch",
    "event_record_path_mismatch": "event_record_path_mismatch",
    "event_semantic_fingerprint_mismatch": "event_semantic_fingerprint_mismatch",
    "manifest_not_completed": "manifest_not_completed",
    "chain_completion_missing": "chain_completion_missing",
    "unexpected_chain_record:run_review_record": "unexpected_chain_record:run_review_record",
    "unexpected_finalization": "unexpected_chain_record:run_review_record",
    "task_state_changed": "unrelated_state_changed",
    "attempt_state_changed": "unrelated_state_changed",
    "result_state_changed": "unrelated_state_changed",
    "verification_state_changed": "unrelated_state_changed",
    "artifact_manifest_changed": "unrelated_state_changed",
    "post_closure_activity": "unrelated_state_changed",
    "unrelated_state_changed": "unrelated_state_changed",
}


def _expected_verification_code(case_id: str) -> str:
    return POST_VERIFY_MISMATCH_EXPECTED_CODE[case_id]




def _assert_ambiguous_post_verify(
    tmp_path: Path,
    issue: dict[str, Any],
    *,
    tamper,
    expected_reason: str,
    run_id: str,
) -> None:
    claim_id = f"claim-verify-{expected_reason}"
    with patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper):
        with pytest.raises(InvokeAmbiguousOutcomeError) as excinfo:
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id=claim_id,
                base_dir=tmp_path,
            )
    err = excinfo.value
    assert err.reason_code == REASON_POST_VERIFICATION_MISMATCH
    assert err.safe_to_retry is False
    assert err.mutation_may_have_committed is True
    codes = err.outcome_evidence["verification_reason_codes"]
    expected = _expected_verification_code(expected_reason)
    assert expected in codes or (
        expected.startswith("completion_record_")
        and "completion_record_fingerprint_mismatch" in codes
    )
    assert marker_present_noncreating(tmp_path, run_id)
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["outcome"]["outcome_class"] == OUTCOME_AMBIGUOUS
    assert bundle["outcome"]["outcome_evidence"]["safe_to_retry"] is False


def _post_invoke_observe_wrapper(real_observe, *, on_post_invoke):
    calls = {"n": 0}

    def observe(run_id_arg, *, base_dir=None):
        calls["n"] += 1
        snapshot = real_observe(run_id_arg, base_dir=base_dir)
        if calls["n"] >= 3:
            on_post_invoke(run_id_arg=run_id_arg, base_dir=base_dir)
        return snapshot

    return observe


STALE_ZERO_INVOKE_CASES = [
    pytest.param(
        "revoked",
        lambda tmp, run, comp, issue: revoke_approval(
            issue["approval_id"], "alice", "cancel", base_dir=tmp
        ),
        InvokeStaleApprovalError,
        "revoked",
        id="revoked",
    ),
    pytest.param(
        "expired",
        lambda tmp, run, comp, issue: None,
        InvokeStaleApprovalError,
        None,
        id="expired",
    ),
    pytest.param(
        "pre_claim",
        lambda tmp, run, comp, issue: claim_approval(
            issue["approval_id"], "claim-pre", "bob", base_dir=tmp
        ),
        InvokeStaleApprovalError,
        "already claimed",
        id="pre_claim",
    ),
    pytest.param(
        "pre_outcome",
        lambda tmp, run, comp, issue: (
            claim_approval(issue["approval_id"], "claim-out", "bob", base_dir=tmp),
            record_use_outcome(
                issue["approval_id"], "claim-out", OUTCOME_CONSUMED, base_dir=tmp
            ),
        ),
        InvokeStaleApprovalError,
        "already claimed",
        id="pre_outcome",
    ),
    pytest.param(
        "checkpoint",
        lambda tmp, run, comp, issue: None,
        InvokeStaleApprovalError,
        "project_repository_checkpoint",
        id="checkpoint",
    ),
    pytest.param(
        "dup_record",
        lambda tmp, run, comp, issue: None,
        InvokeStaleApprovalError,
        "run_completion_record",
        id="dup_record",
    ),
    pytest.param(
        "dup_event",
        lambda tmp, run, comp, issue: events.append_task_event(
            run,
            events.make_event(
                event_type="manual_task_status_changed",
                run_id=run,
                task_id=comp["completed_task_ids"][0],
                event_id=next(
                    entry["value"]
                    for entry in issue["bound_arguments"]["argument_entries"]
                    if entry.get("key") == "event_id" and entry.get("presence") == "value"
                ),
                actor="human",
                previous_status="completed",
                new_status="completed",
                payload={},
            ),
            base_dir=tmp,
        ),
        InvokeStaleApprovalError,
        "event_id",
        id="dup_event",
    ),
    pytest.param(
        "bogus_task",
        lambda tmp, run, comp, issue: None,
        InvokeStaleApprovalError,
        "task",
        id="bogus_task",
    ),
    pytest.param(
        "occupied_marker",
        lambda tmp, run, comp, issue: (tmp / LOCKS_DIR_NAME).mkdir(
            parents=True, exist_ok=True
        )
        or (tmp / LOCKS_DIR_NAME / f"{run}.marker").write_text("{}", encoding="utf-8"),
        RunExecutionLockOccupiedError,
        None,
        id="occupied_marker",
    ),
]


@pytest.mark.parametrize("case_id,setup_fn,exc_type,match", STALE_ZERO_INVOKE_CASES)
def test_stale_paths_never_call_complete_run_manually(
    tmp_path, case_id, setup_fn, exc_type, match
):
    if case_id == "checkpoint":
        run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
        issue, _ = _issue_completion_approval(
            tmp_path, run_id, completion, checkpoint="cp-live"
        )
    elif case_id == "dup_record":
        run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
        issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
        from unittest.mock import MagicMock

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        patcher = patch(
            "htr.invoke_run_completion.run_completion_record_json_path",
            return_value=mock_path,
        )
    elif case_id == "bogus_task":
        run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
        bogus = dict(completion)
        bogus["completed_task_ids"] = [new_task_id()]
        issue, _ = _issue_completion_approval(tmp_path, run_id, bogus)
        patcher = None
    elif case_id == "dup_event":
        run_id, task_id, completion = _run_ready_for_completion(tmp_path)
        event_id = new_event_id()
        issue, _ = _issue_completion_approval(
            tmp_path, run_id, completion, event_id=event_id
        )
        patcher = None
    else:
        run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
        issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
        patcher = None

    setup_fn(tmp_path, run_id, completion, issue)

    ctx = patcher if case_id == "dup_record" else nullcontext()
    if case_id == "expired":
        expired_at = approval_control._parse_utc_iso(issue["expires_at"]) + timedelta(
            seconds=1
        )
        expired_patch = patch(
            "htr.invoke_run_completion._utc_now", return_value=expired_at
        )
    else:
        expired_patch = nullcontext()

    with expired_patch, ctx:
        with patch.object(events, "complete_run_manually") as mocked:
            if match:
                with pytest.raises(exc_type, match=match):
                    invoke_approved_run_completion(
                        issue["approval_id"],
                        claim_id=f"claim-stale-{case_id}",
                        base_dir=tmp_path,
                    )
            else:
                with pytest.raises(exc_type):
                    invoke_approved_run_completion(
                        issue["approval_id"],
                        claim_id=f"claim-stale-{case_id}",
                        base_dir=tmp_path,
                    )
            mocked.assert_not_called()


def test_happy_path_complete_run_manually_called_exactly_once(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)
    counter = {"n": 0}
    real = events.complete_run_manually

    def counting_complete(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    with patch("htr.invoke_run_completion.complete_run_manually", side_effect=counting_complete):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-exactly-once",
            base_dir=tmp_path,
        )
    assert counter["n"] == 1


POST_VERIFY_MISMATCH_CASES = [
    "post_snapshot_not_trustworthy",
    "completion_record_missing",
    "completion_record_schema_invalid",
    "completion_record_wrong_run_id",
    "completion_record_wrong_completed_task_ids",
    "completion_record_wrong_reason",
    "completion_record_null_reason_to_value",
    "completion_record_value_reason_to_null",
    "completion_record_wrong_metadata",
    "completion_record_wrong_created_at",
    "completion_record_fingerprint_mismatch",
    "completion_event_missing",
    "completion_event_count_mismatch",
    "multiple_completion_events",
    "event_wrong_event_id",
    "event_wrong_payload_completed_task_ids",
    "json_without_matching_event",
    "event_without_matching_json",
    "event_run_id_mismatch",
    "event_has_task_id",
    "event_actor_mismatch",
    "event_previous_status_mismatch",
    "event_new_status_mismatch",
    "event_completed_task_ids_mismatch",
    "event_record_fingerprint_mismatch",
    "event_record_path_mismatch",
    "event_semantic_fingerprint_mismatch",
    "manifest_not_completed",
    "chain_completion_missing",
    "unexpected_chain_record:run_review_record",
    "unexpected_finalization",
    "task_state_changed",
    "attempt_state_changed",
    "result_state_changed",
    "verification_state_changed",
    "artifact_manifest_changed",
    "post_closure_activity",
    "unrelated_state_changed",
]


def _rewrite_logged_completion_event(
    run_id: str,
    base_dir: Path,
    event_id: str,
    mutator: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    events_path = paths.task_events_path(run_id, base_dir)
    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rewritten: list[str] = []
    for line in lines:
        event = json.loads(line)
        if (
            event.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
            and event.get("event_id") == event_id
        ):
            event = mutator(dict(event))
        rewritten.append(json.dumps(event, ensure_ascii=False))
    events_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


@pytest.mark.parametrize("expected_reason", POST_VERIFY_MISMATCH_CASES)
def test_post_verification_mismatch_reasons_are_ambiguous(tmp_path, expected_reason):
    from htr.action_plan import _chain_records_map as real_chain_map
    from htr.observe import build_run_snapshot as real_observe

    run_id, task_id, completion = _run_ready_for_completion(tmp_path)
    if expected_reason == "completion_record_value_reason_to_null":
        completion = dict(completion)
        completion["reason"] = "original-reason"
    elif expected_reason == "completion_record_null_reason_to_value":
        completion = dict(completion)
        completion["reason"] = None
    elif expected_reason == "completion_record_wrong_reason":
        completion = dict(completion)
        completion["reason"] = "original-reason"
    issue, event_id = _issue_completion_approval(tmp_path, run_id, completion)

    def on_post_invoke(*, run_id_arg, base_dir):
        record_path = run_completion_record_json_path(run_id_arg, base_dir)
        if expected_reason == "completion_record_missing":
            if record_path.is_file():
                record_path.unlink()
        elif expected_reason == "completion_record_schema_invalid":
            invalid = dict(completion)
            invalid["completed_task_ids"] = "not-a-list"
            io.atomic_write_json(record_path, invalid)
        elif expected_reason == "completion_record_semantic_mismatch":
            bad = dict(completion)
            bad["schema_version"] = completion.get("schema_version", 1) + 999
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_wrong_run_id":
            bad = dict(completion)
            bad["run_id"] = new_run_id()
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_wrong_completed_task_ids":
            bad = dict(completion)
            bad["completed_task_ids"] = [new_task_id()]
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_wrong_reason":
            bad = dict(completion)
            bad["reason"] = "tampered-reason"
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_null_reason_to_value":
            bad = dict(completion)
            bad["reason"] = "now-has-value"
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_value_reason_to_null":
            bad = dict(completion)
            bad["reason"] = None
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_wrong_metadata":
            bad = dict(completion)
            bad["metadata"] = {"tampered": True}
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_record_wrong_created_at":
            bad = dict(completion)
            bad["created_at"] = "1970-01-01T00:00:00Z"
            io.atomic_write_json(record_path, bad)
        elif expected_reason == "completion_event_missing":
            events_path = paths.task_events_path(run_id_arg, base_dir)
            events_path.write_text("", encoding="utf-8")
        elif expected_reason == "event_wrong_event_id":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {**ev, "event_id": new_event_id()},
            )
        elif expected_reason == "event_wrong_payload_completed_task_ids":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {
                    **ev,
                    "payload": {
                        **(ev.get("payload") or {}),
                        "completed_task_ids": [new_task_id()],
                    },
                },
            )
        elif expected_reason == "json_without_matching_event":
            events_path = paths.task_events_path(run_id_arg, base_dir)
            events_path.write_text("", encoding="utf-8")
        elif expected_reason == "event_without_matching_json":
            if record_path.is_file():
                record_path.unlink()
        elif expected_reason == "task_state_changed":
            task_status_path = paths.task_status_path(run_id_arg, task_id, base_dir)
            status = io.read_json(task_status_path)
            status["metadata"] = {"tampered": True}
            io.atomic_write_json(task_status_path, status)
        elif expected_reason == "attempt_state_changed":
            attempt_root = paths.run_root(run_id_arg, base_dir) / "tasks" / task_id / "attempts"
            for attempt_dir in sorted(attempt_root.glob("*")):
                attempt_path = paths.attempt_status_path(
                    run_id_arg, task_id, attempt_dir.name, base_dir
                )
                if attempt_path.is_file():
                    status = io.read_json(attempt_path)
                    status["metadata"] = {"tampered": True}
                    io.atomic_write_json(attempt_path, status)
                    break
        elif expected_reason == "result_state_changed":
            task_status_path = paths.task_status_path(run_id_arg, task_id, base_dir)
            status = io.read_json(task_status_path)
            status["metadata"] = {"tampered": True}
            io.atomic_write_json(task_status_path, status)
        elif expected_reason in {"verification_state_changed", "post_closure_activity", "unrelated_state_changed"}:
            task_status_path = paths.task_status_path(run_id_arg, task_id, base_dir)
            status = io.read_json(task_status_path)
            status["metadata"] = {"tampered": True}
            io.atomic_write_json(task_status_path, status)
        elif expected_reason == "artifact_manifest_changed":
            artifact_manifest = paths.run_root(run_id_arg, base_dir) / "artifacts" / "manifest.json"
            artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
            data = io.read_json(artifact_manifest) if artifact_manifest.is_file() else {}
            data["tampered"] = True
            io.atomic_write_json(artifact_manifest, data)
        elif expected_reason == "multiple_completion_events":
            dup = events.make_run_event(
                event_type=EVENT_TYPE_MANUAL_RUN_COMPLETED,
                run_id=run_id_arg,
                actor="human",
                payload={"completed_task_ids": completion["completed_task_ids"]},
                event_id=new_event_id(),
                previous_status="active",
                new_status=RUN_COMPLETED,
            )
            events.append_run_event(run_id_arg, dup, base_dir)
        elif expected_reason == "event_run_id_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {**ev, "run_id": new_run_id()},
            )
        elif expected_reason == "event_has_task_id":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {**ev, "task_id": task_id},
            )
        elif expected_reason == "event_actor_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {**ev, "actor": "wrong-actor"},
            )
        elif expected_reason == "event_previous_status_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {**ev, "previous_status": "bogus"},
            )
        elif expected_reason == "event_new_status_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {**ev, "new_status": "active"},
            )
        elif expected_reason == "event_completed_task_ids_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {
                    **ev,
                    "payload": {
                        **(ev.get("payload") or {}),
                        "completed_task_ids": [new_task_id()],
                    },
                },
            )
        elif expected_reason == "event_record_fingerprint_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {
                    **ev,
                    "payload": {
                        **(ev.get("payload") or {}),
                        "run_completion_fingerprint": "sha256:" + "f" * 64,
                    },
                },
            )
        elif expected_reason == "event_record_path_mismatch":
            _rewrite_logged_completion_event(
                run_id_arg,
                base_dir,
                event_id,
                lambda ev: {
                    **ev,
                    "payload": {
                        **(ev.get("payload") or {}),
                        "run_completion_record_path": "/tmp/wrong/path.json",
                    },
                },
            )

    real_complete = events.complete_run_manually

    def maybe_tampered_complete(*args, **kwargs):
        event = real_complete(*args, **kwargs)
        if expected_reason == "event_semantic_fingerprint_mismatch":
            event = dict(event)
            event["payload"] = dict(event.get("payload") or {})
            event["payload"]["extra"] = "changes returned fingerprint only"
        return event

    if expected_reason == "post_snapshot_not_trustworthy":

        def observe(run_id_arg, *, base_dir=None):
            observe.calls += 1  # type: ignore[attr-defined]
            snap = real_observe(run_id_arg, base_dir=base_dir)
            if observe.calls >= 3:  # type: ignore[attr-defined]
                ds = dict(snap.get("decision_support") or {})
                ds["snapshot_trustworthy"] = False
                snap = dict(snap)
                snap["decision_support"] = ds
            return snap

        observe.calls = 0  # type: ignore[attr-defined]
        tamper = observe
    elif expected_reason == "chain_completion_missing":
        calls = {"n": 0}

        def fake_chain_map(snapshot):
            mapped = real_chain_map(snapshot)
            calls["n"] += 1
            if calls["n"] >= 2:
                mapped = dict(mapped)
                rc = dict(mapped.get("run_completion_record") or {})
                rc["present"] = False
                mapped["run_completion_record"] = rc
            return mapped

        tamper = _post_invoke_observe_wrapper(real_observe, on_post_invoke=on_post_invoke)
        with patch(
            "htr.invoke_run_completion.complete_run_manually",
            side_effect=maybe_tampered_complete,
        ), patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper), patch(
            "htr.invoke_run_completion._chain_records_map",
            side_effect=fake_chain_map,
        ):
            _assert_ambiguous_post_verify(
                tmp_path,
                issue,
                tamper=tamper,
                expected_reason=expected_reason,
                run_id=run_id,
            )
        return
    elif expected_reason in {"unexpected_chain_record:run_review_record", "unexpected_finalization"}:
        chain_calls = {"n": 0}

        def fake_unexpected_chain(snapshot):
            mapped = real_chain_map(snapshot)
            chain_calls["n"] += 1
            if chain_calls["n"] >= 2:
                mapped = dict(mapped)
                review = dict(mapped.get("run_review_record") or {})
                review["present"] = True
                mapped["run_review_record"] = review
            return mapped

        tamper = _post_invoke_observe_wrapper(real_observe, on_post_invoke=lambda **_k: None)
        with patch(
            "htr.invoke_run_completion.complete_run_manually",
            side_effect=maybe_tampered_complete,
        ), patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper), patch(
            "htr.invoke_run_completion._chain_records_map",
            side_effect=fake_unexpected_chain,
        ):
            _assert_ambiguous_post_verify(
                tmp_path,
                issue,
                tamper=tamper,
                expected_reason=expected_reason,
                run_id=run_id,
            )
        return
    elif expected_reason == "completion_record_fingerprint_mismatch":
        real_fp = run_completion_fingerprint
        fp_calls = {"n": 0}

        def biased_fp(record):
            fp_calls["n"] += 1
            fp = real_fp(record)
            if fp_calls["n"] >= 3:
                return fp + "-tampered"
            return fp

        tamper = _post_invoke_observe_wrapper(real_observe, on_post_invoke=on_post_invoke)
        with patch(
            "htr.invoke_run_completion.complete_run_manually",
            side_effect=maybe_tampered_complete,
        ), patch(
            "htr.invoke_run_completion.run_completion_fingerprint",
            side_effect=biased_fp,
        ), patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper):
            _assert_ambiguous_post_verify(
                tmp_path,
                issue,
                tamper=tamper,
                expected_reason=expected_reason,
                run_id=run_id,
            )
        return
    elif expected_reason == "manifest_not_completed":
        real_atomic = io.atomic_write_json
        manifest_path = paths.run_manifest_path(run_id, tmp_path)

        def selective_atomic(path, data):
            if path == manifest_path:
                raise RuntimeError("manifest write blocked for test")
            return real_atomic(path, data)

        tamper = _post_invoke_observe_wrapper(real_observe, on_post_invoke=on_post_invoke)
        with patch("htr.events.atomic_write_json", side_effect=selective_atomic), patch(
            "htr.invoke_run_completion.complete_run_manually",
            side_effect=maybe_tampered_complete,
        ), patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper):
            with pytest.raises(InvokeAmbiguousOutcomeError) as excinfo:
                invoke_approved_run_completion(
                    issue["approval_id"],
                    claim_id="claim-partial-manifest",
                    base_dir=tmp_path,
                )
            assert excinfo.value.reason_code in {
                REASON_POST_VERIFICATION_MISMATCH,
                REASON_INVOKE_RAISED_COMMIT_UNKNOWN,
            }
            assert excinfo.value.safe_to_retry is False
            assert marker_present_noncreating(tmp_path, run_id)
        return
    elif expected_reason == "completion_event_count_mismatch":
        tamper = _post_invoke_observe_wrapper(real_observe, on_post_invoke=on_post_invoke)

        def complete_delete_events(*args, **kwargs):
            event = real_complete(*args, **kwargs)
            events_path = paths.task_events_path(run_id, tmp_path)
            events_path.write_text("", encoding="utf-8")
            return event

        with patch(
            "htr.invoke_run_completion.complete_run_manually",
            side_effect=complete_delete_events,
        ), patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper):
            _assert_ambiguous_post_verify(
                tmp_path,
                issue,
                tamper=tamper,
                expected_reason=expected_reason,
                run_id=run_id,
            )
        return
    else:
        tamper = _post_invoke_observe_wrapper(real_observe, on_post_invoke=on_post_invoke)

    with patch(
        "htr.invoke_run_completion.complete_run_manually",
        side_effect=maybe_tampered_complete,
    ), patch("htr.invoke_run_completion.build_run_snapshot", side_effect=tamper):
        _assert_ambiguous_post_verify(
            tmp_path,
            issue,
            tamper=tamper,
            expected_reason=expected_reason,
            run_id=run_id,
        )


PARTIAL_WRITE_STAGES = [
    pytest.param("before_completion_json", id="before_completion_json"),
    pytest.param("after_json_before_event", id="after_json_before_event"),
    pytest.param("after_event_before_manifest", id="after_event_before_manifest"),
]


@pytest.mark.parametrize("stage", PARTIAL_WRITE_STAGES)
def test_partial_write_matrix_records_ambiguous_and_preserves_marker(tmp_path, stage):
    from htr.invoke_run_completion import (
        REASON_INVOKE_RAISED_COMMIT_UNKNOWN,
        REASON_LIFECYCLE_WRITE_INDETERMINATE,
    )

    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    record_path = run_completion_record_json_path(run_id, tmp_path)
    manifest_path = paths.run_manifest_path(run_id, tmp_path)
    real_atomic = io.atomic_write_json
    real_append = events.append_run_event
    state = {"completion_json": False, "event": False}

    def staged_atomic(path, data):
        if path == record_path and stage == "before_completion_json":
            raise RunExecutionLockIndeterminateError("json write failed")
        if path == record_path:
            state["completion_json"] = True
            return real_atomic(path, data)
        if path == manifest_path and stage == "after_event_before_manifest":
            if state["event"]:
                raise RunExecutionLockIndeterminateError("manifest write failed")
        return real_atomic(path, data)

    def staged_append(run_id_arg, candidate, base_dir=None):
        if stage == "after_json_before_event" and state["completion_json"]:
            raise RunExecutionLockIndeterminateError("event append failed")
        result = real_append(run_id_arg, candidate, base_dir)
        state["event"] = True
        return result

    with patch("htr.events.atomic_write_json", side_effect=staged_atomic), patch(
        "htr.events.append_run_event", side_effect=staged_append
    ):
        with pytest.raises(InvokeAmbiguousOutcomeError) as excinfo:
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id=f"claim-partial-{stage}",
                base_dir=tmp_path,
            )
    err = excinfo.value
    assert err.reason_code in {
        REASON_INVOKE_RAISED_COMMIT_UNKNOWN,
        REASON_LIFECYCLE_WRITE_INDETERMINATE,
        REASON_POST_VERIFICATION_MISMATCH,
    }
    assert err.safe_to_retry is False
    assert marker_present_noncreating(tmp_path, run_id)
    bundle = get_approval(issue["approval_id"], base_dir=tmp_path)
    assert bundle["outcome"]["outcome_class"] == OUTCOME_AMBIGUOUS


def test_spawn_child_cannot_invoke_while_parent_holds_marker(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    hold = threading.Event()
    release = threading.Event()
    child_outcome: dict[str, str] = {}

    def holder() -> None:
        with approval_control._approval_use_session(run_id, tmp_path):
            hold.set()
            release.wait(timeout=15)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    hold.wait(timeout=5)
    script = f"""
from pathlib import Path
from htr.invoke_run_completion import invoke_approved_run_completion
from htr.execution_lock import RunExecutionLockOccupiedError
base = Path({str(tmp_path)!r})
try:
    invoke_approved_run_completion({issue["approval_id"]!r}, claim_id='claim-child', base_dir=base)
    print('ok')
except RunExecutionLockOccupiedError:
    print('occupied')
except Exception as exc:
    print(type(exc).__name__)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    release.set()
    holder_thread.join(timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "occupied"


def test_cross_run_approval_use_session_rejected_via_invoke(tmp_path):
    run_a, _t_a, completion_a = _run_ready_for_completion(tmp_path)
    issue_a, _ = _issue_completion_approval(tmp_path, run_a, completion_a)
    run_b, _t_b, completion_b = _run_ready_for_completion(tmp_path)
    issue_b, _ = _issue_completion_approval(tmp_path, run_b, completion_b)

    with approval_control._approval_use_session(run_a, tmp_path):
        with pytest.raises(InvokeStaleApprovalError, match="cross-key"):
            invoke_approved_run_completion(
                issue_b["approval_id"],
                claim_id="claim-cross-run",
                base_dir=tmp_path,
            )
    bundle = get_approval(issue_b["approval_id"], base_dir=tmp_path)
    assert bundle["claim"] is None


def test_happy_path_marker_held_until_session_exit(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    seen = {"during_invoke": False}
    real = events.complete_run_manually

    def complete_with_marker_check(*args, **kwargs):
        seen["during_invoke"] = marker_present_noncreating(tmp_path, run_id)
        return real(*args, **kwargs)

    with patch(
        "htr.invoke_run_completion.complete_run_manually",
        side_effect=complete_with_marker_check,
    ):
        invoke_approved_run_completion(
            issue["approval_id"],
            claim_id="claim-marker-hold",
            base_dir=tmp_path,
        )
    assert seen["during_invoke"] is True
    assert not marker_present_noncreating(tmp_path, run_id)


def test_preliminary_untrustworthy_snapshot_rejected_via_invoke(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    from htr.observe import build_run_snapshot as real_observe

    def bad_first_snapshot(run_id_arg, *, base_dir=None):
        snap = real_observe(run_id_arg, base_dir=base_dir)
        ds = dict(snap.get("decision_support") or {})
        ds["snapshot_trustworthy"] = False
        snap = dict(snap)
        snap["decision_support"] = ds
        return snap

    with patch("htr.invoke_run_completion.build_run_snapshot", side_effect=bad_first_snapshot):
        with patch.object(events, "complete_run_manually") as mocked:
            with pytest.raises(InvokeStaleApprovalError, match="trustworthy"):
                invoke_approved_run_completion(
                    issue["approval_id"],
                    claim_id="claim-untrustworthy-pre",
                    base_dir=tmp_path,
                )
            mocked.assert_not_called()


def test_preliminary_observation_digest_mismatch_rejected_via_invoke(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    with patch("htr.invoke_run_completion.compute_source_observation_digest", return_value="sha256:" + "0" * 64):
        with patch.object(events, "complete_run_manually") as mocked:
            with pytest.raises(InvokeStaleApprovalError, match="observation digest"):
                invoke_approved_run_completion(
                    issue["approval_id"],
                    claim_id="claim-pre-digest",
                    base_dir=tmp_path,
                )
            mocked.assert_not_called()


def test_issue_digest_mismatch_rejected_via_invoke(tmp_path):
    run_id, _task_id, completion = _run_ready_for_completion(tmp_path)
    issue, _ = _issue_completion_approval(tmp_path, run_id, completion)
    issue_path = paths.approval_issue_path(issue["approval_id"], tmp_path)
    raw = json.loads(issue_path.read_text(encoding="utf-8"))
    raw["executor_id"] = "charlie"
    issue_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    with patch.object(events, "complete_run_manually") as mocked:
        with pytest.raises(InvokeStaleApprovalError, match="digest mismatch"):
            invoke_approved_run_completion(
                issue["approval_id"],
                claim_id="claim-bad-digest",
                base_dir=tmp_path,
            )
        mocked.assert_not_called()
