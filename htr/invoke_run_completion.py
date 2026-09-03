"""Task 25 — human-gated single-API run-completion invoke pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from htr import contracts, events, io, paths, schemas
from htr.action_plan import (
    STATE_PROPOSABLE,
    PlanningIntent,
    build_action_plan,
    compute_source_observation_digest,
    _chain_records_map,
    _first_absent_chain_index,
    _has_chain_gap,
)
from htr.approval_control import (
    APPROVAL_KIND_LIFECYCLE_MUTATION,
    OUTCOME_AMBIGUOUS,
    OUTCOME_CONSUMED,
    _approval_use_session,
    _argument_entries_to_inputs,
    _canonical_json,
    _compute_approval_digest,
    _load_bundle,
    _materialize_bound_arguments,
    _parse_utc_iso,
    _runs_root_path_digest,
    _utc_now,
    _claim_approval_during_session,
    record_use_outcome,
    _record_use_outcome_during_session,
)
from htr.contracts import (
    PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN,
    run_completion_fingerprint,
    run_completion_record_json_path,
)
from htr.events import EVENT_TYPE_MANUAL_RUN_COMPLETED, complete_run_manually, event_exists
from htr.execution_lock import (
    RunExecutionLockDurabilityError,
    RunExecutionLockError,
    RunExecutionLockIndeterminateError,
    RunExecutionLockOccupiedError,
    marker_present_noncreating,
)
from htr.finalization import SealState, evaluate_run_seal
from htr.ids import validate_id
from htr.observe import ObserveInvocationError, build_run_snapshot
from htr.state import (
    ApprovalStateError,
    ApprovalValidationError,
    InvokeAmbiguousOutcomeError,
    InvokeCleanupDurabilityError,
    InvokeOutcomePersistenceError,
    InvokeRunCompletionResult,
    InvokeStaleApprovalError,
    RUN_COMPLETED,
    TASK_COMPLETED,
)

PILOT_BOUND_API = "complete_run_manually"

REASON_VERIFIED_SUCCESS = "verified_success"
REASON_CLAIMED_INVOKE_NOT_STARTED = "claimed_invoke_not_started"
REASON_INVOKE_RAISED_COMMIT_UNKNOWN = "invoke_raised_commit_unknown"
REASON_LIFECYCLE_WRITE_INDETERMINATE = "lifecycle_write_indeterminate"
REASON_POST_OBSERVATION_FAILED = "post_observation_failed"
REASON_POST_VERIFICATION_MISMATCH = "post_verification_mismatch"


@dataclass(frozen=True)
class _InvokeContext:
    approval_id: str
    claim_id: str
    base_dir: Path | None
    issue: dict[str, Any]
    run_id: str
    completion_record: dict[str, Any]
    actor: str
    event_id: str
    record_fingerprint: str
    pre_observation_digest: str
    pre_manifest_status: str
    unrelated_digest: str


def _semantic_event_fingerprint(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if payload is None:
        payload = {}
    return json.dumps(
        {
            "event_type": event.get("event_type"),
            "run_id": event.get("run_id"),
            "task_id": event.get("task_id"),
            "attempt_id": event.get("attempt_id"),
            "previous_status": event.get("previous_status"),
            "new_status": event.get("new_status"),
            "actor": event.get("actor"),
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _trustworthy_snapshot(snapshot: dict[str, Any]) -> bool:
    ds = snapshot.get("decision_support") or {}
    if not ds.get("snapshot_trustworthy"):
        return False
    if ds.get("integrity_fully_clean") is not True:
        return False
    integrity = snapshot.get("integrity") or {}
    if integrity.get("status") != "pass":
        return False
    return True


def _digest_unrelated_state(run_id: str, base_dir: Path | None) -> str:
    root = paths.run_root(run_id, base_dir)
    if not root.exists():
        return _canonical_json({})
    parts: dict[str, Any] = {}
    for task_dir in sorted(root.glob("tasks/*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        status_path = paths.task_status_path(run_id, task_id, base_dir)
        if status_path.is_file():
            parts[f"task_status:{task_id}"] = io.read_json(status_path)
        for attempt_dir in sorted(task_dir.glob("attempts/*")):
            attempt_id = attempt_dir.name
            attempt_status = paths.attempt_status_path(
                run_id, task_id, attempt_id, base_dir
            )
            if attempt_status.is_file():
                parts[f"attempt_status:{attempt_id}"] = io.read_json(attempt_status)
    artifacts = root / "artifacts"
    if artifacts.is_dir():
        manifest = artifacts / "manifest.json"
        if manifest.is_file():
            parts["artifact_manifest"] = io.read_json(manifest)
    return hashlib.sha256(
        _canonical_json(parts).encode("utf-8")
    ).hexdigest()


def _extract_bound_invocation(issue: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    if issue.get("bound_api") != PILOT_BOUND_API:
        raise InvokeStaleApprovalError(
            f"approval bound_api must be {PILOT_BOUND_API!r}",
            approval_id=issue.get("approval_id"),
        )
    inputs = _argument_entries_to_inputs(issue["bound_arguments"])
    record = inputs.get("record")
    actor = inputs.get("actor")
    event_id = inputs.get("event_id")
    if not isinstance(record, dict):
        raise InvokeStaleApprovalError(
            "bound completion record missing from approval",
            approval_id=issue.get("approval_id"),
        )
    if not actor or not isinstance(event_id, str) or not event_id.strip():
        raise InvokeStaleApprovalError(
            "bound actor and event_id required",
            approval_id=issue.get("approval_id"),
        )
    if not validate_id(event_id, "event"):
        raise InvokeStaleApprovalError(
            f"invalid bound event_id: {event_id!r}",
            approval_id=issue.get("approval_id"),
        )
    return record, actor, event_id.strip()


def _verify_issue_digest(issue: dict[str, Any]) -> None:
    body = dict(issue)
    digest = body.pop("approval_digest", None)
    expected = _compute_approval_digest(body)
    if digest != expected:
        raise InvokeStaleApprovalError(
            "approval digest mismatch",
            approval_id=issue.get("approval_id"),
        )


def _build_evidence(
    *,
    reason_code: str,
    error_classification: str,
    bound_api: str,
    event_id: str,
    pre_observation_digest: str,
    post_observation_digest: str | None,
    mutation_may_have_committed: bool,
    verification_reason_codes: list[str],
    observed_record_fingerprint: str | None,
    observed_event_fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "reason_code": reason_code,
        "error_classification": error_classification,
        "bound_api": bound_api,
        "event_id": event_id,
        "pre_observation_digest": pre_observation_digest,
        "post_observation_digest": post_observation_digest,
        "mutation_may_have_committed": mutation_may_have_committed,
        "safe_to_retry": False,
        "verification_reason_codes": list(verification_reason_codes),
        "observed_record_fingerprint": observed_record_fingerprint,
        "observed_event_fingerprint": observed_event_fingerprint,
    }


def _preliminary_validate(
    approval_id: str,
    claim_id: str,
    base_dir: Path | None,
) -> _InvokeContext:
    validate_id(approval_id, "approval")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise InvokeStaleApprovalError("claim_id must be a non-empty string")
    bundle = _load_bundle(approval_id, base_dir)
    issue = bundle["issue"]
    if issue.get("approval_id") != approval_id:
        raise InvokeStaleApprovalError("approval_id mismatch", approval_id=approval_id)
    _verify_issue_digest(issue)
    if issue.get("approval_kind") != APPROVAL_KIND_LIFECYCLE_MUTATION:
        raise InvokeStaleApprovalError(
            "approval kind does not support lifecycle mutation",
            approval_id=approval_id,
        )
    if bundle["revoke"] is not None:
        raise InvokeStaleApprovalError("approval revoked", approval_id=approval_id)
    if bundle["claim"] is not None:
        raise InvokeStaleApprovalError("approval already claimed", approval_id=approval_id)
    if bundle["outcome"] is not None:
        raise InvokeStaleApprovalError("approval already has outcome", approval_id=approval_id)
    if _utc_now() >= _parse_utc_iso(issue["expires_at"]):
        raise InvokeStaleApprovalError("approval expired", approval_id=approval_id)
    if issue.get("project_repository_checkpoint") is not None:
        raise InvokeStaleApprovalError(
            "project_repository_checkpoint must be null for Task 25 pilot",
            approval_id=approval_id,
        )
    if _runs_root_path_digest(base_dir) != issue.get("htr_runs_root_path_digest"):
        raise InvokeStaleApprovalError(
            "runs-root path digest mismatch",
            approval_id=approval_id,
        )
    completion_record, actor, event_id = _extract_bound_invocation(issue)
    run_id = issue["run_id"]
    if marker_present_noncreating(base_dir, run_id):
        raise RunExecutionLockOccupiedError(run_id=run_id)
    snapshot = build_run_snapshot(run_id, base_dir=base_dir)
    if not _trustworthy_snapshot(snapshot):
        raise InvokeStaleApprovalError(
            "preliminary observation not trustworthy",
            approval_id=approval_id,
        )
    pre_digest = compute_source_observation_digest(snapshot)
    if pre_digest != issue["source_observation_digest"]:
        raise InvokeStaleApprovalError(
            "preliminary observation digest mismatch",
            approval_id=approval_id,
        )
    intent = PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs=_argument_entries_to_inputs(issue["bound_arguments"]),
        htr_runs_root=str(base_dir) if base_dir is not None else None,
    )
    plan = build_action_plan(snapshot, intent)
    if plan.get("plan_digest") != issue["plan_digest"]:
        raise InvokeStaleApprovalError("preliminary plan digest mismatch", approval_id=approval_id)
    seal = evaluate_run_seal(run_id, base_dir)
    if seal.state == SealState.FINALIZED_VALID:
        raise InvokeStaleApprovalError("run is finalized", approval_id=approval_id)
    manifest_path = paths.run_manifest_path(run_id, base_dir)
    if not manifest_path.is_file():
        raise InvokeStaleApprovalError("run manifest missing", approval_id=approval_id)
    manifest = io.read_json(manifest_path)
    return _InvokeContext(
        approval_id=approval_id,
        claim_id=claim_id.strip(),
        base_dir=base_dir,
        issue=issue,
        run_id=run_id,
        completion_record=completion_record,
        actor=actor,
        event_id=event_id,
        record_fingerprint=run_completion_fingerprint(completion_record),
        pre_observation_digest=pre_digest,
        pre_manifest_status=manifest["status"],
        unrelated_digest=_digest_unrelated_state(run_id, base_dir),
    )


def _post_marker_validate(ctx: _InvokeContext) -> None:
    bundle = _load_bundle(ctx.approval_id, ctx.base_dir)
    issue = bundle["issue"]
    if issue["approval_digest"] != ctx.issue["approval_digest"]:
        raise InvokeStaleApprovalError("issue changed after marker", approval_id=ctx.approval_id)
    if bundle["revoke"] is not None:
        raise InvokeStaleApprovalError("approval revoked", approval_id=ctx.approval_id)
    if bundle["claim"] is not None:
        raise InvokeStaleApprovalError("approval already claimed", approval_id=ctx.approval_id)
    if bundle["outcome"] is not None:
        raise InvokeStaleApprovalError("approval already has outcome", approval_id=ctx.approval_id)
    if issue.get("project_repository_checkpoint") is not None:
        raise InvokeStaleApprovalError(
            "project_repository_checkpoint must be null",
            approval_id=ctx.approval_id,
        )
    snapshot = build_run_snapshot(ctx.run_id, base_dir=ctx.base_dir)
    if not _trustworthy_snapshot(snapshot):
        raise InvokeStaleApprovalError(
            "post-marker observation not trustworthy",
            approval_id=ctx.approval_id,
        )
    live_digest = compute_source_observation_digest(snapshot)
    if live_digest != ctx.issue["source_observation_digest"]:
        raise InvokeStaleApprovalError(
            "post-marker observation digest mismatch",
            approval_id=ctx.approval_id,
        )
    intent = PlanningIntent(
        requested_action=PILOT_BOUND_API,
        action_inputs=_argument_entries_to_inputs(issue["bound_arguments"]),
        htr_runs_root=str(ctx.base_dir) if ctx.base_dir is not None else None,
    )
    plan = build_action_plan(snapshot, intent)
    if plan.get("plan_state") != STATE_PROPOSABLE:
        raise InvokeStaleApprovalError("plan not proposable", approval_id=ctx.approval_id)
    if plan.get("plan_digest") != ctx.issue["plan_digest"]:
        raise InvokeStaleApprovalError("post-marker plan digest mismatch", approval_id=ctx.approval_id)
    bound = _materialize_bound_arguments(plan, intent=intent, base_dir=ctx.base_dir)
    if bound["bound_api"] != PILOT_BOUND_API:
        raise InvokeStaleApprovalError("rebuilt API mismatch", approval_id=ctx.approval_id)
    if _canonical_json(bound["bound_arguments"]) != _canonical_json(issue["bound_arguments"]):
        raise InvokeStaleApprovalError("rebuilt arguments mismatch", approval_id=ctx.approval_id)
    if (plan.get("risk") or {}).get("class") != issue.get("risk_class"):
        raise InvokeStaleApprovalError("risk class mismatch", approval_id=ctx.approval_id)
    seal = evaluate_run_seal(ctx.run_id, base_dir=ctx.base_dir)
    if seal.state == SealState.FINALIZED_VALID:
        raise InvokeStaleApprovalError("run finalized", approval_id=ctx.approval_id)
    run_root = paths.run_root(ctx.run_id, ctx.base_dir)
    if not run_root.is_dir():
        raise InvokeStaleApprovalError("run workspace missing", approval_id=ctx.approval_id)
    manifest_path = paths.run_manifest_path(ctx.run_id, ctx.base_dir)
    if not manifest_path.is_file():
        raise InvokeStaleApprovalError("run manifest missing", approval_id=ctx.approval_id)
    manifest = io.read_json(manifest_path)
    schemas.validate(manifest, "run_manifest")
    if manifest["status"] == RUN_COMPLETED:
        raise InvokeStaleApprovalError("run already completed", approval_id=ctx.approval_id)
    if run_completion_record_json_path(ctx.run_id, ctx.base_dir).exists():
        raise InvokeStaleApprovalError(
            "run_completion_record already exists",
            approval_id=ctx.approval_id,
        )
    if event_exists(ctx.run_id, ctx.event_id, ctx.base_dir):
        raise InvokeStaleApprovalError(
            f"event_id {ctx.event_id!r} already exists",
            approval_id=ctx.approval_id,
        )
    if _has_chain_gap(snapshot):
        raise InvokeStaleApprovalError("chain gap detected", approval_id=ctx.approval_id)
    chain_map = _chain_records_map(snapshot)
    if chain_map.get("run_completion_record", {}).get("present"):
        raise InvokeStaleApprovalError(
            "completion record present in observation",
            approval_id=ctx.approval_id,
        )
    if _first_absent_chain_index(snapshot) != 0:
        raise InvokeStaleApprovalError(
            "first absent chain slot is not run_completion_record",
            approval_id=ctx.approval_id,
        )
    for record_type in PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN[1:]:
        if chain_map.get(record_type, {}).get("present"):
            raise InvokeStaleApprovalError(
                f"later chain record {record_type!r} present",
                approval_id=ctx.approval_id,
            )
    schemas.validate(ctx.completion_record, "run_completion_record")
    if ctx.completion_record["run_id"] != ctx.run_id:
        raise InvokeStaleApprovalError("record run_id mismatch", approval_id=ctx.approval_id)
    for task_id in ctx.completion_record["completed_task_ids"]:
        task_status_path = paths.task_status_path(ctx.run_id, task_id, ctx.base_dir)
        if not task_status_path.is_file():
            raise InvokeStaleApprovalError(
                f"task {task_id!r} status missing",
                approval_id=ctx.approval_id,
            )
        task_status = io.read_json(task_status_path)
        if task_status["status"] != TASK_COMPLETED:
            raise InvokeStaleApprovalError(
                f"task {task_id!r} not completed",
                approval_id=ctx.approval_id,
            )
    if run_completion_fingerprint(ctx.completion_record) != ctx.record_fingerprint:
        raise InvokeStaleApprovalError(
            "completion record fingerprint unstable",
            approval_id=ctx.approval_id,
        )


def _verify_post_invoke(
    ctx: _InvokeContext,
    *,
    event: dict[str, Any],
    post_snapshot: dict[str, Any],
    post_digest: str,
) -> list[str]:
    reasons: list[str] = []
    if not _trustworthy_snapshot(post_snapshot):
        reasons.append("post_snapshot_not_trustworthy")
    record_path = run_completion_record_json_path(ctx.run_id, ctx.base_dir)
    if not record_path.is_file():
        reasons.append("completion_record_missing")
        return reasons
    on_disk = io.read_json(record_path)
    record_schema_valid = True
    try:
        schemas.validate(on_disk, "run_completion_record")
    except ValueError:
        reasons.append("completion_record_schema_invalid")
        record_schema_valid = False
    if on_disk != ctx.completion_record:
        expected = ctx.completion_record
        if on_disk.get("run_id") != expected.get("run_id"):
            reasons.append("completion_record_wrong_run_id")
        elif on_disk.get("completed_task_ids") != expected.get("completed_task_ids"):
            reasons.append("completion_record_wrong_completed_task_ids")
        elif on_disk.get("reason") != expected.get("reason"):
            if expected.get("reason") is None and on_disk.get("reason") is not None:
                reasons.append("completion_record_null_reason_to_value")
            elif expected.get("reason") is not None and on_disk.get("reason") is None:
                reasons.append("completion_record_value_reason_to_null")
            else:
                reasons.append("completion_record_wrong_reason")
        elif on_disk.get("metadata") != expected.get("metadata"):
            reasons.append("completion_record_wrong_metadata")
        elif on_disk.get("created_at") != expected.get("created_at"):
            reasons.append("completion_record_wrong_created_at")
        else:
            reasons.append("completion_record_semantic_mismatch")
    if record_schema_valid and run_completion_fingerprint(on_disk) != ctx.record_fingerprint:
        reasons.append("completion_record_fingerprint_mismatch")
    matching = [
        ev
        for ev in events.read_task_events(ctx.run_id, base_dir=ctx.base_dir)
        if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
        and ev.get("event_id") == ctx.event_id
    ]
    if len(matching) != 1:
        reasons.append("completion_event_count_mismatch")
    else:
        ev = matching[0]
        if ev.get("run_id") != ctx.run_id:
            reasons.append("event_run_id_mismatch")
        if "task_id" in ev:
            reasons.append("event_has_task_id")
        if ev.get("actor") != ctx.actor:
            reasons.append("event_actor_mismatch")
        if ev.get("previous_status") != ctx.pre_manifest_status:
            reasons.append("event_previous_status_mismatch")
        if ev.get("new_status") != RUN_COMPLETED:
            reasons.append("event_new_status_mismatch")
        payload = ev.get("payload") or {}
        if payload.get("completed_task_ids") != ctx.completion_record["completed_task_ids"]:
            reasons.append("event_completed_task_ids_mismatch")
        if payload.get("run_completion_fingerprint") != ctx.record_fingerprint:
            reasons.append("event_record_fingerprint_mismatch")
        if payload.get("run_completion_record_path") != str(record_path):
            reasons.append("event_record_path_mismatch")
        if _semantic_event_fingerprint(ev) != _semantic_event_fingerprint(event):
            reasons.append("event_semantic_fingerprint_mismatch")
    manifest = io.read_json(paths.run_manifest_path(ctx.run_id, ctx.base_dir))
    if manifest["status"] != RUN_COMPLETED:
        reasons.append("manifest_not_completed")
    chain_map = _chain_records_map(post_snapshot)
    if not chain_map.get("run_completion_record", {}).get("present"):
        reasons.append("chain_completion_missing")
    for record_type in PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN[1:]:
        if chain_map.get(record_type, {}).get("present"):
            reasons.append(f"unexpected_chain_record:{record_type}")
    if _digest_unrelated_state(ctx.run_id, ctx.base_dir) != ctx.unrelated_digest:
        reasons.append("unrelated_state_changed")
    completion_events = [
        ev
        for ev in events.read_task_events(ctx.run_id, base_dir=ctx.base_dir)
        if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
    ]
    if len(completion_events) != 1:
        reasons.append("multiple_completion_events")
    return reasons


def _persist_ambiguous_and_raise(
    ctx: _InvokeContext,
    *,
    reason_code: str,
    error_classification: str,
    mutation_may_have_committed: bool,
    verification_reason_codes: list[str],
    post_digest: str | None,
    observed_record_fingerprint: str | None,
    observed_event_fingerprint: str | None,
    preserve_marker: bool,
    during_session: bool,
    message: str,
    exc: BaseException | None = None,
) -> None:
    evidence = _build_evidence(
        reason_code=reason_code,
        error_classification=error_classification,
        bound_api=PILOT_BOUND_API,
        event_id=ctx.event_id,
        pre_observation_digest=ctx.pre_observation_digest,
        post_observation_digest=post_digest,
        mutation_may_have_committed=mutation_may_have_committed,
        verification_reason_codes=verification_reason_codes,
        observed_record_fingerprint=observed_record_fingerprint,
        observed_event_fingerprint=observed_event_fingerprint,
    )
    try:
        writer = _record_use_outcome_during_session if during_session else record_use_outcome
        writer(
            ctx.approval_id,
            ctx.claim_id,
            OUTCOME_AMBIGUOUS,
            outcome_evidence=evidence,
            base_dir=ctx.base_dir,
        )
    except Exception as persist_exc:
        raise InvokeOutcomePersistenceError(
            f"failed to persist ambiguous outcome: {persist_exc}",
            approval_id=ctx.approval_id,
            claim_id=ctx.claim_id,
            reason_code="outcome_persistence_failed",
            outcome_evidence=evidence,
        ) from persist_exc
    err = InvokeAmbiguousOutcomeError(
        message,
        approval_id=ctx.approval_id,
        claim_id=ctx.claim_id,
        reason_code=reason_code,
        mutation_may_have_committed=mutation_may_have_committed,
        outcome_evidence=evidence,
    )
    if preserve_marker:
        raise err from exc
    raise err from exc


def invoke_approved_run_completion(
    approval_id: str,
    *,
    claim_id: str,
    base_dir: Path | None = None,
) -> InvokeRunCompletionResult:
    """Human-gated invoke of approved ``complete_run_manually`` only (Task 25 pilot)."""
    ctx = _preliminary_validate(approval_id, claim_id, base_dir)
    claim_durable = False
    invoke_attempted = False
    pending_ambiguous: InvokeAmbiguousOutcomeError | None = None
    outcome_persistence_error: InvokeOutcomePersistenceError | None = None

    try:
        with _approval_use_session(ctx.run_id, ctx.base_dir):
            try:
                _post_marker_validate(ctx)
                _claim_approval_during_session(
                    ctx.approval_id,
                    ctx.claim_id,
                    base_dir=ctx.base_dir,
                )
                claim_durable = True

                invoke_attempted = True
                event = complete_run_manually(
                    ctx.run_id,
                    ctx.completion_record,
                    actor=ctx.actor,
                    event_id=ctx.event_id,
                    base_dir=ctx.base_dir,
                )

                try:
                    post_snapshot = build_run_snapshot(ctx.run_id, base_dir=ctx.base_dir)
                except ObserveInvocationError as exc:
                    _persist_ambiguous_and_raise(
                        ctx,
                        reason_code=REASON_POST_OBSERVATION_FAILED,
                        error_classification=REASON_POST_OBSERVATION_FAILED,
                        mutation_may_have_committed=True,
                        verification_reason_codes=["post_observe_failed"],
                        post_digest=None,
                        observed_record_fingerprint=None,
                        observed_event_fingerprint=_semantic_event_fingerprint(event),
                        preserve_marker=True,
                        during_session=True,
                        message=str(exc),
                        exc=exc,
                    )
                    raise  # pragma: no cover

                post_digest = compute_source_observation_digest(post_snapshot)
                verify_reasons = _verify_post_invoke(
                    ctx, event=event, post_snapshot=post_snapshot, post_digest=post_digest
                )
                if verify_reasons:
                    _persist_ambiguous_and_raise(
                        ctx,
                        reason_code=REASON_POST_VERIFICATION_MISMATCH,
                        error_classification=REASON_POST_VERIFICATION_MISMATCH,
                        mutation_may_have_committed=True,
                        verification_reason_codes=verify_reasons,
                        post_digest=post_digest,
                        observed_record_fingerprint=ctx.record_fingerprint,
                        observed_event_fingerprint=_semantic_event_fingerprint(event),
                        preserve_marker=True,
                        during_session=True,
                        message="post-invoke verification failed",
                    )

                evidence = _build_evidence(
                    reason_code=REASON_VERIFIED_SUCCESS,
                    error_classification=REASON_VERIFIED_SUCCESS,
                    bound_api=PILOT_BOUND_API,
                    event_id=ctx.event_id,
                    pre_observation_digest=ctx.pre_observation_digest,
                    post_observation_digest=post_digest,
                    mutation_may_have_committed=True,
                    verification_reason_codes=[],
                    observed_record_fingerprint=ctx.record_fingerprint,
                    observed_event_fingerprint=_semantic_event_fingerprint(event),
                )
                try:
                    outcome = _record_use_outcome_during_session(
                        ctx.approval_id,
                        ctx.claim_id,
                        OUTCOME_CONSUMED,
                        outcome_evidence=evidence,
                        base_dir=ctx.base_dir,
                    )
                except Exception as persist_exc:
                    raise InvokeOutcomePersistenceError(
                        f"failed to persist consumed outcome: {persist_exc}",
                        approval_id=ctx.approval_id,
                        claim_id=ctx.claim_id,
                        reason_code="outcome_persistence_failed",
                    ) from persist_exc
                outcome_digest = outcome.get("outcome_digest") or outcome.get("claim_digest", "")
                return InvokeRunCompletionResult(
                    approval_id=ctx.approval_id,
                    claim_id=ctx.claim_id,
                    run_id=ctx.run_id,
                    event_id=ctx.event_id,
                    completion_record_fingerprint=ctx.record_fingerprint,
                    event_semantic_fingerprint=_semantic_event_fingerprint(event),
                    pre_observation_digest=ctx.pre_observation_digest,
                    post_observation_digest=post_digest,
                    outcome_digest=outcome_digest,
                )
            except (
                InvokeAmbiguousOutcomeError,
                InvokeStaleApprovalError,
            ):
                raise
            except InvokeOutcomePersistenceError as exc:
                outcome_persistence_error = exc
                raise
            except Exception as exc:
                bundle_now = _load_bundle(ctx.approval_id, ctx.base_dir)
                has_claim = bundle_now.get("claim") is not None
                if has_claim and not invoke_attempted:
                    evidence = _build_evidence(
                        reason_code=REASON_CLAIMED_INVOKE_NOT_STARTED,
                        error_classification=REASON_CLAIMED_INVOKE_NOT_STARTED,
                        bound_api=PILOT_BOUND_API,
                        event_id=ctx.event_id,
                        pre_observation_digest=ctx.pre_observation_digest,
                        post_observation_digest=None,
                        mutation_may_have_committed=False,
                        verification_reason_codes=[type(exc).__name__],
                        observed_record_fingerprint=None,
                        observed_event_fingerprint=None,
                    )
                    try:
                        _record_use_outcome_during_session(
                            ctx.approval_id,
                            ctx.claim_id,
                            OUTCOME_AMBIGUOUS,
                            outcome_evidence=evidence,
                            base_dir=ctx.base_dir,
                        )
                    except Exception as persist_exc:
                        outcome_persistence_error = InvokeOutcomePersistenceError(
                            f"failed to persist ambiguous outcome: {persist_exc}",
                            approval_id=ctx.approval_id,
                            claim_id=ctx.claim_id,
                            reason_code="outcome_persistence_failed",
                            outcome_evidence=evidence,
                        )
                        raise outcome_persistence_error from persist_exc
                    pending_ambiguous = InvokeAmbiguousOutcomeError(
                        str(exc),
                        approval_id=ctx.approval_id,
                        claim_id=ctx.claim_id,
                        reason_code=REASON_CLAIMED_INVOKE_NOT_STARTED,
                        mutation_may_have_committed=False,
                        outcome_evidence=evidence,
                    )
                elif has_claim and invoke_attempted:
                    reason = REASON_INVOKE_RAISED_COMMIT_UNKNOWN
                    if isinstance(exc, RunExecutionLockError):
                        reason = REASON_LIFECYCLE_WRITE_INDETERMINATE
                    _persist_ambiguous_and_raise(
                        ctx,
                        reason_code=reason,
                        error_classification=reason,
                        mutation_may_have_committed=True,
                        verification_reason_codes=[type(exc).__name__],
                        post_digest=None,
                        observed_record_fingerprint=None,
                        observed_event_fingerprint=None,
                        preserve_marker=True,
                        during_session=True,
                        message=str(exc),
                        exc=exc,
                    )
                else:
                    raise

        if pending_ambiguous is not None:
            raise pending_ambiguous

    except InvokeAmbiguousOutcomeError:
        raise
    except InvokeOutcomePersistenceError:
        raise
    except InvokeStaleApprovalError:
        raise
    except RunExecutionLockOccupiedError:
        raise
    except RunExecutionLockDurabilityError as exc:
        bundle = _load_bundle(ctx.approval_id, ctx.base_dir)
        if bundle.get("outcome") and bundle["outcome"].get("outcome_class") == OUTCOME_CONSUMED:
            raise InvokeCleanupDurabilityError(
                str(exc),
                approval_id=ctx.approval_id,
                claim_id=ctx.claim_id,
                mutation_may_have_committed=True,
            ) from exc
        if claim_durable and invoke_attempted:
            _persist_ambiguous_and_raise(
                ctx,
                reason_code=REASON_LIFECYCLE_WRITE_INDETERMINATE,
                error_classification=REASON_LIFECYCLE_WRITE_INDETERMINATE,
                mutation_may_have_committed=True,
                verification_reason_codes=["lifecycle_lock_durability"],
                post_digest=None,
                observed_record_fingerprint=None,
                observed_event_fingerprint=None,
                preserve_marker=True,
                during_session=False,
                message=str(exc),
                exc=exc,
            )
        raise InvokeStaleApprovalError(str(exc), approval_id=ctx.approval_id) from exc
    except RunExecutionLockIndeterminateError as exc:
        if outcome_persistence_error is not None:
            raise outcome_persistence_error from exc
        if claim_durable and invoke_attempted:
            _persist_ambiguous_and_raise(
                ctx,
                reason_code=REASON_LIFECYCLE_WRITE_INDETERMINATE,
                error_classification=REASON_LIFECYCLE_WRITE_INDETERMINATE,
                mutation_may_have_committed=True,
                verification_reason_codes=["lifecycle_lock_indeterminate"],
                post_digest=None,
                observed_record_fingerprint=None,
                observed_event_fingerprint=None,
                preserve_marker=True,
                during_session=False,
                message=str(exc),
                exc=exc,
            )
        raise InvokeStaleApprovalError(str(exc), approval_id=ctx.approval_id) from exc
    except InvokeOutcomePersistenceError:
        raise
    except Exception as exc:
        if pending_ambiguous is not None:
            raise pending_ambiguous
        raise InvokeStaleApprovalError(str(exc), approval_id=ctx.approval_id) from exc
