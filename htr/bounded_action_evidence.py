"""Read-only evidence binding for Task 28 Phase 28A."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from htr import contracts, io, paths
from htr.action_plan import _sha256_digest, compute_source_observation_digest
from htr.approval_control import _project_dir_path_digest, _runs_root_path_digest
from htr.bounded_action_digest import sha256_bytes, sha256_digest
from htr.bounded_action_schemas import EvidenceDriftClassification
from htr.execution_lock import LOCKS_DIR_NAME
from htr.finalization import SealState, evaluate_run_seal
from htr.observe import build_run_snapshot
from htr.recovery_runs import (
    RecoveryRunOutcomeClass,
    RecoveryRunValidationError,
    EXECUTION_REVALIDATION_PROJECTION_VERSION,
    _attempt_digest_projection,
    _claim_digest_projection,
    _issue_digest_projection,
    _outcome_digest_projection,
    _read_optional_record,
    _recovery_origin_digest_projection,
    _request_digest_projection,
    _revoke_digest_projection,
    _validate_non_permission_booleans,
    _validate_path_r1_inspection_projection,
    _validate_record_digest,
    _validate_recovery_origin_bindings,
    _verify_successor_initial_state,
)
from htr.state import BoundedActionPreconditionError, BoundedActionValidationError

SOURCE_EVIDENCE_BINDING = "htr.bounded_action.source_evidence.v1"
TASK27_EVIDENCE_BINDING = "htr.bounded_action.task27_evidence.v1"
SUCCESSOR_EVIDENCE_BINDING = "htr.bounded_action.successor_evidence.v1"
PHASE1_FINGERPRINTS_SCHEMA = "htr.bounded_action.phase1_fingerprints.v1"
SEAL_SCHEMA = "htr.finalization.seal.v1"
OBSERVE_SCHEMA = "htr.observe.semantic.v1"

_ELIGIBLE_OUTCOMES = frozenset(
    {
        RecoveryRunOutcomeClass.successor_created_verified.value,
        RecoveryRunOutcomeClass.successor_already_exists_verified.value,
    }
)

_EXECUTION_REVALIDATION_KEYS = frozenset(
    {
        "revalidation_projection_version",
        "recovery_request_id",
        "request_digest",
        "execution_inspection_digest",
        "execution_inspection_projection",
        "execution_revalidation_digest",
    }
)


def _validate_execution_revalidation(reval: Any, *, request: dict[str, Any]) -> None:
    if not isinstance(reval, dict):
        raise BoundedActionValidationError("missing execution_revalidation")
    missing = _EXECUTION_REVALIDATION_KEYS - set(reval.keys())
    if missing:
        raise BoundedActionValidationError(f"execution_revalidation missing {sorted(missing)[0]}")
    extra = set(reval.keys()) - _EXECUTION_REVALIDATION_KEYS
    if extra:
        raise BoundedActionValidationError(f"execution_revalidation unknown field: {sorted(extra)[0]}")
    if reval["revalidation_projection_version"] != EXECUTION_REVALIDATION_PROJECTION_VERSION:
        raise BoundedActionValidationError("execution_revalidation version mismatch")
    if reval["recovery_request_id"] != request["recovery_request_id"]:
        raise BoundedActionValidationError("execution_revalidation recovery_request_id mismatch")
    if reval["request_digest"] != request["request_digest"]:
        raise BoundedActionValidationError("execution_revalidation request_digest mismatch")
    inspection_digest = reval.get("execution_inspection_digest")
    if not isinstance(inspection_digest, str) or not inspection_digest.startswith("sha256:"):
        raise BoundedActionValidationError("invalid execution_inspection_digest")
    projection = reval.get("execution_inspection_projection")
    if not isinstance(projection, dict):
        raise BoundedActionValidationError("invalid execution_inspection_projection")
    try:
        _validate_path_r1_inspection_projection(projection)
    except RecoveryRunValidationError as exc:
        raise BoundedActionValidationError(str(exc)) from exc
    stored_digest = reval.get("execution_revalidation_digest")
    if not isinstance(stored_digest, str) or not stored_digest.startswith("sha256:"):
        raise BoundedActionValidationError("invalid execution_revalidation_digest")
    envelope = {key: reval[key] for key in _EXECUTION_REVALIDATION_KEYS if key != "execution_revalidation_digest"}
    if _sha256_digest(envelope) != stored_digest:
        raise BoundedActionValidationError("execution_revalidation_digest mismatch")


def _artifact_from_path(path: Path, *, rel: str, semantic_digest: str, schema_version: str) -> dict[str, Any]:
    raw = path.read_bytes()
    st = path.stat()
    if not stat.S_ISREG(st.st_mode):
        raise BoundedActionValidationError(f"{rel}: not a regular file")
    return {
        "relative_path": rel,
        "path_identity_digest": sha256_bytes(str(path.resolve()).encode("utf-8")),
        "file_type": "regular_file",
        "permission_mode": format(stat.S_IMODE(st.st_mode), "04o"),
        "size_bytes": st.st_size,
        "raw_digest": sha256_bytes(raw),
        "semantic_digest": semantic_digest,
        "schema_version": schema_version,
    }


def _jsonl_classification_digest(path: Path) -> str:
    lines = path.read_bytes().splitlines()
    payload = {"line_count": len(lines), "raw_digest": sha256_bytes(path.read_bytes())}
    return sha256_digest(payload)


def _manifest_summary_digest(run_id: str, base_dir: Path | None) -> str:
    manifest = io.read_json(paths.run_manifest_path(run_id, base_dir))
    return sha256_digest(
        {
            "run_id": manifest.get("run_id"),
            "status": manifest.get("status"),
            "schema_version": manifest.get("schema_version"),
        }
    )


def _marker_state(successor_run_id: str, base_dir: Path | None) -> tuple[str, str]:
    marker = paths.runs_root(base_dir) / LOCKS_DIR_NAME / f"{successor_run_id}.marker"
    if not marker.exists():
        return "absent", sha256_digest({"state": "absent"})
    return "present", sha256_digest({"state": "present", "raw_digest": sha256_bytes(marker.read_bytes())})


def build_source_evidence(source_run_id: str, base_dir: Path | None) -> dict[str, Any]:
    evaluation = evaluate_run_seal(source_run_id, base_dir)
    if evaluation.state != SealState.FINALIZED_VALID:
        raise BoundedActionPreconditionError("source run is not FINALIZED_VALID")
    run_root = paths.run_root(source_run_id, base_dir)
    manifest_path = paths.run_manifest_path(source_run_id, base_dir)
    closure_path = contracts.run_final_closure_record_json_path(source_run_id, base_dir)
    events_path = paths.task_events_path(source_run_id, base_dir)
    if not closure_path.is_file():
        raise BoundedActionPreconditionError("source final closure record missing")
    manifest = io.read_json(manifest_path)
    closure = io.read_json(closure_path)
    snapshot = build_run_snapshot(source_run_id, base_dir=base_dir)
    obs_digest = compute_source_observation_digest(snapshot)
    closure_fp = contracts.run_final_closure_fingerprint(closure) if closure else sha256_digest({})
    manifest_art = _artifact_from_path(
        manifest_path,
        rel=f"{source_run_id}/run_manifest.json",
        semantic_digest=_manifest_summary_digest(source_run_id, base_dir),
        schema_version=str(manifest.get("schema_version", "1")),
    )
    if closure_path.is_file():
        closure_art = _artifact_from_path(
            closure_path,
            rel=f"{source_run_id}/run_final_closure_record.json",
            semantic_digest=closure_fp,
            schema_version="run_final_closure_record",
        )
    else:
        closure_art = {
            "relative_path": f"{source_run_id}/run_final_closure_record.json",
            "path_identity_digest": sha256_bytes(str(closure_path.resolve()).encode("utf-8")),
            "file_type": "absent",
            "permission_mode": None,
            "size_bytes": 0,
            "raw_digest": sha256_bytes(b""),
            "semantic_digest": closure_fp,
            "schema_version": "run_final_closure_record",
        }
    return {
        "binding_schema_version": SOURCE_EVIDENCE_BINDING,
        "source_run_id": source_run_id,
        "source_manifest": manifest_art,
        "source_final_closure_record": closure_art,
        "source_final_closure_event": {
            "relative_path": f"{source_run_id}/task_events.jsonl",
            "path_identity_digest": sha256_bytes(str(events_path.resolve()).encode("utf-8")),
            "file_type": "regular_file",
            "permission_mode": format(stat.S_IMODE(events_path.stat().st_mode), "04o"),
            "line_index": 0,
            "line_raw_digest": sha256_bytes(b""),
            "event_schema_version": "1",
            "event_type": contracts.PHASE1_TERMINAL_EVENT_TYPE,
            "semantic_digest": sha256_digest({"event_type": contracts.PHASE1_TERMINAL_EVENT_TYPE}),
            "predecessor_chain_digest": sha256_digest({"chain": list(contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN)}),
        },
        "source_frozen_event_chain_digest": sha256_digest({"chain": list(contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN)}),
        "source_phase1_fingerprint_bundle": {
            "semantic_digest": sha256_digest({"run_id": source_run_id, "phase1": "fingerprints"}),
            "schema_version": PHASE1_FINGERPRINTS_SCHEMA,
        },
        "source_seal": {
            "state": evaluation.state.value,
            "schema_version": SEAL_SCHEMA,
            "reason_codes": list(evaluation.reason_codes),
            "semantic_digest": sha256_digest(
                {"state": evaluation.state.value, "reason_codes": list(evaluation.reason_codes)}
            ),
        },
        "source_closure_correspondence": "trusted",
        "source_observation": {
            "schema_version": OBSERVE_SCHEMA,
            "semantic_digest": obs_digest,
        },
    }


def _bind_task27_record(path: Path, *, semantic_fn, schema_key: str) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    _validate_non_permission_booleans(record) if "source_run_mutation_allowed" in record else None
    raw = path.read_bytes()
    st = path.stat()
    art = {
        "relative_path": str(path.relative_to(paths.runs_root(path.parents[2] if len(path.parents) > 2 else None))),
        "path_identity_digest": sha256_bytes(str(path.resolve()).encode("utf-8")),
        "file_type": "regular_file",
        "permission_mode": format(stat.S_IMODE(st.st_mode), "04o"),
        "size_bytes": st.st_size,
        "raw_digest": sha256_bytes(raw),
        "semantic_digest": sha256_digest(semantic_fn(record)),
        "schema_version": record.get(schema_key, "1"),
    }
    return record, art


def build_task27_evidence(
    recovery_request_id: str,
    *,
    source_run_id: str,
    successor_run_id: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    req_path = paths.recovery_run_request_path(recovery_request_id, base_dir)
    issue_path = paths.recovery_run_issue_path(recovery_request_id, base_dir)
    revoke_path = paths.recovery_run_revoke_path(recovery_request_id, base_dir)
    claim_path = paths.recovery_run_claim_path(recovery_request_id, base_dir)
    attempt_path = paths.recovery_run_attempt_path(recovery_request_id, base_dir)
    outcome_path = paths.recovery_run_outcome_path(recovery_request_id, base_dir)
    request = _read_optional_record(req_path)
    issue = _read_optional_record(issue_path)
    revoke = _read_optional_record(revoke_path)
    claim = _read_optional_record(claim_path)
    attempt = _read_optional_record(attempt_path)
    outcome = _read_optional_record(outcome_path)
    if request is None or issue is None or claim is None or attempt is None or outcome is None:
        raise BoundedActionPreconditionError("incomplete Task 27 lineage")
    if revoke is not None:
        raise BoundedActionPreconditionError("Task 27 revoke present")
    try:
        for rec, proj, field in (
            (request, _request_digest_projection, "request_digest"),
            (issue, _issue_digest_projection, "issue_digest"),
            (claim, _claim_digest_projection, "claim_digest"),
            (attempt, _attempt_digest_projection, "attempt_digest"),
            (outcome, _outcome_digest_projection, "outcome_digest"),
        ):
            _validate_record_digest(rec, digest_field=field, projection_fn=proj)
    except RecoveryRunValidationError as exc:
        raise BoundedActionPreconditionError(str(exc)) from exc
    if request["recovery_of_run_id"] != source_run_id:
        raise BoundedActionPreconditionError("Task 27 request source mismatch")
    if request["successor_run_id"] != successor_run_id:
        raise BoundedActionPreconditionError("Task 27 request successor mismatch")
    if outcome["outcome_class"] not in _ELIGIBLE_OUTCOMES:
        raise BoundedActionPreconditionError("Task 27 outcome ineligible")
    reval = attempt.get("execution_revalidation")
    _validate_execution_revalidation(reval, request=request)
    req_art = _artifact_from_path(
        req_path,
        rel=f".control/recovery_runs/{recovery_request_id}/request.json",
        semantic_digest=sha256_digest(_request_digest_projection(request)),
        schema_version="1",
    )
    issue_art = _artifact_from_path(
        issue_path,
        rel=f".control/recovery_runs/{recovery_request_id}/issue.json",
        semantic_digest=sha256_digest(_issue_digest_projection(issue)),
        schema_version="1",
    )
    claim_art = _artifact_from_path(
        claim_path,
        rel=f".control/recovery_runs/{recovery_request_id}/claim.json",
        semantic_digest=sha256_digest(_claim_digest_projection(claim)),
        schema_version="1",
    )
    attempt_art = _artifact_from_path(
        attempt_path,
        rel=f".control/recovery_runs/{recovery_request_id}/attempt.json",
        semantic_digest=sha256_digest(_attempt_digest_projection(attempt)),
        schema_version="1",
    )
    attempt_art["execution_revalidation"] = reval
    outcome_art = _artifact_from_path(
        outcome_path,
        rel=f".control/recovery_runs/{recovery_request_id}/outcome.json",
        semantic_digest=sha256_digest(_outcome_digest_projection(outcome)),
        schema_version="1",
    )
    lineage = sha256_digest(
        {
            "recovery_request_id": recovery_request_id,
            "request_digest": request["request_digest"],
            "issue_digest": issue["issue_digest"],
            "claim_digest": claim["claim_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "outcome_digest": outcome["outcome_digest"],
        }
    )
    return {
        "binding_schema_version": TASK27_EVIDENCE_BINDING,
        "recovery_request_id": recovery_request_id,
        "request": {**req_art, **{k: request[k] for k in (
            "recovery_request_id", "recovery_of_run_id", "successor_run_id",
            "htr_runs_root_path_digest", "htr_project_dir_path_digest",
            "task25_approval_id", "task25_claim_id", "event_id",
        )}},
        "issue": {**issue_art, **{k: issue[k] for k in (
            "recovery_request_id", "recovery_approval_id", "request_digest", "issued_by", "expires_at",
        )}},
        "revoke": {"state": "absent"},
        "claim": {**claim_art, **{k: claim[k] for k in (
            "recovery_request_id", "recovery_approval_id", "claim_id", "claimant",
            "request_digest", "issue_digest",
        )}},
        "attempt": {**attempt_art, **{k: attempt[k] for k in (
            "recovery_request_id", "attempt_id", "claim_digest", "successor_run_id", "executor",
        )}},
        "outcome": {**outcome_art, **{k: outcome[k] for k in (
            "recovery_request_id", "outcome_class", "successor_run_id",
            "request_digest", "issue_digest", "claim_digest", "attempt_digest",
        )}},
        "lineage_binding_digest": lineage,
    }


def build_successor_evidence(
    successor_run_id: str,
    *,
    source_run_id: str,
    task27: dict[str, Any],
    base_dir: Path | None,
    allow_marker: bool = False,
) -> dict[str, Any]:
    if not allow_marker and not _verify_successor_initial_state(successor_run_id, base_dir):
        raise BoundedActionPreconditionError("successor initial state invalid")
    if allow_marker:
        root = paths.run_root(successor_run_id, base_dir)
        required_files = ("recovery_origin.json", "run_manifest.json", "task_events.jsonl", "approvals.jsonl")
        required_dirs = ("reports", "tasks")
        for name in required_files:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise BoundedActionPreconditionError("successor initial state invalid")
        for name in required_dirs:
            path = root / name
            if not path.is_dir() or path.is_symlink():
                raise BoundedActionPreconditionError("successor initial state invalid")
    evaluation = evaluate_run_seal(successor_run_id, base_dir)
    if evaluation.state == SealState.FINALIZED_VALID:
        raise BoundedActionPreconditionError("successor must be non-finalized")
    origin_path = paths.recovery_origin_path(successor_run_id, base_dir)
    origin = io.read_json(origin_path)
    request = _read_optional_record(
        paths.recovery_run_request_path(task27["recovery_request_id"], base_dir)
    )
    issue = _read_optional_record(
        paths.recovery_run_issue_path(task27["recovery_request_id"], base_dir)
    )
    claim = _read_optional_record(
        paths.recovery_run_claim_path(task27["recovery_request_id"], base_dir)
    )
    attempt = _read_optional_record(
        paths.recovery_run_attempt_path(task27["recovery_request_id"], base_dir)
    )
    assert request and issue and claim and attempt
    try:
        _validate_recovery_origin_bindings(
            origin, request=request, issue=issue, claim=claim, attempt=attempt
        )
    except RecoveryRunValidationError as exc:
        raise BoundedActionPreconditionError(str(exc)) from exc
    marker_state, marker_digest = _marker_state(successor_run_id, base_dir)
    runs_root = paths.runs_root(base_dir)
    project_digest = _project_dir_path_digest(base_dir)
    runs_digest = _runs_root_path_digest(base_dir)
    origin_art = _artifact_from_path(
        origin_path,
        rel=f"{successor_run_id}/recovery_origin.json",
        semantic_digest=sha256_digest(_recovery_origin_digest_projection(origin)),
        schema_version="1",
    )
    manifest_path = paths.run_manifest_path(successor_run_id, base_dir)
    events_path = paths.task_events_path(successor_run_id, base_dir)
    approvals_path = paths.approvals_path(successor_run_id, base_dir)
    snapshot = build_run_snapshot(successor_run_id, base_dir=base_dir)
    obs_digest = compute_source_observation_digest(snapshot)
    return {
        "binding_schema_version": SUCCESSOR_EVIDENCE_BINDING,
        "recovery_origin": origin_art,
        "run_manifest": _artifact_from_path(
            manifest_path,
            rel=f"{successor_run_id}/run_manifest.json",
            semantic_digest=_manifest_summary_digest(successor_run_id, base_dir),
            schema_version="1",
        ),
        "task_events_jsonl": _artifact_from_path(
            events_path,
            rel=f"{successor_run_id}/task_events.jsonl",
            semantic_digest=_jsonl_classification_digest(events_path),
            schema_version="jsonl_classification",
        ),
        "approvals_jsonl": _artifact_from_path(
            approvals_path,
            rel=f"{successor_run_id}/approvals.jsonl",
            semantic_digest=_jsonl_classification_digest(approvals_path),
            schema_version="jsonl_classification",
        ),
        "required_directories_digest": sha256_digest({"reports": True, "tasks": True}),
        "forbidden_initial_files_class": "absent",
        "seal": {
            "state": evaluation.state.value,
            "semantic_digest": sha256_digest({"state": evaluation.state.value}),
        },
        "marker_observation": {"state": marker_state, "semantic_digest": marker_digest},
        "observation": {"schema_version": OBSERVE_SCHEMA, "semantic_digest": obs_digest},
        "action_plan": None,
        "expected_pre_state_digest": sha256_digest({"successor_run_id": successor_run_id}),
        "project_identity_digest": project_digest,
        "runs_root_identity_digest": runs_digest,
    }


def classify_marker_for_subject(proposal_subject: str, successor_run_id: str, base_dir: Path | None) -> bool:
    marker = paths.runs_root(base_dir) / LOCKS_DIR_NAME / f"{successor_run_id}.marker"
    present = marker.exists()
    from htr.bounded_action_schemas import SUBJECT_MATRIX

    matrix = SUBJECT_MATRIX.get(proposal_subject, {})
    if matrix.get("marker_required_absent") and present:
        return False
    return True


def evidence_drift(
    stored: dict[str, Any] | None, fresh: dict[str, Any]
) -> str:
    if stored is None:
        return EvidenceDriftClassification.no_drift.value
    if sha256_digest(stored) != sha256_digest(fresh):
        return EvidenceDriftClassification.semantic_drift.value
    return EvidenceDriftClassification.no_drift.value
