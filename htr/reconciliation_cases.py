"""Task 26B — durable reconciliation case control records."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, NoReturn

from htr import paths
from htr.action_plan import _canonical_json, _normalize_path_for_digest, _sha256_digest
from htr.approval_control import (
    _argument_entries_to_inputs,
    _compute_approval_digest,
    _project_dir_path_digest,
    _runs_root_path_digest,
)
from htr.ids import generate_id, validate_id
from htr.io import read_json
from htr.reconciliation_inspection import (
    INSPECTION_DIGEST_PROJECTION_VERSION,
    PILOT_BOUND_API,
    compute_inspection_semantic_digest,
    inspect_run_completion_reconciliation,
)
from htr.state import (
    ReconciliationCaseBundle,
    ReconciliationCaseOpenRecord,
    ReconciliationConflictError,
    ReconciliationDecisionRecord,
    ReconciliationDurabilityError,
    ReconciliationEvidenceIntegrityError,
    ReconciliationInspectionError,
    ReconciliationObservationRecord,
    ReconciliationRecordName,
    ReconciliationStateError,
    ReconciliationUnsupportedApprovalError,
    ReconciliationValidationError,
    ReconciliationWriteMetadata,
    RunCompletionReconciliationInspection,
)

CASE_SCHEMA_VERSION = "1"
OPEN_DIGEST_PROJECTION_VERSION = "htr.reconciliation.open.digest.v1"
OPEN_REQUEST_PROJECTION_VERSION = "htr.reconciliation.open.request.v1"
OBSERVATION_SCHEMA_VERSION = "1"
OBSERVATION_DIGEST_PROJECTION_VERSION = "htr.reconciliation.observation.digest.v1"
OBSERVATION_REQUEST_PROJECTION_VERSION = "htr.reconciliation.observation.request.v1"
DECISION_SCHEMA_VERSION = "1"
DECISION_DIGEST_PROJECTION_VERSION = "htr.reconciliation.decision.digest.v1"
DECISION_REQUEST_PROJECTION_VERSION = "htr.reconciliation.decision.request.v1"
DECISION_REVALIDATION_PROJECTION_VERSION = "htr.reconciliation.decision_revalidation.digest.v1"

EVIDENCE_CAPTURE_MODE = "derived_readonly_inspection"
EVIDENCE_CAPTURE_DISCLAIMER = (
    "Point-in-time derived view from Task 26A inspection; "
    "filesystem state may differ on re-inspection."
)

_NON_PERMISSION_BOOLEANS = (
    "safe_to_retry",
    "marker_disposition_allowed",
    "invoke_allowed",
    "repair_allowed",
    "recovery_run_creation_allowed",
    "outcome_rewrite_allowed",
)

_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_DIRECTORY = os.O_DIRECTORY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

_CONTROL_FILE_MODE = 0o600
_CONTROL_DIR_MODE = 0o700

_OPEN_ALLOWED = frozenset({"open.json"})
_OBSERVED_ALLOWED = frozenset({"open.json", "observation.json"})
_DECIDED_ALLOWED = frozenset({"open.json", "observation.json", "decision.json"})

_MAX_ACTOR_LEN = 256


class ReconciliationScopeReason(str, Enum):
    ambiguous_completion_reconciliation = "ambiguous_completion_reconciliation"


class ReconciliationDecisionClass(str, Enum):
    completion_verified_by_reconciliation = "completion_verified_by_reconciliation"
    evidence_conflict_confirmed = "evidence_conflict_confirmed"
    partial_commit_confirmed = "partial_commit_confirmed"
    integrity_blocked_confirmed = "integrity_blocked_confirmed"
    indeterminate_insufficient_evidence = "indeterminate_insufficient_evidence"
    case_closed_no_action_required = "case_closed_no_action_required"
    case_closed_deferred_to_protocol = "case_closed_deferred_to_protocol"


class ReconciliationNextProtocol(str, Enum):
    none = "none"
    marker_disposition_review = "marker_disposition_review"
    recovery_run_review = "recovery_run_review"
    human_review = "human_review"
    retry_review = "retry_review"


class ReconciliationRationaleCode(str, Enum):
    verified_completion_marker_residue = "verified_completion_marker_residue"
    ambiguous_outcome_lifecycle_complete = "ambiguous_outcome_lifecycle_complete"
    partial_lifecycle_commit = "partial_lifecycle_commit"
    control_lifecycle_evidence_conflict = "control_lifecycle_evidence_conflict"
    consumed_outcome_evidence_mismatch = "consumed_outcome_evidence_mismatch"
    integrity_blocked = "integrity_blocked"
    inspection_drift_detected = "inspection_drift_detected"
    claim_missing_or_invalid = "claim_missing_or_invalid"
    no_reconciliation_needed = "no_reconciliation_needed"


def generate_reconciliation_case_id() -> str:
    """Mint ``rcn_…`` ID only — no filesystem side effects."""
    return generate_id("reconciliation")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_actor(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReconciliationValidationError(f"{field} must be a non-empty string")
    actor = value.strip()
    if len(actor) > _MAX_ACTOR_LEN:
        raise ReconciliationValidationError(f"{field} exceeds {_MAX_ACTOR_LEN} characters")
    if not actor.isprintable():
        raise ReconciliationValidationError(f"{field} must be printable")
    return actor


def _raise_unsafe_path(context: str, exc: BaseException) -> NoReturn:
    raise ReconciliationValidationError(f"unsafe reconciliation control path ({context})") from exc


def _open_control_dir_no_follow(path: Path, *, context: str) -> int:
    try:
        return os.open(str(path), _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
    except OSError as exc:
        _raise_unsafe_path(context, exc)


def _openat_control_dir_no_follow(dir_fd: int, name: str, *, context: str) -> int:
    try:
        return os.open(
            name,
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        _raise_unsafe_path(f"{context}/{name}", exc)


def _openat_control_file_no_follow(
    dir_fd: int,
    name: str,
    flags: int,
    mode: int = 0,
    *,
    context: str,
) -> int:
    try:
        if mode:
            return os.open(name, flags, mode, dir_fd=dir_fd)
        return os.open(name, flags, dir_fd=dir_fd)
    except FileExistsError:
        raise
    except OSError as exc:
        _raise_unsafe_path(f"{context}/{name}", exc)


def _mkdirat_control(dir_fd: int, name: str, mode: int, *, context: str) -> bool:
    try:
        os.mkdir(name, mode, dir_fd=dir_fd)
        return True
    except FileExistsError:
        return False
    except OSError as exc:
        _raise_unsafe_path(f"{context}/{name}", exc)


def _fsync_dir_fd(dir_fd: int, *, case_id: str, record_name: str, stage: str) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise ReconciliationDurabilityError(
            f"directory fsync failed at {stage}: {exc}",
            record_may_have_committed=True,
            exact_replay_status="indeterminate",
            durability_stage=stage,  # type: ignore[arg-type]
            case_id=case_id,
            record_name=record_name,  # type: ignore[arg-type]
        ) from exc


def _fsync_file_fd(file_fd: int, *, case_id: str, record_name: str) -> None:
    try:
        os.fsync(file_fd)
    except OSError as exc:
        raise ReconciliationDurabilityError(
            f"file fsync failed: {exc}",
            record_may_have_committed=True,
            exact_replay_status="indeterminate",
            durability_stage="record_fsync",
            case_id=case_id,
            record_name=record_name,
        ) from exc


def _write_all(
    fd: int,
    payload: bytes,
    *,
    case_id: str,
    record_name: ReconciliationRecordName,
) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise ReconciliationDurabilityError(
                "short write while persisting reconciliation record",
                record_may_have_committed=False,
                exact_replay_status="no",
                durability_stage="record_write",
                case_id=case_id,
                record_name=record_name,
            )
        offset += written


def _read_json_fd(fd: int) -> dict[str, Any]:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 65536)
        if not block:
            break
        chunks.append(block)
    data = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(data, dict):
        raise ReconciliationValidationError("expected JSON object record")
    return data


def _fstat_identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


def _list_dir_entries(dir_fd: int) -> frozenset[str]:
    with os.scandir(dir_fd) as it:
        return frozenset(entry.name for entry in it)


def _read_optional_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ReconciliationValidationError(f"unsafe symlink record: {path}")
    try:
        return read_json(path)
    except json.JSONDecodeError as exc:
        raise ReconciliationValidationError(f"malformed JSON record: {path}") from exc


def _read_optional_record_if_published(path: Path) -> dict[str, Any] | None:
    """Return a durably published record, or None if absent or still being published."""
    try:
        return _read_optional_record(path)
    except ReconciliationValidationError as exc:
        if str(exc).startswith("malformed JSON record:"):
            return None
        raise


def _extract_event_id(issue: dict[str, Any]) -> str:
    inputs = _argument_entries_to_inputs(issue["bound_arguments"])
    event_id = inputs.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ReconciliationValidationError("bound event_id missing from approval issue")
    event_id = event_id.strip()
    if not validate_id(event_id, "event"):
        raise ReconciliationValidationError(f"invalid bound event_id: {event_id!r}")
    return event_id


def _extract_project_dir_digest(issue: dict[str, Any], base_dir: Path | None) -> str:
    entries = (issue.get("bound_arguments") or {}).get("argument_entries") or []
    for entry in entries:
        if entry.get("key") == "project_dir" and entry.get("presence") == "value":
            digest = entry.get("path_digest")
            if isinstance(digest, str) and digest:
                return digest
    if base_dir is not None:
        return _project_dir_path_digest(str(base_dir))
    raise ReconciliationValidationError("could not resolve htr_project_dir_path_digest")


def _read_issue_readonly(approval_id: str, base_dir: Path | None) -> dict[str, Any]:
    validate_id(approval_id, "approval")
    issue_path = paths.approval_issue_path(approval_id, base_dir)
    issue = _read_optional_record(issue_path)
    if issue is None:
        raise ReconciliationValidationError(
            f"approval {approval_id!r} is missing issue.json",
        )
    stored_digest = issue.get("approval_digest")
    computed = _compute_approval_digest(issue)
    if stored_digest != computed:
        raise ReconciliationValidationError("approval issue digest mismatch")
    if issue.get("bound_api") != PILOT_BOUND_API:
        raise ReconciliationUnsupportedApprovalError(
            f"approval bound_api must be {PILOT_BOUND_API!r}",
            approval_id=approval_id,
        )
    expected_runs_root = _runs_root_path_digest(base_dir)
    if issue.get("htr_runs_root_path_digest") != expected_runs_root:
        raise ReconciliationValidationError("htr_runs_root_path_digest mismatch")
    return issue


def _open_request_projection(
    *,
    case_id: str,
    approval_id: str,
    opened_by: str,
    scope_reason: ReconciliationScopeReason,
    approval_issue_digest: str,
    run_id: str,
    event_id: str,
    htr_runs_root_path_digest: str,
    htr_project_dir_path_digest: str,
) -> dict[str, Any]:
    return {
        "open_request_projection_version": OPEN_REQUEST_PROJECTION_VERSION,
        "case_id": case_id,
        "approval_id": approval_id,
        "opened_by": opened_by,
        "scope_reason": scope_reason.value,
        "approval_issue_digest": approval_issue_digest,
        "run_id": run_id,
        "bound_api": PILOT_BOUND_API,
        "event_id": event_id,
        "htr_runs_root_path_digest": htr_runs_root_path_digest,
        "htr_project_dir_path_digest": htr_project_dir_path_digest,
    }


def _open_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_schema_version": body["case_schema_version"],
        "case_open_digest_projection_version": body["case_open_digest_projection_version"],
        "case_id": body["case_id"],
        "approval_id": body["approval_id"],
        "approval_issue_digest": body["approval_issue_digest"],
        "run_id": body["run_id"],
        "bound_api": body["bound_api"],
        "event_id": body["event_id"],
        "htr_runs_root_path_digest": body["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": body["htr_project_dir_path_digest"],
        "opened_by": body["opened_by"],
        "scope_reason": body["scope_reason"],
        "opened_at": body["opened_at"],
    }


def _compute_open_digest(body: dict[str, Any]) -> str:
    return _sha256_digest(_open_digest_projection(body))


def _open_record_from_request(
    request: dict[str, Any],
    *,
    opened_at: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "case_schema_version": CASE_SCHEMA_VERSION,
        "case_id": request["case_id"],
        "case_open_digest_projection_version": OPEN_DIGEST_PROJECTION_VERSION,
        "approval_id": request["approval_id"],
        "approval_issue_digest": request["approval_issue_digest"],
        "run_id": request["run_id"],
        "bound_api": request["bound_api"],
        "event_id": request["event_id"],
        "htr_runs_root_path_digest": request["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": request["htr_project_dir_path_digest"],
        "opened_by": request["opened_by"],
        "scope_reason": request["scope_reason"],
        "opened_at": opened_at,
    }
    body["case_open_digest"] = _compute_open_digest(body)
    return body


def _open_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_request_projection_version": OPEN_REQUEST_PROJECTION_VERSION,
        "case_id": record["case_id"],
        "approval_id": record["approval_id"],
        "opened_by": record["opened_by"],
        "scope_reason": record["scope_reason"],
        "approval_issue_digest": record["approval_issue_digest"],
        "run_id": record["run_id"],
        "bound_api": record["bound_api"],
        "event_id": record["event_id"],
        "htr_runs_root_path_digest": record["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": record["htr_project_dir_path_digest"],
    }


def _build_inspection_semantic_projection_from_result(
    result: RunCompletionReconciliationInspection,
) -> dict[str, Any]:
    """Build the Task 26A inspection semantic projection (26B-local mirror)."""
    return {
        "inspection_schema_version": result.inspection_schema_version,
        "inspection_projection_version": result.inspection_projection_version,
        "approval_id": result.approval_id,
        "approval_digest": result.approval_digest,
        "claim_id": result.claim_id,
        "claim_digest": result.claim_digest,
        "outcome_class": result.outcome_class,
        "outcome_digest": result.outcome_digest,
        "run_id": result.run_id,
        "bound_api": result.bound_api,
        "event_id": result.event_id,
        "htr_runs_root_path_digest": result.htr_runs_root_path_digest,
        "approval_control_state": result.approval_control_state,
        "marker_state": result.marker_state,
        "lifecycle_evidence_state": result.lifecycle_evidence_state,
        "integrity_state": result.integrity_state,
        "overall_classification": result.overall_classification,
        "reason_codes": list(result.reason_codes),
        "observed_completion_record_fingerprint": result.observed_completion_record_fingerprint,
        "observed_event_semantic_fingerprint": result.observed_event_semantic_fingerprint,
        "observed_manifest_status": result.observed_manifest_status,
        "current_observation_semantic_digest": result.current_observation_semantic_digest,
        "source_observation_digest": result.source_observation_digest,
        "safe_to_retry": result.safe_to_retry,
        "marker_disposition_allowed": result.marker_disposition_allowed,
        "reconciliation_case_required": result.reconciliation_case_required,
        "recovery_protocol_required": result.recovery_protocol_required,
    }


def _prove_inspection_projection(
    result: RunCompletionReconciliationInspection,
) -> tuple[dict[str, Any], str]:
    projection = _build_inspection_semantic_projection_from_result(result)
    local_digest = _sha256_digest(projection)
    if local_digest != result.inspection_semantic_digest:
        raise ReconciliationValidationError("inspection semantic digest mismatch against result")
    if local_digest != compute_inspection_semantic_digest(result):
        raise ReconciliationValidationError(
            "inspection semantic digest mismatch against compute_inspection_semantic_digest",
        )
    return projection, local_digest


def _observation_request_projection(*, case_id: str, observed_by: str) -> dict[str, Any]:
    return {
        "observation_request_projection_version": OBSERVATION_REQUEST_PROJECTION_VERSION,
        "case_id": case_id,
        "observed_by": observed_by,
    }


def _observation_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_schema_version": body["observation_schema_version"],
        "observation_digest_projection_version": body["observation_digest_projection_version"],
        "case_id": body["case_id"],
        "case_open_digest": body["case_open_digest"],
        "observed_by": body["observed_by"],
        "observed_at": body["observed_at"],
        "evidence_capture_mode": body["evidence_capture_mode"],
        "evidence_capture_disclaimer": body["evidence_capture_disclaimer"],
        "inspection_semantic_digest": body["inspection_semantic_digest"],
        "inspection_semantic_projection": body["inspection_semantic_projection"],
        "inspection_semantic_projection_version": body["inspection_semantic_projection_version"],
    }


def _compute_observation_digest(body: dict[str, Any]) -> str:
    return _sha256_digest(_observation_digest_projection(body))


def _observation_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_request_projection_version": OBSERVATION_REQUEST_PROJECTION_VERSION,
        "case_id": record["case_id"],
        "observed_by": record["observed_by"],
    }


def _decision_request_projection(
    *,
    case_id: str,
    expected_observation_digest: str,
    requested_decision_class: ReconciliationDecisionClass,
    decided_by: str,
    requested_rationale_codes: tuple[ReconciliationRationaleCode, ...],
    recommended_next_protocol: ReconciliationNextProtocol | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "decision_request_projection_version": DECISION_REQUEST_PROJECTION_VERSION,
        "case_id": case_id,
        "expected_observation_digest": expected_observation_digest,
        "requested_decision_class": requested_decision_class.value,
        "decided_by": decided_by,
        "rationale_codes": sorted(code.value for code in requested_rationale_codes),
        "recommended_next_protocol": (
            recommended_next_protocol or ReconciliationNextProtocol.none
        ).value,
    }
    return payload


def _decision_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_schema_version": body["decision_schema_version"],
        "decision_digest_projection_version": body["decision_digest_projection_version"],
        "case_id": body["case_id"],
        "case_open_digest": body["case_open_digest"],
        "observation_digest": body["observation_digest"],
        "decided_by": body["decided_by"],
        "decided_at": body["decided_at"],
        "requested_decision_class": body["requested_decision_class"],
        "decision_class": body["decision_class"],
        "decision_basis": body["decision_basis"],
        "derived_rationale_codes": list(body["derived_rationale_codes"]),
        "requested_rationale_codes": list(body.get("requested_rationale_codes") or []),
        "observation_decision_drift": body["observation_decision_drift"],
        "decision_time_revalidation": body["decision_time_revalidation"],
        "completion_verification": body.get("completion_verification"),
        "safe_to_retry": body["safe_to_retry"],
        "marker_disposition_allowed": body["marker_disposition_allowed"],
        "invoke_allowed": body["invoke_allowed"],
        "repair_allowed": body["repair_allowed"],
        "recovery_run_creation_allowed": body["recovery_run_creation_allowed"],
        "outcome_rewrite_allowed": body["outcome_rewrite_allowed"],
        "recommended_next_protocol": body["recommended_next_protocol"],
        "recommended_next_protocol_authority": body["recommended_next_protocol_authority"],
    }


def _compute_decision_digest(body: dict[str, Any]) -> str:
    return _sha256_digest(_decision_digest_projection(body))


def _decision_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_request_projection_version": DECISION_REQUEST_PROJECTION_VERSION,
        "case_id": record["case_id"],
        "expected_observation_digest": record["observation_digest"],
        "requested_decision_class": record["requested_decision_class"],
        "decided_by": record["decided_by"],
        "rationale_codes": sorted(record.get("requested_rationale_codes") or []),
        "recommended_next_protocol": record.get("recommended_next_protocol")
        or ReconciliationNextProtocol.none.value,
    }


def _decision_revalidation_digest_projection(envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        "revalidation_projection_version": envelope["revalidation_projection_version"],
        "case_id": envelope["case_id"],
        "case_open_digest": envelope["case_open_digest"],
        "observation_digest": envelope["observation_digest"],
        "inspection_semantic_projection": envelope["inspection_semantic_projection"],
    }


def _validate_record_digest(
    record: dict[str, Any],
    *,
    digest_field: str,
    projection_fn,
) -> None:
    stored = record.get(digest_field)
    if not isinstance(stored, str) or not stored:
        raise ReconciliationValidationError(f"missing {digest_field}")
    computed = _sha256_digest(projection_fn(record))
    if stored != computed:
        raise ReconciliationValidationError(f"{digest_field} mismatch")


def _validate_non_permission_booleans(record: dict[str, Any]) -> None:
    for field in _NON_PERMISSION_BOOLEANS:
        value = record.get(field)
        if value is not False:
            raise ReconciliationValidationError(f"{field} must be false")


def _validate_observation_record(record: dict[str, Any]) -> None:
    _validate_record_digest(
        record,
        digest_field="observation_digest",
        projection_fn=_observation_digest_projection,
    )
    projection = record.get("inspection_semantic_projection")
    if not isinstance(projection, dict):
        raise ReconciliationValidationError("missing inspection_semantic_projection")
    local = _sha256_digest(projection)
    if local != record.get("inspection_semantic_digest"):
        raise ReconciliationValidationError("inspection_semantic_projection digest mismatch")


def _validate_open_record(record: dict[str, Any], *, case_id: str) -> None:
    if record.get("case_id") != case_id:
        raise ReconciliationValidationError("case_id mismatch in open.json")
    _validate_record_digest(
        record,
        digest_field="case_open_digest",
        projection_fn=_open_digest_projection,
    )


def _validate_decision_record(record: dict[str, Any], *, case_id: str) -> None:
    if record.get("case_id") != case_id:
        raise ReconciliationValidationError("case_id mismatch in decision.json")
    _validate_record_digest(
        record,
        digest_field="decision_digest",
        projection_fn=_decision_digest_projection,
    )
    _validate_non_permission_booleans(record)
    envelope = (record.get("decision_time_revalidation") or {})
    stored_rev = envelope.get("decision_revalidation_record_digest")
    if isinstance(stored_rev, str) and stored_rev:
        computed = _sha256_digest(_decision_revalidation_digest_projection(envelope))
        if stored_rev != computed:
            raise ReconciliationValidationError("decision_revalidation_record_digest mismatch")


def _validate_directory_state(entries: frozenset[str], *, allowed: frozenset[str], case_id: str) -> None:
    if entries <= allowed:
        return
    raise ReconciliationStateError(
        f"unexpected reconciliation case directory entries: {sorted(entries)}",
        case_id=case_id,
    )


def _exact_replay_published_record_under_lock(
    *,
    case_id: str,
    case_fd: int,
    filename: Literal["open.json", "observation.json", "decision.json"],
    request_projection: dict[str, Any],
    intent_from_record: Callable[[dict[str, Any]], dict[str, Any]],
    validate_existing: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Reload a published immutable record under case ``flock(LOCK_EX)`` for exact replay."""
    flags_read = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    record_ctx = f"reconciliation/{case_id}/{filename}"
    fcntl.flock(case_fd, fcntl.LOCK_EX)
    try:
        existing_fd = _openat_control_file_no_follow(
            case_fd,
            filename,
            flags_read,
            context=record_ctx,
        )
        try:
            existing = _read_json_fd(existing_fd)
        finally:
            os.close(existing_fd)
        validate_existing(existing)
        if intent_from_record(existing) != request_projection:
            raise ReconciliationConflictError(
                f"{filename} already exists with conflicting semantics",
                case_id=case_id,
            )
        return existing
    finally:
        fcntl.flock(case_fd, fcntl.LOCK_UN)


def _bootstrap_reconciliation_tree(
    case_id: str,
    base_dir: Path | None,
) -> tuple[int, int, tuple[int, int]]:
    """Return ``(reconciliation_fd, case_fd, case_identity)`` after bootstrap."""
    runs_root = paths.runs_root(base_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    runs_fd = _open_control_dir_no_follow(runs_root, context="runs_root")
    try:
        created_control = _mkdirat_control(
            runs_fd,
            paths.CONTROL_DIR_NAME,
            _CONTROL_DIR_MODE,
            context="runs_root",
        )
        control_fd = _openat_control_dir_no_follow(
            runs_fd,
            paths.CONTROL_DIR_NAME,
            context="runs_root/.control",
        )
        try:
            created_reconciliation = _mkdirat_control(
                control_fd,
                paths.RECONCILIATION_DIR_NAME,
                _CONTROL_DIR_MODE,
                context="runs_root/.control",
            )
            reconciliation_fd = _openat_control_dir_no_follow(
                control_fd,
                paths.RECONCILIATION_DIR_NAME,
                context="runs_root/.control/reconciliation",
            )
            try:
                created_case = _mkdirat_control(
                    reconciliation_fd,
                    case_id,
                    _CONTROL_DIR_MODE,
                    context=f"reconciliation/{case_id}",
                )
                case_fd = _openat_control_dir_no_follow(
                    reconciliation_fd,
                    case_id,
                    context=f"reconciliation/{case_id}",
                )
                case_identity = _fstat_identity(case_fd)
                if created_case:
                    _fsync_dir_fd(
                        reconciliation_fd,
                        case_id=case_id,
                        record_name="open.json",
                        stage="case_dir_fsync",
                    )
                if created_reconciliation:
                    _fsync_dir_fd(
                        control_fd,
                        case_id=case_id,
                        record_name="open.json",
                        stage="control_dir_fsync",
                    )
                if created_control:
                    _fsync_dir_fd(
                        runs_fd,
                        case_id=case_id,
                        record_name="open.json",
                        stage="parent_dir_fsync",
                    )
                return reconciliation_fd, case_fd, case_identity
            except Exception:
                os.close(reconciliation_fd)
                raise
        finally:
            os.close(control_fd)
    finally:
        os.close(runs_fd)
    raise ReconciliationStateError("unreachable bootstrap failure", case_id=case_id)



def _create_immutable_record(
    *,
    case_id: str,
    case_fd: int,
    filename: Literal["open.json", "observation.json", "decision.json"],
    record: dict[str, Any],
    request_projection: dict[str, Any],
    intent_from_record: Callable[[dict[str, Any]], dict[str, Any]],
    validate_existing: Callable[[dict[str, Any]], None] | None = None,
    base_dir: Path | None,
) -> tuple[dict[str, Any], bool]:
    """Persist or replay an immutable record.

    Case-directory ``flock(LOCK_EX)`` serializes creators so replay readers
    never observe partial JSON. On O_EXCL loss, reload the winner and compare
    timestamp-free request intent projections only.
    """
    del base_dir  # retained for call-site stability
    flags = _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
    flags_read = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    record_ctx = f"reconciliation/{case_id}/{filename}"
    fcntl.flock(case_fd, fcntl.LOCK_EX)
    file_fd: int | None = None
    try:
        try:
            file_fd = _openat_control_file_no_follow(
                case_fd,
                filename,
                flags,
                _CONTROL_FILE_MODE,
                context=record_ctx,
            )
        except FileExistsError:
            existing_fd = _openat_control_file_no_follow(
                case_fd,
                filename,
                flags_read,
                context=record_ctx,
            )
            try:
                existing = _read_json_fd(existing_fd)
            finally:
                os.close(existing_fd)
            if validate_existing is not None:
                validate_existing(existing)
            if intent_from_record(existing) != request_projection:
                raise ReconciliationConflictError(
                    f"{filename} already exists with conflicting semantics",
                    case_id=case_id,
                )
            return existing, True
        payload = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            _write_all(file_fd, payload, case_id=case_id, record_name=filename)
            _fsync_file_fd(file_fd, case_id=case_id, record_name=filename)
            _fsync_dir_fd(case_fd, case_id=case_id, record_name=filename, stage="case_dir_fsync")
        except Exception:
            try:
                os.unlink(filename, dir_fd=case_fd)
            except OSError:
                pass
            raise
        finally:
            if file_fd is not None:
                os.close(file_fd)
                file_fd = None
        return record, False
    finally:
        if file_fd is not None:
            os.close(file_fd)
        fcntl.flock(case_fd, fcntl.LOCK_UN)


def _write_metadata(*, exact_replay: bool) -> ReconciliationWriteMetadata:
    return ReconciliationWriteMetadata(
        exact_replay=exact_replay,
        exact_replay_status="yes" if exact_replay else "no",
        record_may_have_committed=True,
        durability_indeterminate=False,
    )


def _to_open_record(body: dict[str, Any]) -> ReconciliationCaseOpenRecord:
    return ReconciliationCaseOpenRecord(
        case_id=body["case_id"],
        case_open_digest=body["case_open_digest"],
        approval_id=body["approval_id"],
        approval_issue_digest=body["approval_issue_digest"],
        run_id=body["run_id"],
        bound_api=body["bound_api"],
        event_id=body["event_id"],
        htr_runs_root_path_digest=body["htr_runs_root_path_digest"],
        htr_project_dir_path_digest=body["htr_project_dir_path_digest"],
        opened_by=body["opened_by"],
        scope_reason=body["scope_reason"],
        opened_at=body["opened_at"],
    )


def _to_observation_record(body: dict[str, Any]) -> ReconciliationObservationRecord:
    return ReconciliationObservationRecord(
        case_id=body["case_id"],
        observation_digest=body["observation_digest"],
        case_open_digest=body["case_open_digest"],
        observed_by=body["observed_by"],
        observed_at=body["observed_at"],
        inspection_semantic_digest=body["inspection_semantic_digest"],
    )


def _to_decision_record(body: dict[str, Any]) -> ReconciliationDecisionRecord:
    return ReconciliationDecisionRecord(
        case_id=body["case_id"],
        decision_digest=body["decision_digest"],
        case_open_digest=body["case_open_digest"],
        observation_digest=body["observation_digest"],
        decided_by=body["decided_by"],
        decided_at=body["decided_at"],
        requested_decision_class=body["requested_decision_class"],
        decision_class=body["decision_class"],
        derived_rationale_codes=tuple(body["derived_rationale_codes"]),
    )


def _load_validated_open(case_id: str, base_dir: Path | None) -> dict[str, Any]:
    record = _read_optional_record(paths.reconciliation_open_path(case_id, base_dir))
    if record is None:
        raise ReconciliationStateError(f"case {case_id!r} is missing open.json", case_id=case_id)
    _validate_open_record(record, case_id=case_id)
    return record


def _load_validated_observation(case_id: str, base_dir: Path | None) -> dict[str, Any]:
    record = _read_optional_record(paths.reconciliation_observation_path(case_id, base_dir))
    if record is None:
        raise ReconciliationStateError(
            f"case {case_id!r} is missing observation.json",
            case_id=case_id,
        )
    _validate_observation_record(record)
    return record


def _completion_verification_eligible(
    projection: dict[str, Any],
    *,
    open_record: dict[str, Any],
    drift_detected: bool,
) -> bool:
    if drift_detected:
        return False
    if open_record.get("bound_api") != PILOT_BOUND_API:
        return False
    if projection.get("bound_api") != PILOT_BOUND_API:
        return False
    if not projection.get("claim_id") or not projection.get("claim_digest"):
        return False
    outcome = projection.get("outcome_class")
    if outcome == "consumed":
        return False
    if outcome not in (None, "ambiguous"):
        return False
    if projection.get("lifecycle_evidence_state") != "lifecycle_complete_observed":
        return False
    if projection.get("integrity_state") != "clean":
        return False
    if not projection.get("current_observation_semantic_digest"):
        return False
    if "multiple_completion_events" in (projection.get("reason_codes") or []):
        return False
    blocking = {
        "completion_record_semantic_mismatch",
        "event_semantic_mismatch",
        "event_actor_mismatch",
        "consumed_outcome_current_evidence_mismatch",
        "control_evidence_replaced_during_read",
    }
    if blocking.intersection(projection.get("reason_codes") or []):
        return False
    for field in ("run_id", "event_id", "htr_runs_root_path_digest"):
        if projection.get(field) != open_record.get(field if field != "htr_runs_root_path_digest" else field):
            return False
    if projection.get("approval_id") != open_record.get("approval_id"):
        return False
    return True


def _derive_rationale_codes(
    projection: dict[str, Any],
    *,
    drift_detected: bool,
) -> list[ReconciliationRationaleCode]:
    codes: list[ReconciliationRationaleCode] = []
    if drift_detected:
        codes.append(ReconciliationRationaleCode.inspection_drift_detected)
    overall = projection.get("overall_classification")
    if overall == "no_reconciliation_needed":
        codes.append(ReconciliationRationaleCode.no_reconciliation_needed)
    if overall == "verified_completion_marker_residue":
        codes.append(ReconciliationRationaleCode.verified_completion_marker_residue)
    if overall in {"completion_observed_outcome_missing", "reconciliation_inspection_required"}:
        if projection.get("lifecycle_evidence_state") == "lifecycle_complete_observed":
            codes.append(ReconciliationRationaleCode.ambiguous_outcome_lifecycle_complete)
    if overall == "partial_lifecycle_commit":
        codes.append(ReconciliationRationaleCode.partial_lifecycle_commit)
    if overall == "control_lifecycle_evidence_conflict":
        codes.append(ReconciliationRationaleCode.control_lifecycle_evidence_conflict)
        if "consumed_outcome_current_evidence_mismatch" in (projection.get("reason_codes") or []):
            codes.append(ReconciliationRationaleCode.consumed_outcome_evidence_mismatch)
    if overall == "integrity_blocked" or projection.get("integrity_state") != "clean":
        codes.append(ReconciliationRationaleCode.integrity_blocked)
    if not projection.get("claim_id"):
        codes.append(ReconciliationRationaleCode.claim_missing_or_invalid)
    return sorted(set(codes), key=lambda c: c.value)


def _derive_allowed_decision_classes(
    projection: dict[str, Any],
    *,
    open_record: dict[str, Any],
    drift_detected: bool,
) -> frozenset[ReconciliationDecisionClass]:
    allowed: set[ReconciliationDecisionClass] = set()
    overall = projection.get("overall_classification")
    control = projection.get("approval_control_state")
    outcome = projection.get("outcome_class")

    if drift_detected:
        allowed.update(
            {
                ReconciliationDecisionClass.indeterminate_insufficient_evidence,
                ReconciliationDecisionClass.evidence_conflict_confirmed,
                ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            }
        )
        return frozenset(allowed)

    if overall == "no_reconciliation_needed":
        allowed.add(ReconciliationDecisionClass.case_closed_no_action_required)
        return frozenset(allowed)

    if control == "consumed_outcome" and overall == "verified_completion_marker_residue":
        allowed.update(
            {
                ReconciliationDecisionClass.case_closed_no_action_required,
                ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            }
        )
        return frozenset(allowed)

    if control == "consumed_outcome" and (
        overall == "control_lifecycle_evidence_conflict"
        or "consumed_outcome_current_evidence_mismatch" in (projection.get("reason_codes") or [])
    ):
        allowed.update(
            {
                ReconciliationDecisionClass.evidence_conflict_confirmed,
                ReconciliationDecisionClass.indeterminate_insufficient_evidence,
            }
        )
        return frozenset(allowed)

    if overall == "partial_lifecycle_commit":
        allowed.update(
            {
                ReconciliationDecisionClass.partial_commit_confirmed,
                ReconciliationDecisionClass.indeterminate_insufficient_evidence,
                ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            }
        )
        return frozenset(allowed)

    if overall == "control_lifecycle_evidence_conflict":
        allowed.update(
            {
                ReconciliationDecisionClass.evidence_conflict_confirmed,
                ReconciliationDecisionClass.indeterminate_insufficient_evidence,
            }
        )
        return frozenset(allowed)

    if overall == "integrity_blocked" or projection.get("integrity_state") == "seal_blocked":
        allowed.update(
            {
                ReconciliationDecisionClass.integrity_blocked_confirmed,
                ReconciliationDecisionClass.case_closed_deferred_to_protocol,
            }
        )
        return frozenset(allowed)

    if not projection.get("claim_id") and projection.get("lifecycle_evidence_state") != "no_lifecycle_evidence_observed":
        allowed.update(
            {
                ReconciliationDecisionClass.evidence_conflict_confirmed,
                ReconciliationDecisionClass.indeterminate_insufficient_evidence,
            }
        )
        return frozenset(allowed)

    if _completion_verification_eligible(
        projection,
        open_record=open_record,
        drift_detected=drift_detected,
    ):
        allowed.update(
            {
                ReconciliationDecisionClass.completion_verified_by_reconciliation,
                ReconciliationDecisionClass.case_closed_deferred_to_protocol,
                ReconciliationDecisionClass.case_closed_no_action_required,
            }
        )
        return frozenset(allowed)

    allowed.update(
        {
            ReconciliationDecisionClass.indeterminate_insufficient_evidence,
            ReconciliationDecisionClass.evidence_conflict_confirmed,
            ReconciliationDecisionClass.case_closed_deferred_to_protocol,
        }
    )
    if outcome == "consumed":
        allowed.discard(ReconciliationDecisionClass.completion_verified_by_reconciliation)
    return frozenset(allowed)


def _allowed_next_protocols(
    decision_class: ReconciliationDecisionClass,
    projection: dict[str, Any],
) -> frozenset[ReconciliationNextProtocol]:
    allowed = {ReconciliationNextProtocol.none}
    marker = projection.get("marker_state")
    marker_present = marker in {"present_valid_metadata", "present_malformed_metadata"}
    if marker_present and decision_class in {
        ReconciliationDecisionClass.case_closed_no_action_required,
        ReconciliationDecisionClass.case_closed_deferred_to_protocol,
    }:
        allowed.add(ReconciliationNextProtocol.marker_disposition_review)
    if decision_class in {
        ReconciliationDecisionClass.evidence_conflict_confirmed,
        ReconciliationDecisionClass.partial_commit_confirmed,
        ReconciliationDecisionClass.integrity_blocked_confirmed,
    }:
        allowed.add(ReconciliationNextProtocol.recovery_run_review)
    if decision_class == ReconciliationDecisionClass.indeterminate_insufficient_evidence:
        allowed.add(ReconciliationNextProtocol.human_review)
        allowed.add(ReconciliationNextProtocol.retry_review)
    if decision_class in {
        ReconciliationDecisionClass.evidence_conflict_confirmed,
        ReconciliationDecisionClass.partial_commit_confirmed,
    }:
        allowed.add(ReconciliationNextProtocol.retry_review)
    return frozenset(allowed)


def open_reconciliation_case(
    case_id: str,
    approval_id: str,
    *,
    base_dir: Path | None = None,
    opened_by: str,
    scope_reason: ReconciliationScopeReason,
) -> tuple[ReconciliationCaseOpenRecord, ReconciliationWriteMetadata]:
    validate_id(case_id, "reconciliation")
    opened_by = _validate_actor(opened_by, field="opened_by")

    issue = _read_issue_readonly(approval_id, base_dir)
    approval_issue_digest = issue["approval_digest"]
    run_id = issue["run_id"]
    event_id = _extract_event_id(issue)
    htr_runs_root_path_digest = issue["htr_runs_root_path_digest"]
    htr_project_dir_path_digest = _extract_project_dir_digest(issue, base_dir)

    request = _open_request_projection(
        case_id=case_id,
        approval_id=approval_id,
        opened_by=opened_by,
        scope_reason=scope_reason,
        approval_issue_digest=approval_issue_digest,
        run_id=run_id,
        event_id=event_id,
        htr_runs_root_path_digest=htr_runs_root_path_digest,
        htr_project_dir_path_digest=htr_project_dir_path_digest,
    )

    open_path = paths.reconciliation_open_path(case_id, base_dir)
    existing = _read_optional_record_if_published(open_path)
    if existing is not None:
        _validate_open_record(existing, case_id=case_id)
        if _open_intent_projection(existing) != request:
            raise ReconciliationConflictError(
                f"open.json already exists with conflicting semantics for {case_id!r}",
                case_id=case_id,
            )
        return _to_open_record(existing), _write_metadata(exact_replay=True)

    case_dir = paths.reconciliation_case_dir(case_id, base_dir)
    reconciliation_fd: int | None = None
    case_fd: int | None = None
    try:
        if case_dir.is_dir() and not case_dir.is_symlink():
            case_fd = _open_control_dir_no_follow(case_dir, context=f"reconciliation/{case_id}")
            pre_identity = _fstat_identity(case_fd)
            entries = _list_dir_entries(case_fd)
            if entries == frozenset({"open.json"}):
                existing = _read_optional_record_if_published(open_path)
                if existing is None:
                    existing = _exact_replay_published_record_under_lock(
                        case_id=case_id,
                        case_fd=case_fd,
                        filename="open.json",
                        request_projection=request,
                        intent_from_record=_open_intent_projection,
                        validate_existing=lambda body: _validate_open_record(body, case_id=case_id),
                    )
                _validate_open_record(existing, case_id=case_id)
                if _open_intent_projection(existing) != request:
                    raise ReconciliationConflictError(
                        f"open.json already exists with conflicting semantics for {case_id!r}",
                        case_id=case_id,
                    )
                return _to_open_record(existing), _write_metadata(exact_replay=True)
            if entries:
                raise ReconciliationStateError(
                    f"case directory {case_id!r} has unexpected entries before open",
                    case_id=case_id,
                )
            reconciliation_fd = _open_control_dir_no_follow(
                paths.control_reconciliation_root(base_dir),
                context="reconciliation",
            )
        else:
            reconciliation_fd, case_fd, pre_identity = _bootstrap_reconciliation_tree(
                case_id,
                base_dir,
            )

        intended = _open_record_from_request(request, opened_at=_utc_now_iso())
        persisted, exact_replay = _create_immutable_record(
            case_id=case_id,
            case_fd=case_fd,
            filename="open.json",
            record=intended,
            request_projection=request,
            intent_from_record=_open_intent_projection,
            validate_existing=lambda body: _validate_open_record(body, case_id=case_id),
            base_dir=base_dir,
        )
        post_identity = _fstat_identity(case_fd)
        if post_identity != pre_identity:
            raise ReconciliationStateError(
                "case directory identity changed during open",
                case_id=case_id,
            )
        post_entries = _list_dir_entries(case_fd)
        _validate_directory_state(post_entries, allowed=_OPEN_ALLOWED, case_id=case_id)
        return _to_open_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        if case_fd is not None:
            os.close(case_fd)
        if reconciliation_fd is not None:
            os.close(reconciliation_fd)


def record_reconciliation_observation(
    case_id: str,
    *,
    base_dir: Path | None = None,
    observed_by: str,
) -> tuple[ReconciliationObservationRecord, ReconciliationWriteMetadata]:
    validate_id(case_id, "reconciliation")
    observed_by = _validate_actor(observed_by, field="observed_by")

    open_record = _load_validated_open(case_id, base_dir)
    request = _observation_request_projection(case_id=case_id, observed_by=observed_by)

    obs_path = paths.reconciliation_observation_path(case_id, base_dir)
    existing = _read_optional_record_if_published(obs_path)
    if existing is not None:
        _validate_observation_record(existing)
        if _observation_intent_projection(existing) != request:
            raise ReconciliationConflictError(
                f"observation.json already exists with conflicting semantics for {case_id!r}",
                case_id=case_id,
            )
        return _to_observation_record(existing), _write_metadata(exact_replay=True)

    case_dir = paths.reconciliation_case_dir(case_id, base_dir)
    if not case_dir.is_dir() or case_dir.is_symlink():
        raise ReconciliationStateError(f"case directory missing for {case_id!r}", case_id=case_id)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"reconciliation/{case_id}")
    try:
        entries = _list_dir_entries(case_fd)
        if entries == _OBSERVED_ALLOWED:
            existing = _exact_replay_published_record_under_lock(
                case_id=case_id,
                case_fd=case_fd,
                filename="observation.json",
                request_projection=request,
                intent_from_record=_observation_intent_projection,
                validate_existing=_validate_observation_record,
            )
            return _to_observation_record(existing), _write_metadata(exact_replay=True)

        _validate_directory_state(entries, allowed=_OBSERVED_ALLOWED - {"observation.json"}, case_id=case_id)

        approval_id = open_record["approval_id"]
        try:
            result = inspect_run_completion_reconciliation(approval_id, base_dir=base_dir)
        except (
            ReconciliationInspectionError,
            ReconciliationEvidenceIntegrityError,
            ReconciliationUnsupportedApprovalError,
        ):
            raise
        except Exception as exc:
            raise ReconciliationInspectionError(str(exc), approval_id=approval_id) from exc

        projection, digest = _prove_inspection_projection(result)
        observed_at = _utc_now_iso()
        body: dict[str, Any] = {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "case_id": case_id,
            "case_open_digest": open_record["case_open_digest"],
            "observation_digest_projection_version": OBSERVATION_DIGEST_PROJECTION_VERSION,
            "observed_by": observed_by,
            "observed_at": observed_at,
            "evidence_capture_mode": EVIDENCE_CAPTURE_MODE,
            "evidence_capture_disclaimer": EVIDENCE_CAPTURE_DISCLAIMER,
            "inspection_semantic_digest": digest,
            "inspection_semantic_projection": projection,
            "inspection_semantic_projection_version": INSPECTION_DIGEST_PROJECTION_VERSION,
        }
        body["observation_digest"] = _compute_observation_digest(body)

        persisted, exact_replay = _create_immutable_record(
            case_id=case_id,
            case_fd=case_fd,
            filename="observation.json",
            record=body,
            request_projection=request,
            intent_from_record=_observation_intent_projection,
            validate_existing=_validate_observation_record,
            base_dir=base_dir,
        )
        post_entries = _list_dir_entries(case_fd)
        _validate_directory_state(post_entries, allowed=_OBSERVED_ALLOWED, case_id=case_id)
        return _to_observation_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)


def record_reconciliation_decision(
    case_id: str,
    *,
    base_dir: Path | None = None,
    expected_observation_digest: str,
    requested_decision_class: ReconciliationDecisionClass,
    decided_by: str,
    requested_rationale_codes: tuple[ReconciliationRationaleCode, ...] = (),
    recommended_next_protocol: ReconciliationNextProtocol | None = None,
) -> tuple[ReconciliationDecisionRecord, ReconciliationWriteMetadata]:
    validate_id(case_id, "reconciliation")
    decided_by = _validate_actor(decided_by, field="decided_by")

    open_record = _load_validated_open(case_id, base_dir)
    observation_record = _load_validated_observation(case_id, base_dir)
    if observation_record["observation_digest"] != expected_observation_digest:
        raise ReconciliationValidationError("expected_observation_digest mismatch")

    request = _decision_request_projection(
        case_id=case_id,
        expected_observation_digest=expected_observation_digest,
        requested_decision_class=requested_decision_class,
        decided_by=decided_by,
        requested_rationale_codes=requested_rationale_codes,
        recommended_next_protocol=recommended_next_protocol,
    )

    decision_path = paths.reconciliation_decision_path(case_id, base_dir)
    existing = _read_optional_record_if_published(decision_path)
    if existing is not None:
        _validate_decision_record(existing, case_id=case_id)
        if _decision_intent_projection(existing) != request:
            raise ReconciliationConflictError(
                f"decision.json already exists with conflicting semantics for {case_id!r}",
                case_id=case_id,
            )
        return _to_decision_record(existing), _write_metadata(exact_replay=True)

    case_dir = paths.reconciliation_case_dir(case_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"reconciliation/{case_id}")
    try:
        entries = _list_dir_entries(case_fd)
        if entries == _DECIDED_ALLOWED:
            existing = _exact_replay_published_record_under_lock(
                case_id=case_id,
                case_fd=case_fd,
                filename="decision.json",
                request_projection=request,
                intent_from_record=_decision_intent_projection,
                validate_existing=lambda body: _validate_decision_record(body, case_id=case_id),
            )
            return _to_decision_record(existing), _write_metadata(exact_replay=True)

        _validate_directory_state(entries, allowed=_DECIDED_ALLOWED - {"decision.json"}, case_id=case_id)

        approval_id = open_record["approval_id"]
        try:
            revalidation = inspect_run_completion_reconciliation(approval_id, base_dir=base_dir)
        except (
            ReconciliationInspectionError,
            ReconciliationEvidenceIntegrityError,
            ReconciliationUnsupportedApprovalError,
        ):
            raise
        except Exception as exc:
            raise ReconciliationInspectionError(str(exc), approval_id=approval_id) from exc

        revalidation_projection, revalidation_digest = _prove_inspection_projection(revalidation)
        obs_digest_at_observation = observation_record["inspection_semantic_digest"]
        obs_digest_at_decision = revalidation_digest
        drift_detected = obs_digest_at_observation != obs_digest_at_decision
        drift_reason_codes: list[str] = []
        if drift_detected:
            drift_reason_codes.append("inspection_semantic_digest_mismatch")

        allowed_classes = _derive_allowed_decision_classes(
            revalidation_projection,
            open_record=open_record,
            drift_detected=drift_detected,
        )
        if requested_decision_class not in allowed_classes:
            raise ReconciliationValidationError(
                f"requested decision class {requested_decision_class.value!r} not allowed by policy",
            )

        derived_codes = _derive_rationale_codes(
            revalidation_projection,
            drift_detected=drift_detected,
        )
        requested_values = {code.value for code in requested_rationale_codes}
        derived_values = {code.value for code in derived_codes}
        if not requested_values.issubset(derived_values):
            raise ReconciliationValidationError(
                "requested_rationale_codes must be a subset of derived rationale codes",
            )

        protocol = recommended_next_protocol or ReconciliationNextProtocol.none
        allowed_protocols = _allowed_next_protocols(requested_decision_class, revalidation_projection)
        if protocol not in allowed_protocols:
            raise ReconciliationValidationError(
                f"recommended_next_protocol {protocol.value!r} not allowed for decision class",
            )

        revalidation_envelope: dict[str, Any] = {
            "revalidation_projection_version": DECISION_REVALIDATION_PROJECTION_VERSION,
            "case_id": case_id,
            "case_open_digest": open_record["case_open_digest"],
            "observation_digest": observation_record["observation_digest"],
            "inspection_semantic_projection": revalidation_projection,
        }
        revalidation_envelope["decision_revalidation_record_digest"] = _sha256_digest(
            _decision_revalidation_digest_projection(revalidation_envelope),
        )

        completion_verification: dict[str, Any] | None = None
        if requested_decision_class == ReconciliationDecisionClass.completion_verified_by_reconciliation:
            completion_verification = {
                "original_outcome_class": revalidation_projection.get("outcome_class"),
                "original_outcome_digest": revalidation_projection.get("outcome_digest"),
                "outcome_record_modified": False,
            }

        decided_at = _utc_now_iso()
        body: dict[str, Any] = {
            "decision_schema_version": DECISION_SCHEMA_VERSION,
            "case_id": case_id,
            "case_open_digest": open_record["case_open_digest"],
            "observation_digest": observation_record["observation_digest"],
            "decision_digest_projection_version": DECISION_DIGEST_PROJECTION_VERSION,
            "decided_by": decided_by,
            "decided_at": decided_at,
            "requested_decision_class": requested_decision_class.value,
            "decision_class": requested_decision_class.value,
            "decision_basis": "observation_and_revalidation",
            "derived_rationale_codes": [code.value for code in derived_codes],
            "requested_rationale_codes": sorted(requested_values),
            "observation_decision_drift": {
                "drift_detected": drift_detected,
                "inspection_semantic_digest_at_observation": obs_digest_at_observation,
                "inspection_semantic_digest_at_decision": obs_digest_at_decision,
                "drift_reason_codes": drift_reason_codes,
            },
            "decision_time_revalidation": revalidation_envelope,
            "completion_verification": completion_verification,
            "safe_to_retry": False,
            "marker_disposition_allowed": False,
            "invoke_allowed": False,
            "repair_allowed": False,
            "recovery_run_creation_allowed": False,
            "outcome_rewrite_allowed": False,
            "recommended_next_protocol": protocol.value,
            "recommended_next_protocol_authority": "advisory_only",
        }
        body["decision_digest"] = _compute_decision_digest(body)

        persisted, exact_replay = _create_immutable_record(
            case_id=case_id,
            case_fd=case_fd,
            filename="decision.json",
            record=body,
            request_projection=request,
            intent_from_record=_decision_intent_projection,
            validate_existing=lambda body: _validate_decision_record(body, case_id=case_id),
            base_dir=base_dir,
        )
        post_entries = _list_dir_entries(case_fd)
        _validate_directory_state(post_entries, allowed=_DECIDED_ALLOWED, case_id=case_id)
        return _to_decision_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)


def load_reconciliation_case(
    case_id: str,
    *,
    base_dir: Path | None = None,
) -> ReconciliationCaseBundle:
    validate_id(case_id, "reconciliation")
    open_body = _load_validated_open(case_id, base_dir)
    obs_body = _read_optional_record(paths.reconciliation_observation_path(case_id, base_dir))
    dec_body = _read_optional_record(paths.reconciliation_decision_path(case_id, base_dir))
    observation = None
    decision = None
    if obs_body is not None:
        _validate_observation_record(obs_body)
        observation = _to_observation_record(obs_body)
    if dec_body is not None:
        _validate_decision_record(dec_body, case_id=case_id)
        decision = _to_decision_record(dec_body)
    return ReconciliationCaseBundle(
        case_id=case_id,
        open_record=_to_open_record(open_body),
        observation_record=observation,
        decision_record=decision,
    )


__all__ = [
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
]
