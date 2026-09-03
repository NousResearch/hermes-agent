"""Task 26C — approved marker disposition protocol (Path A only)."""

from __future__ import annotations

import fcntl
import json
import os
import stat
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, NoReturn

from htr import paths
from htr.action_plan import _canonical_json, _normalize_path_for_digest, _sha256_digest
from htr.approval_control import (
    OUTCOME_CONSUMED,
    _compute_approval_digest,
    _compute_claim_digest,
    _compute_outcome_digest,
    _project_dir_path_digest,
    _runs_root_path_digest,
)
from htr.execution_lock import (
    LOCKS_DIR_NAME,
    RunExecutionLockIndeterminateError,
    RunExecutionLockPathUnsafeError,
    RunExecutionLockReleaseConflictError,
    acquire_marker_directory_entry_coordination,
    disposition_unlink_marker,
    lock_directory_identity,
    pin_lock_directory,
    read_marker_metadata_at,
    release_marker_directory_entry_coordination,
)
from htr.finalization import SealState, evaluate_run_seal
from htr.ids import (
    IdKind,
    generate_marker_disposition_approval_id,
    generate_marker_disposition_attempt_id,
    generate_marker_disposition_claim_id,
    generate_marker_disposition_id,
    validate_id,
)
from htr.io import read_json
from htr.reconciliation_cases import (
    DECISION_REVALIDATION_PROJECTION_VERSION,
    ReconciliationDecisionClass,
    ReconciliationNextProtocol,
    load_reconciliation_case,
)
from htr.reconciliation_inspection import (
    INSPECTION_DIGEST_PROJECTION_VERSION,
    PILOT_BOUND_API,
    compute_inspection_semantic_digest,
    inspect_run_completion_reconciliation,
)
from htr.state import (
    MarkerDispositionAttemptRecord,
    MarkerDispositionBundle,
    MarkerDispositionClaimRecord,
    MarkerDispositionConflictError,
    MarkerDispositionDurabilityError,
    MarkerDispositionExecutionResult,
    MarkerDispositionIssueRecord,
    MarkerDispositionOutcomeRecord,
    MarkerDispositionReconcileResult,
    MarkerDispositionRecordName,
    MarkerDispositionRequestRecord,
    MarkerDispositionStateError,
    MarkerDispositionValidationError,
    MarkerDispositionWriteMetadata,
    ReconciliationEvidenceIntegrityError,
    ReconciliationInspectionError,
    ReconciliationStateError,
    ReconciliationUnsupportedApprovalError,
    ReconciliationValidationError,
    RunCompletionReconciliationInspection,
)

REQUEST_SCHEMA_VERSION = "1"
ISSUE_SCHEMA_VERSION = "1"
REVOKE_SCHEMA_VERSION = "1"
CLAIM_SCHEMA_VERSION = "1"
ATTEMPT_SCHEMA_VERSION = "1"
OUTCOME_SCHEMA_VERSION = "1"

REQUEST_DIGEST_PROJECTION_VERSION = "htr.marker_disposition.request.digest.v1"
REQUEST_INTENT_PROJECTION_VERSION = "htr.marker_disposition.request.intent.v1"
ISSUE_DIGEST_PROJECTION_VERSION = "htr.marker_disposition.issue.digest.v1"
ISSUE_INTENT_PROJECTION_VERSION = "htr.marker_disposition.issue.intent.v1"
REVOKE_DIGEST_PROJECTION_VERSION = "htr.marker_disposition.revoke.digest.v1"
REVOKE_INTENT_PROJECTION_VERSION = "htr.marker_disposition.revoke.intent.v1"
CLAIM_DIGEST_PROJECTION_VERSION = "htr.marker_disposition.claim.digest.v1"
CLAIM_INTENT_PROJECTION_VERSION = "htr.marker_disposition.claim.intent.v1"
ATTEMPT_DIGEST_PROJECTION_VERSION = "htr.marker_disposition.attempt.digest.v1"
ATTEMPT_INTENT_PROJECTION_VERSION = "htr.marker_disposition.attempt.intent.v1"
OUTCOME_DIGEST_PROJECTION_VERSION = "htr.marker_disposition.outcome.digest.v1"
EXECUTION_REVALIDATION_PROJECTION_VERSION = "htr.marker_disposition.execution_revalidation.digest.v1"

MAX_APPROVAL_LIFETIME = timedelta(minutes=15)

_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_DIRECTORY = os.O_DIRECTORY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

_CONTROL_FILE_MODE = 0o600
_CONTROL_DIR_MODE = 0o700

_REQUEST_ALLOWED = frozenset({"request.json"})
_ISSUED_ALLOWED = frozenset({"request.json", "issue.json"})
_REVOKED_ALLOWED = frozenset({"request.json", "issue.json", "revoke.json"})
_CLAIMED_ALLOWED = frozenset({"request.json", "issue.json", "revoke.json", "claim.json"})
_ATTEMPTED_ALLOWED = _CLAIMED_ALLOWED | {"attempt.json"}
_TERMINAL_ALLOWED = _ATTEMPTED_ALLOWED | {"outcome.json"}

_MAX_ACTOR_LEN = 256

_NON_PERMISSION_BOOLEANS = (
    "safe_to_retry",
    "invoke_allowed",
    "repair_allowed",
    "recovery_run_creation_allowed",
    "outcome_rewrite_allowed",
    "further_marker_disposition_allowed",
)

_PATH_A_DECISION_CLASSES = frozenset(
    {
        ReconciliationDecisionClass.case_closed_no_action_required.value,
        ReconciliationDecisionClass.case_closed_deferred_to_protocol.value,
    }
)


class MarkerDispositionOutcomeClass(str, Enum):
    disposed_verified = "disposed_verified"
    already_absent_observed = "already_absent_observed"
    marker_changed = "marker_changed"
    approval_invalid = "approval_invalid"
    evidence_drifted = "evidence_drifted"
    integrity_blocked = "integrity_blocked"
    unlink_failed = "unlink_failed"
    lock_directory_fsync_failed = "lock_directory_fsync_failed"
    outcome_durability_indeterminate = "outcome_durability_indeterminate"
    execution_ambiguous = "execution_ambiguous"


def _require_valid_id(value: str, kind: IdKind) -> None:
    if not validate_id(value, kind):
        raise MarkerDispositionValidationError(f"invalid {kind} id: {value!r}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_utc_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise MarkerDispositionValidationError(f"timestamp must be timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _validate_actor(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarkerDispositionValidationError(f"{field} must be a non-empty string")
    actor = value.strip()
    if len(actor) > _MAX_ACTOR_LEN:
        raise MarkerDispositionValidationError(f"{field} exceeds {_MAX_ACTOR_LEN} characters")
    if not actor.isprintable():
        raise MarkerDispositionValidationError(f"{field} must be printable")
    return actor


def _validate_expiry(*, issued_at: str, expires_at: str) -> None:
    issued = _parse_utc_iso(issued_at)
    expires = _parse_utc_iso(expires_at)
    if expires <= issued:
        raise MarkerDispositionValidationError("expires_at must be after issued_at")
    if expires - issued > MAX_APPROVAL_LIFETIME:
        raise MarkerDispositionValidationError(
            f"approval lifetime exceeds {MAX_APPROVAL_LIFETIME.total_seconds()} seconds"
        )


def _raise_unsafe_path(context: str, exc: BaseException) -> NoReturn:
    raise MarkerDispositionValidationError(
        f"unsafe marker disposition control path ({context})"
    ) from exc


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


def _fsync_disposition_durability_chain(
    case_fd: int,
    *,
    disposition_id: str,
    record_name: str,
) -> None:
    _fsync_dir_fd(
        case_fd,
        disposition_id=disposition_id,
        record_name=record_name,
        stage="disposition_dir_fsync",
    )
    parent_fd = os.open(
        f"/proc/self/fd/{case_fd}/..",
        _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
    )
    try:
        _fsync_dir_fd(
            parent_fd,
            disposition_id=disposition_id,
            record_name=record_name,
            stage="control_dir_fsync",
        )
        grandparent_fd = os.open(
            f"/proc/self/fd/{parent_fd}/..",
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
        try:
            _fsync_dir_fd(
                grandparent_fd,
                disposition_id=disposition_id,
                record_name=record_name,
                stage="parent_dir_fsync",
            )
        finally:
            os.close(grandparent_fd)
    finally:
        os.close(parent_fd)


def _fsync_dir_fd(
    dir_fd: int,
    *,
    disposition_id: str,
    record_name: MarkerDispositionRecordName,
    stage: str,
) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise MarkerDispositionDurabilityError(
            f"directory fsync failed at {stage}: {exc}",
            disposition_id=disposition_id,
            record_name=record_name,
            durability_stage=stage,  # type: ignore[arg-type]
            record_may_have_committed=True,
            exact_replay_status="indeterminate",
        ) from exc


def _fsync_file_fd(
    file_fd: int,
    *,
    disposition_id: str,
    record_name: MarkerDispositionRecordName,
) -> None:
    try:
        os.fsync(file_fd)
    except OSError as exc:
        raise MarkerDispositionDurabilityError(
            f"file fsync failed: {exc}",
            disposition_id=disposition_id,
            record_name=record_name,
            durability_stage="record_fsync",
            record_may_have_committed=True,
            exact_replay_status="indeterminate",
        ) from exc


def _write_all(
    fd: int,
    payload: bytes,
    *,
    disposition_id: str,
    record_name: MarkerDispositionRecordName,
) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise MarkerDispositionDurabilityError(
                "short write while persisting marker disposition record",
                disposition_id=disposition_id,
                record_name=record_name,
                durability_stage="record_write",
                record_may_have_committed=False,
                exact_replay_status="no",
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
        raise MarkerDispositionValidationError("expected JSON object record")
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
        raise MarkerDispositionValidationError(f"unsafe symlink record: {path}")
    try:
        return read_json(path)
    except json.JSONDecodeError as exc:
        raise MarkerDispositionValidationError(f"malformed JSON record: {path}") from exc


def _validate_record_digest(
    record: dict[str, Any],
    *,
    digest_field: str,
    projection_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    stored = record.get(digest_field)
    if not isinstance(stored, str) or not stored:
        raise MarkerDispositionValidationError(f"missing {digest_field}")
    computed = _sha256_digest(projection_fn(record))
    if stored != computed:
        raise MarkerDispositionValidationError(f"{digest_field} mismatch")


def _validate_non_permission_booleans(record: dict[str, Any]) -> None:
    for field in _NON_PERMISSION_BOOLEANS:
        value = record.get(field)
        if value is not False:
            raise MarkerDispositionValidationError(f"{field} must be false")


def _write_metadata(*, exact_replay: bool) -> MarkerDispositionWriteMetadata:
    return MarkerDispositionWriteMetadata(
        exact_replay=exact_replay,
        exact_replay_status="yes" if exact_replay else "no",
        record_may_have_committed=True,
        durability_indeterminate=False,
    )


def _marker_path_digest(base_dir: Path | None, run_id: str) -> str:
    runs_path = paths.runs_root(base_dir)
    marker_rel = f"{LOCKS_DIR_NAME}/{run_id}.marker"
    normalized = os.path.normpath(os.path.join(str(runs_path), marker_rel))
    return _sha256_digest({"normalized_path": _normalize_path_for_digest(normalized)})


def _marker_metadata_digest(metadata: dict[str, Any]) -> str:
    return _sha256_digest({"marker_metadata": metadata})


def _seal_evidence(run_id: str, base_dir: Path | None) -> dict[str, Any]:
    evaluation = evaluate_run_seal(run_id, base_dir)
    return {
        "seal_state": evaluation.state.value,
        "reason_codes": list(evaluation.reason_codes),
    }


def _build_inspection_projection(result: RunCompletionReconciliationInspection) -> dict[str, Any]:
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


def _load_task25_lineage(
    approval_id: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    issue_path = paths.approval_issue_path(approval_id, base_dir)
    claim_path = paths.approval_claim_path(approval_id, base_dir)
    outcome_path = paths.approval_outcome_path(approval_id, base_dir)
    issue = _read_optional_record(issue_path)
    claim = _read_optional_record(claim_path)
    outcome = _read_optional_record(outcome_path)
    if issue is None:
        raise MarkerDispositionValidationError("Task 25 approval issue missing")
    if claim is None:
        raise MarkerDispositionValidationError("Task 25 approval claim missing")
    if outcome is None:
        raise MarkerDispositionValidationError("Task 25 approval outcome missing")
    if issue.get("approval_digest") != _compute_approval_digest(issue):
        raise MarkerDispositionValidationError("Task 25 issue digest mismatch")
    if claim.get("claim_digest") != _compute_claim_digest(claim):
        raise MarkerDispositionValidationError("Task 25 claim digest mismatch")
    if outcome.get("outcome_class") != OUTCOME_CONSUMED:
        raise MarkerDispositionValidationError("Task 25 outcome is not consumed")
    outcome_digest = _compute_outcome_digest(outcome)
    if outcome.get("outcome_digest") != outcome_digest:
        raise MarkerDispositionValidationError("Task 25 outcome digest mismatch")
    return {
        "approval_id": approval_id,
        "approval_issue_digest": issue["approval_digest"],
        "claim_id": claim["claim_id"],
        "claim_digest": claim["claim_digest"],
        "consumed_outcome_digest": outcome_digest,
    }


def _load_decision_raw(case_id: str, base_dir: Path | None) -> dict[str, Any]:
    path = paths.reconciliation_decision_path(case_id, base_dir)
    record = _read_optional_record(path)
    if record is None:
        raise MarkerDispositionValidationError("reconciliation decision missing")
    return record


def _fresh_inspection(approval_id: str, base_dir: Path | None) -> RunCompletionReconciliationInspection:
    try:
        return inspect_run_completion_reconciliation(approval_id, base_dir=base_dir)
    except (
        ReconciliationInspectionError,
        ReconciliationEvidenceIntegrityError,
        ReconciliationUnsupportedApprovalError,
    ):
        raise
    except Exception as exc:
        raise ReconciliationInspectionError(str(exc), approval_id=approval_id) from exc


def _validate_path_a_inspection_projection(projection: dict[str, Any]) -> None:
    if projection.get("lifecycle_evidence_state") != "verified_completed":
        raise MarkerDispositionValidationError("lifecycle state is not verified_completed")
    if projection.get("overall_classification") != "verified_completion_marker_residue":
        raise MarkerDispositionValidationError("overall classification mismatch for Path A")
    if projection.get("marker_state") != "present_valid_metadata":
        raise MarkerDispositionValidationError("marker state is not present_valid_metadata")
    if projection.get("approval_control_state") != "consumed_outcome":
        raise MarkerDispositionValidationError("approval control state is not consumed_outcome")
    if projection.get("outcome_class") != OUTCOME_CONSUMED:
        raise MarkerDispositionValidationError("outcome class is not consumed")


def _prove_path_a_eligibility(
    *,
    case_id: str,
    base_dir: Path | None,
    require_seal: bool = True,
) -> dict[str, Any]:
    bundle = load_reconciliation_case(case_id, base_dir=base_dir)
    if bundle.observation_record is None or bundle.decision_record is None:
        raise MarkerDispositionValidationError("reconciliation case incomplete for Path A")
    decision_raw = _load_decision_raw(case_id, base_dir)
    decision_class = decision_raw.get("decision_class")
    if decision_class not in _PATH_A_DECISION_CLASSES:
        raise MarkerDispositionValidationError("decision class not eligible for marker disposition")
    protocol = decision_raw.get("recommended_next_protocol")
    if protocol != ReconciliationNextProtocol.marker_disposition_review.value:
        raise MarkerDispositionValidationError("recommended_next_protocol must be marker_disposition_review")

    open_record = bundle.open_record
    approval_id = open_record.approval_id

    revalidation = decision_raw.get("decision_time_revalidation") or {}
    stored_rev = revalidation.get("decision_revalidation_record_digest")
    rev_projection = revalidation.get("inspection_semantic_projection") or {}
    if isinstance(stored_rev, str) and stored_rev:
        envelope = {
            "revalidation_projection_version": revalidation.get("revalidation_projection_version"),
            "case_id": revalidation.get("case_id"),
            "case_open_digest": revalidation.get("case_open_digest"),
            "observation_digest": revalidation.get("observation_digest"),
            "inspection_semantic_projection": rev_projection,
        }
        computed_rev = _sha256_digest(
            {
                "revalidation_projection_version": envelope["revalidation_projection_version"],
                "case_id": envelope["case_id"],
                "case_open_digest": envelope["case_open_digest"],
                "observation_digest": envelope["observation_digest"],
                "inspection_semantic_projection": envelope["inspection_semantic_projection"],
            }
        )
        if stored_rev != computed_rev:
            raise MarkerDispositionValidationError("decision revalidation digest mismatch")

    seal = _seal_evidence(open_record.run_id, base_dir)
    seal_finalized = seal["seal_state"] == SealState.FINALIZED_VALID.value
    if require_seal and not seal_finalized:
        raise MarkerDispositionValidationError("run seal is not finalized_valid")

    obs_digest = bundle.observation_record.inspection_semantic_digest

    if seal_finalized:
        if not rev_projection:
            raise MarkerDispositionValidationError("decision-time revalidation projection missing")
        _validate_path_a_inspection_projection(rev_projection)
        drift = decision_raw.get("observation_decision_drift") or {}
        decision_inspection_digest = drift.get("inspection_semantic_digest_at_decision")
        if not isinstance(decision_inspection_digest, str) or not decision_inspection_digest:
            raise MarkerDispositionValidationError("decision-time inspection digest missing")
        execution_projection = rev_projection
        execution_inspection_digest = decision_inspection_digest
    else:
        inspection = _fresh_inspection(approval_id, base_dir)
        projection = _build_inspection_projection(inspection)
        inspection_digest = compute_inspection_semantic_digest(inspection)
        if inspection_digest != inspection.inspection_semantic_digest:
            raise MarkerDispositionValidationError("inspection semantic digest mismatch")
        _validate_path_a_inspection_projection(projection)
        if obs_digest != inspection_digest:
            raise MarkerDispositionValidationError("Task 26A observation inspection drift detected")
        if rev_projection != projection:
            raise MarkerDispositionValidationError("Task 26B decision-time revalidation drift detected")
        execution_projection = projection
        execution_inspection_digest = inspection_digest

    task25 = _load_task25_lineage(approval_id, base_dir)
    runs_root_fd, lock_root_fd = pin_lock_directory(base_dir)
    try:
        metadata, marker_identity = read_marker_metadata_at(lock_root_fd, open_record.run_id)
    except RunExecutionLockIndeterminateError as exc:
        if "not present" in str(exc).lower():
            raise MarkerDispositionValidationError("marker not present at request time") from exc
        raise MarkerDispositionValidationError(str(exc)) from exc
    finally:
        os.close(lock_root_fd)
        os.close(runs_root_fd)

    acquisition_id = metadata.get("acquisition_id")
    if not isinstance(acquisition_id, str) or not acquisition_id:
        raise MarkerDispositionValidationError("marker acquisition_id missing")
    if metadata.get("run_id") != open_record.run_id:
        raise MarkerDispositionValidationError("marker run_id mismatch")

    lock_dir_identity = None
    runs_root_fd2, lock_root_fd2 = pin_lock_directory(base_dir)
    try:
        lock_dir_identity = lock_directory_identity(lock_root_fd2)
    finally:
        os.close(lock_root_fd2)
        os.close(runs_root_fd2)

    return {
        "case_id": case_id,
        "case_open_digest": open_record.case_open_digest,
        "observation_digest": bundle.observation_record.observation_digest,
        "decision_digest": bundle.decision_record.decision_digest,
        "decision_class": decision_class,
        "recommended_next_protocol": protocol,
        "task26a_observation_inspection_digest": obs_digest,
        "task26b_decision_revalidation_inspection_digest": execution_inspection_digest,
        "approval_id": approval_id,
        "run_id": open_record.run_id,
        "event_id": open_record.event_id,
        "htr_runs_root_path_digest": open_record.htr_runs_root_path_digest,
        "htr_project_dir_path_digest": open_record.htr_project_dir_path_digest,
        "canonical_marker_path_digest": _marker_path_digest(base_dir, open_record.run_id),
        "marker_run_id": open_record.run_id,
        "marker_acquisition_id": acquisition_id,
        "marker_metadata_digest": _marker_metadata_digest(metadata),
        "request_time_marker_device": marker_identity[0],
        "request_time_marker_inode": marker_identity[1],
        "finalized_valid_seal_evidence": seal,
        "task25": task25,
        "execution_inspection_projection": execution_projection,
        "execution_inspection_digest": execution_inspection_digest,
        "lock_directory_device": lock_dir_identity[0] if lock_dir_identity else None,
        "lock_directory_inode": lock_dir_identity[1] if lock_dir_identity else None,
    }


def _prove_path_a_execution_eligibility(
    *,
    request: dict[str, Any],
    base_dir: Path | None,
) -> dict[str, Any]:
    """Execution-time Path A proof: bind request digests; marker absence handled under flock."""
    case_id = request["reconciliation_case_id"]
    bundle = load_reconciliation_case(case_id, base_dir=base_dir)
    if bundle.observation_record is None or bundle.decision_record is None:
        raise MarkerDispositionValidationError("reconciliation case incomplete for Path A")

    open_record = bundle.open_record
    if open_record.case_open_digest != request["case_open_digest"]:
        raise MarkerDispositionValidationError("case_open_digest drift detected")
    if bundle.observation_record.observation_digest != request["observation_digest"]:
        raise MarkerDispositionValidationError("observation_digest drift detected")
    if bundle.decision_record.decision_digest != request["decision_digest"]:
        raise MarkerDispositionValidationError("decision_digest drift detected")

    decision_raw = _load_decision_raw(case_id, base_dir)
    decision_class = decision_raw.get("decision_class")
    if decision_class not in _PATH_A_DECISION_CLASSES:
        raise MarkerDispositionValidationError("decision class not eligible for marker disposition")
    protocol = decision_raw.get("recommended_next_protocol")
    if protocol != ReconciliationNextProtocol.marker_disposition_review.value:
        raise MarkerDispositionValidationError("recommended_next_protocol must be marker_disposition_review")

    seal = _seal_evidence(open_record.run_id, base_dir)
    if seal["seal_state"] != SealState.FINALIZED_VALID.value:
        raise MarkerDispositionValidationError("run seal is not finalized_valid")
    stored_seal = request.get("finalized_valid_seal_evidence") or {}
    if stored_seal.get("seal_state") != SealState.FINALIZED_VALID.value:
        raise MarkerDispositionValidationError("request seal evidence not finalized_valid")

    task25 = _load_task25_lineage(request["task25_approval_id"], base_dir)
    if task25["approval_issue_digest"] != request["task25_approval_issue_digest"]:
        raise MarkerDispositionValidationError("Task 25 issue digest drift detected")
    if task25["claim_id"] != request["task25_claim_id"]:
        raise MarkerDispositionValidationError("Task 25 claim id drift detected")
    if task25["claim_digest"] != request["task25_claim_digest"]:
        raise MarkerDispositionValidationError("Task 25 claim digest drift detected")
    if task25["consumed_outcome_digest"] != request["task25_consumed_outcome_digest"]:
        raise MarkerDispositionValidationError("Task 25 outcome digest drift detected")

    runs_root_fd, lock_root_fd = pin_lock_directory(base_dir)
    try:
        lock_dir_identity = lock_directory_identity(lock_root_fd)
    finally:
        os.close(lock_root_fd)
        os.close(runs_root_fd)

    revalidation = decision_raw.get("decision_time_revalidation") or {}
    rev_projection = revalidation.get("inspection_semantic_projection") or {}
    if request["task26a_observation_inspection_digest"] != bundle.observation_record.inspection_semantic_digest:
        raise MarkerDispositionValidationError("Task 26A observation inspection drift detected")

    return {
        "case_id": case_id,
        "run_id": open_record.run_id,
        "execution_inspection_projection": rev_projection,
        "execution_inspection_digest": request["task26b_decision_revalidation_inspection_digest"],
        "lock_directory_device": lock_dir_identity[0] if lock_dir_identity else None,
        "lock_directory_inode": lock_dir_identity[1] if lock_dir_identity else None,
    }


def _request_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_intent_projection_version": REQUEST_INTENT_PROJECTION_VERSION,
        "disposition_id": record["disposition_id"],
        "reconciliation_case_id": record["reconciliation_case_id"],
        "requested_by": record["requested_by"],
        "run_id": record["run_id"],
        "event_id": record["event_id"],
        "decision_class": record["decision_class"],
        "recommended_next_protocol": record["recommended_next_protocol"],
        "marker_acquisition_id": record["marker_acquisition_id"],
    }


def _request_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_schema_version": body["request_schema_version"],
        "request_digest_projection_version": body["request_digest_projection_version"],
        "disposition_id": body["disposition_id"],
        "reconciliation_case_id": body["reconciliation_case_id"],
        "case_open_digest": body["case_open_digest"],
        "observation_digest": body["observation_digest"],
        "decision_digest": body["decision_digest"],
        "decision_class": body["decision_class"],
        "recommended_next_protocol": body["recommended_next_protocol"],
        "task26a_observation_inspection_digest": body["task26a_observation_inspection_digest"],
        "task26b_decision_revalidation_inspection_digest": body[
            "task26b_decision_revalidation_inspection_digest"
        ],
        "task25_approval_id": body["task25_approval_id"],
        "task25_approval_issue_digest": body["task25_approval_issue_digest"],
        "task25_claim_id": body["task25_claim_id"],
        "task25_claim_digest": body["task25_claim_digest"],
        "task25_consumed_outcome_digest": body["task25_consumed_outcome_digest"],
        "run_id": body["run_id"],
        "event_id": body["event_id"],
        "htr_runs_root_path_digest": body["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": body["htr_project_dir_path_digest"],
        "canonical_marker_path_digest": body["canonical_marker_path_digest"],
        "marker_run_id": body["marker_run_id"],
        "marker_acquisition_id": body["marker_acquisition_id"],
        "marker_metadata_digest": body["marker_metadata_digest"],
        "request_time_marker_device": body["request_time_marker_device"],
        "request_time_marker_inode": body["request_time_marker_inode"],
        "finalized_valid_seal_evidence": body["finalized_valid_seal_evidence"],
        "requested_by": body["requested_by"],
        "requested_at": body["requested_at"],
    }


def _issue_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_intent_projection_version": ISSUE_INTENT_PROJECTION_VERSION,
        "disposition_id": record["disposition_id"],
        "disposition_approval_id": record["disposition_approval_id"],
        "issued_by": record["issued_by"],
        "expires_at": record["expires_at"],
    }


def _issue_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_schema_version": body["issue_schema_version"],
        "issue_digest_projection_version": body["issue_digest_projection_version"],
        "disposition_id": body["disposition_id"],
        "disposition_approval_id": body["disposition_approval_id"],
        "request_digest": body["request_digest"],
        "issued_by": body["issued_by"],
        "issued_at": body["issued_at"],
        "expires_at": body["expires_at"],
    }


def _revoke_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "revoke_intent_projection_version": REVOKE_INTENT_PROJECTION_VERSION,
        "disposition_id": record["disposition_id"],
        "revoked_by": record["revoked_by"],
    }


def _revoke_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "revoke_schema_version": body["revoke_schema_version"],
        "revoke_digest_projection_version": body["revoke_digest_projection_version"],
        "disposition_id": body["disposition_id"],
        "issue_digest": body["issue_digest"],
        "request_digest": body["request_digest"],
        "revoked_by": body["revoked_by"],
        "revoked_at": body["revoked_at"],
        "reason": body["reason"],
    }


def _claim_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_intent_projection_version": CLAIM_INTENT_PROJECTION_VERSION,
        "disposition_id": record["disposition_id"],
        "claim_id": record["claim_id"],
        "claimant": record["claimant"],
    }


def _claim_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_schema_version": body["claim_schema_version"],
        "claim_digest_projection_version": body["claim_digest_projection_version"],
        "disposition_id": body["disposition_id"],
        "disposition_approval_id": body["disposition_approval_id"],
        "issue_digest": body["issue_digest"],
        "request_digest": body["request_digest"],
        "claim_id": body["claim_id"],
        "claimant": body["claimant"],
        "claimed_at": body["claimed_at"],
    }


def _attempt_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_intent_projection_version": ATTEMPT_INTENT_PROJECTION_VERSION,
        "disposition_id": record["disposition_id"],
        "attempt_id": record["attempt_id"],
        "executor": record["executor"],
    }


def _attempt_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_schema_version": body["attempt_schema_version"],
        "attempt_digest_projection_version": body["attempt_digest_projection_version"],
        "disposition_id": body["disposition_id"],
        "request_digest": body["request_digest"],
        "issue_digest": body["issue_digest"],
        "claim_digest": body["claim_digest"],
        "attempt_id": body["attempt_id"],
        "executor": body["executor"],
        "attempted_at": body["attempted_at"],
        "execution_revalidation": body["execution_revalidation"],
        "marker_device": body["marker_device"],
        "marker_inode": body["marker_inode"],
        "lock_directory_device": body["lock_directory_device"],
        "lock_directory_inode": body["lock_directory_inode"],
    }


def _outcome_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_schema_version": body["outcome_schema_version"],
        "outcome_digest_projection_version": body["outcome_digest_projection_version"],
        "disposition_id": body["disposition_id"],
        "outcome_class": body["outcome_class"],
        "request_digest": body["request_digest"],
        "issue_digest": body["issue_digest"],
        "claim_digest": body["claim_digest"],
        "attempt_digest": body["attempt_digest"],
        "recorded_at": body["recorded_at"],
        "marker_removed_by_this_execution": body["marker_removed_by_this_execution"],
        "lock_directory_fsync_succeeded": body["lock_directory_fsync_succeeded"],
        "safe_to_retry": body["safe_to_retry"],
        "invoke_allowed": body["invoke_allowed"],
        "repair_allowed": body["repair_allowed"],
        "recovery_run_creation_allowed": body["recovery_run_creation_allowed"],
        "outcome_rewrite_allowed": body["outcome_rewrite_allowed"],
        "further_marker_disposition_allowed": body["further_marker_disposition_allowed"],
    }


def _bootstrap_disposition_tree(
    disposition_id: str,
    base_dir: Path | None,
) -> tuple[int, int, tuple[int, int]]:
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
            created_dispositions = _mkdirat_control(
                control_fd,
                paths.MARKER_DISPOSITIONS_DIR_NAME,
                _CONTROL_DIR_MODE,
                context="runs_root/.control",
            )
            dispositions_fd = _openat_control_dir_no_follow(
                control_fd,
                paths.MARKER_DISPOSITIONS_DIR_NAME,
                context="runs_root/.control/marker_dispositions",
            )
            try:
                created_case = _mkdirat_control(
                    dispositions_fd,
                    disposition_id,
                    _CONTROL_DIR_MODE,
                    context=f"marker_dispositions/{disposition_id}",
                )
                case_fd = _openat_control_dir_no_follow(
                    dispositions_fd,
                    disposition_id,
                    context=f"marker_dispositions/{disposition_id}",
                )
                case_identity = _fstat_identity(case_fd)
                if created_case:
                    _fsync_dir_fd(
                        dispositions_fd,
                        disposition_id=disposition_id,
                        record_name="request.json",
                        stage="disposition_dir_fsync",
                    )
                if created_dispositions:
                    _fsync_dir_fd(
                        control_fd,
                        disposition_id=disposition_id,
                        record_name="request.json",
                        stage="control_dir_fsync",
                    )
                if created_control:
                    _fsync_dir_fd(
                        runs_fd,
                        disposition_id=disposition_id,
                        record_name="request.json",
                        stage="parent_dir_fsync",
                    )
                return dispositions_fd, case_fd, case_identity
            except Exception:
                os.close(dispositions_fd)
                raise
        finally:
            os.close(control_fd)
    finally:
        os.close(runs_fd)
    raise MarkerDispositionStateError("unreachable bootstrap failure", disposition_id=disposition_id)


def _create_immutable_record(
    *,
    disposition_id: str,
    case_fd: int,
    filename: MarkerDispositionRecordName,
    record: dict[str, Any] | None = None,
    record_factory: Callable[[], dict[str, Any]] | None = None,
    request_projection: dict[str, Any] | Callable[[], dict[str, Any]],
    intent_from_record: Callable[[dict[str, Any]], dict[str, Any]],
    validate_existing: Callable[[dict[str, Any]], None] | None = None,
    actor_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    if record is None and record_factory is None:
        raise MarkerDispositionValidationError("record or record_factory required")
    if record is not None and record_factory is not None:
        raise MarkerDispositionValidationError("record and record_factory are mutually exclusive")

    def _resolved_projection() -> dict[str, Any]:
        return request_projection() if callable(request_projection) else request_projection

    flags = _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
    flags_read = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    record_ctx = f"marker_dispositions/{disposition_id}/{filename}"
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
                try:
                    existing = _read_json_fd(existing_fd)
                except json.JSONDecodeError as exc:
                    raise MarkerDispositionValidationError(
                        f"{filename} exists but is not complete valid JSON"
                    ) from exc
            finally:
                os.close(existing_fd)
            if actor_fields is not None:
                for field, expected in actor_fields.items():
                    if existing.get(field) != expected:
                        raise MarkerDispositionConflictError(
                            f"{filename} already exists with conflicting semantics",
                            disposition_id=disposition_id,
                        )
            if validate_existing is not None:
                validate_existing(existing)
            projection = _resolved_projection()
            if intent_from_record(existing) != projection:
                raise MarkerDispositionConflictError(
                    f"{filename} already exists with conflicting semantics",
                    disposition_id=disposition_id,
                )
            return existing, True
        record_body = record if record is not None else record_factory()
        payload = (json.dumps(record_body, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            _write_all(file_fd, payload, disposition_id=disposition_id, record_name=filename)
            _fsync_file_fd(file_fd, disposition_id=disposition_id, record_name=filename)
            _fsync_disposition_durability_chain(
                case_fd,
                disposition_id=disposition_id,
                record_name=filename,
            )
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
        return record_body, False
    finally:
        if file_fd is not None:
            os.close(file_fd)
        fcntl.flock(case_fd, fcntl.LOCK_UN)


def _validate_directory_state(
    entries: frozenset[str],
    *,
    allowed: frozenset[str],
    disposition_id: str,
) -> None:
    if entries <= allowed:
        return
    raise MarkerDispositionStateError(
        f"unexpected marker disposition directory entries: {sorted(entries)}",
        disposition_id=disposition_id,
    )


def _load_validated_request(disposition_id: str, base_dir: Path | None) -> dict[str, Any]:
    record = _read_optional_record(paths.marker_disposition_request_path(disposition_id, base_dir))
    if record is None:
        raise MarkerDispositionStateError(
            f"disposition {disposition_id!r} is missing request.json",
            disposition_id=disposition_id,
        )
    _validate_record_digest(
        record,
        digest_field="request_digest",
        projection_fn=_request_digest_projection,
    )
    return record


def _to_request_record(body: dict[str, Any]) -> MarkerDispositionRequestRecord:
    return MarkerDispositionRequestRecord(
        disposition_id=body["disposition_id"],
        request_digest=body["request_digest"],
        reconciliation_case_id=body["reconciliation_case_id"],
        run_id=body["run_id"],
        requested_by=body["requested_by"],
        requested_at=body["requested_at"],
    )


def _to_issue_record(body: dict[str, Any]) -> MarkerDispositionIssueRecord:
    return MarkerDispositionIssueRecord(
        disposition_id=body["disposition_id"],
        disposition_approval_id=body["disposition_approval_id"],
        issue_digest=body["issue_digest"],
        issued_by=body["issued_by"],
        issued_at=body["issued_at"],
        expires_at=body["expires_at"],
    )


def _to_claim_record(body: dict[str, Any]) -> MarkerDispositionClaimRecord:
    return MarkerDispositionClaimRecord(
        disposition_id=body["disposition_id"],
        claim_id=body["claim_id"],
        claim_digest=body["claim_digest"],
        claimant=body["claimant"],
        claimed_at=body["claimed_at"],
    )


def _to_attempt_record(body: dict[str, Any]) -> MarkerDispositionAttemptRecord:
    return MarkerDispositionAttemptRecord(
        disposition_id=body["disposition_id"],
        attempt_id=body["attempt_id"],
        attempt_digest=body["attempt_digest"],
        executor=body["executor"],
        attempted_at=body["attempted_at"],
    )


def _to_outcome_record(body: dict[str, Any]) -> MarkerDispositionOutcomeRecord:
    return MarkerDispositionOutcomeRecord(
        disposition_id=body["disposition_id"],
        outcome_class=body["outcome_class"],
        outcome_digest=body["outcome_digest"],
        recorded_at=body["recorded_at"],
    )


def _assert_approval_active(
    *,
    issue: dict[str, Any],
    revoke: dict[str, Any] | None,
    claim: dict[str, Any] | None,
    now: datetime | None = None,
) -> None:
    now = now or _utc_now()
    if revoke is not None and claim is None:
        raise MarkerDispositionValidationError("disposition approval revoked before claim")
    expires = _parse_utc_iso(issue["expires_at"])
    if now >= expires:
        raise MarkerDispositionValidationError("disposition approval expired")


def _outcome_body(
    *,
    disposition_id: str,
    outcome_class: MarkerDispositionOutcomeClass,
    request_digest: str,
    issue_digest: str,
    claim_digest: str,
    attempt_digest: str | None,
    marker_removed_by_this_execution: bool,
    lock_directory_fsync_succeeded: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "outcome_digest_projection_version": OUTCOME_DIGEST_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "outcome_class": outcome_class.value,
        "request_digest": request_digest,
        "issue_digest": issue_digest,
        "claim_digest": claim_digest,
        "attempt_digest": attempt_digest,
        "recorded_at": _utc_now_iso(),
        "marker_removed_by_this_execution": marker_removed_by_this_execution,
        "lock_directory_fsync_succeeded": lock_directory_fsync_succeeded,
        "safe_to_retry": False,
        "invoke_allowed": False,
        "repair_allowed": False,
        "recovery_run_creation_allowed": False,
        "outcome_rewrite_allowed": False,
        "further_marker_disposition_allowed": False,
    }
    body["outcome_digest"] = _sha256_digest(_outcome_digest_projection(body))
    return body


def _persist_outcome_private(
    *,
    disposition_id: str,
    case_fd: int,
    body: dict[str, Any],
    intent_projection: dict[str, Any],
) -> dict[str, Any]:
    def _validate(body: dict[str, Any]) -> None:
        _validate_record_digest(
            body,
            digest_field="outcome_digest",
            projection_fn=_outcome_digest_projection,
        )
        _validate_non_permission_booleans(body)

    persisted, exact_replay = _create_immutable_record(
        disposition_id=disposition_id,
        case_fd=case_fd,
        filename="outcome.json",
        record=body,
        request_projection=intent_projection,
        intent_from_record=lambda rec: {
            "outcome_class": rec["outcome_class"],
            "request_digest": rec["request_digest"],
            "issue_digest": rec["issue_digest"],
            "claim_digest": rec["claim_digest"],
            "attempt_digest": rec["attempt_digest"],
        },
        validate_existing=_validate,
    )
    return persisted


def create_marker_disposition_request(
    disposition_id: str,
    reconciliation_case_id: str,
    *,
    requested_by: str,
    base_dir: Path | None = None,
) -> tuple[MarkerDispositionRequestRecord, MarkerDispositionWriteMetadata]:
    _require_valid_id(disposition_id, "marker_disposition")
    _require_valid_id(reconciliation_case_id, "reconciliation")
    requested_by = _validate_actor(requested_by, field="requested_by")

    proof_cache: dict[str, Any] = {}

    def _proof_bundle() -> dict[str, Any]:
        if "proof" not in proof_cache:
            proof_cache["proof"] = _prove_path_a_eligibility(
                case_id=reconciliation_case_id, base_dir=base_dir
            )
        return proof_cache["proof"]

    def _request_intent() -> dict[str, Any]:
        proof = _proof_bundle()
        return {
            "request_intent_projection_version": REQUEST_INTENT_PROJECTION_VERSION,
            "disposition_id": disposition_id,
            "reconciliation_case_id": reconciliation_case_id,
            "requested_by": requested_by,
            "run_id": proof["run_id"],
            "event_id": proof["event_id"],
            "decision_class": proof["decision_class"],
            "recommended_next_protocol": proof["recommended_next_protocol"],
            "marker_acquisition_id": proof["marker_acquisition_id"],
        }

    def _request_body() -> dict[str, Any]:
        proof = _proof_bundle()
        task25 = proof["task25"]
        requested_at = _utc_now_iso()
        body: dict[str, Any] = {
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "request_digest_projection_version": REQUEST_DIGEST_PROJECTION_VERSION,
            "disposition_id": disposition_id,
            "reconciliation_case_id": reconciliation_case_id,
            "case_open_digest": proof["case_open_digest"],
            "observation_digest": proof["observation_digest"],
            "decision_digest": proof["decision_digest"],
            "decision_class": proof["decision_class"],
            "recommended_next_protocol": proof["recommended_next_protocol"],
            "task26a_observation_inspection_digest": proof["task26a_observation_inspection_digest"],
            "task26b_decision_revalidation_inspection_digest": proof[
                "task26b_decision_revalidation_inspection_digest"
            ],
            "task25_approval_id": task25["approval_id"],
            "task25_approval_issue_digest": task25["approval_issue_digest"],
            "task25_claim_id": task25["claim_id"],
            "task25_claim_digest": task25["claim_digest"],
            "task25_consumed_outcome_digest": task25["consumed_outcome_digest"],
            "run_id": proof["run_id"],
            "event_id": proof["event_id"],
            "htr_runs_root_path_digest": proof["htr_runs_root_path_digest"],
            "htr_project_dir_path_digest": proof["htr_project_dir_path_digest"],
            "canonical_marker_path_digest": proof["canonical_marker_path_digest"],
            "marker_run_id": proof["marker_run_id"],
            "marker_acquisition_id": proof["marker_acquisition_id"],
            "marker_metadata_digest": proof["marker_metadata_digest"],
            "request_time_marker_device": proof["request_time_marker_device"],
            "request_time_marker_inode": proof["request_time_marker_inode"],
            "finalized_valid_seal_evidence": proof["finalized_valid_seal_evidence"],
            "requested_by": requested_by,
            "requested_at": requested_at,
        }
        body["request_digest"] = _sha256_digest(_request_digest_projection(body))
        return body

    _dispositions_fd, case_fd, _identity = _bootstrap_disposition_tree(disposition_id, base_dir)
    try:
        persisted, exact_replay = _create_immutable_record(
            disposition_id=disposition_id,
            case_fd=case_fd,
            filename="request.json",
            record_factory=_request_body,
            request_projection=_request_intent,
            actor_fields={"requested_by": requested_by},
            intent_from_record=_request_intent_projection,
            validate_existing=lambda rec: _validate_record_digest(
                rec,
                digest_field="request_digest",
                projection_fn=_request_digest_projection,
            ),
        )
        post = _list_dir_entries(case_fd)
        _validate_directory_state(post, allowed=_REQUEST_ALLOWED, disposition_id=disposition_id)
        return _to_request_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)
        os.close(_dispositions_fd)


def issue_marker_disposition_approval(
    disposition_id: str,
    disposition_approval_id: str,
    *,
    issued_by: str,
    expires_at: str,
    base_dir: Path | None = None,
) -> tuple[MarkerDispositionIssueRecord, MarkerDispositionWriteMetadata]:
    _require_valid_id(disposition_id, "marker_disposition")
    _require_valid_id(disposition_approval_id, "marker_disposition_approval")
    issued_by = _validate_actor(issued_by, field="issued_by")

    request = _load_validated_request(disposition_id, base_dir)
    issued_at = _utc_now_iso()
    _validate_expiry(issued_at=issued_at, expires_at=expires_at)

    intent = {
        "issue_intent_projection_version": ISSUE_INTENT_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "disposition_approval_id": disposition_approval_id,
        "issued_by": issued_by,
        "expires_at": expires_at,
    }

    replay_issued_at = issued_at
    body: dict[str, Any] = {
        "issue_schema_version": ISSUE_SCHEMA_VERSION,
        "issue_digest_projection_version": ISSUE_DIGEST_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "disposition_approval_id": disposition_approval_id,
        "request_digest": request["request_digest"],
        "issued_by": issued_by,
        "issued_at": replay_issued_at,
        "expires_at": expires_at,
    }
    body["issue_digest"] = _sha256_digest(_issue_digest_projection(body))

    case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"marker_dispositions/{disposition_id}")
    try:
        persisted, exact_replay = _create_immutable_record(
            disposition_id=disposition_id,
            case_fd=case_fd,
            filename="issue.json",
            record=body,
            request_projection=intent,
            actor_fields={"issued_by": issued_by},
            intent_from_record=_issue_intent_projection,
            validate_existing=lambda rec: _validate_record_digest(
                rec,
                digest_field="issue_digest",
                projection_fn=_issue_digest_projection,
            ),
        )
        post = _list_dir_entries(case_fd)
        _validate_directory_state(post, allowed=_ISSUED_ALLOWED, disposition_id=disposition_id)
        return _to_issue_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)


def revoke_marker_disposition_approval(
    disposition_id: str,
    *,
    revoked_by: str,
    reason: str,
    base_dir: Path | None = None,
) -> tuple[dict[str, Any], MarkerDispositionWriteMetadata]:
    _require_valid_id(disposition_id, "marker_disposition")
    revoked_by = _validate_actor(revoked_by, field="revoked_by")
    if not isinstance(reason, str) or not reason.strip():
        raise MarkerDispositionValidationError("revoke reason must be a non-empty string")

    request = _load_validated_request(disposition_id, base_dir)
    issue = _read_optional_record(paths.marker_disposition_issue_path(disposition_id, base_dir))
    if issue is None:
        raise MarkerDispositionStateError("issue.json missing", disposition_id=disposition_id)
    _validate_record_digest(issue, digest_field="issue_digest", projection_fn=_issue_digest_projection)

    claim = _read_optional_record(paths.marker_disposition_claim_path(disposition_id, base_dir))
    if claim is not None:
        raise MarkerDispositionConflictError(
            "revoke after claim is conflicting/ineffective",
            disposition_id=disposition_id,
        )

    intent = {
        "revoke_intent_projection_version": REVOKE_INTENT_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "revoked_by": revoked_by,
    }

    revoked_at = _utc_now_iso()
    body: dict[str, Any] = {
        "revoke_schema_version": REVOKE_SCHEMA_VERSION,
        "revoke_digest_projection_version": REVOKE_DIGEST_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "issue_digest": issue["issue_digest"],
        "request_digest": request["request_digest"],
        "revoked_by": revoked_by,
        "revoked_at": revoked_at,
        "reason": reason.strip(),
    }
    body["revoke_digest"] = _sha256_digest(_revoke_digest_projection(body))

    case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"marker_dispositions/{disposition_id}")
    try:
        persisted, exact_replay = _create_immutable_record(
            disposition_id=disposition_id,
            case_fd=case_fd,
            filename="revoke.json",
            record=body,
            request_projection=intent,
            actor_fields={"revoked_by": revoked_by},
            intent_from_record=_revoke_intent_projection,
            validate_existing=lambda rec: _validate_record_digest(
                rec,
                digest_field="revoke_digest",
                projection_fn=_revoke_digest_projection,
            ),
        )
        post = _list_dir_entries(case_fd)
        _validate_directory_state(post, allowed=_REVOKED_ALLOWED, disposition_id=disposition_id)
        return persisted, _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)


def claim_marker_disposition_approval(
    disposition_id: str,
    claim_id: str,
    *,
    claimant: str,
    base_dir: Path | None = None,
) -> tuple[MarkerDispositionClaimRecord, MarkerDispositionWriteMetadata]:
    _require_valid_id(disposition_id, "marker_disposition")
    _require_valid_id(claim_id, "marker_disposition_claim")
    claimant = _validate_actor(claimant, field="claimant")

    request = _load_validated_request(disposition_id, base_dir)
    issue = _read_optional_record(paths.marker_disposition_issue_path(disposition_id, base_dir))
    if issue is None:
        raise MarkerDispositionStateError("issue.json missing", disposition_id=disposition_id)
    _validate_record_digest(issue, digest_field="issue_digest", projection_fn=_issue_digest_projection)
    revoke = _read_optional_record(paths.marker_disposition_revoke_path(disposition_id, base_dir))
    if revoke is not None:
        raise MarkerDispositionValidationError("cannot claim revoked disposition approval")

    _assert_approval_active(issue=issue, revoke=revoke, claim=None)

    intent = {
        "claim_intent_projection_version": CLAIM_INTENT_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "claim_id": claim_id,
        "claimant": claimant,
    }

    claimed_at = _utc_now_iso()
    body: dict[str, Any] = {
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "claim_digest_projection_version": CLAIM_DIGEST_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "disposition_approval_id": issue["disposition_approval_id"],
        "issue_digest": issue["issue_digest"],
        "request_digest": request["request_digest"],
        "claim_id": claim_id,
        "claimant": claimant,
        "claimed_at": claimed_at,
    }
    body["claim_digest"] = _sha256_digest(_claim_digest_projection(body))

    case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"marker_dispositions/{disposition_id}")
    try:
        if revoke is not None:
            raise MarkerDispositionValidationError("cannot claim revoked disposition approval")
        persisted, exact_replay = _create_immutable_record(
            disposition_id=disposition_id,
            case_fd=case_fd,
            filename="claim.json",
            record=body,
            request_projection=intent,
            actor_fields={"claimant": claimant},
            intent_from_record=_claim_intent_projection,
            validate_existing=lambda rec: _validate_record_digest(
                rec,
                digest_field="claim_digest",
                projection_fn=_claim_digest_projection,
            ),
        )
        post = _list_dir_entries(case_fd)
        _validate_directory_state(post, allowed=_CLAIMED_ALLOWED, disposition_id=disposition_id)
        return _to_claim_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)


def load_marker_disposition_bundle(
    disposition_id: str,
    *,
    base_dir: Path | None = None,
) -> MarkerDispositionBundle:
    _require_valid_id(disposition_id, "marker_disposition")
    request = _read_optional_record(paths.marker_disposition_request_path(disposition_id, base_dir))
    issue = _read_optional_record(paths.marker_disposition_issue_path(disposition_id, base_dir))
    revoke = _read_optional_record(paths.marker_disposition_revoke_path(disposition_id, base_dir))
    claim = _read_optional_record(paths.marker_disposition_claim_path(disposition_id, base_dir))
    attempt = _read_optional_record(paths.marker_disposition_attempt_path(disposition_id, base_dir))
    outcome = _read_optional_record(paths.marker_disposition_outcome_path(disposition_id, base_dir))

    req_rec = None
    if request is not None:
        _validate_record_digest(
            request,
            digest_field="request_digest",
            projection_fn=_request_digest_projection,
        )
        req_rec = _to_request_record(request)
    if issue is not None:
        _validate_record_digest(
            issue,
            digest_field="issue_digest",
            projection_fn=_issue_digest_projection,
        )
    if revoke is not None:
        _validate_record_digest(
            revoke,
            digest_field="revoke_digest",
            projection_fn=_revoke_digest_projection,
        )
    if claim is not None:
        _validate_record_digest(
            claim,
            digest_field="claim_digest",
            projection_fn=_claim_digest_projection,
        )
    if attempt is not None:
        _validate_record_digest(
            attempt,
            digest_field="attempt_digest",
            projection_fn=_attempt_digest_projection,
        )
    if outcome is not None:
        _validate_non_permission_booleans(outcome)
        _validate_record_digest(
            outcome,
            digest_field="outcome_digest",
            projection_fn=_outcome_digest_projection,
        )

    return MarkerDispositionBundle(
        disposition_id=disposition_id,
        request_record=req_rec,
        issue_record=_to_issue_record(issue) if issue else None,
        revoke_record=revoke,
        claim_record=_to_claim_record(claim) if claim else None,
        attempt_record=_to_attempt_record(attempt) if attempt else None,
        outcome_record=_to_outcome_record(outcome) if outcome else None,
    )




def _execution_outcome_after_coordination(
    *,
    disposition_id: str,
    attempt_id: str,
    executor: str,
    base_dir: Path | None,
    request: dict[str, Any],
    issue: dict[str, Any],
    claim: dict[str, Any],
) -> MarkerDispositionExecutionResult | None:
    """Re-check durable records after acquiring coordination flock."""
    exec_intent = {
        "attempt_intent_projection_version": ATTEMPT_INTENT_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "attempt_id": attempt_id,
        "executor": executor,
    }
    existing_attempt = _read_optional_record(
        paths.marker_disposition_attempt_path(disposition_id, base_dir)
    )
    existing_outcome = _read_optional_record(
        paths.marker_disposition_outcome_path(disposition_id, base_dir)
    )
    if existing_outcome is not None:
        _validate_record_digest(
            existing_outcome,
            digest_field="outcome_digest",
            projection_fn=_outcome_digest_projection,
        )
        _validate_non_permission_booleans(existing_outcome)
        if existing_attempt is not None and _attempt_intent_projection(existing_attempt) != exec_intent:
            return MarkerDispositionExecutionResult(
                disposition_id=disposition_id,
                outcome_class=MarkerDispositionOutcomeClass.execution_ambiguous,
                outcome_digest=existing_outcome["outcome_digest"],
                exact_replay=False,
                marker_removed_by_this_execution=False,
            )
        if (
            existing_outcome.get("request_digest") == request["request_digest"]
            and existing_outcome.get("issue_digest") == issue["issue_digest"]
            and existing_outcome.get("claim_digest") == claim["claim_digest"]
            and (
                existing_attempt is None
                or existing_outcome.get("attempt_digest") == existing_attempt.get("attempt_digest")
            )
        ):
            return MarkerDispositionExecutionResult(
                disposition_id=disposition_id,
                outcome_class=existing_outcome["outcome_class"],
                outcome_digest=existing_outcome["outcome_digest"],
                exact_replay=True,
                marker_removed_by_this_execution=bool(
                    existing_outcome.get("marker_removed_by_this_execution")
                ),
            )
        raise MarkerDispositionConflictError(
            "outcome already exists with conflicting lineage",
            disposition_id=disposition_id,
        )
    if existing_attempt is not None:
        _validate_record_digest(
            existing_attempt,
            digest_field="attempt_digest",
            projection_fn=_attempt_digest_projection,
        )
        if _attempt_intent_projection(existing_attempt) != exec_intent:
            return MarkerDispositionExecutionResult(
                disposition_id=disposition_id,
                outcome_class=MarkerDispositionOutcomeClass.execution_ambiguous,
                outcome_digest=existing_attempt["attempt_digest"],
                exact_replay=False,
                marker_removed_by_this_execution=False,
            )
        return None
    return None


def execute_approved_marker_disposition(
    disposition_id: str,
    attempt_id: str,
    *,
    executor: str,
    base_dir: Path | None = None,
) -> MarkerDispositionExecutionResult:
    _require_valid_id(disposition_id, "marker_disposition")
    _require_valid_id(attempt_id, "marker_disposition_attempt")
    executor = _validate_actor(executor, field="executor")

    request = _load_validated_request(disposition_id, base_dir)
    issue = _read_optional_record(paths.marker_disposition_issue_path(disposition_id, base_dir))
    if issue is None:
        raise MarkerDispositionStateError("issue.json missing", disposition_id=disposition_id)
    _validate_record_digest(issue, digest_field="issue_digest", projection_fn=_issue_digest_projection)
    revoke = _read_optional_record(paths.marker_disposition_revoke_path(disposition_id, base_dir))
    claim = _read_optional_record(paths.marker_disposition_claim_path(disposition_id, base_dir))
    if claim is None:
        raise MarkerDispositionValidationError("claim.json required before execution")
    _validate_record_digest(claim, digest_field="claim_digest", projection_fn=_claim_digest_projection)
    try:
        _assert_approval_active(issue=issue, revoke=revoke, claim=claim)
    except MarkerDispositionValidationError:
        case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
        case_fd = _open_control_dir_no_follow(
            case_dir, context=f"marker_dispositions/{disposition_id}"
        )
        try:
            body = _outcome_body(
                disposition_id=disposition_id,
                outcome_class=MarkerDispositionOutcomeClass.approval_invalid,
                request_digest=request["request_digest"],
                issue_digest=issue["issue_digest"],
                claim_digest=claim["claim_digest"],
                attempt_digest=None,
                marker_removed_by_this_execution=False,
                lock_directory_fsync_succeeded=False,
            )
            intent = {
                "outcome_class": body["outcome_class"],
                "request_digest": body["request_digest"],
                "issue_digest": body["issue_digest"],
                "claim_digest": body["claim_digest"],
                "attempt_digest": body["attempt_digest"],
            }
            persisted = _persist_outcome_private(
                disposition_id=disposition_id,
                case_fd=case_fd,
                body=body,
                intent_projection=intent,
            )
            return MarkerDispositionExecutionResult(
                disposition_id=disposition_id,
                outcome_class=persisted["outcome_class"],
                outcome_digest=persisted["outcome_digest"],
                exact_replay=False,
                marker_removed_by_this_execution=False,
            )
        finally:
            os.close(case_fd)

    exec_intent = {
        "attempt_intent_projection_version": ATTEMPT_INTENT_PROJECTION_VERSION,
        "disposition_id": disposition_id,
        "attempt_id": attempt_id,
        "executor": executor,
    }

    try:
        proof = _prove_path_a_execution_eligibility(request=request, base_dir=base_dir)
    except (MarkerDispositionValidationError, ReconciliationValidationError):
        case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
        case_fd = _open_control_dir_no_follow(
            case_dir, context=f"marker_dispositions/{disposition_id}"
        )
        try:
            body = _outcome_body(
                disposition_id=disposition_id,
                outcome_class=MarkerDispositionOutcomeClass.evidence_drifted,
                request_digest=request["request_digest"],
                issue_digest=issue["issue_digest"],
                claim_digest=claim["claim_digest"],
                attempt_digest=None,
                marker_removed_by_this_execution=False,
                lock_directory_fsync_succeeded=False,
            )
            intent = {
                "outcome_class": body["outcome_class"],
                "request_digest": body["request_digest"],
                "issue_digest": body["issue_digest"],
                "claim_digest": body["claim_digest"],
                "attempt_digest": body["attempt_digest"],
            }
            persisted = _persist_outcome_private(
                disposition_id=disposition_id,
                case_fd=case_fd,
                body=body,
                intent_projection=intent,
            )
            return MarkerDispositionExecutionResult(
                disposition_id=disposition_id,
                outcome_class=persisted["outcome_class"],
                outcome_digest=persisted["outcome_digest"],
                exact_replay=False,
                marker_removed_by_this_execution=False,
            )
        finally:
            os.close(case_fd)

    run_id = request["marker_run_id"]
    expected_acquisition = request["marker_acquisition_id"]
    expected_identity = (
        int(request["request_time_marker_device"]),
        int(request["request_time_marker_inode"]),
    )
    expected_lock_identity = (
        int(proof["lock_directory_device"]),
        int(proof["lock_directory_inode"]),
    )

    runs_root_fd, lock_root_fd = pin_lock_directory(base_dir)
    lock_dir_fsync_ok = False
    marker_removed = False
    outcome_class = MarkerDispositionOutcomeClass.unlink_failed
    attempt_body: dict[str, Any] | None = None

    try:
        acquire_marker_directory_entry_coordination(lock_root_fd)
        try:
            raced = _execution_outcome_after_coordination(
                disposition_id=disposition_id,
                attempt_id=attempt_id,
                executor=executor,
                base_dir=base_dir,
                request=request,
                issue=issue,
                claim=claim,
            )
            if raced is not None:
                return raced
            if lock_directory_identity(lock_root_fd) != expected_lock_identity:
                outcome_class = MarkerDispositionOutcomeClass.integrity_blocked
                raise MarkerDispositionValidationError("lock directory identity mismatch")

            try:
                metadata, marker_identity = read_marker_metadata_at(lock_root_fd, run_id)
            except RunExecutionLockIndeterminateError as exc:
                if "not present" in str(exc).lower():
                    outcome_class = MarkerDispositionOutcomeClass.already_absent_observed
                    raise
                outcome_class = MarkerDispositionOutcomeClass.integrity_blocked
                raise
            except RunExecutionLockPathUnsafeError as exc:
                outcome_class = MarkerDispositionOutcomeClass.integrity_blocked
                raise MarkerDispositionValidationError(str(exc)) from exc

            if _marker_metadata_digest(metadata) != request["marker_metadata_digest"]:
                outcome_class = MarkerDispositionOutcomeClass.marker_changed
                raise MarkerDispositionValidationError("marker metadata digest mismatch")
            if marker_identity != expected_identity:
                outcome_class = MarkerDispositionOutcomeClass.marker_changed
                raise MarkerDispositionValidationError("marker identity mismatch")

            revalidation_envelope = {
                "revalidation_projection_version": EXECUTION_REVALIDATION_PROJECTION_VERSION,
                "disposition_id": disposition_id,
                "request_digest": request["request_digest"],
                "execution_inspection_digest": proof["execution_inspection_digest"],
                "execution_inspection_projection": proof["execution_inspection_projection"],
            }
            revalidation_envelope["execution_revalidation_digest"] = _sha256_digest(
                revalidation_envelope
            )

            attempted_at = _utc_now_iso()
            attempt_body = {
                "attempt_schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_digest_projection_version": ATTEMPT_DIGEST_PROJECTION_VERSION,
                "disposition_id": disposition_id,
                "request_digest": request["request_digest"],
                "issue_digest": issue["issue_digest"],
                "claim_digest": claim["claim_digest"],
                "attempt_id": attempt_id,
                "executor": executor,
                "attempted_at": attempted_at,
                "execution_revalidation": revalidation_envelope,
                "marker_device": marker_identity[0],
                "marker_inode": marker_identity[1],
                "lock_directory_device": expected_lock_identity[0],
                "lock_directory_inode": expected_lock_identity[1],
            }
            attempt_body["attempt_digest"] = _sha256_digest(_attempt_digest_projection(attempt_body))

            case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
            case_fd = _open_control_dir_no_follow(
                case_dir, context=f"marker_dispositions/{disposition_id}"
            )
            try:
                entries = _list_dir_entries(case_fd)
                _validate_directory_state(
                    entries, allowed=_CLAIMED_ALLOWED, disposition_id=disposition_id
                )
                try:
                    persisted_attempt, _ = _create_immutable_record(
                        disposition_id=disposition_id,
                        case_fd=case_fd,
                        filename="attempt.json",
                        record=attempt_body,
                        request_projection=exec_intent,
                        intent_from_record=_attempt_intent_projection,
                        validate_existing=lambda rec: _validate_record_digest(
                            rec,
                            digest_field="attempt_digest",
                            projection_fn=_attempt_digest_projection,
                        ),
                    )
                    attempt_body = persisted_attempt
                except MarkerDispositionConflictError:
                    outcome_class = MarkerDispositionOutcomeClass.execution_ambiguous
                    raise MarkerDispositionValidationError(
                        "concurrent attempt conflict"
                    ) from None
            finally:
                os.close(case_fd)

            metadata2, identity2 = read_marker_metadata_at(lock_root_fd, run_id)
            if identity2 != marker_identity or metadata2.get("acquisition_id") != expected_acquisition:
                outcome_class = MarkerDispositionOutcomeClass.marker_changed
                raise MarkerDispositionValidationError("marker changed before unlink")

            try:
                disposition_unlink_marker(
                        lock_root_fd,
                        run_id,
                        expected_identity=marker_identity,
                        expected_acquisition_id=expected_acquisition,
                    )
                lock_dir_fsync_ok = True
                marker_removed = True
                outcome_class = MarkerDispositionOutcomeClass.disposed_verified
            except (RunExecutionLockReleaseConflictError, RunExecutionLockPathUnsafeError):
                outcome_class = MarkerDispositionOutcomeClass.unlink_failed
            except RunExecutionLockIndeterminateError as exc:
                if "fsync" in str(exc).lower():
                    outcome_class = MarkerDispositionOutcomeClass.lock_directory_fsync_failed
                else:
                    outcome_class = MarkerDispositionOutcomeClass.unlink_failed

            if attempt_body is not None:
                case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
                case_fd = _open_control_dir_no_follow(
                    case_dir, context=f"marker_dispositions/{disposition_id}"
                )
                try:
                    attempt_digest = attempt_body["attempt_digest"]
                    body = _outcome_body(
                        disposition_id=disposition_id,
                        outcome_class=outcome_class,
                        request_digest=request["request_digest"],
                        issue_digest=issue["issue_digest"],
                        claim_digest=claim["claim_digest"],
                        attempt_digest=attempt_digest,
                        marker_removed_by_this_execution=marker_removed,
                        lock_directory_fsync_succeeded=lock_dir_fsync_ok,
                    )
                    intent = {
                        "outcome_class": body["outcome_class"],
                        "request_digest": body["request_digest"],
                        "issue_digest": body["issue_digest"],
                        "claim_digest": body["claim_digest"],
                        "attempt_digest": body["attempt_digest"],
                    }
                    persisted = _persist_outcome_private(
                        disposition_id=disposition_id,
                        case_fd=case_fd,
                        body=body,
                        intent_projection=intent,
                    )
                    return MarkerDispositionExecutionResult(
                        disposition_id=disposition_id,
                        outcome_class=persisted["outcome_class"],
                        outcome_digest=persisted["outcome_digest"],
                        exact_replay=False,
                        marker_removed_by_this_execution=marker_removed,
                    )
                finally:
                    os.close(case_fd)
        finally:
            release_marker_directory_entry_coordination(lock_root_fd)
    except MarkerDispositionValidationError:
        pass
    except RunExecutionLockIndeterminateError as exc:
        if outcome_class == MarkerDispositionOutcomeClass.already_absent_observed:
            pass
        elif "fsync" in str(exc).lower():
            outcome_class = MarkerDispositionOutcomeClass.lock_directory_fsync_failed
        else:
            if outcome_class == MarkerDispositionOutcomeClass.unlink_failed:
                pass
    finally:
        os.close(lock_root_fd)
        os.close(runs_root_fd)

    case_dir = paths.marker_disposition_dir(disposition_id, base_dir)
    case_fd = _open_control_dir_no_follow(
        case_dir, context=f"marker_dispositions/{disposition_id}"
    )
    try:
        if attempt_body is None and outcome_class == MarkerDispositionOutcomeClass.already_absent_observed:
            attempted_at = _utc_now_iso()
            attempt_body = {
                "attempt_schema_version": ATTEMPT_SCHEMA_VERSION,
                "attempt_digest_projection_version": ATTEMPT_DIGEST_PROJECTION_VERSION,
                "disposition_id": disposition_id,
                "request_digest": request["request_digest"],
                "issue_digest": issue["issue_digest"],
                "claim_digest": claim["claim_digest"],
                "attempt_id": attempt_id,
                "executor": executor,
                "attempted_at": attempted_at,
                "execution_revalidation": {
                    "revalidation_projection_version": EXECUTION_REVALIDATION_PROJECTION_VERSION,
                    "disposition_id": disposition_id,
                    "request_digest": request["request_digest"],
                    "execution_inspection_digest": proof.get("execution_inspection_digest"),
                    "execution_inspection_projection": proof.get("execution_inspection_projection"),
                    "execution_revalidation_digest": _sha256_digest(
                        {
                            "revalidation_projection_version": EXECUTION_REVALIDATION_PROJECTION_VERSION,
                            "disposition_id": disposition_id,
                            "request_digest": request["request_digest"],
                            "execution_inspection_digest": proof.get("execution_inspection_digest"),
                            "execution_inspection_projection": proof.get(
                                "execution_inspection_projection"
                            ),
                        }
                    ),
                },
                "marker_device": expected_identity[0],
                "marker_inode": expected_identity[1],
                "lock_directory_device": expected_lock_identity[0],
                "lock_directory_inode": expected_lock_identity[1],
            }
            attempt_body["attempt_digest"] = _sha256_digest(_attempt_digest_projection(attempt_body))
            _create_immutable_record(
                disposition_id=disposition_id,
                case_fd=case_fd,
                filename="attempt.json",
                record=attempt_body,
                request_projection=exec_intent,
                intent_from_record=_attempt_intent_projection,
                validate_existing=lambda rec: _validate_record_digest(
                    rec,
                    digest_field="attempt_digest",
                    projection_fn=_attempt_digest_projection,
                ),
            )

        if attempt_body is not None:
            attempt_digest = attempt_body["attempt_digest"]
        else:
            attempt_digest = None
        body = _outcome_body(
            disposition_id=disposition_id,
            outcome_class=outcome_class,
            request_digest=request["request_digest"],
            issue_digest=issue["issue_digest"],
            claim_digest=claim["claim_digest"],
            attempt_digest=attempt_digest,
            marker_removed_by_this_execution=marker_removed,
            lock_directory_fsync_succeeded=lock_dir_fsync_ok,
        )
        intent = {
            "outcome_class": body["outcome_class"],
            "request_digest": body["request_digest"],
            "issue_digest": body["issue_digest"],
            "claim_digest": body["claim_digest"],
            "attempt_digest": body["attempt_digest"],
        }
        persisted = _persist_outcome_private(
            disposition_id=disposition_id,
            case_fd=case_fd,
            body=body,
            intent_projection=intent,
        )
        return MarkerDispositionExecutionResult(
            disposition_id=disposition_id,
            outcome_class=persisted["outcome_class"],
            outcome_digest=persisted["outcome_digest"],
            exact_replay=False,
            marker_removed_by_this_execution=marker_removed,
        )
    finally:
        os.close(case_fd)


def reconcile_marker_disposition_outcome(
    disposition_id: str,
    *,
    base_dir: Path | None = None,
) -> MarkerDispositionReconcileResult:
    """Literal read-only reconciliation — performs zero writes."""
    _require_valid_id(disposition_id, "marker_disposition")
    request = _read_optional_record(paths.marker_disposition_request_path(disposition_id, base_dir))
    attempt = _read_optional_record(paths.marker_disposition_attempt_path(disposition_id, base_dir))
    outcome = _read_optional_record(paths.marker_disposition_outcome_path(disposition_id, base_dir))
    claim = _read_optional_record(paths.marker_disposition_claim_path(disposition_id, base_dir))

    notes: list[str] = []
    outcome_rec = None
    marker_present: bool | None = None

    if outcome is not None:
        try:
            _validate_non_permission_booleans(outcome)
            _validate_record_digest(
                outcome,
                digest_field="outcome_digest",
                projection_fn=_outcome_digest_projection,
            )
            outcome_rec = _to_outcome_record(outcome)
            return MarkerDispositionReconcileResult(
                disposition_id=disposition_id,
                classification="valid_durable_outcome",
                outcome_record=outcome_rec,
                marker_present=None,
                notes=tuple(notes),
            )
        except MarkerDispositionValidationError as exc:
            return MarkerDispositionReconcileResult(
                disposition_id=disposition_id,
                classification="malformed_outcome",
                outcome_record=None,
                marker_present=None,
                notes=(str(exc),),
            )

    if claim is not None and attempt is None:
        return MarkerDispositionReconcileResult(
            disposition_id=disposition_id,
            classification="claim_without_attempt",
            outcome_record=None,
            marker_present=None,
            notes=tuple(notes),
        )

    if attempt is not None and request is not None:
        run_id = request.get("marker_run_id")
        if isinstance(run_id, str):
            runs_root_fd, lock_root_fd = pin_lock_directory(base_dir)
            try:
                try:
                    read_marker_metadata_at(lock_root_fd, run_id)
                    marker_present = True
                    classification = "attempt_with_marker_present"
                except RunExecutionLockIndeterminateError:
                    marker_present = False
                    classification = "attempt_with_marker_absent"
            finally:
                os.close(lock_root_fd)
                os.close(runs_root_fd)
            return MarkerDispositionReconcileResult(
                disposition_id=disposition_id,
                classification=classification,
                outcome_record=None,
                marker_present=marker_present,
                notes=tuple(notes),
            )

    if request is None:
        return MarkerDispositionReconcileResult(
            disposition_id=disposition_id,
            classification="missing_request",
            outcome_record=None,
            marker_present=None,
            notes=tuple(notes),
        )

    return MarkerDispositionReconcileResult(
        disposition_id=disposition_id,
        classification="incomplete",
        outcome_record=None,
        marker_present=None,
        notes=tuple(notes),
    )
