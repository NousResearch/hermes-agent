"""Task 27 — approved recovery run creation protocol (Path R1 only)."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import stat
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, NoReturn

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
from htr.execution_lock import LOCKS_DIR_NAME
from htr.io import (
    RunRootReservation,
    RunRootReservationError,
    bootstrap_reserved_run_workspace,
    read_json,
    release_run_root_reservation,
    reserve_run_root_exclusive,
)
from htr.finalization import SealState, evaluate_run_seal
from htr.ids import (
    IdKind,
    generate_recovery_approval_id,
    generate_recovery_attempt_id,
    generate_recovery_claim_id,
    generate_recovery_request_id,
    generate_successor_run_id,
    validate_id,
)
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
    RecoveryRunAttemptRecord,
    RecoveryRunBundle,
    RecoveryRunClaimRecord,
    RecoveryRunConflictError,
    RecoveryRunDurabilityError,
    RecoveryRunExecutionResult,
    RecoveryRunIssueRecord,
    RecoveryRunOutcomeRecord,
    RecoveryRunReconcileResult,
    RecoveryRunRecordName,
    RecoveryRunRequestRecord,
    RecoveryRunStateError,
    RecoveryRunValidationError,
    RecoveryRunWriteMetadata,
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

REQUEST_DIGEST_PROJECTION_VERSION = "htr.recovery_run.request.digest.v1"
REQUEST_INTENT_PROJECTION_VERSION = "htr.recovery_run.request.intent.v1"
ISSUE_DIGEST_PROJECTION_VERSION = "htr.recovery_run.issue.digest.v1"
ISSUE_INTENT_PROJECTION_VERSION = "htr.recovery_run.issue.intent.v1"
REVOKE_DIGEST_PROJECTION_VERSION = "htr.recovery_run.revoke.digest.v1"
REVOKE_INTENT_PROJECTION_VERSION = "htr.recovery_run.revoke.intent.v1"
CLAIM_DIGEST_PROJECTION_VERSION = "htr.recovery_run.claim.digest.v1"
CLAIM_INTENT_PROJECTION_VERSION = "htr.recovery_run.claim.intent.v1"
ATTEMPT_DIGEST_PROJECTION_VERSION = "htr.recovery_run.attempt.digest.v1"
ATTEMPT_INTENT_PROJECTION_VERSION = "htr.recovery_run.attempt.intent.v1"
OUTCOME_DIGEST_PROJECTION_VERSION = "htr.recovery_run.outcome.digest.v1"
RECOVERY_ORIGIN_DIGEST_PROJECTION_VERSION = "htr.recovery_run.recovery_origin.digest.v1"
RECOVERY_ORIGIN_SCHEMA_VERSION = "1"
EXECUTION_REVALIDATION_PROJECTION_VERSION = "htr.recovery_run.execution_revalidation.digest.v1"

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
    "source_run_mutation_allowed",
    "retry_allowed",
    "repair_allowed",
    "invoke_allowed",
    "automatic_execution_allowed",
    "outcome_rewrite_allowed",
)

_PATH_R1_DECISION_CLASSES = frozenset(
    {ReconciliationDecisionClass.evidence_conflict_confirmed.value}
)

_PERMITTED_RECOVERY_SCOPES = frozenset(
    {
        "diagnostic_only",
        "verification_only",
        "artifact_reconstruction_review",
        "controlled_follow_up",
    }
)


class RecoveryScope(str, Enum):
    diagnostic_only = "diagnostic_only"
    verification_only = "verification_only"
    artifact_reconstruction_review = "artifact_reconstruction_review"
    controlled_follow_up = "controlled_follow_up"
    forward_fix = "forward_fix"


class RecoveryRunOutcomeClass(str, Enum):
    successor_created_verified = "successor_created_verified"
    successor_already_exists_verified = "successor_already_exists_verified"
    source_evidence_drifted = "source_evidence_drifted"
    approval_invalid = "approval_invalid"
    successor_id_conflict = "successor_id_conflict"
    creation_failed = "creation_failed"
    creation_partial = "creation_partial"
    outcome_durability_indeterminate = "outcome_durability_indeterminate"
    execution_ambiguous = "execution_ambiguous"
    integrity_blocked = "integrity_blocked"


def _require_valid_id(value: str, kind: IdKind) -> None:
    if not validate_id(value, kind):
        raise RecoveryRunValidationError(f"invalid {kind} id: {value!r}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_utc_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise RecoveryRunValidationError(f"timestamp must be timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _validate_actor(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryRunValidationError(f"{field} must be a non-empty string")
    actor = value.strip()
    if len(actor) > _MAX_ACTOR_LEN:
        raise RecoveryRunValidationError(f"{field} exceeds {_MAX_ACTOR_LEN} characters")
    if not actor.isprintable():
        raise RecoveryRunValidationError(f"{field} must be printable")
    return actor


def _validate_expiry(*, issued_at: str, expires_at: str) -> None:
    issued = _parse_utc_iso(issued_at)
    expires = _parse_utc_iso(expires_at)
    if expires <= issued:
        raise RecoveryRunValidationError("expires_at must be after issued_at")
    if expires - issued > MAX_APPROVAL_LIFETIME:
        raise RecoveryRunValidationError(
            f"approval lifetime exceeds {MAX_APPROVAL_LIFETIME.total_seconds()} seconds"
        )


def _raise_unsafe_path(context: str, exc: BaseException) -> NoReturn:
    raise RecoveryRunValidationError(
        f"unsafe recovery run control path ({context})"
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


def _fsync_recovery_durability_chain(
    case_fd: int,
    *,
    recovery_request_id: str,
    record_name: str,
) -> None:
    _fsync_dir_fd(
        case_fd,
        recovery_request_id=recovery_request_id,
        record_name=record_name,
        stage="recovery_case_dir_fsync",
    )
    parent_fd = os.open(
        f"/proc/self/fd/{case_fd}/..",
        _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
    )
    try:
        _fsync_dir_fd(
            parent_fd,
            recovery_request_id=recovery_request_id,
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
                recovery_request_id=recovery_request_id,
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
    recovery_request_id: str,
    record_name: RecoveryRunRecordName,
    stage: str,
) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise RecoveryRunDurabilityError(
            f"directory fsync failed at {stage}: {exc}",
            recovery_request_id=recovery_request_id,
            record_name=record_name,
            durability_stage=stage,  # type: ignore[arg-type]
            record_may_have_committed=True,
            successor_may_have_been_created=False,
            exact_replay_status="indeterminate",
        ) from exc


def _fsync_file_fd(
    file_fd: int,
    *,
    recovery_request_id: str,
    record_name: RecoveryRunRecordName,
) -> None:
    try:
        os.fsync(file_fd)
    except OSError as exc:
        raise RecoveryRunDurabilityError(
            f"file fsync failed: {exc}",
            recovery_request_id=recovery_request_id,
            record_name=record_name,
            durability_stage="record_fsync",
            record_may_have_committed=True,
            successor_may_have_been_created=False,
            exact_replay_status="indeterminate",
        ) from exc


def _write_all(
    fd: int,
    payload: bytes,
    *,
    recovery_request_id: str,
    record_name: RecoveryRunRecordName,
) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise RecoveryRunDurabilityError(
                "short write while persisting recovery run record",
                recovery_request_id=recovery_request_id,
                record_name=record_name,
                durability_stage="record_write",
                record_may_have_committed=False,
                successor_may_have_been_created=False,
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
        raise RecoveryRunValidationError("expected JSON object record")
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
        raise RecoveryRunValidationError(f"unsafe symlink record: {path}")
    try:
        return read_json(path)
    except json.JSONDecodeError as exc:
        raise RecoveryRunValidationError(f"malformed JSON record: {path}") from exc


def _read_optional_record_at(case_fd: int, filename: str) -> dict[str, Any] | None:
    flags_read = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    try:
        file_fd = os.open(filename, flags_read, dir_fd=case_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        _raise_unsafe_path(f"recovery_runs/{filename}", exc)
    try:
        return _read_json_fd(file_fd)
    except json.JSONDecodeError as exc:
        raise RecoveryRunValidationError(f"malformed JSON record: {filename}") from exc
    finally:
        os.close(file_fd)


def _validate_record_digest(
    record: dict[str, Any],
    *,
    digest_field: str,
    projection_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    stored = record.get(digest_field)
    if not isinstance(stored, str) or not stored:
        raise RecoveryRunValidationError(f"missing {digest_field}")
    computed = _sha256_digest(projection_fn(record))
    if stored != computed:
        raise RecoveryRunValidationError(f"{digest_field} mismatch")


def _validate_non_permission_booleans(record: dict[str, Any]) -> None:
    for field in _NON_PERMISSION_BOOLEANS:
        value = record.get(field)
        if value is not False:
            raise RecoveryRunValidationError(f"{field} must be false")


def _write_metadata(*, exact_replay: bool) -> RecoveryRunWriteMetadata:
    return RecoveryRunWriteMetadata(
        exact_replay=exact_replay,
        exact_replay_status="yes" if exact_replay else "no",
        record_may_have_committed=True,
        durability_indeterminate=False,
    )


def _open_recovery_case_fd(recovery_request_id: str, base_dir: Path | None) -> int:
    case_dir = paths.recovery_run_control_dir(recovery_request_id, base_dir)
    return _open_control_dir_no_follow(
        case_dir,
        context=f"recovery_runs/{recovery_request_id}",
    )


@contextmanager
def _case_execution_barrier(case_fd: int) -> Iterator[None]:
    """Serialize execution for one recovery request across processes."""
    fcntl.flock(case_fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(case_fd, fcntl.LOCK_UN)




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
        raise RecoveryRunValidationError("Task 25 approval issue missing")
    if claim is None:
        raise RecoveryRunValidationError("Task 25 approval claim missing")
    if outcome is None:
        raise RecoveryRunValidationError("Task 25 approval outcome missing")
    if issue.get("approval_digest") != _compute_approval_digest(issue):
        raise RecoveryRunValidationError("Task 25 issue digest mismatch")
    if claim.get("claim_digest") != _compute_claim_digest(claim):
        raise RecoveryRunValidationError("Task 25 claim digest mismatch")
    if outcome.get("outcome_class") != OUTCOME_CONSUMED:
        raise RecoveryRunValidationError("Task 25 outcome is not consumed")
    outcome_digest = _compute_outcome_digest(outcome)
    if outcome.get("outcome_digest") != outcome_digest:
        raise RecoveryRunValidationError("Task 25 outcome digest mismatch")
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
        raise RecoveryRunValidationError("reconciliation decision missing")
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



def _validate_recovery_scope(scope: RecoveryScope | str) -> str:
    if isinstance(scope, RecoveryScope):
        value = scope.value
    elif isinstance(scope, str):
        value = scope.strip()
    else:
        raise RecoveryRunValidationError("recovery_scope must be a RecoveryScope or string")
    if value == RecoveryScope.forward_fix.value:
        raise RecoveryRunValidationError("forward_fix scope is not permitted in v1")
    if value not in _PERMITTED_RECOVERY_SCOPES:
        raise RecoveryRunValidationError(f"recovery_scope not permitted: {value!r}")
    return value


def _validate_recovery_reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason.strip():
        raise RecoveryRunValidationError("recovery_reason must be a non-empty string")
    return reason.strip()


def _validate_path_r1_inspection_projection(projection: dict[str, Any]) -> None:
    overall = projection.get("overall_classification")
    if overall == "partial_lifecycle_commit":
        raise RecoveryRunValidationError("partial_lifecycle_commit is not eligible for Path R1")
    if overall == "integrity_blocked" or projection.get("integrity_state") != "clean":
        raise RecoveryRunValidationError("integrity_blocked overall is not eligible for Path R1")
    if overall == "verified_completion_marker_residue":
        raise RecoveryRunValidationError("verified_completion_marker_residue requires marker disposition path")
    if overall != "control_lifecycle_evidence_conflict":
        raise RecoveryRunValidationError("overall classification mismatch for Path R1")
    if projection.get("approval_control_state") != "consumed_outcome":
        raise RecoveryRunValidationError("approval control state is not consumed_outcome")
    if projection.get("outcome_class") != OUTCOME_CONSUMED:
        raise RecoveryRunValidationError("outcome class is not consumed")



def _prove_path_r1_eligibility(
    *,
    case_id: str,
    base_dir: Path | None,
    require_seal: bool = True,
) -> dict[str, Any]:
    bundle = load_reconciliation_case(case_id, base_dir=base_dir)
    if bundle.observation_record is None or bundle.decision_record is None:
        raise RecoveryRunValidationError("reconciliation case incomplete for Path R1")
    decision_raw = _load_decision_raw(case_id, base_dir)
    decision_class = decision_raw.get("decision_class")
    if decision_class not in _PATH_R1_DECISION_CLASSES:
        raise RecoveryRunValidationError("decision class not eligible for recovery run creation")
    protocol = decision_raw.get("recommended_next_protocol")
    if protocol != ReconciliationNextProtocol.recovery_run_review.value:
        raise RecoveryRunValidationError("recommended_next_protocol must be recovery_run_review")
    if protocol == ReconciliationNextProtocol.marker_disposition_review.value:
        raise RecoveryRunValidationError("marker_disposition_review path rejected for Path R1")

    drift = decision_raw.get("observation_decision_drift") or {}
    if drift.get("drift_detected") is not False:
        raise RecoveryRunValidationError("decision-time drift detected; Path R1 requires drift false")

    open_record = bundle.open_record
    approval_id = open_record.approval_id
    revalidation = decision_raw.get("decision_time_revalidation") or {}
    rev_projection = revalidation.get("inspection_semantic_projection") or {}
    seal = _seal_evidence(open_record.run_id, base_dir)
    seal_finalized = seal["seal_state"] == SealState.FINALIZED_VALID.value
    if require_seal and not seal_finalized:
        raise RecoveryRunValidationError("run seal is not finalized_valid")

    obs_digest = bundle.observation_record.inspection_semantic_digest
    if seal_finalized:
        if not rev_projection:
            raise RecoveryRunValidationError("decision-time revalidation projection missing")
        _validate_path_r1_inspection_projection(rev_projection)
        execution_projection = rev_projection
        execution_inspection_digest = drift.get("inspection_semantic_digest_at_decision")
        if not isinstance(execution_inspection_digest, str) or not execution_inspection_digest:
            raise RecoveryRunValidationError("decision-time inspection digest missing")
    else:
        inspection = _fresh_inspection(approval_id, base_dir)
        projection = _build_inspection_projection(inspection)
        inspection_digest = compute_inspection_semantic_digest(inspection)
        if inspection_digest != inspection.inspection_semantic_digest:
            raise RecoveryRunValidationError("inspection semantic digest mismatch")
        _validate_path_r1_inspection_projection(projection)
        if obs_digest != inspection_digest:
            raise RecoveryRunValidationError("Task 26A observation inspection drift detected")
        execution_projection = projection
        execution_inspection_digest = inspection_digest

    task25 = _load_task25_lineage(approval_id, base_dir)
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
        "recovery_of_run_id": open_record.run_id,
        "event_id": open_record.event_id,
        "htr_runs_root_path_digest": open_record.htr_runs_root_path_digest,
        "htr_project_dir_path_digest": open_record.htr_project_dir_path_digest,
        "finalized_valid_seal_evidence": seal,
        "task25": task25,
        "execution_inspection_projection": execution_projection,
        "execution_inspection_digest": execution_inspection_digest,
    }



def _prove_path_r1_execution_eligibility(
    *,
    request: dict[str, Any],
    base_dir: Path | None,
) -> dict[str, Any]:
    case_id = request["reconciliation_case_id"]
    bundle = load_reconciliation_case(case_id, base_dir=base_dir)
    if bundle.observation_record is None or bundle.decision_record is None:
        raise RecoveryRunValidationError("reconciliation case incomplete for Path R1")
    open_record = bundle.open_record
    if open_record.case_open_digest != request["case_open_digest"]:
        raise RecoveryRunValidationError("case_open_digest drift detected")
    if bundle.observation_record.observation_digest != request["observation_digest"]:
        raise RecoveryRunValidationError("observation_digest drift detected")
    if bundle.decision_record.decision_digest != request["decision_digest"]:
        raise RecoveryRunValidationError("decision_digest drift detected")

    decision_raw = _load_decision_raw(case_id, base_dir)
    if decision_raw.get("decision_class") not in _PATH_R1_DECISION_CLASSES:
        raise RecoveryRunValidationError("decision class not eligible for recovery run creation")
    if decision_raw.get("recommended_next_protocol") != ReconciliationNextProtocol.recovery_run_review.value:
        raise RecoveryRunValidationError("recommended_next_protocol must be recovery_run_review")
    drift = decision_raw.get("observation_decision_drift") or {}
    if drift.get("drift_detected") is not False:
        raise RecoveryRunValidationError("decision-time drift detected at execution")

    seal = _seal_evidence(open_record.run_id, base_dir)
    if seal["seal_state"] != SealState.FINALIZED_VALID.value:
        raise RecoveryRunValidationError("run seal is not finalized_valid")
    stored_seal = request.get("finalized_valid_seal_evidence") or {}
    if stored_seal.get("seal_state") != SealState.FINALIZED_VALID.value:
        raise RecoveryRunValidationError("request seal evidence not finalized_valid")

    task25 = _load_task25_lineage(request["task25_approval_id"], base_dir)
    if task25["approval_issue_digest"] != request["task25_approval_issue_digest"]:
        raise RecoveryRunValidationError("Task 25 issue digest drift detected")
    if task25["claim_id"] != request["task25_claim_id"]:
        raise RecoveryRunValidationError("Task 25 claim id drift detected")
    if task25["claim_digest"] != request["task25_claim_digest"]:
        raise RecoveryRunValidationError("Task 25 claim digest drift detected")
    if task25["consumed_outcome_digest"] != request["task25_consumed_outcome_digest"]:
        raise RecoveryRunValidationError("Task 25 outcome digest drift detected")

    revalidation = decision_raw.get("decision_time_revalidation") or {}
    rev_projection = revalidation.get("inspection_semantic_projection") or {}
    if request["task26a_observation_inspection_digest"] != bundle.observation_record.inspection_semantic_digest:
        raise RecoveryRunValidationError("Task 26A observation inspection drift detected")
    current_digest = compute_inspection_semantic_digest(_fresh_inspection(open_record.approval_id, base_dir))
    bound_digest = request["task26b_decision_revalidation_inspection_digest"]
    if current_digest != bound_digest:
        raise RecoveryRunValidationError("execution-time inspection drift detected")

    return {
        "case_id": case_id,
        "recovery_of_run_id": open_record.run_id,
        "execution_inspection_projection": rev_projection,
        "execution_inspection_digest": bound_digest,
    }



def _request_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_intent_projection_version": REQUEST_INTENT_PROJECTION_VERSION,
        "recovery_request_id": record["recovery_request_id"],
        "reconciliation_case_id": record["reconciliation_case_id"],
        "requested_by": record["requested_by"],
        "recovery_of_run_id": record["recovery_of_run_id"],
        "successor_run_id": record["successor_run_id"],
        "recovery_scope": record["recovery_scope"],
        "recovery_reason": record["recovery_reason"],
        "decision_class": record["decision_class"],
        "recommended_next_protocol": record["recommended_next_protocol"],
    }


def _request_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_schema_version": body["request_schema_version"],
        "request_digest_projection_version": body["request_digest_projection_version"],
        "recovery_request_id": body["recovery_request_id"],
        "reconciliation_case_id": body["reconciliation_case_id"],
        "case_open_digest": body["case_open_digest"],
        "observation_digest": body["observation_digest"],
        "decision_digest": body["decision_digest"],
        "decision_class": body["decision_class"],
        "recommended_next_protocol": body["recommended_next_protocol"],
        "task26a_observation_inspection_digest": body["task26a_observation_inspection_digest"],
        "task26b_decision_revalidation_inspection_digest": body["task26b_decision_revalidation_inspection_digest"],
        "task25_approval_id": body["task25_approval_id"],
        "task25_approval_issue_digest": body["task25_approval_issue_digest"],
        "task25_claim_id": body["task25_claim_id"],
        "task25_claim_digest": body["task25_claim_digest"],
        "task25_consumed_outcome_digest": body["task25_consumed_outcome_digest"],
        "recovery_of_run_id": body["recovery_of_run_id"],
        "successor_run_id": body["successor_run_id"],
        "recovery_scope": body["recovery_scope"],
        "recovery_reason": body["recovery_reason"],
        "event_id": body["event_id"],
        "htr_runs_root_path_digest": body["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": body["htr_project_dir_path_digest"],
        "finalized_valid_seal_evidence": body["finalized_valid_seal_evidence"],
        "requested_by": body["requested_by"],
        "requested_at": body["requested_at"],
    }


def _issue_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_intent_projection_version": ISSUE_INTENT_PROJECTION_VERSION,
        "recovery_request_id": record["recovery_request_id"],
        "recovery_approval_id": record["recovery_approval_id"],
        "issued_by": record["issued_by"],
        "expires_at": record["expires_at"],
    }


def _issue_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_schema_version": body["issue_schema_version"],
        "issue_digest_projection_version": body["issue_digest_projection_version"],
        "recovery_request_id": body["recovery_request_id"],
        "recovery_approval_id": body["recovery_approval_id"],
        "request_digest": body["request_digest"],
        "issued_by": body["issued_by"],
        "issued_at": body["issued_at"],
        "expires_at": body["expires_at"],
    }


def _revoke_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "revoke_intent_projection_version": REVOKE_INTENT_PROJECTION_VERSION,
        "recovery_request_id": record["recovery_request_id"],
        "revoked_by": record["revoked_by"],
    }


def _revoke_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "revoke_schema_version": body["revoke_schema_version"],
        "revoke_digest_projection_version": body["revoke_digest_projection_version"],
        "recovery_request_id": body["recovery_request_id"],
        "issue_digest": body["issue_digest"],
        "request_digest": body["request_digest"],
        "revoked_by": body["revoked_by"],
        "revoked_at": body["revoked_at"],
        "reason": body["reason"],
    }


def _claim_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_intent_projection_version": CLAIM_INTENT_PROJECTION_VERSION,
        "recovery_request_id": record["recovery_request_id"],
        "claim_id": record["claim_id"],
        "claimant": record["claimant"],
    }


def _claim_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_schema_version": body["claim_schema_version"],
        "claim_digest_projection_version": body["claim_digest_projection_version"],
        "recovery_request_id": body["recovery_request_id"],
        "recovery_approval_id": body["recovery_approval_id"],
        "issue_digest": body["issue_digest"],
        "request_digest": body["request_digest"],
        "claim_id": body["claim_id"],
        "claimant": body["claimant"],
        "claimed_at": body["claimed_at"],
    }


def _attempt_intent_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_intent_projection_version": ATTEMPT_INTENT_PROJECTION_VERSION,
        "recovery_request_id": record["recovery_request_id"],
        "attempt_id": record["attempt_id"],
        "executor": record["executor"],
    }



def _attempt_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_schema_version": body["attempt_schema_version"],
        "attempt_digest_projection_version": body["attempt_digest_projection_version"],
        "recovery_request_id": body["recovery_request_id"],
        "request_digest": body["request_digest"],
        "issue_digest": body["issue_digest"],
        "claim_digest": body["claim_digest"],
        "attempt_id": body["attempt_id"],
        "executor": body["executor"],
        "attempted_at": body["attempted_at"],
        "successor_run_id": body["successor_run_id"],
        "execution_revalidation": body["execution_revalidation"],
    }



def _outcome_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_schema_version": body["outcome_schema_version"],
        "outcome_digest_projection_version": body["outcome_digest_projection_version"],
        "recovery_request_id": body["recovery_request_id"],
        "outcome_class": body["outcome_class"],
        "request_digest": body["request_digest"],
        "issue_digest": body["issue_digest"],
        "claim_digest": body["claim_digest"],
        "attempt_digest": body["attempt_digest"],
        "successor_run_id": body["successor_run_id"],
        "recorded_at": body["recorded_at"],
        "source_run_mutation_allowed": body["source_run_mutation_allowed"],
        "retry_allowed": body["retry_allowed"],
        "repair_allowed": body["repair_allowed"],
        "invoke_allowed": body["invoke_allowed"],
        "automatic_execution_allowed": body["automatic_execution_allowed"],
        "outcome_rewrite_allowed": body["outcome_rewrite_allowed"],
    }


def _bootstrap_recovery_tree(
    recovery_request_id: str,
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
                paths.RECOVERY_RUNS_DIR_NAME,
                _CONTROL_DIR_MODE,
                context="runs_root/.control",
            )
            dispositions_fd = _openat_control_dir_no_follow(
                control_fd,
                paths.RECOVERY_RUNS_DIR_NAME,
                context="runs_root/.control/recovery_runs",
            )
            try:
                created_case = _mkdirat_control(
                    dispositions_fd,
                    recovery_request_id,
                    _CONTROL_DIR_MODE,
                    context=f"recovery_runs/{recovery_request_id}",
                )
                case_fd = _openat_control_dir_no_follow(
                    dispositions_fd,
                    recovery_request_id,
                    context=f"recovery_runs/{recovery_request_id}",
                )
                case_identity = _fstat_identity(case_fd)
                if created_case:
                    _fsync_dir_fd(
                        dispositions_fd,
                        recovery_request_id=recovery_request_id,
                        record_name="request.json",
                        stage="recovery_case_dir_fsync",
                    )
                if created_dispositions:
                    _fsync_dir_fd(
                        control_fd,
                        recovery_request_id=recovery_request_id,
                        record_name="request.json",
                        stage="control_dir_fsync",
                    )
                if created_control:
                    _fsync_dir_fd(
                        runs_fd,
                        recovery_request_id=recovery_request_id,
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
    raise RecoveryRunStateError("unreachable bootstrap failure", recovery_request_id=recovery_request_id)


def _create_immutable_record(
    *,
    recovery_request_id: str,
    case_fd: int,
    filename: RecoveryRunRecordName,
    record: dict[str, Any] | None = None,
    record_factory: Callable[[], dict[str, Any]] | None = None,
    request_projection: dict[str, Any] | Callable[[], dict[str, Any]],
    intent_from_record: Callable[[dict[str, Any]], dict[str, Any]],
    validate_existing: Callable[[dict[str, Any]], None] | None = None,
    actor_fields: dict[str, Any] | None = None,
    case_lock_already_held: bool = False,
) -> tuple[dict[str, Any], bool]:
    if record is None and record_factory is None:
        raise RecoveryRunValidationError("record or record_factory required")
    if record is not None and record_factory is not None:
        raise RecoveryRunValidationError("record and record_factory are mutually exclusive")

    def _resolved_projection() -> dict[str, Any]:
        return request_projection() if callable(request_projection) else request_projection

    flags = _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
    flags_read = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
    record_ctx = f"recovery_runs/{recovery_request_id}/{filename}"
    if not case_lock_already_held:
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
                    raise RecoveryRunValidationError(
                        f"{filename} exists but is not complete valid JSON"
                    ) from exc
            finally:
                os.close(existing_fd)
            if actor_fields is not None:
                for field, expected in actor_fields.items():
                    if existing.get(field) != expected:
                        raise RecoveryRunConflictError(
                            f"{filename} already exists with conflicting semantics",
                            recovery_request_id=recovery_request_id,
                        )
            if validate_existing is not None:
                validate_existing(existing)
            projection = _resolved_projection()
            if intent_from_record(existing) != projection:
                raise RecoveryRunConflictError(
                    f"{filename} already exists with conflicting semantics",
                    recovery_request_id=recovery_request_id,
                )
            return existing, True
        record_body = record if record is not None else record_factory()
        payload = (json.dumps(record_body, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            _write_all(file_fd, payload, recovery_request_id=recovery_request_id, record_name=filename)
            _fsync_file_fd(file_fd, recovery_request_id=recovery_request_id, record_name=filename)
            _fsync_recovery_durability_chain(
                case_fd,
                recovery_request_id=recovery_request_id,
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
        if not case_lock_already_held:
            fcntl.flock(case_fd, fcntl.LOCK_UN)


def _validate_directory_state(
    entries: frozenset[str],
    *,
    allowed: frozenset[str],
    recovery_request_id: str,
) -> None:
    if entries <= allowed:
        return
    raise RecoveryRunStateError(
        f"unexpected recovery run directory entries: {sorted(entries)}",
        recovery_request_id=recovery_request_id,
    )


def _load_validated_request(recovery_request_id: str, base_dir: Path | None) -> dict[str, Any]:
    record = _read_optional_record(paths.recovery_run_request_path(recovery_request_id, base_dir))
    if record is None:
        raise RecoveryRunStateError(
            f"disposition {recovery_request_id!r} is missing request.json",
            recovery_request_id=recovery_request_id,
        )
    _validate_record_digest(
        record,
        digest_field="request_digest",
        projection_fn=_request_digest_projection,
    )
    return record



def _to_request_record(body: dict[str, Any]) -> RecoveryRunRequestRecord:
    return RecoveryRunRequestRecord(
        recovery_request_id=body["recovery_request_id"],
        request_digest=body["request_digest"],
        reconciliation_case_id=body["reconciliation_case_id"],
        recovery_of_run_id=body["recovery_of_run_id"],
        successor_run_id=body["successor_run_id"],
        recovery_scope=body["recovery_scope"],
        requested_by=body["requested_by"],
        requested_at=body["requested_at"],
    )


def _to_issue_record(body: dict[str, Any]) -> RecoveryRunIssueRecord:
    return RecoveryRunIssueRecord(
        recovery_request_id=body["recovery_request_id"],
        recovery_approval_id=body["recovery_approval_id"],
        issue_digest=body["issue_digest"],
        issued_by=body["issued_by"],
        issued_at=body["issued_at"],
        expires_at=body["expires_at"],
    )


def _to_claim_record(body: dict[str, Any]) -> RecoveryRunClaimRecord:
    return RecoveryRunClaimRecord(
        recovery_request_id=body["recovery_request_id"],
        claim_id=body["claim_id"],
        claim_digest=body["claim_digest"],
        claimant=body["claimant"],
        claimed_at=body["claimed_at"],
    )


def _to_attempt_record(body: dict[str, Any]) -> RecoveryRunAttemptRecord:
    return RecoveryRunAttemptRecord(
        recovery_request_id=body["recovery_request_id"],
        attempt_id=body["attempt_id"],
        attempt_digest=body["attempt_digest"],
        executor=body["executor"],
        attempted_at=body["attempted_at"],
    )


def _to_outcome_record(body: dict[str, Any]) -> RecoveryRunOutcomeRecord:
    return RecoveryRunOutcomeRecord(
        recovery_request_id=body["recovery_request_id"],
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
        raise RecoveryRunValidationError("recovery approval revoked before claim")
    expires = _parse_utc_iso(issue["expires_at"])
    if now >= expires:
        raise RecoveryRunValidationError("recovery approval expired")



def _outcome_body(
    *,
    recovery_request_id: str,
    outcome_class: RecoveryRunOutcomeClass,
    request_digest: str,
    issue_digest: str,
    claim_digest: str,
    attempt_digest: str | None,
    successor_run_id: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "outcome_digest_projection_version": OUTCOME_DIGEST_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "outcome_class": outcome_class.value,
        "request_digest": request_digest,
        "issue_digest": issue_digest,
        "claim_digest": claim_digest,
        "attempt_digest": attempt_digest,
        "successor_run_id": successor_run_id,
        "recorded_at": _utc_now_iso(),
        "source_run_mutation_allowed": False,
        "retry_allowed": False,
        "repair_allowed": False,
        "invoke_allowed": False,
        "automatic_execution_allowed": False,
        "outcome_rewrite_allowed": False,
    }
    body["outcome_digest"] = _sha256_digest(_outcome_digest_projection(body))
    return body


def _persist_outcome_private(
    *,
    recovery_request_id: str,
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
        recovery_request_id=recovery_request_id,
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



def create_recovery_run_request(
    recovery_request_id: str,
    reconciliation_case_id: str,
    *,
    recovery_scope: RecoveryScope | str,
    recovery_reason: str,
    successor_run_id: str,
    requested_by: str,
    base_dir: Path | None = None,
) -> tuple[RecoveryRunRequestRecord, RecoveryRunWriteMetadata]:
    _require_valid_id(recovery_request_id, "recovery_run_request")
    _require_valid_id(reconciliation_case_id, "reconciliation")
    _require_valid_id(successor_run_id, "run")
    requested_by = _validate_actor(requested_by, field="requested_by")
    scope_value = _validate_recovery_scope(recovery_scope)
    reason_value = _validate_recovery_reason(recovery_reason)

    proof_cache: dict[str, Any] = {}

    def _proof_bundle() -> dict[str, Any]:
        if "proof" not in proof_cache:
            proof_cache["proof"] = _prove_path_r1_eligibility(
                case_id=reconciliation_case_id, base_dir=base_dir
            )
        return proof_cache["proof"]

    def _request_intent() -> dict[str, Any]:
        proof = _proof_bundle()
        return {
            "request_intent_projection_version": REQUEST_INTENT_PROJECTION_VERSION,
            "recovery_request_id": recovery_request_id,
            "reconciliation_case_id": reconciliation_case_id,
            "requested_by": requested_by,
            "recovery_of_run_id": proof["recovery_of_run_id"],
            "successor_run_id": successor_run_id,
            "recovery_scope": scope_value,
            "recovery_reason": reason_value,
            "decision_class": proof["decision_class"],
            "recommended_next_protocol": proof["recommended_next_protocol"],
        }

    def _request_body() -> dict[str, Any]:
        proof = _proof_bundle()
        task25 = proof["task25"]
        requested_at = _utc_now_iso()
        body: dict[str, Any] = {
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "request_digest_projection_version": REQUEST_DIGEST_PROJECTION_VERSION,
            "recovery_request_id": recovery_request_id,
            "reconciliation_case_id": reconciliation_case_id,
            "case_open_digest": proof["case_open_digest"],
            "observation_digest": proof["observation_digest"],
            "decision_digest": proof["decision_digest"],
            "decision_class": proof["decision_class"],
            "recommended_next_protocol": proof["recommended_next_protocol"],
            "task26a_observation_inspection_digest": proof["task26a_observation_inspection_digest"],
            "task26b_decision_revalidation_inspection_digest": proof["task26b_decision_revalidation_inspection_digest"],
            "task25_approval_id": task25["approval_id"],
            "task25_approval_issue_digest": task25["approval_issue_digest"],
            "task25_claim_id": task25["claim_id"],
            "task25_claim_digest": task25["claim_digest"],
            "task25_consumed_outcome_digest": task25["consumed_outcome_digest"],
            "recovery_of_run_id": proof["recovery_of_run_id"],
            "successor_run_id": successor_run_id,
            "recovery_scope": scope_value,
            "recovery_reason": reason_value,
            "event_id": proof["event_id"],
            "htr_runs_root_path_digest": proof["htr_runs_root_path_digest"],
            "htr_project_dir_path_digest": proof["htr_project_dir_path_digest"],
            "finalized_valid_seal_evidence": proof["finalized_valid_seal_evidence"],
            "requested_by": requested_by,
            "requested_at": requested_at,
        }
        body["request_digest"] = _sha256_digest(_request_digest_projection(body))
        return body

    _recovery_fd, case_fd, _identity = _bootstrap_recovery_tree(recovery_request_id, base_dir)
    try:
        persisted, exact_replay = _create_immutable_record(
            recovery_request_id=recovery_request_id,
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
        _validate_directory_state(post, allowed=_REQUEST_ALLOWED, recovery_request_id=recovery_request_id)
        return _to_request_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)
        os.close(_recovery_fd)


def issue_recovery_run_approval(
    recovery_request_id: str,
    recovery_approval_id: str,
    *,
    issued_by: str,
    expires_at: str,
    base_dir: Path | None = None,
) -> tuple[RecoveryRunIssueRecord, RecoveryRunWriteMetadata]:
    _require_valid_id(recovery_request_id, "recovery_run_request")
    _require_valid_id(recovery_approval_id, "recovery_run_approval")
    issued_by = _validate_actor(issued_by, field="issued_by")

    request = _load_validated_request(recovery_request_id, base_dir)
    issued_at = _utc_now_iso()
    _validate_expiry(issued_at=issued_at, expires_at=expires_at)

    intent = {
        "issue_intent_projection_version": ISSUE_INTENT_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "recovery_approval_id": recovery_approval_id,
        "issued_by": issued_by,
        "expires_at": expires_at,
    }

    replay_issued_at = issued_at
    body: dict[str, Any] = {
        "issue_schema_version": ISSUE_SCHEMA_VERSION,
        "issue_digest_projection_version": ISSUE_DIGEST_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "recovery_approval_id": recovery_approval_id,
        "request_digest": request["request_digest"],
        "issued_by": issued_by,
        "issued_at": replay_issued_at,
        "expires_at": expires_at,
    }
    body["issue_digest"] = _sha256_digest(_issue_digest_projection(body))

    case_dir = paths.recovery_run_control_dir(recovery_request_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"recovery_runs/{recovery_request_id}")
    try:
        persisted, exact_replay = _create_immutable_record(
            recovery_request_id=recovery_request_id,
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
        _validate_directory_state(post, allowed=_ISSUED_ALLOWED, recovery_request_id=recovery_request_id)
        return _to_issue_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        os.close(case_fd)


def revoke_recovery_run_approval(
    recovery_request_id: str,
    *,
    revoked_by: str,
    reason: str,
    base_dir: Path | None = None,
) -> tuple[dict[str, Any], RecoveryRunWriteMetadata]:
    _require_valid_id(recovery_request_id, "recovery_run_request")
    revoked_by = _validate_actor(revoked_by, field="revoked_by")
    if not isinstance(reason, str) or not reason.strip():
        raise RecoveryRunValidationError("revoke reason must be a non-empty string")

    request = _load_validated_request(recovery_request_id, base_dir)
    issue = _read_optional_record(paths.recovery_run_issue_path(recovery_request_id, base_dir))
    if issue is None:
        raise RecoveryRunStateError("issue.json missing", recovery_request_id=recovery_request_id)
    _validate_record_digest(issue, digest_field="issue_digest", projection_fn=_issue_digest_projection)

    intent = {
        "revoke_intent_projection_version": REVOKE_INTENT_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "revoked_by": revoked_by,
    }

    revoked_at = _utc_now_iso()
    body: dict[str, Any] = {
        "revoke_schema_version": REVOKE_SCHEMA_VERSION,
        "revoke_digest_projection_version": REVOKE_DIGEST_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "issue_digest": issue["issue_digest"],
        "request_digest": request["request_digest"],
        "revoked_by": revoked_by,
        "revoked_at": revoked_at,
        "reason": reason.strip(),
    }
    body["revoke_digest"] = _sha256_digest(_revoke_digest_projection(body))

    case_dir = paths.recovery_run_control_dir(recovery_request_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"recovery_runs/{recovery_request_id}")
    fcntl.flock(case_fd, fcntl.LOCK_EX)
    try:
        claim = _read_optional_record_at(case_fd, "claim.json")
        if claim is not None:
            _validate_record_digest(
                claim,
                digest_field="claim_digest",
                projection_fn=_claim_digest_projection,
            )
            raise RecoveryRunConflictError(
                "revoke after claim is conflicting/ineffective",
                recovery_request_id=recovery_request_id,
            )
        persisted, exact_replay = _create_immutable_record(
            recovery_request_id=recovery_request_id,
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
            case_lock_already_held=True,
        )
        post = _list_dir_entries(case_fd)
        _validate_directory_state(post, allowed=_REVOKED_ALLOWED, recovery_request_id=recovery_request_id)
        return persisted, _write_metadata(exact_replay=exact_replay)
    finally:
        fcntl.flock(case_fd, fcntl.LOCK_UN)
        os.close(case_fd)


def claim_recovery_run_approval(
    recovery_request_id: str,
    claim_id: str,
    *,
    claimant: str,
    base_dir: Path | None = None,
) -> tuple[RecoveryRunClaimRecord, RecoveryRunWriteMetadata]:
    _require_valid_id(recovery_request_id, "recovery_run_request")
    _require_valid_id(claim_id, "recovery_run_claim")
    claimant = _validate_actor(claimant, field="claimant")

    request = _load_validated_request(recovery_request_id, base_dir)
    issue = _read_optional_record(paths.recovery_run_issue_path(recovery_request_id, base_dir))
    if issue is None:
        raise RecoveryRunStateError("issue.json missing", recovery_request_id=recovery_request_id)
    _validate_record_digest(issue, digest_field="issue_digest", projection_fn=_issue_digest_projection)

    intent = {
        "claim_intent_projection_version": CLAIM_INTENT_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "claim_id": claim_id,
        "claimant": claimant,
    }

    claimed_at = _utc_now_iso()
    body: dict[str, Any] = {
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "claim_digest_projection_version": CLAIM_DIGEST_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "recovery_approval_id": issue["recovery_approval_id"],
        "issue_digest": issue["issue_digest"],
        "request_digest": request["request_digest"],
        "claim_id": claim_id,
        "claimant": claimant,
        "claimed_at": claimed_at,
    }
    body["claim_digest"] = _sha256_digest(_claim_digest_projection(body))

    case_dir = paths.recovery_run_control_dir(recovery_request_id, base_dir)
    case_fd = _open_control_dir_no_follow(case_dir, context=f"recovery_runs/{recovery_request_id}")
    fcntl.flock(case_fd, fcntl.LOCK_EX)
    try:
        revoke = _read_optional_record_at(case_fd, "revoke.json")
        if revoke is not None:
            _validate_record_digest(
                revoke,
                digest_field="revoke_digest",
                projection_fn=_revoke_digest_projection,
            )
            raise RecoveryRunValidationError("cannot claim revoked recovery approval")
        _assert_approval_active(issue=issue, revoke=revoke, claim=None)
        persisted, exact_replay = _create_immutable_record(
            recovery_request_id=recovery_request_id,
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
            case_lock_already_held=True,
        )
        post = _list_dir_entries(case_fd)
        _validate_directory_state(post, allowed=_CLAIMED_ALLOWED, recovery_request_id=recovery_request_id)
        return _to_claim_record(persisted), _write_metadata(exact_replay=exact_replay)
    finally:
        fcntl.flock(case_fd, fcntl.LOCK_UN)
        os.close(case_fd)


def load_recovery_run_bundle(
    recovery_request_id: str,
    *,
    base_dir: Path | None = None,
) -> RecoveryRunBundle:
    _require_valid_id(recovery_request_id, "recovery_run_request")
    request = _read_optional_record(paths.recovery_run_request_path(recovery_request_id, base_dir))
    issue = _read_optional_record(paths.recovery_run_issue_path(recovery_request_id, base_dir))
    revoke = _read_optional_record(paths.recovery_run_revoke_path(recovery_request_id, base_dir))
    claim = _read_optional_record(paths.recovery_run_claim_path(recovery_request_id, base_dir))
    attempt = _read_optional_record(paths.recovery_run_attempt_path(recovery_request_id, base_dir))
    outcome = _read_optional_record(paths.recovery_run_outcome_path(recovery_request_id, base_dir))

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

    return RecoveryRunBundle(
        recovery_request_id=recovery_request_id,
        request_record=req_rec,
        issue_record=_to_issue_record(issue) if issue else None,
        revoke_record=revoke,
        claim_record=_to_claim_record(claim) if claim else None,
        attempt_record=_to_attempt_record(attempt) if attempt else None,
        outcome_record=_to_outcome_record(outcome) if outcome else None,
    )






def _recovery_origin_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "recovery_origin_schema_version": body["recovery_origin_schema_version"],
        "recovery_origin_digest_projection_version": body["recovery_origin_digest_projection_version"],
        "successor_run_id": body["successor_run_id"],
        "recovery_of_run_id": body["recovery_of_run_id"],
        "finalized_valid_seal_evidence": body["finalized_valid_seal_evidence"],
        "htr_runs_root_path_digest": body["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": body["htr_project_dir_path_digest"],
        "task26a_observation_inspection_digest": body["task26a_observation_inspection_digest"],
        "task26b_decision_revalidation_inspection_digest": body["task26b_decision_revalidation_inspection_digest"],
        "task25_consumed_outcome_digest": body["task25_consumed_outcome_digest"],
        "recovery_request_digest": body["recovery_request_digest"],
        "recovery_issue_digest": body["recovery_issue_digest"],
        "recovery_claim_digest": body["recovery_claim_digest"],
        "recovery_attempt_digest": body["recovery_attempt_digest"],
        "recovery_scope": body["recovery_scope"],
        "created_at": body["created_at"],
        "source_run_mutation_allowed": body["source_run_mutation_allowed"],
        "retry_allowed": body["retry_allowed"],
        "repair_allowed": body["repair_allowed"],
        "invoke_allowed": body["invoke_allowed"],
        "automatic_execution_allowed": body["automatic_execution_allowed"],
        "outcome_rewrite_allowed": body["outcome_rewrite_allowed"],
    }


def _build_recovery_origin_body(
    *,
    request: dict[str, Any],
    issue: dict[str, Any],
    claim: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    created_at = _utc_now_iso()
    body: dict[str, Any] = {
        "recovery_origin_schema_version": RECOVERY_ORIGIN_SCHEMA_VERSION,
        "recovery_origin_digest_projection_version": RECOVERY_ORIGIN_DIGEST_PROJECTION_VERSION,
        "successor_run_id": request["successor_run_id"],
        "recovery_of_run_id": request["recovery_of_run_id"],
        "finalized_valid_seal_evidence": request["finalized_valid_seal_evidence"],
        "htr_runs_root_path_digest": request["htr_runs_root_path_digest"],
        "htr_project_dir_path_digest": request["htr_project_dir_path_digest"],
        "task26a_observation_inspection_digest": request["task26a_observation_inspection_digest"],
        "task26b_decision_revalidation_inspection_digest": request["task26b_decision_revalidation_inspection_digest"],
        "task25_consumed_outcome_digest": request["task25_consumed_outcome_digest"],
        "recovery_request_digest": request["request_digest"],
        "recovery_issue_digest": issue["issue_digest"],
        "recovery_claim_digest": claim["claim_digest"],
        "recovery_attempt_digest": attempt["attempt_digest"],
        "recovery_scope": request["recovery_scope"],
        "created_at": created_at,
        "source_run_mutation_allowed": False,
        "retry_allowed": False,
        "repair_allowed": False,
        "invoke_allowed": False,
        "automatic_execution_allowed": False,
        "outcome_rewrite_allowed": False,
    }
    body["recovery_origin_digest"] = _sha256_digest(_recovery_origin_digest_projection(body))
    return body


def _write_recovery_origin_exclusive(reservation: RunRootReservation, body: dict[str, Any]) -> None:
    payload = (json.dumps(body, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    file_fd = _openat_control_file_no_follow(
        reservation.run_root_fd,
        "recovery_origin.json",
        _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC,
        _CONTROL_FILE_MODE,
        context="recovery_origin.json",
    )
    try:
        _write_all(file_fd, payload, recovery_request_id=body.get("successor_run_id", ""), record_name="recovery_origin.json")
        _fsync_file_fd(file_fd, recovery_request_id=body.get("successor_run_id", ""), record_name="recovery_origin.json")
    finally:
        os.close(file_fd)
    _fsync_dir_fd(reservation.run_root_fd, recovery_request_id=body.get("successor_run_id", ""), record_name="recovery_origin.json", stage="successor_root_fsync")
    _fsync_dir_fd(reservation.runs_root_fd, recovery_request_id=body.get("successor_run_id", ""), record_name="recovery_origin.json", stage="runs_root_fsync")


def _verify_successor_initial_state(successor_run_id: str, base_dir: Path | None) -> bool:
    root = paths.run_root(successor_run_id, base_dir)
    required_files = ("recovery_origin.json", "run_manifest.json", "task_events.jsonl", "approvals.jsonl")
    required_dirs = ("reports", "tasks")
    for name in required_files:
        path = root / name
        if not path.is_file() or path.is_symlink():
            return False
    for name in required_dirs:
        path = root / name
        if not path.is_dir() or path.is_symlink():
            return False
    marker = paths.runs_root(base_dir) / LOCKS_DIR_NAME / f"{successor_run_id}.marker"
    if marker.exists():
        return False
    manifest = read_json(paths.run_manifest_path(successor_run_id, base_dir))
    if manifest.get("status") != "created":
        return False
    if any(paths.tasks_dir(successor_run_id, base_dir).iterdir()):
        return False
    return True


def _validate_recovery_origin_bindings(
    origin: dict[str, Any],
    *,
    request: dict[str, Any],
    issue: dict[str, Any],
    claim: dict[str, Any],
    attempt: dict[str, Any],
) -> None:
    stored_digest = origin.get("recovery_origin_digest")
    if not isinstance(stored_digest, str) or not stored_digest:
        raise RecoveryRunValidationError("missing recovery_origin_digest")
    computed = _sha256_digest(_recovery_origin_digest_projection(origin))
    if stored_digest != computed:
        raise RecoveryRunValidationError("recovery_origin_digest mismatch")
    _validate_non_permission_booleans(origin)
    bindings = (
        ("successor_run_id", request["successor_run_id"]),
        ("recovery_of_run_id", request["recovery_of_run_id"]),
        ("finalized_valid_seal_evidence", request["finalized_valid_seal_evidence"]),
        ("htr_runs_root_path_digest", request["htr_runs_root_path_digest"]),
        ("htr_project_dir_path_digest", request["htr_project_dir_path_digest"]),
        ("task26a_observation_inspection_digest", request["task26a_observation_inspection_digest"]),
        (
            "task26b_decision_revalidation_inspection_digest",
            request["task26b_decision_revalidation_inspection_digest"],
        ),
        ("task25_consumed_outcome_digest", request["task25_consumed_outcome_digest"]),
        ("recovery_request_digest", request["request_digest"]),
        ("recovery_issue_digest", issue["issue_digest"]),
        ("recovery_claim_digest", claim["claim_digest"]),
        ("recovery_attempt_digest", attempt["attempt_digest"]),
        ("recovery_scope", request["recovery_scope"]),
    )
    for field, expected in bindings:
        if origin.get(field) != expected:
            raise RecoveryRunValidationError(f"recovery_origin {field} binding mismatch")


def _successor_exists_verified(request: dict[str, Any], issue: dict[str, Any], claim: dict[str, Any], attempt: dict[str, Any], base_dir: Path | None) -> bool:
    successor_run_id = request["successor_run_id"]
    origin_path = paths.recovery_origin_path(successor_run_id, base_dir)
    if not origin_path.is_file() or origin_path.is_symlink():
        return False
    try:
        origin = read_json(origin_path)
        _validate_recovery_origin_bindings(
            origin,
            request=request,
            issue=issue,
            claim=claim,
            attempt=attempt,
        )
    except (RecoveryRunValidationError, json.JSONDecodeError, OSError):
        return False
    return _verify_successor_initial_state(successor_run_id, base_dir)


def _execution_outcome_after_coordination(
    *,
    recovery_request_id: str,
    attempt_id: str,
    executor: str,
    base_dir: Path | None,
    request: dict[str, Any],
    issue: dict[str, Any],
    claim: dict[str, Any],
) -> RecoveryRunExecutionResult | None:
    exec_intent = {
        "attempt_intent_projection_version": ATTEMPT_INTENT_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "attempt_id": attempt_id,
        "executor": executor,
    }
    existing_attempt = _read_optional_record(paths.recovery_run_attempt_path(recovery_request_id, base_dir))
    existing_outcome = _read_optional_record(paths.recovery_run_outcome_path(recovery_request_id, base_dir))
    if existing_outcome is not None:
        _validate_record_digest(existing_outcome, digest_field="outcome_digest", projection_fn=_outcome_digest_projection)
        _validate_non_permission_booleans(existing_outcome)
        if existing_attempt is not None and _attempt_intent_projection(existing_attempt) != exec_intent:
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=request["successor_run_id"],
                outcome_class=RecoveryRunOutcomeClass.execution_ambiguous.value,
                outcome_digest=existing_outcome["outcome_digest"],
                exact_replay=False,
            )
        if (
            existing_outcome.get("request_digest") == request["request_digest"]
            and existing_outcome.get("issue_digest") == issue["issue_digest"]
            and existing_outcome.get("claim_digest") == claim["claim_digest"]
        ):
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=existing_outcome.get("successor_run_id") or request["successor_run_id"],
                outcome_class=existing_outcome["outcome_class"],
                outcome_digest=existing_outcome["outcome_digest"],
                exact_replay=True,
            )
        raise RecoveryRunConflictError(
            "outcome already exists with conflicting lineage",
            recovery_request_id=recovery_request_id,
        )
    if existing_attempt is not None:
        _validate_record_digest(existing_attempt, digest_field="attempt_digest", projection_fn=_attempt_digest_projection)
        if _attempt_intent_projection(existing_attempt) != exec_intent:
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=request["successor_run_id"],
                outcome_class=RecoveryRunOutcomeClass.execution_ambiguous.value,
                outcome_digest=existing_attempt["attempt_digest"],
                exact_replay=False,
            )
        if _successor_exists_verified(request, issue, claim, existing_attempt, base_dir):
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=request["successor_run_id"],
                outcome_class=RecoveryRunOutcomeClass.successor_already_exists_verified.value,
                outcome_digest=existing_attempt["attempt_digest"],
                exact_replay=True,
            )
        return None
    return None


def execute_approved_successor_run_creation(
    recovery_request_id: str,
    attempt_id: str,
    *,
    executor: str,
    base_dir: Path | None = None,
) -> RecoveryRunExecutionResult:
    _require_valid_id(recovery_request_id, "recovery_run_request")
    _require_valid_id(attempt_id, "recovery_run_attempt")
    executor = _validate_actor(executor, field="executor")

    request = _load_validated_request(recovery_request_id, base_dir)
    issue = _read_optional_record(paths.recovery_run_issue_path(recovery_request_id, base_dir))
    if issue is None:
        raise RecoveryRunStateError("issue.json missing", recovery_request_id=recovery_request_id)
    _validate_record_digest(issue, digest_field="issue_digest", projection_fn=_issue_digest_projection)
    revoke = _read_optional_record(paths.recovery_run_revoke_path(recovery_request_id, base_dir))
    claim = _read_optional_record(paths.recovery_run_claim_path(recovery_request_id, base_dir))
    if claim is None:
        raise RecoveryRunValidationError("claim.json required before execution")
    _validate_record_digest(claim, digest_field="claim_digest", projection_fn=_claim_digest_projection)

    raced = _execution_outcome_after_coordination(
        recovery_request_id=recovery_request_id,
        attempt_id=attempt_id,
        executor=executor,
        base_dir=base_dir,
        request=request,
        issue=issue,
        claim=claim,
    )
    if raced is not None:
        return raced

    try:
        _assert_approval_active(issue=issue, revoke=revoke, claim=claim)
    except RecoveryRunValidationError:
        case_fd = _open_control_dir_no_follow(
            paths.recovery_run_control_dir(recovery_request_id, base_dir),
            context=f"recovery_runs/{recovery_request_id}",
        )
        try:
            body = _outcome_body(
                recovery_request_id=recovery_request_id,
                outcome_class=RecoveryRunOutcomeClass.approval_invalid,
                request_digest=request["request_digest"],
                issue_digest=issue["issue_digest"],
                claim_digest=claim["claim_digest"],
                attempt_digest=None,
                successor_run_id=request["successor_run_id"],
            )
            persisted = _persist_outcome_private(
                recovery_request_id=recovery_request_id,
                case_fd=case_fd,
                body=body,
                intent_projection={
                    "outcome_class": body["outcome_class"],
                    "request_digest": body["request_digest"],
                    "issue_digest": body["issue_digest"],
                    "claim_digest": body["claim_digest"],
                    "attempt_digest": body["attempt_digest"],
                },
            )
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=request["successor_run_id"],
                outcome_class=persisted["outcome_class"],
                outcome_digest=persisted["outcome_digest"],
                exact_replay=False,
            )
        finally:
            os.close(case_fd)

    try:
        proof = _prove_path_r1_execution_eligibility(request=request, base_dir=base_dir)
    except (RecoveryRunValidationError, ReconciliationValidationError):
        case_fd = _open_control_dir_no_follow(
            paths.recovery_run_control_dir(recovery_request_id, base_dir),
            context=f"recovery_runs/{recovery_request_id}",
        )
        try:
            body = _outcome_body(
                recovery_request_id=recovery_request_id,
                outcome_class=RecoveryRunOutcomeClass.source_evidence_drifted,
                request_digest=request["request_digest"],
                issue_digest=issue["issue_digest"],
                claim_digest=claim["claim_digest"],
                attempt_digest=None,
                successor_run_id=request["successor_run_id"],
            )
            persisted = _persist_outcome_private(
                recovery_request_id=recovery_request_id,
                case_fd=case_fd,
                body=body,
                intent_projection={
                    "outcome_class": body["outcome_class"],
                    "request_digest": body["request_digest"],
                    "issue_digest": body["issue_digest"],
                    "claim_digest": body["claim_digest"],
                    "attempt_digest": body["attempt_digest"],
                },
            )
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=request["successor_run_id"],
                outcome_class=persisted["outcome_class"],
                outcome_digest=persisted["outcome_digest"],
                exact_replay=False,
            )
        finally:
            os.close(case_fd)

    successor_run_id = request["successor_run_id"]
    existing_attempt_pre = _read_optional_record(
        paths.recovery_run_attempt_path(recovery_request_id, base_dir)
    )
    if paths.run_root(successor_run_id, base_dir).exists():
        if (
            existing_attempt_pre is not None
            and _successor_exists_verified(
                request, issue, claim, existing_attempt_pre, base_dir
            )
        ):
            case_fd = _open_control_dir_no_follow(
                paths.recovery_run_control_dir(recovery_request_id, base_dir),
                context=f"recovery_runs/{recovery_request_id}",
            )
            try:
                body = _outcome_body(
                    recovery_request_id=recovery_request_id,
                    outcome_class=RecoveryRunOutcomeClass.successor_already_exists_verified,
                    request_digest=request["request_digest"],
                    issue_digest=issue["issue_digest"],
                    claim_digest=claim["claim_digest"],
                    attempt_digest=None,
                    successor_run_id=successor_run_id,
                )
                persisted = _persist_outcome_private(
                    recovery_request_id=recovery_request_id,
                    case_fd=case_fd,
                    body=body,
                    intent_projection={
                        "outcome_class": body["outcome_class"],
                        "request_digest": body["request_digest"],
                        "issue_digest": body["issue_digest"],
                        "claim_digest": body["claim_digest"],
                        "attempt_digest": body["attempt_digest"],
                    },
                )
                return RecoveryRunExecutionResult(
                    recovery_request_id=recovery_request_id,
                    successor_run_id=successor_run_id,
                    outcome_class=persisted["outcome_class"],
                    outcome_digest=persisted["outcome_digest"],
                    exact_replay=True,
                )
            finally:
                os.close(case_fd)
        case_fd = _open_control_dir_no_follow(
            paths.recovery_run_control_dir(recovery_request_id, base_dir),
            context=f"recovery_runs/{recovery_request_id}",
        )
        try:
            body = _outcome_body(
                recovery_request_id=recovery_request_id,
                outcome_class=RecoveryRunOutcomeClass.successor_id_conflict,
                request_digest=request["request_digest"],
                issue_digest=issue["issue_digest"],
                claim_digest=claim["claim_digest"],
                attempt_digest=None,
                successor_run_id=successor_run_id,
            )
            persisted = _persist_outcome_private(
                recovery_request_id=recovery_request_id,
                case_fd=case_fd,
                body=body,
                intent_projection={
                    "outcome_class": body["outcome_class"],
                    "request_digest": body["request_digest"],
                    "issue_digest": body["issue_digest"],
                    "claim_digest": body["claim_digest"],
                    "attempt_digest": body["attempt_digest"],
                },
            )
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=successor_run_id,
                outcome_class=persisted["outcome_class"],
                outcome_digest=persisted["outcome_digest"],
                exact_replay=False,
            )
        finally:
            os.close(case_fd)

    exec_intent = {
        "attempt_intent_projection_version": ATTEMPT_INTENT_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "attempt_id": attempt_id,
        "executor": executor,
    }
    revalidation_envelope = {
        "revalidation_projection_version": EXECUTION_REVALIDATION_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "request_digest": request["request_digest"],
        "execution_inspection_digest": proof["execution_inspection_digest"],
        "execution_inspection_projection": proof["execution_inspection_projection"],
    }
    revalidation_envelope["execution_revalidation_digest"] = _sha256_digest(revalidation_envelope)
    attempted_at = _utc_now_iso()
    attempt_body = {
        "attempt_schema_version": ATTEMPT_SCHEMA_VERSION,
        "attempt_digest_projection_version": ATTEMPT_DIGEST_PROJECTION_VERSION,
        "recovery_request_id": recovery_request_id,
        "request_digest": request["request_digest"],
        "issue_digest": issue["issue_digest"],
        "claim_digest": claim["claim_digest"],
        "attempt_id": attempt_id,
        "executor": executor,
        "attempted_at": attempted_at,
        "successor_run_id": successor_run_id,
        "execution_revalidation": revalidation_envelope,
    }
    attempt_body["attempt_digest"] = _sha256_digest(_attempt_digest_projection(attempt_body))

    case_fd = _open_recovery_case_fd(recovery_request_id, base_dir)
    outcome_class = RecoveryRunOutcomeClass.creation_failed
    reservation: RunRootReservation | None = None
    try:
        with _case_execution_barrier(case_fd):
            raced = _execution_outcome_after_coordination(
                recovery_request_id=recovery_request_id,
                attempt_id=attempt_id,
                executor=executor,
                base_dir=base_dir,
                request=request,
                issue=issue,
                claim=claim,
            )
            if raced is not None:
                return raced
            entries = _list_dir_entries(case_fd)
            _validate_directory_state(entries, allowed=_CLAIMED_ALLOWED, recovery_request_id=recovery_request_id)
            try:
                persisted_attempt, _ = _create_immutable_record(
                    recovery_request_id=recovery_request_id,
                    case_fd=case_fd,
                    filename="attempt.json",
                    record=attempt_body,
                    request_projection=exec_intent,
                    intent_from_record=_attempt_intent_projection,
                    validate_existing=lambda rec: _validate_record_digest(
                        rec, digest_field="attempt_digest", projection_fn=_attempt_digest_projection
                    ),
                )
                attempt_body = persisted_attempt
            except RecoveryRunConflictError:
                body = _outcome_body(
                    recovery_request_id=recovery_request_id,
                    outcome_class=RecoveryRunOutcomeClass.execution_ambiguous,
                    request_digest=request["request_digest"],
                    issue_digest=issue["issue_digest"],
                    claim_digest=claim["claim_digest"],
                    attempt_digest=attempt_body.get("attempt_digest"),
                    successor_run_id=successor_run_id,
                )
                persisted = _persist_outcome_private(
                    recovery_request_id=recovery_request_id,
                    case_fd=case_fd,
                    body=body,
                    intent_projection={
                        "outcome_class": body["outcome_class"],
                        "request_digest": body["request_digest"],
                        "issue_digest": body["issue_digest"],
                        "claim_digest": body["claim_digest"],
                        "attempt_digest": body["attempt_digest"],
                    },
                )
                return RecoveryRunExecutionResult(
                    recovery_request_id=recovery_request_id,
                    successor_run_id=successor_run_id,
                    outcome_class=persisted["outcome_class"],
                    outcome_digest=persisted["outcome_digest"],
                    exact_replay=False,
                )

            try:
                try:
                    reservation = reserve_run_root_exclusive(successor_run_id, base_dir)
                except RunRootReservationError:
                    outcome_class = RecoveryRunOutcomeClass.successor_id_conflict
                    raise
                origin_body = _build_recovery_origin_body(
                    request=request, issue=issue, claim=claim, attempt=attempt_body
                )
                _write_recovery_origin_exclusive(reservation, origin_body)
                bootstrap_reserved_run_workspace(successor_run_id, base_dir, reservation=reservation)
                if not _verify_successor_initial_state(successor_run_id, base_dir):
                    outcome_class = RecoveryRunOutcomeClass.creation_partial
                    raise RecoveryRunValidationError("successor initial state verification failed")
                outcome_class = RecoveryRunOutcomeClass.successor_created_verified
            except RecoveryRunDurabilityError:
                raise
            except Exception:
                if reservation is not None:
                    release_run_root_reservation(reservation)
            finally:
                if reservation is not None and outcome_class == RecoveryRunOutcomeClass.successor_created_verified:
                    release_run_root_reservation(reservation)

            body = _outcome_body(
                recovery_request_id=recovery_request_id,
                outcome_class=outcome_class,
                request_digest=request["request_digest"],
                issue_digest=issue["issue_digest"],
                claim_digest=claim["claim_digest"],
                attempt_digest=attempt_body["attempt_digest"],
                successor_run_id=successor_run_id,
            )
            persisted = _persist_outcome_private(
                recovery_request_id=recovery_request_id,
                case_fd=case_fd,
                body=body,
                intent_projection={
                    "outcome_class": body["outcome_class"],
                    "request_digest": body["request_digest"],
                    "issue_digest": body["issue_digest"],
                    "claim_digest": body["claim_digest"],
                    "attempt_digest": body["attempt_digest"],
                },
            )
            return RecoveryRunExecutionResult(
                recovery_request_id=recovery_request_id,
                successor_run_id=successor_run_id,
                outcome_class=persisted["outcome_class"],
                outcome_digest=persisted["outcome_digest"],
                exact_replay=False,
            )
    finally:
        os.close(case_fd)



def reconcile_recovery_run_creation(
    recovery_request_id: str,
    *,
    base_dir: Path | None = None,
) -> RecoveryRunReconcileResult:
    """Literal read-only reconciliation — performs zero writes."""
    _require_valid_id(recovery_request_id, "recovery_run_request")
    request = _read_optional_record(paths.recovery_run_request_path(recovery_request_id, base_dir))
    attempt = _read_optional_record(paths.recovery_run_attempt_path(recovery_request_id, base_dir))
    outcome = _read_optional_record(paths.recovery_run_outcome_path(recovery_request_id, base_dir))
    claim = _read_optional_record(paths.recovery_run_claim_path(recovery_request_id, base_dir))
    issue = _read_optional_record(paths.recovery_run_issue_path(recovery_request_id, base_dir))

    notes: list[str] = []
    outcome_rec = None
    successor_present: bool | None = None

    if outcome is not None:
        try:
            _validate_non_permission_booleans(outcome)
            _validate_record_digest(outcome, digest_field="outcome_digest", projection_fn=_outcome_digest_projection)
            outcome_rec = _to_outcome_record(outcome)
            return RecoveryRunReconcileResult(
                recovery_request_id=recovery_request_id,
                classification="valid_durable_outcome",
                outcome_record=outcome_rec,
                successor_present=None,
                notes=tuple(notes),
            )
        except RecoveryRunValidationError as exc:
            return RecoveryRunReconcileResult(
                recovery_request_id=recovery_request_id,
                classification="malformed_outcome",
                outcome_record=None,
                successor_present=None,
                notes=(str(exc),),
            )

    if claim is not None and attempt is None:
        return RecoveryRunReconcileResult(
            recovery_request_id=recovery_request_id,
            classification="claim_without_attempt",
            outcome_record=None,
            successor_present=None,
            notes=tuple(notes),
        )

    if attempt is not None and request is not None and issue is not None and claim is not None:
        successor_run_id = request.get("successor_run_id")
        if isinstance(successor_run_id, str):
            successor_present = paths.run_root(successor_run_id, base_dir).exists()
            if successor_present and _successor_exists_verified(request, issue, claim, attempt, base_dir):
                classification = "attempt_with_verified_successor"
            elif successor_present:
                classification = "attempt_with_unverified_successor"
            else:
                classification = "attempt_without_successor"
            return RecoveryRunReconcileResult(
                recovery_request_id=recovery_request_id,
                classification=classification,
                outcome_record=None,
                successor_present=successor_present,
                notes=tuple(notes),
            )

    if request is None:
        return RecoveryRunReconcileResult(
            recovery_request_id=recovery_request_id,
            classification="missing_request",
            outcome_record=None,
            successor_present=None,
            notes=tuple(notes),
        )

    return RecoveryRunReconcileResult(
        recovery_request_id=recovery_request_id,
        classification="incomplete",
        outcome_record=None,
        successor_present=None,
        notes=tuple(notes),
    )


__all__ = [
    "RecoveryScope",
    "RecoveryRunOutcomeClass",
    "generate_recovery_request_id",
    "generate_recovery_approval_id",
    "generate_recovery_claim_id",
    "generate_recovery_attempt_id",
    "generate_successor_run_id",
    "create_recovery_run_request",
    "issue_recovery_run_approval",
    "revoke_recovery_run_approval",
    "claim_recovery_run_approval",
    "execute_approved_successor_run_creation",
    "load_recovery_run_bundle",
    "reconcile_recovery_run_creation",
]
