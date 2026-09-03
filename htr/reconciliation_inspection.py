"""Task 26A — read-only run-completion reconciliation inspection."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from htr import contracts, events, io, paths, schemas
from htr.action_plan import (
    _chain_records_map,
    _first_absent_chain_index,
    _has_chain_gap,
    compute_source_observation_digest,
)
from htr.contracts import (
    PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN,
    run_completion_fingerprint,
    run_completion_record_json_path,
)
from htr.approval_control import (
    OUTCOME_AMBIGUOUS,
    OUTCOME_CONSUMED,
    OUTCOME_SCHEMA_VERSION,
    OUTCOME_SCHEMA_VERSION_V2,
    _argument_entries_to_inputs,
    _canonical_json,
    _compute_approval_digest,
    _compute_claim_digest,
    _compute_outcome_digest,
    _load_bundle,
    _runs_root_path_digest,
)
from htr.events import EVENT_TYPE_MANUAL_RUN_COMPLETED
from htr.execution_lock import LOCKS_DIR_NAME
from htr.finalization import SealState, evaluate_run_seal
from htr.ids import validate_id
from htr.observe import build_run_snapshot
from htr.state import (
    RUN_COMPLETED,
    RUN_LEGAL_TRANSITIONS,
    ReconciliationEvidenceIntegrityError,
    ReconciliationInspectionError,
    ReconciliationUnsupportedApprovalError,
    RunCompletionReconciliationInspection,
)

PILOT_BOUND_API = "complete_run_manually"
INSPECTION_DIGEST_PROJECTION_VERSION = "htr.reconciliation.inspection.digest.v1"
INSPECTION_SCHEMA_VERSION = "1"

_O_RDONLY = os.O_RDONLY
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_MARKER_SUFFIX = ".marker"
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _sha256_digest(payload: dict[str, Any]) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_runs_root_path(base_dir: Path | None) -> Path:
    return paths.runs_root(base_dir)


def _marker_name(run_id: str) -> str:
    validate_id(run_id, "run")
    return f"{run_id}{_MARKER_SUFFIX}"


def _open_dir_no_follow(path: Path) -> int:
    flags = _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
    return os.open(str(path), flags)


def _fstat_identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


def _stat_entry_identity(dir_fd: int, name: str) -> tuple[int, int]:
    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    return st.st_dev, st.st_ino


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


def _extract_bound_invocation(
    issue: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    if issue.get("bound_api") != PILOT_BOUND_API:
        raise ReconciliationUnsupportedApprovalError(
            f"approval bound_api must be {PILOT_BOUND_API!r}",
            approval_id=issue.get("approval_id"),
        )
    inputs = _argument_entries_to_inputs(issue["bound_arguments"])
    record = inputs.get("record")
    actor = inputs.get("actor")
    event_id = inputs.get("event_id")
    if not isinstance(record, dict):
        raise ReconciliationEvidenceIntegrityError(
            "bound completion record missing from approval",
            approval_id=issue.get("approval_id"),
        )
    if not actor or not isinstance(event_id, str) or not event_id.strip():
        raise ReconciliationEvidenceIntegrityError(
            "bound actor and event_id required",
            approval_id=issue.get("approval_id"),
        )
    if not validate_id(event_id, "event"):
        raise ReconciliationEvidenceIntegrityError(
            f"invalid bound event_id: {event_id!r}",
            approval_id=issue.get("approval_id"),
        )
    return record, actor, event_id.strip()


def _canonicalize_reason_codes(reasons: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(reasons)))


def _control_bundle_fingerprint(bundle: dict[str, Any]) -> str:
    payload: dict[str, Any] = {}
    for key in ("issue", "revoke", "claim", "outcome"):
        record = bundle.get(key)
        if record is not None:
            payload[key] = record
    return _sha256_digest(payload)


def _append_consumed_outcome_mismatch_reasons(
    *,
    control_state: str,
    outcome: dict[str, Any] | None,
    lifecycle_reasons: list[str],
    observed_record_fp: str | None,
    observed_event_fp: str | None,
    current_observation_digest: str | None,
    lifecycle_complete: bool,
    verified_completed: bool,
) -> None:
    if control_state != "consumed_outcome" or outcome is None:
        return
    evidence = outcome.get("outcome_evidence") or {}
    mismatch = False
    stored_record_fp = evidence.get("observed_record_fingerprint")
    stored_event_fp = evidence.get("observed_event_fingerprint")
    stored_post_digest = evidence.get("post_observation_digest")
    if lifecycle_reasons:
        mismatch = True
    if stored_record_fp is not None and observed_record_fp != stored_record_fp:
        mismatch = True
    if stored_event_fp is not None and observed_event_fp != stored_event_fp:
        mismatch = True
    if (
        stored_post_digest is not None
        and current_observation_digest is not None
        and current_observation_digest != stored_post_digest
    ):
        mismatch = True
    if not verified_completed and lifecycle_complete is False and control_state == "consumed_outcome":
        if lifecycle_reasons or stored_record_fp is not None:
            mismatch = True
    if mismatch:
        lifecycle_reasons.append("consumed_outcome_current_evidence_mismatch")


def _build_inspection_result(
    *,
    approval_id: str,
    approval_digest: str,
    claim_id: str | None,
    claim_digest: str | None,
    outcome_class: str | None,
    outcome_digest: str | None,
    run_id: str,
    bound_api: str,
    event_id: str,
    expected_runs_root_digest: str,
    approval_control_state: str,
    marker_state: str,
    lifecycle_evidence_state: str,
    integrity_state: str,
    overall_classification: str,
    reason_codes: tuple[str, ...],
    observed_completion_record_fingerprint: str | None,
    observed_event_semantic_fingerprint: str | None,
    observed_manifest_status: str | None,
    current_observation_semantic_digest: str | None,
    source_observation_digest: str,
    reconciliation_case_required: bool,
    recovery_protocol_required: bool,
) -> RunCompletionReconciliationInspection:
    observed_at = _utc_now_iso()
    preliminary = RunCompletionReconciliationInspection(
        inspection_schema_version=INSPECTION_SCHEMA_VERSION,
        inspection_projection_version=INSPECTION_DIGEST_PROJECTION_VERSION,
        approval_id=approval_id,
        approval_digest=approval_digest,
        claim_id=claim_id,
        claim_digest=claim_digest,
        outcome_class=outcome_class,
        outcome_digest=outcome_digest,
        run_id=run_id,
        bound_api=bound_api,
        event_id=event_id,
        htr_runs_root_path_digest=expected_runs_root_digest,
        approval_control_state=approval_control_state,
        marker_state=marker_state,
        lifecycle_evidence_state=lifecycle_evidence_state,
        integrity_state=integrity_state,
        overall_classification=overall_classification,
        reason_codes=reason_codes,
        observed_completion_record_fingerprint=observed_completion_record_fingerprint,
        observed_event_semantic_fingerprint=observed_event_semantic_fingerprint,
        observed_manifest_status=observed_manifest_status,
        current_observation_semantic_digest=current_observation_semantic_digest,
        source_observation_digest=source_observation_digest,
        inspection_semantic_digest="",
        safe_to_retry=False,
        marker_disposition_allowed=False,
        reconciliation_case_required=reconciliation_case_required,
        recovery_protocol_required=recovery_protocol_required,
        observed_at=observed_at,
    )
    digest = compute_inspection_semantic_digest(preliminary)
    return RunCompletionReconciliationInspection(
        inspection_schema_version=preliminary.inspection_schema_version,
        inspection_projection_version=preliminary.inspection_projection_version,
        approval_id=preliminary.approval_id,
        approval_digest=preliminary.approval_digest,
        claim_id=preliminary.claim_id,
        claim_digest=preliminary.claim_digest,
        outcome_class=preliminary.outcome_class,
        outcome_digest=preliminary.outcome_digest,
        run_id=preliminary.run_id,
        bound_api=preliminary.bound_api,
        event_id=preliminary.event_id,
        htr_runs_root_path_digest=preliminary.htr_runs_root_path_digest,
        approval_control_state=preliminary.approval_control_state,
        marker_state=preliminary.marker_state,
        lifecycle_evidence_state=preliminary.lifecycle_evidence_state,
        integrity_state=preliminary.integrity_state,
        overall_classification=preliminary.overall_classification,
        reason_codes=preliminary.reason_codes,
        observed_completion_record_fingerprint=preliminary.observed_completion_record_fingerprint,
        observed_event_semantic_fingerprint=preliminary.observed_event_semantic_fingerprint,
        observed_manifest_status=preliminary.observed_manifest_status,
        current_observation_semantic_digest=preliminary.current_observation_semantic_digest,
        source_observation_digest=preliminary.source_observation_digest,
        inspection_semantic_digest=digest,
        safe_to_retry=False,
        marker_disposition_allowed=False,
        reconciliation_case_required=preliminary.reconciliation_case_required,
        recovery_protocol_required=preliminary.recovery_protocol_required,
        observed_at=preliminary.observed_at,
    )


def _inspect_unsupported_approval(
    approval_id: str,
    bundle: dict[str, Any],
    *,
    base_dir: Path | None,
    expected_runs_root_digest: str,
) -> RunCompletionReconciliationInspection:
    """Readable issue bound to a non-pilot API — control axis only."""
    issue = bundle["issue"]
    control_reasons: list[str] = ["unsupported_bound_api"]
    try:
        computed_issue_digest = _compute_approval_digest(issue)
        approval_digest = issue.get("approval_digest") or computed_issue_digest
        if issue.get("approval_digest") != computed_issue_digest:
            control_reasons.append("issue_digest_mismatch")
    except Exception:
        approval_digest = issue.get("approval_digest") or ""
        control_reasons.append("issue_digest_computation_failed")

    if issue.get("htr_runs_root_path_digest") != expected_runs_root_digest:
        control_reasons.append("issue_runs_root_digest_mismatch")

    control_state = "unsupported_approval"
    if any(
        code.endswith("_mismatch") or "malformed" in code or "computation_failed" in code
        for code in control_reasons
        if code != "issue_digest_mismatch"
    ):
        control_state = "malformed_control_evidence"

    reason_codes = _canonicalize_reason_codes(control_reasons)
    integrity_state = (
        "control_integrity_issue"
        if control_state == "malformed_control_evidence"
        else "clean"
    )
    overall = "indeterminate"
    if control_state == "unsupported_approval":
        overall = "integrity_blocked"
    elif integrity_state != "clean":
        overall = "integrity_blocked"

    run_id = issue.get("run_id") or ""
    source_observation_digest = issue.get("source_observation_digest") or _sha256_digest({})
    bound_api = str(issue.get("bound_api") or "")

    return _build_inspection_result(
        approval_id=approval_id,
        approval_digest=approval_digest,
        claim_id=None,
        claim_digest=None,
        outcome_class=None,
        outcome_digest=None,
        run_id=run_id,
        bound_api=bound_api,
        event_id="",
        expected_runs_root_digest=expected_runs_root_digest,
        approval_control_state=control_state,
        marker_state="absent",
        lifecycle_evidence_state="no_lifecycle_evidence_observed",
        integrity_state=integrity_state,
        overall_classification=overall,
        reason_codes=reason_codes,
        observed_completion_record_fingerprint=None,
        observed_event_semantic_fingerprint=None,
        observed_manifest_status=None,
        current_observation_semantic_digest=None,
        source_observation_digest=source_observation_digest,
        reconciliation_case_required=True,
        recovery_protocol_required=False,
    )


def _inspect_marker_readonly(
    base_dir: Path | None,
    run_id: str,
    *,
    runs_root_digest: str,
) -> tuple[str, list[str], dict[str, Any] | None]:
    reasons: list[str] = []
    runs_path = _canonical_runs_root_path(base_dir)
    lock_root_path = runs_path / LOCKS_DIR_NAME
    if not runs_path.is_dir() or not lock_root_path.is_dir():
        return "absent", reasons, None

    lock_root_fd: int | None = None
    marker_fd: int | None = None
    try:
        if lock_root_path.is_symlink():
            reasons.append("lock_directory_symlink")
            return "present_unsafe_path", reasons, None
        lock_root_fd = _open_dir_no_follow(lock_root_path)
        marker_name = _marker_name(run_id)
        try:
            st = os.stat(marker_name, dir_fd=lock_root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return "absent", reasons, None
        except OSError as exc:
            reasons.append("marker_lookup_failed")
            return "indeterminate_marker_state", reasons, None

        if not os.path.stat.S_ISREG(st.st_mode):
            reasons.append("marker_not_regular_file")
            return "present_unsafe_path", reasons, None

        open_flags = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
        try:
            marker_fd = os.open(marker_name, open_flags, dir_fd=lock_root_fd)
        except OSError as exc:
            reasons.append("marker_open_failed")
            return "present_unsafe_path", reasons, None

        if _fstat_identity(marker_fd) != _stat_entry_identity(lock_root_fd, marker_name):
            reasons.append("marker_identity_mismatch_on_open")
            return "present_identity_mismatch", reasons, None

        raw = os.read(marker_fd, 65536)
        if _fstat_identity(marker_fd) != _stat_entry_identity(lock_root_fd, marker_name):
            reasons.append("marker_replaced_during_read")
            return "present_identity_mismatch", reasons, None

        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            reasons.append("marker_malformed_json")
            return "present_malformed_metadata", reasons, None

        if not isinstance(metadata, dict):
            reasons.append("marker_metadata_not_object")
            return "present_malformed_metadata", reasons, None

        acquisition_id = metadata.get("acquisition_id")
        pid = metadata.get("pid")
        hostname = metadata.get("hostname")
        marker_run_id = metadata.get("run_id")
        if not isinstance(acquisition_id, str) or not _UUID_RE.match(acquisition_id):
            reasons.append("marker_missing_acquisition_id")
            return "present_malformed_metadata", reasons, metadata
        if not isinstance(pid, int) or pid <= 0:
            reasons.append("marker_malformed_pid")
            return "present_malformed_metadata", reasons, metadata
        if not isinstance(hostname, str) or not hostname.strip():
            reasons.append("marker_malformed_hostname")
            return "present_malformed_metadata", reasons, metadata
        if marker_run_id != run_id:
            reasons.append("marker_run_id_mismatch")
            return "present_malformed_metadata", reasons, metadata

        return "present_valid_metadata", reasons, metadata
    finally:
        if marker_fd is not None:
            try:
                os.close(marker_fd)
            except OSError:
                pass
        if lock_root_fd is not None:
            try:
                os.close(lock_root_fd)
            except OSError:
                pass


def _evaluate_control_state(
    bundle: dict[str, Any],
    *,
    expected_runs_root_digest: str,
) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    issue = bundle["issue"]
    revoke = bundle["revoke"]
    claim = bundle["claim"]
    outcome = bundle["outcome"]

    try:
        computed_issue_digest = _compute_approval_digest(issue)
    except Exception:
        reasons.append("issue_digest_computation_failed")
        return "malformed_control_evidence", reasons, issue

    if issue.get("approval_digest") != computed_issue_digest:
        reasons.append("issue_digest_mismatch")

    if issue.get("htr_runs_root_path_digest") != expected_runs_root_digest:
        reasons.append("issue_runs_root_digest_mismatch")

    if revoke is not None and claim is None:
        try:
            if revoke.get("approval_digest") != issue.get("approval_digest"):
                reasons.append("revoke_approval_digest_mismatch")
        except Exception:
            reasons.append("revoke_malformed")
        if reasons:
            return "conflicting_control_evidence", reasons, issue
        return "revoked_before_claim", reasons, issue

    if claim is None and outcome is None:
        if reasons:
            return "malformed_control_evidence", reasons, issue
        return "issue_only", reasons, issue

    if claim is not None:
        try:
            computed_claim_digest = _compute_claim_digest(claim)
        except Exception:
            reasons.append("claim_digest_computation_failed")
            return "malformed_control_evidence", reasons, issue
        if claim.get("claim_digest") != computed_claim_digest:
            reasons.append("claim_digest_mismatch")
        if claim.get("approval_digest") != issue.get("approval_digest"):
            reasons.append("claim_approval_digest_mismatch")
        if claim.get("bound_api") != PILOT_BOUND_API:
            reasons.append("claim_bound_api_mismatch")

    if claim is not None and outcome is None:
        if reasons:
            return "conflicting_control_evidence", reasons, issue
        return "claimed_without_outcome", reasons, issue

    if outcome is not None:
        try:
            computed_outcome_digest = _compute_outcome_digest(outcome)
        except Exception:
            reasons.append("outcome_digest_computation_failed")
            return "malformed_control_evidence", reasons, issue
        if outcome.get("approval_digest") != issue.get("approval_digest"):
            reasons.append("outcome_approval_digest_mismatch")
        if claim is not None and outcome.get("claim_id") != claim.get("claim_id"):
            reasons.append("outcome_claim_id_mismatch")
        if claim is not None and outcome.get("claim_digest") != claim.get("claim_digest"):
            reasons.append("outcome_claim_digest_mismatch")
        outcome_class = outcome.get("outcome_class")
        if outcome.get("outcome_schema_version") == OUTCOME_SCHEMA_VERSION:
            reasons.append("outcome_v1_no_task25_diagnostics")
            stored_digest = outcome.get("outcome_digest")
            if stored_digest is not None and stored_digest != computed_outcome_digest:
                reasons.append("outcome_digest_mismatch")
        elif outcome.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION_V2:
            reasons.append("outcome_schema_unsupported")
        elif outcome.get("outcome_digest") != computed_outcome_digest:
            reasons.append("outcome_digest_mismatch")

        if reasons and any(
            code.endswith("_mismatch") or "malformed" in code or "computation_failed" in code
            for code in reasons
        ):
            return "conflicting_control_evidence", reasons, issue

        if outcome_class == OUTCOME_CONSUMED:
            return "consumed_outcome", reasons, issue
        if outcome_class == OUTCOME_AMBIGUOUS:
            return "ambiguous_outcome", reasons, issue
        reasons.append("outcome_class_unrecognized")
        return "malformed_control_evidence", reasons, issue

    if reasons:
        return "conflicting_control_evidence", reasons, issue
    return "indeterminate_control_state", reasons, issue


def _lifecycle_verification_reasons(
    *,
    run_id: str,
    base_dir: Path | None,
    completion_record: dict[str, Any],
    actor: str,
    event_id: str,
    record_fingerprint: str,
    pre_manifest_status: str | None,
    snapshot: dict[str, Any] | None,
) -> tuple[list[str], bool, bool, bool, str | None, str | None, str | None]:
    """Mirror Task 25 post-invoke verification read-only."""
    reasons: list[str] = []
    record_path = run_completion_record_json_path(run_id, base_dir)
    json_present = record_path.is_file()
    event_present = False
    manifest_complete = False
    observed_record_fp: str | None = None
    observed_event_fp: str | None = None
    manifest_status: str | None = None

    matching_events: list[dict[str, Any]] = []
    if json_present:
        on_disk = io.read_json(record_path)
        try:
            schemas.validate(on_disk, "run_completion_record")
        except ValueError:
            reasons.append("completion_record_schema_invalid")
        if on_disk != completion_record:
            reasons.append("completion_record_semantic_mismatch")
        try:
            observed_record_fp = run_completion_fingerprint(on_disk)
            if observed_record_fp != record_fingerprint:
                reasons.append("completion_record_fingerprint_mismatch")
        except ValueError:
            reasons.append("completion_record_fingerprint_unavailable")

        matching_events = [
            ev
            for ev in events.read_task_events(run_id, base_dir=base_dir)
            if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
            and ev.get("event_id") == event_id
        ]

    if json_present and len(matching_events) == 1:
        event_present = True
        ev = matching_events[0]
        observed_event_fp = _semantic_event_fingerprint(ev)
        if ev.get("run_id") != run_id:
            reasons.append("event_run_id_mismatch")
        if "task_id" in ev:
            reasons.append("event_has_task_id")
        if ev.get("actor") != actor:
            reasons.append("event_actor_mismatch")
        previous_status = ev.get("previous_status")
        if pre_manifest_status is not None:
            if previous_status != pre_manifest_status:
                reasons.append("event_previous_status_mismatch")
        elif previous_status not in RUN_LEGAL_TRANSITIONS:
            reasons.append("event_previous_status_invalid")
        elif RUN_COMPLETED not in RUN_LEGAL_TRANSITIONS.get(previous_status, frozenset()):
            reasons.append("event_previous_status_not_legal")
        if ev.get("new_status") != RUN_COMPLETED:
            reasons.append("event_new_status_mismatch")
        payload = ev.get("payload") or {}
        if payload.get("completed_task_ids") != completion_record["completed_task_ids"]:
            reasons.append("event_completed_task_ids_mismatch")
        if payload.get("run_completion_fingerprint") != record_fingerprint:
            reasons.append("event_record_fingerprint_mismatch")
        if payload.get("run_completion_record_path") != str(record_path):
            reasons.append("event_record_path_mismatch")
    elif json_present:
        if len(matching_events) == 0:
            reasons.append("completion_event_missing")
        else:
            reasons.append("completion_event_count_mismatch")

    if not json_present:
        orphan_events = [
            ev
            for ev in events.read_task_events(run_id, base_dir=base_dir)
            if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
        ]
        if orphan_events:
            reasons.append("completion_event_without_json")
            event_present = True

    manifest_path = paths.run_manifest_path(run_id, base_dir)
    if manifest_path.is_file():
        try:
            manifest = io.read_json(manifest_path)
            schemas.validate(manifest, "run_manifest")
            manifest_status = manifest.get("status")
            if manifest_status == RUN_COMPLETED:
                manifest_complete = True
            elif json_present or event_present:
                reasons.append("manifest_not_completed")
        except ValueError:
            reasons.append("manifest_schema_invalid")
    elif json_present or event_present:
        reasons.append("manifest_missing")

    if snapshot is not None and _trustworthy_snapshot(snapshot):
        chain_map = _chain_records_map(snapshot)
        if json_present and not chain_map.get("run_completion_record", {}).get("present"):
            reasons.append("chain_completion_missing")
        if _has_chain_gap(snapshot):
            reasons.append("chain_gap_detected")
        first_absent = _first_absent_chain_index(snapshot)
        if json_present:
            if chain_map.get("run_completion_record", {}).get("present"):
                if first_absent not in (1, None):
                    reasons.append("first_absent_chain_not_after_completion")
            elif first_absent not in (0, None):
                reasons.append("first_absent_chain_not_completion")
        for record_type in PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN[1:]:
            if chain_map.get(record_type, {}).get("present"):
                reasons.append(f"unexpected_chain_record:{record_type}")
        completion_events = [
            ev
            for ev in events.read_task_events(run_id, base_dir=base_dir)
            if ev.get("event_type") == EVENT_TYPE_MANUAL_RUN_COMPLETED
        ]
        if len(completion_events) > 1:
            reasons.append("multiple_completion_events")
    elif snapshot is not None:
        reasons.append("snapshot_not_trustworthy")

    seal = evaluate_run_seal(run_id, base_dir)
    if seal.state in (SealState.CLOSURE_PRESENT_UNTRUSTED, SealState.INDETERMINATE):
        reasons.append("seal_blocked")
    elif seal.state == SealState.FINALIZED_VALID and reasons:
        reasons.append("finalized_run_lifecycle_conflict")

    lifecycle_complete = (
        json_present
        and event_present
        and manifest_complete
        and not reasons
    )
    return (
        reasons,
        json_present,
        event_present,
        lifecycle_complete,
        observed_record_fp,
        observed_event_fp,
        manifest_status,
    )


def _classify_lifecycle_state(
    *,
    reasons: list[str],
    json_present: bool,
    event_present: bool,
    lifecycle_complete: bool,
    verified_completed: bool,
) -> str:
    if verified_completed:
        return "verified_completed"
    if lifecycle_complete:
        return "lifecycle_complete_observed"
    if json_present and event_present and not lifecycle_complete:
        return "completion_json_and_event_manifest_incomplete"
    if json_present and not event_present:
        return "completion_json_only"
    if not json_present and event_present:
        return "lifecycle_evidence_conflict"
    if reasons and any("seal_blocked" in r or "manifest" in r or "chain" in r for r in reasons):
        return "lifecycle_integrity_blocked"
    if reasons:
        return "lifecycle_evidence_conflict"
    return "no_lifecycle_evidence_observed"


def _derive_integrity_state(
    *,
    control_state: str,
    marker_state: str,
    lifecycle_state: str,
    reason_codes: tuple[str, ...],
) -> str:
    if any("seal_blocked" in code for code in reason_codes):
        return "seal_blocked"
    if control_state in {
        "malformed_control_evidence",
        "conflicting_control_evidence",
    }:
        return "control_integrity_issue"
    if marker_state in {
        "present_malformed_metadata",
        "present_identity_mismatch",
        "present_unsafe_path",
        "indeterminate_marker_state",
    }:
        return "marker_integrity_issue"
    if lifecycle_state in {
        "lifecycle_evidence_conflict",
        "lifecycle_integrity_blocked",
    }:
        return "lifecycle_integrity_issue"
    if lifecycle_state == "indeterminate_lifecycle_state":
        return "indeterminate"
    if control_state == "indeterminate_control_state":
        return "indeterminate"
    return "clean"


def _derive_overall_classification(
    *,
    control_state: str,
    marker_state: str,
    lifecycle_state: str,
    integrity_state: str,
    reason_codes: tuple[str, ...],
) -> tuple[str, bool, bool]:
    reconciliation_case_required = False
    recovery_protocol_required = False

    if integrity_state == "seal_blocked":
        recovery_protocol_required = True
        return "integrity_blocked", reconciliation_case_required, recovery_protocol_required

    if integrity_state != "clean":
        if control_state == "conflicting_control_evidence" or lifecycle_state == "lifecycle_evidence_conflict":
            reconciliation_case_required = True
            return "control_lifecycle_evidence_conflict", reconciliation_case_required, recovery_protocol_required
        reconciliation_case_required = True
        return "integrity_blocked", reconciliation_case_required, recovery_protocol_required

    if any(
        code in reason_codes
        for code in (
            "control_evidence_replaced_during_read",
            "consumed_outcome_current_evidence_mismatch",
        )
    ):
        reconciliation_case_required = True
        return "control_lifecycle_evidence_conflict", reconciliation_case_required, recovery_protocol_required

    if control_state == "consumed_outcome":
        if lifecycle_state == "verified_completed":
            if marker_state == "absent":
                return "no_reconciliation_needed", False, False
            if marker_state == "present_valid_metadata":
                return "verified_completion_marker_residue", True, False
        elif lifecycle_state == "lifecycle_complete_observed":
            if marker_state == "absent":
                return "no_reconciliation_needed", False, False
            return "verified_completion_marker_residue", True, False
        reconciliation_case_required = True
        return "control_lifecycle_evidence_conflict", True, False

    if control_state == "unsupported_approval":
        reconciliation_case_required = True
        return "integrity_blocked", reconciliation_case_required, recovery_protocol_required

    if control_state in {"ambiguous_outcome", "claimed_without_outcome"}:
        if lifecycle_state in {
            "lifecycle_complete_observed",
            "verified_completed",
        }:
            reconciliation_case_required = True
            return "completion_observed_outcome_missing", True, False
        if lifecycle_state == "completion_json_only":
            reconciliation_case_required = True
            return "partial_lifecycle_commit", True, False
        if lifecycle_state == "completion_json_and_event_manifest_incomplete":
            reconciliation_case_required = True
            return "partial_lifecycle_commit", True, False

    if lifecycle_state == "completion_json_only":
        reconciliation_case_required = True
        return "partial_lifecycle_commit", True, False

    if lifecycle_state == "completion_json_and_event_manifest_incomplete":
        reconciliation_case_required = True
        return "partial_lifecycle_commit", True, False

    if control_state == "claimed_without_outcome":
        reconciliation_case_required = True
        return "reconciliation_inspection_required", True, False

    if control_state == "issue_only" and lifecycle_state != "no_lifecycle_evidence_observed":
        reconciliation_case_required = True
        return "reconciliation_inspection_required", True, False

    if lifecycle_state == "no_lifecycle_evidence_observed" and control_state in {
        "claimed_without_outcome",
        "ambiguous_outcome",
    }:
        reconciliation_case_required = True
        return "reconciliation_inspection_required", True, False

    if any("run_id" in code for code in reason_codes):
        reconciliation_case_required = True
        return "control_lifecycle_evidence_conflict", True, False

    return "indeterminate", reconciliation_case_required, recovery_protocol_required


def _inspection_digest_projection(result: RunCompletionReconciliationInspection) -> dict[str, Any]:
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


def compute_inspection_semantic_digest(
    result: RunCompletionReconciliationInspection,
) -> str:
    return _sha256_digest(_inspection_digest_projection(result))


def inspect_run_completion_reconciliation(
    approval_id: str,
    *,
    base_dir: Path | None = None,
) -> RunCompletionReconciliationInspection:
    """Read-only reconciliation inspection for Task 25 run-completion pilot."""
    validate_id(approval_id, "approval")
    expected_runs_root_digest = _runs_root_path_digest(base_dir)

    try:
        bundle = _load_bundle(approval_id, base_dir)
    except Exception as exc:
        raise ReconciliationEvidenceIntegrityError(
            f"cannot load approval bundle: {exc}",
            approval_id=approval_id,
        ) from exc

    issue = bundle["issue"]
    if issue.get("bound_api") != PILOT_BOUND_API:
        return _inspect_unsupported_approval(
            approval_id,
            bundle,
            base_dir=base_dir,
            expected_runs_root_digest=expected_runs_root_digest,
        )

    initial_control_fingerprint = _control_bundle_fingerprint(bundle)

    completion_record, actor, event_id = _extract_bound_invocation(issue)
    run_id = issue["run_id"]
    record_fingerprint = run_completion_fingerprint(completion_record)
    source_observation_digest = issue["source_observation_digest"]
    approval_digest = issue.get("approval_digest") or _compute_approval_digest(issue)

    control_state, control_reasons, _issue = _evaluate_control_state(
        bundle,
        expected_runs_root_digest=expected_runs_root_digest,
    )

    marker_state, marker_reasons, _marker_meta = _inspect_marker_readonly(
        base_dir,
        run_id,
        runs_root_digest=expected_runs_root_digest,
    )

    snapshot: dict[str, Any] | None = None
    current_observation_digest: str | None = None
    pre_manifest_status: str | None = None
    observation_reasons: list[str] = []
    run_root = paths.run_root(run_id, base_dir)
    if run_root.is_dir():
        try:
            snapshot = build_run_snapshot(run_id, base_dir=base_dir)
            if _trustworthy_snapshot(snapshot):
                current_observation_digest = compute_source_observation_digest(snapshot)
            else:
                observation_reasons.append("current_snapshot_not_trustworthy")
            manifest_path = paths.run_manifest_path(run_id, base_dir)
            if manifest_path.is_file():
                current_status = io.read_json(manifest_path).get("status")
                if current_status != RUN_COMPLETED:
                    pre_manifest_status = current_status
        except Exception as exc:
            observation_reasons.append("snapshot_build_failed")
            snapshot = None
    else:
        observation_reasons.append("run_workspace_absent")

    lifecycle_reasons, json_present, event_present, lifecycle_complete, observed_record_fp, observed_event_fp, manifest_status = (
        _lifecycle_verification_reasons(
            run_id=run_id,
            base_dir=base_dir,
            completion_record=completion_record,
            actor=actor,
            event_id=event_id,
            record_fingerprint=record_fingerprint,
            pre_manifest_status=pre_manifest_status,
            snapshot=snapshot,
        )
    )

    verified_completed = False
    outcome = bundle.get("outcome")
    outcome_class = outcome.get("outcome_class") if outcome else None
    if (
        control_state == "consumed_outcome"
        and outcome is not None
        and outcome.get("outcome_schema_version") == OUTCOME_SCHEMA_VERSION_V2
        and lifecycle_complete
        and not lifecycle_reasons
        and snapshot is not None
        and _trustworthy_snapshot(snapshot)
    ):
        evidence = outcome.get("outcome_evidence") or {}
        if evidence.get("reason_code") == "verified_success":
            if evidence.get("observed_record_fingerprint") == observed_record_fp:
                if evidence.get("observed_event_fingerprint") == observed_event_fp:
                    if current_observation_digest == evidence.get("post_observation_digest"):
                        verified_completed = True

    _append_consumed_outcome_mismatch_reasons(
        control_state=control_state,
        outcome=outcome,
        lifecycle_reasons=lifecycle_reasons,
        observed_record_fp=observed_record_fp,
        observed_event_fp=observed_event_fp,
        current_observation_digest=current_observation_digest,
        lifecycle_complete=lifecycle_complete,
        verified_completed=verified_completed,
    )
    if "consumed_outcome_current_evidence_mismatch" in lifecycle_reasons:
        verified_completed = False

    lifecycle_state = _classify_lifecycle_state(
        reasons=lifecycle_reasons,
        json_present=json_present,
        event_present=event_present,
        lifecycle_complete=lifecycle_complete,
        verified_completed=verified_completed,
    )

    try:
        final_bundle = _load_bundle(approval_id, base_dir)
        if _control_bundle_fingerprint(final_bundle) != initial_control_fingerprint:
            control_reasons.append("control_evidence_replaced_during_read")
    except Exception:
        control_reasons.append("control_evidence_reread_failed")

    all_reasons = _canonicalize_reason_codes(
        list(control_reasons)
        + list(marker_reasons)
        + list(lifecycle_reasons)
        + observation_reasons
    )

    integrity_state = _derive_integrity_state(
        control_state=control_state,
        marker_state=marker_state,
        lifecycle_state=lifecycle_state,
        reason_codes=all_reasons,
    )

    overall, reconciliation_case_required, recovery_protocol_required = _derive_overall_classification(
        control_state=control_state,
        marker_state=marker_state,
        lifecycle_state=lifecycle_state,
        integrity_state=integrity_state,
        reason_codes=all_reasons,
    )

    claim = bundle.get("claim")
    claim_id = claim.get("claim_id") if claim else None
    claim_digest = claim.get("claim_digest") if claim else None
    outcome_digest = outcome.get("outcome_digest") if outcome else None

    observed_at = _utc_now_iso()

    preliminary = RunCompletionReconciliationInspection(
        inspection_schema_version=INSPECTION_SCHEMA_VERSION,
        inspection_projection_version=INSPECTION_DIGEST_PROJECTION_VERSION,
        approval_id=approval_id,
        approval_digest=approval_digest,
        claim_id=claim_id,
        claim_digest=claim_digest,
        outcome_class=outcome_class,
        outcome_digest=outcome_digest,
        run_id=run_id,
        bound_api=PILOT_BOUND_API,
        event_id=event_id,
        htr_runs_root_path_digest=expected_runs_root_digest,
        approval_control_state=control_state,
        marker_state=marker_state,
        lifecycle_evidence_state=lifecycle_state,
        integrity_state=integrity_state,
        overall_classification=overall,
        reason_codes=all_reasons,
        observed_completion_record_fingerprint=observed_record_fp,
        observed_event_semantic_fingerprint=observed_event_fp,
        observed_manifest_status=manifest_status,
        current_observation_semantic_digest=current_observation_digest,
        source_observation_digest=source_observation_digest,
        inspection_semantic_digest="",
        safe_to_retry=False,
        marker_disposition_allowed=False,
        reconciliation_case_required=reconciliation_case_required,
        recovery_protocol_required=recovery_protocol_required,
        observed_at=observed_at,
    )
    digest = compute_inspection_semantic_digest(preliminary)
    return RunCompletionReconciliationInspection(
        inspection_schema_version=preliminary.inspection_schema_version,
        inspection_projection_version=preliminary.inspection_projection_version,
        approval_id=preliminary.approval_id,
        approval_digest=preliminary.approval_digest,
        claim_id=preliminary.claim_id,
        claim_digest=preliminary.claim_digest,
        outcome_class=preliminary.outcome_class,
        outcome_digest=preliminary.outcome_digest,
        run_id=preliminary.run_id,
        bound_api=preliminary.bound_api,
        event_id=preliminary.event_id,
        htr_runs_root_path_digest=preliminary.htr_runs_root_path_digest,
        approval_control_state=preliminary.approval_control_state,
        marker_state=preliminary.marker_state,
        lifecycle_evidence_state=preliminary.lifecycle_evidence_state,
        integrity_state=preliminary.integrity_state,
        overall_classification=preliminary.overall_classification,
        reason_codes=preliminary.reason_codes,
        observed_completion_record_fingerprint=preliminary.observed_completion_record_fingerprint,
        observed_event_semantic_fingerprint=preliminary.observed_event_semantic_fingerprint,
        observed_manifest_status=preliminary.observed_manifest_status,
        current_observation_semantic_digest=preliminary.current_observation_semantic_digest,
        source_observation_digest=preliminary.source_observation_digest,
        inspection_semantic_digest=digest,
        safe_to_retry=False,
        marker_disposition_allowed=False,
        reconciliation_case_required=preliminary.reconciliation_case_required,
        recovery_protocol_required=preliminary.recovery_protocol_required,
        observed_at=preliminary.observed_at,
    )


__all__ = [
    "INSPECTION_DIGEST_PROJECTION_VERSION",
    "INSPECTION_SCHEMA_VERSION",
    "PILOT_BOUND_API",
    "compute_inspection_semantic_digest",
    "inspect_run_completion_reconciliation",
]
