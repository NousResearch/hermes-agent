"""Authoritative approval control schema and API (Task 24)."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, NoReturn

from htr import paths
from htr.action_plan import (
    POLICY_C_VERSION,
    PROJECT_DIR_BINDING_EXPLICIT,
    PROJECT_DIR_BINDING_OBSERVER,
    STATE_PROPOSABLE,
    PlanningIntent,
    build_action_plan,
    compute_source_observation_digest,
    _CATALOG,
    _canonical_json,
    _normalize_path_for_digest,
    _sha256_digest,
)
from htr.events import _find_event_by_id
from htr.execution_lock import (
    RunExecutionLockBoundaryViolationError,
    RunExecutionLockIndeterminateError,
    RunExecutionLockOccupiedError,
    RunWriteContext,
    _acquire_outer_run_marker,
    _exit_outer_barrier,
    _find_nested_entry,
    _require_platform,
    _thread_active_entry,
    begin_run_write,
    bind_active_write_context,
    marker_present_noncreating,
)
from htr.finalization import SealState, evaluate_run_seal
from htr.ids import new_approval_id, validate_id
from htr.io import read_json
from htr.observe import ObserveInvocationError, build_run_snapshot
from htr.state import (
    ApprovalConflictError,
    ApprovalFinalizedRunError,
    ApprovalStateError,
    ApprovalValidationError,
    RunSealBlockedError,
)

APPROVAL_SCHEMA_VERSION = "1"
APPROVAL_DIGEST_PROJECTION_VERSION = "htr.approval.digest.v1"
CLAIM_SCHEMA_VERSION = "1"
CLAIM_DIGEST_PROJECTION_VERSION = "htr.approval.claim.digest.v1"
REVOKE_SCHEMA_VERSION = "1"
OUTCOME_SCHEMA_VERSION = "1"
OUTCOME_SCHEMA_VERSION_V2 = "2"

APPROVAL_KIND_LIFECYCLE_MUTATION = "lifecycle_mutation"

OUTCOME_CONSUMED = "consumed"
OUTCOME_AMBIGUOUS = "ambiguous"

REVOKE_DIGEST_PROJECTION_VERSION = "htr.approval.revoke.digest.v1"
OUTCOME_DIGEST_PROJECTION_VERSION = "htr.approval.outcome.digest.v1"
OUTCOME_DIGEST_PROJECTION_VERSION_V2 = "htr.approval.outcome.digest.v2"

MAX_APPROVAL_LIFETIME = timedelta(hours=24)

_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_DIRECTORY = os.O_DIRECTORY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

_CONTROL_FILE_MODE = 0o600
_CONTROL_DIR_MODE = 0o700


def _project_dir_path_digest(path: str) -> str:
    return _sha256_digest({"normalized_path": _normalize_path_for_digest(path)})


def _event_id_exists(run_id: str, event_id: str, base_dir: Path | None) -> bool:
    return _find_event_by_id(run_id, event_id, base_dir) is not None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_utc_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ApprovalValidationError(f"timestamp must be timezone-aware: {value!r}")
    return parsed.astimezone(timezone.utc)


def _validate_identity_string(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApprovalValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _runs_root_path_digest(base_dir: Path | None) -> str:
    runs_path = paths.runs_root(base_dir)
    normalized = os.path.normpath(os.path.abspath(str(runs_path)))
    return _sha256_digest({"normalized_path": _normalize_path_for_digest(normalized)})


def _record_fingerprint(record: dict[str, Any], digest_field: str) -> str:
    digest = record.get(digest_field)
    if not isinstance(digest, str) or not digest:
        raise ApprovalValidationError(f"missing {digest_field}")
    return digest


def _materialize_bound_arguments(
    plan: dict[str, Any],
    *,
    intent: PlanningIntent | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    candidate = plan.get("candidate_action") or {}
    api = candidate.get("api")
    if not isinstance(api, str) or not api:
        raise ApprovalValidationError("plan has no candidate API")
    if api not in _CATALOG:
        raise ApprovalValidationError(f"unsupported API: {api!r}")

    catalog = _CATALOG[api]
    supplied = plan.get("arguments", {}).get("supplied")
    if not isinstance(supplied, dict):
        supplied = {}
    derived = plan.get("arguments", {}).get("derived")
    if not isinstance(derived, dict):
        derived = {}

    allowed = {"record", "actor", "executor", "event_id", "notes", "metadata", "project_dir"}
    all_keys = sorted(set(supplied.keys()) | set(derived.keys()) | allowed)
    entries: list[dict[str, Any]] = []
    for key in all_keys:
        if key in supplied:
            value = supplied[key]
            if value is None:
                entries.append({"key": key, "presence": "null"})
            else:
                entries.append({"key": key, "presence": "value", "value": value})
        elif key in derived:
            entries.append({"key": key, "presence": "derived", "value": derived[key]})
        else:
            entries.append({"key": key, "presence": "omitted"})

    if catalog["requires_record"]:
        record = supplied.get("record")
        if not isinstance(record, dict):
            raise ApprovalValidationError("explicit record required in plan arguments")
    if catalog["requires_actor"]:
        actor = supplied.get("actor")
        if not actor:
            raise ApprovalValidationError("explicit actor required in plan arguments")
    if catalog["requires_executor"]:
        executor = supplied.get("executor")
        if not executor:
            raise ApprovalValidationError("explicit executor required in plan arguments")

    event_id = supplied.get("event_id")
    if expected_events := catalog.get("expected_events"):
        if expected_events and (not isinstance(event_id, str) or not event_id.strip()):
            raise ApprovalValidationError(
                "explicit event_id required for event-appending lifecycle API"
            )
        if not validate_id(event_id, "event"):
            raise ApprovalValidationError(f"invalid event_id format: {event_id!r}")

    if catalog.get("uses_project_dir"):
        binding = derived.get("project_dir_binding")
        if not isinstance(binding, dict):
            raise ApprovalValidationError("project_dir_binding required for API")
        project_dir_value: str | None = None
        explicit = supplied.get("project_dir")
        if isinstance(explicit, str) and explicit.strip():
            project_dir_value = explicit.strip()
        elif intent is not None and isinstance(intent.action_inputs.get("project_dir"), str):
            candidate = intent.action_inputs["project_dir"].strip()
            if candidate:
                project_dir_value = candidate
        elif binding.get("binding") == PROJECT_DIR_BINDING_OBSERVER:
            if base_dir is not None:
                project_dir_value = str(base_dir)
            elif intent is not None and intent.htr_runs_root:
                project_dir_value = intent.htr_runs_root
        elif binding.get("binding") == PROJECT_DIR_BINDING_EXPLICIT and intent is not None:
            candidate = intent.action_inputs.get("project_dir")
            if isinstance(candidate, str) and candidate.strip():
                project_dir_value = candidate.strip()
        if project_dir_value is None:
            raise ApprovalValidationError("could not materialize project_dir for approval binding")
        normalized = _normalize_path_for_digest(project_dir_value)
        path_digest = _project_dir_path_digest(project_dir_value)
        if binding.get("path_digest") and binding["path_digest"] != path_digest:
            raise ApprovalValidationError("project_dir path digest mismatch against plan binding")
        for index, entry in enumerate(entries):
            if entry.get("key") == "project_dir":
                entries[index] = {
                    "key": "project_dir",
                    "presence": "value",
                    "value": normalized,
                    "path_digest": path_digest,
                }
                break

    extra = set(supplied) - allowed
    if extra:
        raise ApprovalValidationError(f"unsupported supplied argument keys: {sorted(extra)}")

    return {"bound_api": api, "bound_arguments": {"argument_entries": entries}}


def _approval_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "approval_digest_projection_version": APPROVAL_DIGEST_PROJECTION_VERSION,
        "approval_schema_version": body["approval_schema_version"],
        "policy_version": body["policy_version"],
        "approval_id": body["approval_id"],
        "approval_kind": body["approval_kind"],
        "htr_runs_root_path_digest": body["htr_runs_root_path_digest"],
        "run_id": body["run_id"],
        "source_observation_digest": body["source_observation_digest"],
        "plan_digest": body["plan_digest"],
        "bound_api": body["bound_api"],
        "bound_arguments": body["bound_arguments"],
        "project_repository_checkpoint": body.get("project_repository_checkpoint"),
        "risk_class": body["risk_class"],
        "approver_id": body["approver_id"],
        "executor_id": body["executor_id"],
        "issued_at": body["issued_at"],
        "expires_at": body["expires_at"],
    }


def _compute_approval_digest(body: dict[str, Any]) -> str:
    return _sha256_digest(_approval_digest_projection(body))


def _claim_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_digest_projection_version": CLAIM_DIGEST_PROJECTION_VERSION,
        "claim_schema_version": body["claim_schema_version"],
        "approval_id": body["approval_id"],
        "approval_digest": body["approval_digest"],
        "claim_id": body["claim_id"],
        "claimant_id": body["claimant_id"],
        "executor_id": body["executor_id"],
        "source_observation_digest": body["source_observation_digest"],
        "plan_digest": body["plan_digest"],
        "bound_api": body["bound_api"],
        "bound_arguments": body["bound_arguments"],
        "claimed_at": body["claimed_at"],
    }


def _compute_claim_digest(body: dict[str, Any]) -> str:
    return _sha256_digest(_claim_digest_projection(body))


def _revoke_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "revoke_digest_projection_version": REVOKE_DIGEST_PROJECTION_VERSION,
        "revoke_schema_version": body["revoke_schema_version"],
        "approval_id": body["approval_id"],
        "approval_digest": body["approval_digest"],
        "revoked_by": body["revoked_by"],
        "revoked_at": body["revoked_at"],
        "reason": body["reason"],
    }


def _compute_revoke_digest(body: dict[str, Any]) -> str:
    return _sha256_digest(_revoke_digest_projection(body))


def _outcome_digest_projection(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_digest_projection_version": OUTCOME_DIGEST_PROJECTION_VERSION,
        "outcome_schema_version": body["outcome_schema_version"],
        "approval_id": body["approval_id"],
        "approval_digest": body["approval_digest"],
        "claim_id": body["claim_id"],
        "claim_digest": body["claim_digest"],
        "outcome_class": body["outcome_class"],
        "recorded_at": body["recorded_at"],
    }


def _compute_outcome_digest(body: dict[str, Any]) -> str:
    if body.get("outcome_schema_version") == OUTCOME_SCHEMA_VERSION_V2:
        return _sha256_digest(_outcome_digest_projection_v2(body))
    return _sha256_digest(_outcome_digest_projection(body))


def _outcome_digest_projection_v2(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_digest_projection_version": OUTCOME_DIGEST_PROJECTION_VERSION_V2,
        "outcome_schema_version": body["outcome_schema_version"],
        "approval_id": body["approval_id"],
        "approval_digest": body["approval_digest"],
        "claim_id": body["claim_id"],
        "claim_digest": body["claim_digest"],
        "outcome_class": body["outcome_class"],
        "recorded_at": body["recorded_at"],
        "outcome_evidence": body["outcome_evidence"],
    }


def _validate_outcome_evidence_v2(
    evidence: dict[str, Any],
    *,
    outcome_class: str,
) -> None:
    if not isinstance(evidence, dict):
        raise ApprovalValidationError("outcome_evidence must be a JSON object")
    required = (
        "reason_code",
        "error_classification",
        "bound_api",
        "event_id",
        "pre_observation_digest",
        "mutation_may_have_committed",
        "safe_to_retry",
        "verification_reason_codes",
    )
    for field in required:
        if field not in evidence:
            raise ApprovalValidationError(f"outcome_evidence missing {field!r}")
    if "post_observation_digest" not in evidence:
        raise ApprovalValidationError("outcome_evidence missing post_observation_digest")
    if not isinstance(evidence["reason_code"], str) or not evidence["reason_code"]:
        raise ApprovalValidationError("outcome_evidence.reason_code must be non-empty")
    if not isinstance(evidence["safe_to_retry"], bool):
        raise ApprovalValidationError("outcome_evidence.safe_to_retry must be boolean")
    if evidence["safe_to_retry"] is not False:
        raise ApprovalValidationError("Task 25 outcome_evidence.safe_to_retry must be false")
    if not isinstance(evidence["verification_reason_codes"], list):
        raise ApprovalValidationError("outcome_evidence.verification_reason_codes must be a list")
    if not isinstance(evidence["mutation_may_have_committed"], bool):
        raise ApprovalValidationError(
            "outcome_evidence.mutation_may_have_committed must be boolean"
        )
    if outcome_class == OUTCOME_CONSUMED and evidence["reason_code"] != "verified_success":
        raise ApprovalValidationError("consumed v2 outcome requires reason_code verified_success")
    if outcome_class == OUTCOME_AMBIGUOUS:
        if evidence["reason_code"] == "verified_success":
            raise ApprovalValidationError(
                "ambiguous v2 outcome cannot use reason_code verified_success"
            )
        if evidence.get("error_classification") == "verified_success":
            raise ApprovalValidationError(
                "ambiguous v2 outcome cannot use error_classification verified_success"
            )


@contextmanager
def _approval_control_barrier(
    run_id: str,
    base_dir: Path | None = None,
) -> Iterator[RunWriteContext]:
    """Acquire Task 23 marker for approval-control writes only (no lifecycle seal policy)."""
    validate_id(run_id, "run")
    _require_platform()

    nested = _find_nested_entry(run_id)
    if nested is not None:
        nested.depth += 1
        ctx = RunWriteContext(
            run_id=run_id,
            base_dir=base_dir,
            key=nested.key,
            token=nested.token,
            is_outermost=False,
            run_write_started=nested.run_write_started,
        )
        try:
            with bind_active_write_context(ctx):
                yield ctx
        finally:
            nested.depth -= 1
            ctx.run_write_started = nested.run_write_started
        return

    active_other = _thread_active_entry()
    if active_other is not None and active_other.key[2] != run_id:
        raise RunExecutionLockBoundaryViolationError(
            "cross-key nested mutation is not allowed"
        )

    ctx: RunWriteContext | None = None
    entry = None
    exc_info: BaseException | None = None
    try:
        ctx, entry = _acquire_outer_run_marker(run_id, base_dir)
        with bind_active_write_context(ctx):
            yield ctx
    except BaseException as exc:
        exc_info = exc
        raise
    finally:
        if ctx is not None and entry is not None:
            _exit_outer_barrier(ctx, entry, exc=exc_info)


@contextmanager
def _approval_use_session(
    run_id: str,
    base_dir: Path | None = None,
) -> Iterator[RunWriteContext]:
    """Internal Task 25 hook: hold marker across validate/claim/invoke/outcome."""
    validate_id(run_id, "run")
    prior = getattr(_approval_use_session_local, "session", None)
    with _approval_control_barrier(run_id, base_dir) as ctx:
        entry = _thread_active_entry()
        if entry is None or entry.depth <= 0:
            raise RunExecutionLockBoundaryViolationError(
                "approval-use session requires active run marker"
            )
        _approval_use_session_local.session = {
            "run_id": run_id,
            "base_dir": base_dir,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "token": entry.token,
            "key": entry.key,
            "ctx": ctx,
        }
        try:
            yield ctx
        finally:
            if prior is None:
                if hasattr(_approval_use_session_local, "session"):
                    delattr(_approval_use_session_local, "session")
            else:
                _approval_use_session_local.session = prior


_approval_use_session_local = threading.local()


def _require_active_approval_use_session(
    approval_id: str,
    base_dir: Path | None,
) -> str:
    """Validate Task 25 in-session helper ownership and approval/run binding."""
    session = getattr(_approval_use_session_local, "session", None)
    if session is None:
        raise ApprovalStateError(
            "approval-use session required for in-session control write",
            approval_id=approval_id,
        )
    if session["pid"] != os.getpid():
        raise RunExecutionLockBoundaryViolationError(
            "approval-use session is not owned by this process"
        )
    if session["thread_id"] != threading.get_ident():
        raise RunExecutionLockBoundaryViolationError(
            "approval-use session is not owned by this thread"
        )
    entry = _thread_active_entry()
    if entry is None or entry.depth <= 0:
        raise ApprovalStateError(
            "approval-use session requires active run marker depth",
            approval_id=approval_id,
        )
    if entry.owner_pid != os.getpid() or entry.owner_thread_id != threading.get_ident():
        raise RunExecutionLockBoundaryViolationError(
            "active run marker is not owned by this thread"
        )
    if entry.token != session["token"] or entry.key != session["key"]:
        raise RunExecutionLockBoundaryViolationError(
            "active run marker token does not match approval-use session"
        )
    if entry.key[2] != session["run_id"]:
        raise RunExecutionLockBoundaryViolationError(
            "cross-run approval-use session mismatch"
        )
    if session["base_dir"] != base_dir:
        raise ApprovalValidationError(
            "base_dir does not match active approval-use session"
        )
    bundle = _load_bundle(approval_id, base_dir)
    run_id = bundle["issue"]["run_id"]
    if run_id != session["run_id"]:
        raise ApprovalStateError(
            "approval run_id does not match active approval-use session",
            approval_id=approval_id,
        )
    return run_id


def _mark_control_write_started() -> None:
    """Mark Task 23 write-start immediately before first control-record mutation."""
    begin_run_write()


def _reject_replay_under_occupied_marker(run_id: str, base_dir: Path | None) -> None:
    if marker_present_noncreating(base_dir, run_id):
        raise RunExecutionLockOccupiedError(run_id=run_id)


def _readonly_exact_replay(
    existing: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    run_id: str,
    base_dir: Path | None,
    approval_id: str,
    label: str,
) -> dict[str, Any] | None:
    if existing is None:
        return None
    if _canonical_json(existing) == _canonical_json(expected):
        entry = _thread_active_entry()
        if entry is None or entry.key[2] != run_id:
            _reject_replay_under_occupied_marker(run_id, base_dir)
        return existing
    raise ApprovalConflictError(
        f"{label} already exists with conflicting semantics for {approval_id!r}",
        approval_id=approval_id,
    )


def _raise_unsafe_control_path(context: str, exc: BaseException) -> NoReturn:
    raise ApprovalValidationError(f"unsafe approval control path ({context})") from exc


def _open_control_dir_no_follow(path: Path, *, context: str) -> int:
    try:
        return os.open(str(path), _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
    except OSError as exc:
        _raise_unsafe_control_path(context, exc)


def _openat_control_dir_no_follow(dir_fd: int, name: str, *, context: str) -> int:
    try:
        return os.open(
            name,
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        _raise_unsafe_control_path(f"{context}/{name}", exc)


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
        _raise_unsafe_control_path(f"{context}/{name}", exc)


def _mkdirat_control(dir_fd: int, name: str, mode: int, *, context: str) -> None:
    try:
        os.mkdir(name, mode, dir_fd=dir_fd)
    except FileExistsError:
        return
    except OSError as exc:
        _raise_unsafe_control_path(f"{context}/{name}", exc)


def _open_dir_no_follow(path: Path) -> int:
    try:
        return os.open(str(path), _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(f"cannot open directory {path}: {exc}") from exc


def _fsync_dir_fd(dir_fd: int) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(f"directory fsync failed: {exc}") from exc


def _fsync_file_fd(file_fd: int) -> None:
    try:
        os.fsync(file_fd)
    except OSError as exc:
        raise RunExecutionLockIndeterminateError(f"file fsync failed: {exc}") from exc


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise RunExecutionLockIndeterminateError("short write while persisting record")
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
        raise ApprovalValidationError("expected JSON object record")
    return data


def _bootstrap_control_tree(base_dir: Path | None) -> int:
    runs_root = paths.runs_root(base_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    runs_fd = _open_control_dir_no_follow(runs_root, context="runs_root")
    try:
        _mkdirat_control(
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
            _mkdirat_control(
                control_fd,
                paths.APPROVALS_DIR_NAME,
                _CONTROL_DIR_MODE,
                context="runs_root/.control",
            )
            _fsync_dir_fd(control_fd)
        finally:
            os.close(control_fd)
        _fsync_dir_fd(runs_fd)
    finally:
        os.close(runs_fd)
    return _open_control_dir_no_follow(
        paths.control_approvals_root(base_dir),
        context="approvals_root",
    )


def _bootstrap_approval_dir_fd(approval_id: str, base_dir: Path | None) -> tuple[int, int]:
    approvals_fd = _bootstrap_control_tree(base_dir)
    try:
        _mkdirat_control(
            approvals_fd,
            approval_id,
            _CONTROL_DIR_MODE,
            context="approvals_root",
        )
        approval_fd = _openat_control_dir_no_follow(
            approvals_fd,
            approval_id,
            context=f"approvals_root/{approval_id}",
        )
        _fsync_dir_fd(approvals_fd)
        return approvals_fd, approval_fd
    except Exception:
        os.close(approvals_fd)
        raise


def _create_immutable_record(
    approval_id: str,
    filename: str,
    record: dict[str, Any],
    *,
    digest_field: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    approvals_fd, approval_fd = _bootstrap_approval_dir_fd(approval_id, base_dir)
    try:
        flags = _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
        record_ctx = f"approvals_root/{approval_id}/{filename}"
        try:
            file_fd = _openat_control_file_no_follow(
                approval_fd,
                filename,
                flags,
                _CONTROL_FILE_MODE,
                context=record_ctx,
            )
        except FileExistsError:
            flags_read = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
            existing_fd = _openat_control_file_no_follow(
                approval_fd,
                filename,
                flags_read,
                context=record_ctx,
            )
            try:
                existing = _read_json_fd(existing_fd)
            finally:
                os.close(existing_fd)
            if _canonical_json(existing) == _canonical_json(record):
                return existing
            raise ApprovalConflictError(
                f"{filename} already exists with conflicting semantics",
                approval_id=approval_id,
            )
        try:
            payload = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            _write_all(file_fd, payload)
            _fsync_file_fd(file_fd)
            _fsync_dir_fd(approval_fd)
        finally:
            os.close(file_fd)
        return record
    finally:
        os.close(approval_fd)
        os.close(approvals_fd)


def _read_optional_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ApprovalValidationError(f"unsafe symlink record: {path}")
    try:
        return read_json(path)
    except json.JSONDecodeError as exc:
        raise ApprovalValidationError(f"malformed JSON record: {path}") from exc


def _load_bundle(approval_id: str, base_dir: Path | None) -> dict[str, Any]:
    validate_id(approval_id, "approval")
    root = paths.approval_control_dir(approval_id, base_dir)
    issue = _read_optional_record(paths.approval_issue_path(approval_id, base_dir))
    if issue is None:
        raise ApprovalStateError(f"approval {approval_id!r} is missing issue.json", approval_id=approval_id)
    return {
        "approval_id": approval_id,
        "root": root,
        "issue": issue,
        "revoke": _read_optional_record(paths.approval_revoke_path(approval_id, base_dir)),
        "claim": _read_optional_record(paths.approval_claim_path(approval_id, base_dir)),
        "outcome": _read_optional_record(paths.approval_outcome_path(approval_id, base_dir)),
    }


def _evaluate_seal_for_lifecycle_issue(run_id: str, base_dir: Path | None) -> None:
    evaluation = evaluate_run_seal(run_id, base_dir)
    if evaluation.state == SealState.FINALIZED_VALID:
        raise ApprovalFinalizedRunError(
            f"cannot issue lifecycle approval for finalized run {run_id!r}",
            approval_id=None,
        )
    if evaluation.state in (SealState.CLOSURE_PRESENT_UNTRUSTED, SealState.INDETERMINATE):
        raise RunSealBlockedError(
            run_id=run_id,
            reason_codes=evaluation.reason_codes,
        )


def _evaluate_seal_for_lifecycle_claim(run_id: str, base_dir: Path | None) -> None:
    _evaluate_seal_for_lifecycle_issue(run_id, base_dir)


def _validate_expiry(issued_at: str, expires_at: str) -> None:
    issued = _parse_utc_iso(issued_at)
    expires = _parse_utc_iso(expires_at)
    if issued >= expires:
        raise ApprovalValidationError("issued_at must be strictly before expires_at")
    if expires - issued > MAX_APPROVAL_LIFETIME:
        raise ApprovalValidationError("approval lifetime exceeds 24 hours")


def _build_issue_record(
    *,
    approval_id: str,
    run_id: str,
    plan: dict[str, Any],
    approver_id: str,
    executor_id: str,
    issued_at: str,
    expires_at: str,
    base_dir: Path | None,
    requester_id: str | None,
    intent: PlanningIntent | None = None,
) -> dict[str, Any]:
    if plan.get("plan_state") != STATE_PROPOSABLE:
        raise ApprovalValidationError("plan must be proposable to issue approval")
    bound = _materialize_bound_arguments(plan, intent=intent, base_dir=base_dir)
    _validate_expiry(issued_at, expires_at)
    event_entries = [
        e for e in bound["bound_arguments"]["argument_entries"] if e["key"] == "event_id"
    ]
    if event_entries and event_entries[0]["presence"] == "value":
        event_id = event_entries[0]["value"]
        if _event_id_exists(run_id, event_id, base_dir):
            raise ApprovalValidationError(
                f"event_id {event_id!r} already exists in run event log"
            )

    body: dict[str, Any] = {
        "record_kind": "approval_issue",
        "approval_schema_version": APPROVAL_SCHEMA_VERSION,
        "policy_version": POLICY_C_VERSION,
        "approval_id": approval_id,
        "approval_kind": APPROVAL_KIND_LIFECYCLE_MUTATION,
        "htr_runs_root_path_digest": _runs_root_path_digest(base_dir),
        "run_id": run_id,
        "source_observation_digest": plan["source"]["source_observation_digest"],
        "plan_digest": plan["plan_digest"],
        "bound_api": bound["bound_api"],
        "bound_arguments": bound["bound_arguments"],
        "project_repository_checkpoint": plan.get("source", {}).get(
            "project_repository_checkpoint"
        ),
        "risk_class": (plan.get("risk") or {}).get("class"),
        "approver_id": _validate_identity_string(approver_id, field="approver_id"),
        "executor_id": _validate_identity_string(executor_id, field="executor_id"),
        "issued_at": issued_at,
        "expires_at": expires_at,
    }
    if requester_id is not None:
        body["requester_id"] = _validate_identity_string(requester_id, field="requester_id")
    body["approval_digest"] = _compute_approval_digest(body)
    return body


def _revalidated_plan_for_issue(
    run_id: str,
    intent: PlanningIntent,
    base_dir: Path | None,
    *,
    expected_plan_digest: str | None = None,
) -> dict[str, Any]:
    try:
        snapshot = build_run_snapshot(run_id, base_dir=base_dir)
    except ObserveInvocationError as exc:
        raise ApprovalValidationError(str(exc)) from exc
    runs_root = str(base_dir) if base_dir is not None else intent.htr_runs_root
    enriched = PlanningIntent(
        requested_action=intent.requested_action,
        action_inputs=intent.action_inputs,
        project_repository_checkpoint=intent.project_repository_checkpoint,
        htr_runs_root=runs_root,
        remediation_oriented=intent.remediation_oriented,
    )
    plan = build_action_plan(snapshot, enriched)
    if expected_plan_digest is not None and plan["plan_digest"] != expected_plan_digest:
        raise ApprovalValidationError("current plan digest does not match bound approval")
    live_obs = compute_source_observation_digest(snapshot)
    if live_obs != plan["source"]["source_observation_digest"]:
        raise ApprovalValidationError("observation digest mismatch during revalidation")
    return plan


def _derived_classifications(
    bundle: dict[str, Any],
    *,
    base_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or _utc_now()
    issue = bundle["issue"]
    derived: dict[str, Any] = {
        "authoritative_state": "issued",
        "expired": False,
        "invalidated": False,
        "blocked_finalized": False,
    }
    if bundle["revoke"] is not None:
        derived["authoritative_state"] = "revoked"
    elif bundle["outcome"] is not None:
        derived["authoritative_state"] = bundle["outcome"]["outcome_class"]
    elif bundle["claim"] is not None:
        derived["authoritative_state"] = "claimed"

    expires = _parse_utc_iso(issue["expires_at"])
    if now >= expires and derived["authoritative_state"] == "issued":
        derived["expired"] = True

    evaluation = evaluate_run_seal(issue["run_id"], base_dir)
    if evaluation.state == SealState.FINALIZED_VALID:
        derived["blocked_finalized"] = True
    return derived


def _validate_live_bindings(bundle: dict[str, Any], base_dir: Path | None) -> list[str]:
    reasons: list[str] = []
    issue = bundle["issue"]
    run_id = issue["run_id"]
    try:
        snapshot = build_run_snapshot(run_id, base_dir=base_dir)
    except Exception as exc:
        reasons.append(f"snapshot_unavailable:{exc}")
        return reasons
    live_obs = compute_source_observation_digest(snapshot)
    if live_obs != issue["source_observation_digest"]:
        reasons.append("source_observation_digest_mismatch")
    if _runs_root_path_digest(base_dir) != issue["htr_runs_root_path_digest"]:
        reasons.append("htr_runs_root_path_digest_mismatch")
    try:
        intent = PlanningIntent(
            requested_action=issue["bound_api"],
            action_inputs=_argument_entries_to_inputs(issue["bound_arguments"]),
            project_repository_checkpoint=issue.get("project_repository_checkpoint"),
            htr_runs_root=str(base_dir) if base_dir is not None else None,
        )
        plan = build_action_plan(snapshot, intent)
        if plan.get("plan_digest") != issue["plan_digest"]:
            reasons.append("plan_digest_mismatch")
    except Exception as exc:
        reasons.append(f"plan_revalidation_failed:{exc}")
    evaluation = evaluate_run_seal(run_id, base_dir)
    if evaluation.state == SealState.FINALIZED_VALID:
        reasons.append("blocked_finalized")
    elif evaluation.state in (SealState.CLOSURE_PRESENT_UNTRUSTED, SealState.INDETERMINATE):
        reasons.append("seal_untrusted_or_indeterminate")
    return reasons


def get_approval(approval_id: str, *, base_dir: Path | None = None) -> dict[str, Any]:
    bundle = _load_bundle(approval_id, base_dir)
    derived = _derived_classifications(bundle, base_dir=base_dir)
    return {
        "approval_id": approval_id,
        "issue": bundle["issue"],
        "revoke": bundle["revoke"],
        "claim": bundle["claim"],
        "outcome": bundle["outcome"],
        "derived": derived,
    }


def list_approvals(
    *,
    base_dir: Path | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    root = paths.control_approvals_root(base_dir)
    if not root.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.is_symlink():
            continue
        approval_id = entry.name
        if not validate_id(approval_id, "approval"):
            continue
        issue_path = entry / "issue.json"
        if not issue_path.is_file() or issue_path.is_symlink():
            continue
        try:
            view = get_approval(approval_id, base_dir=base_dir)
        except Exception:
            continue
        if run_id is not None and view["issue"].get("run_id") != run_id:
            continue
        results.append(view)
    return results


def validate_approval(approval_id: str, *, base_dir: Path | None = None) -> dict[str, Any]:
    view = get_approval(approval_id, base_dir=base_dir)
    derived = view["derived"]
    reasons = _validate_live_bindings(
        {
            "issue": view["issue"],
            "revoke": view["revoke"],
            "claim": view["claim"],
            "outcome": view["outcome"],
        },
        base_dir,
    )
    if reasons:
        derived["invalidated"] = True
    derived["validation_reasons"] = reasons
    derived["advisory_only"] = True
    return {"approval_id": approval_id, "derived": derived, "issue": view["issue"]}


def issue_approval(
    run_id: str,
    intent: PlanningIntent,
    approver_id: str,
    executor_id: str,
    expires_at: str,
    *,
    approval_id: str | None = None,
    requester_id: str | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    validate_id(run_id, "run")
    approval_id = approval_id or new_approval_id()
    validate_id(approval_id, "approval")
    issued_at = _utc_now_iso()
    _validate_expiry(issued_at, expires_at)

    issue_path = paths.approval_issue_path(approval_id, base_dir)
    existing_issue = _read_optional_record(issue_path)

    _evaluate_seal_for_lifecycle_issue(run_id, base_dir)
    plan = _revalidated_plan_for_issue(run_id, intent, base_dir)
    replay_issued_at = existing_issue["issued_at"] if existing_issue else issued_at
    expected = _build_issue_record(
        approval_id=approval_id,
        run_id=run_id,
        plan=plan,
        approver_id=approver_id,
        executor_id=executor_id,
        issued_at=replay_issued_at,
        expires_at=expires_at,
        base_dir=base_dir,
        requester_id=requester_id,
        intent=intent,
    )
    replay = _readonly_exact_replay(
        existing_issue,
        expected,
        run_id=run_id,
        base_dir=base_dir,
        approval_id=approval_id,
        label="issue.json",
    )
    if replay is not None:
        return replay

    with _approval_control_barrier(run_id, base_dir):
        _evaluate_seal_for_lifecycle_issue(run_id, base_dir)
        plan = _revalidated_plan_for_issue(run_id, intent, base_dir)
        replay_issued_at = existing_issue["issued_at"] if existing_issue else issued_at
        record = _build_issue_record(
            approval_id=approval_id,
            run_id=run_id,
            plan=plan,
            approver_id=approver_id,
            executor_id=executor_id,
            issued_at=replay_issued_at,
            expires_at=expires_at,
            base_dir=base_dir,
            requester_id=requester_id,
            intent=intent,
        )
        fresh_existing = _read_optional_record(issue_path)
        replay = _readonly_exact_replay(
            fresh_existing,
            record,
            run_id=run_id,
            base_dir=base_dir,
            approval_id=approval_id,
            label="issue.json",
        )
        if replay is not None:
            return replay
        _mark_control_write_started()
        created = _create_immutable_record(
            approval_id,
            "issue.json",
            record,
            digest_field="approval_digest",
            base_dir=base_dir,
        )
    return created


def revoke_approval(
    approval_id: str,
    approver_id: str,
    reason: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    validate_id(approval_id, "approval")
    bundle = _load_bundle(approval_id, base_dir)
    run_id = bundle["issue"]["run_id"]
    approver = _validate_identity_string(approver_id, field="approver_id")
    if not isinstance(reason, str) or not reason.strip():
        raise ApprovalValidationError("revoke reason must be a non-empty string")

    replay_revoked_at = (
        bundle["revoke"]["revoked_at"] if bundle["revoke"] is not None else _utc_now_iso()
    )
    record = {
        "record_kind": "approval_revoke",
        "revoke_schema_version": REVOKE_SCHEMA_VERSION,
        "approval_id": approval_id,
        "approval_digest": _record_fingerprint(bundle["issue"], "approval_digest"),
        "revoked_by": approver,
        "revoked_at": replay_revoked_at,
        "reason": reason.strip(),
    }
    replay = _readonly_exact_replay(
        bundle["revoke"],
        record,
        run_id=run_id,
        base_dir=base_dir,
        approval_id=approval_id,
        label="revoke.json",
    )
    if replay is not None:
        return replay

    with _approval_control_barrier(run_id, base_dir):
        fresh = _load_bundle(approval_id, base_dir)
        replay_revoked_at = (
            fresh["revoke"]["revoked_at"] if fresh["revoke"] is not None else _utc_now_iso()
        )
        record = {
            "record_kind": "approval_revoke",
            "revoke_schema_version": REVOKE_SCHEMA_VERSION,
            "approval_id": approval_id,
            "approval_digest": _record_fingerprint(fresh["issue"], "approval_digest"),
            "revoked_by": approver,
            "revoked_at": replay_revoked_at,
            "reason": reason.strip(),
        }
        replay = _readonly_exact_replay(
            fresh["revoke"],
            record,
            run_id=run_id,
            base_dir=base_dir,
            approval_id=approval_id,
            label="revoke.json",
        )
        if replay is not None:
            return replay
        _mark_control_write_started()
        created = _create_immutable_record(
            approval_id,
            "revoke.json",
            record,
            digest_field="approval_digest",
            base_dir=base_dir,
        )
    return created


def _assert_approval_active_for_claim(bundle: dict[str, Any], *, now: datetime | None = None) -> None:
    issue = bundle["issue"]
    if bundle["revoke"] is not None:
        raise ApprovalStateError("approval is revoked", approval_id=issue["approval_id"])
    if bundle["claim"] is not None:
        existing_claim_id = bundle["claim"].get("claim_id")
        raise ApprovalStateError(
            f"approval already claimed as {existing_claim_id!r}",
            approval_id=issue["approval_id"],
        )
    if bundle["outcome"] is not None:
        raise ApprovalStateError("approval already has outcome", approval_id=issue["approval_id"])
    now = now or _utc_now()
    if now >= _parse_utc_iso(issue["expires_at"]):
        raise ApprovalStateError("approval is expired", approval_id=issue["approval_id"])


def _build_claim_record_body(
    fresh: dict[str, Any],
    approval_id: str,
    claim_id: str,
    claimant: str,
    base_dir: Path | None,
) -> dict[str, Any]:
    issue = fresh["issue"]
    run_id = issue["run_id"]
    intent = PlanningIntent(
        requested_action=issue["bound_api"],
        action_inputs=_argument_entries_to_inputs(issue["bound_arguments"]),
        project_repository_checkpoint=issue.get("project_repository_checkpoint"),
        htr_runs_root=str(base_dir) if base_dir is not None else None,
    )
    plan = _revalidated_plan_for_issue(
        run_id,
        intent,
        base_dir,
        expected_plan_digest=issue["plan_digest"],
    )
    if plan["plan_digest"] != issue["plan_digest"]:
        raise ApprovalValidationError("plan digest mismatch at claim")
    if plan["source"]["source_observation_digest"] != issue["source_observation_digest"]:
        raise ApprovalValidationError("observation digest mismatch at claim")
    replay_claimed_at = (
        fresh["claim"]["claimed_at"] if fresh["claim"] is not None else _utc_now_iso()
    )
    body = {
        "record_kind": "approval_claim",
        "claim_schema_version": CLAIM_SCHEMA_VERSION,
        "approval_id": approval_id,
        "approval_digest": issue["approval_digest"],
        "claim_id": claim_id,
        "claimant_id": claimant,
        "executor_id": issue["executor_id"],
        "source_observation_digest": plan["source"]["source_observation_digest"],
        "plan_digest": plan["plan_digest"],
        "bound_api": issue["bound_api"],
        "bound_arguments": issue["bound_arguments"],
        "claimed_at": replay_claimed_at,
    }
    body["claim_digest"] = _compute_claim_digest(body)
    return body


def _claim_approval_during_session(
    approval_id: str,
    claim_id: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist claim.json while the caller holds ``_approval_use_session`` (Task 25)."""
    validate_id(approval_id, "approval")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ApprovalValidationError("claim_id must be a non-empty string")
    run_id = _require_active_approval_use_session(approval_id, base_dir)
    bundle = _load_bundle(approval_id, base_dir)
    issue = bundle["issue"]
    if issue["run_id"] != run_id:
        raise ApprovalStateError(
            "approval run_id does not match active approval-use session",
            approval_id=approval_id,
        )
    claimant = _validate_identity_string(issue["executor_id"], field="claimant_id")
    if bundle["claim"] is not None:
        raise ApprovalStateError("approval already claimed", approval_id=approval_id)
    _evaluate_seal_for_lifecycle_claim(run_id, base_dir)
    body = _build_claim_record_body(bundle, approval_id, claim_id.strip(), claimant, base_dir)
    _assert_approval_active_for_claim(bundle)
    _mark_control_write_started()
    return _create_immutable_record(
        approval_id,
        "claim.json",
        body,
        digest_field="claim_digest",
        base_dir=base_dir,
    )


def claim_approval(
    approval_id: str,
    claim_id: str,
    claimant_id: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    validate_id(approval_id, "approval")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ApprovalValidationError("claim_id must be a non-empty string")
    claimant = _validate_identity_string(claimant_id, field="claimant_id")
    bundle = _load_bundle(approval_id, base_dir)
    issue = bundle["issue"]
    run_id = issue["run_id"]
    if claimant != issue["executor_id"]:
        raise ApprovalValidationError("claimant_id must equal issue executor_id")

    def _build_claim_body(fresh: dict[str, Any]) -> dict[str, Any]:
        return _build_claim_record_body(
            fresh, approval_id, claim_id.strip(), claimant, base_dir
        )

    _evaluate_seal_for_lifecycle_claim(run_id, base_dir)
    body = _build_claim_body(bundle)
    if bundle["claim"] is not None:
        if bundle["claim"].get("claim_id") != body["claim_id"]:
            raise ApprovalStateError(
                "approval already claimed by a different claim_id",
                approval_id=approval_id,
            )
        replay = _readonly_exact_replay(
            bundle["claim"],
            body,
            run_id=run_id,
            base_dir=base_dir,
            approval_id=approval_id,
            label="claim.json",
        )
        if replay is not None:
            return replay

    with _approval_control_barrier(run_id, base_dir):
        _evaluate_seal_for_lifecycle_claim(run_id, base_dir)
        fresh = _load_bundle(approval_id, base_dir)
        body = _build_claim_body(fresh)
        if fresh["claim"] is not None:
            if fresh["claim"].get("claim_id") != body["claim_id"]:
                raise ApprovalStateError(
                    "approval already claimed by a different claim_id",
                    approval_id=approval_id,
                )
            replay = _readonly_exact_replay(
                fresh["claim"],
                body,
                run_id=run_id,
                base_dir=base_dir,
                approval_id=approval_id,
                label="claim.json",
            )
            if replay is not None:
                return replay
            raise ApprovalConflictError(
                "claim.json already exists with conflicting semantics",
                approval_id=approval_id,
            )
        _assert_approval_active_for_claim(fresh)
        _mark_control_write_started()
        created = _create_immutable_record(
            approval_id,
            "claim.json",
            body,
            digest_field="claim_digest",
            base_dir=base_dir,
        )
    return created


def _argument_entries_to_inputs(bound_arguments: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for entry in bound_arguments.get("argument_entries", []):
        presence = entry.get("presence")
        key = entry.get("key")
        if presence == "value":
            inputs[key] = entry.get("value")
        elif presence == "null":
            inputs[key] = None
    return inputs


def _build_outcome_record(
    *,
    approval_id: str,
    issue: dict[str, Any],
    claim: dict[str, Any],
    claim_id: str,
    outcome_class: str,
    recorded_at: str,
    outcome_evidence: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    if outcome_evidence is not None:
        _validate_outcome_evidence_v2(evidence=outcome_evidence, outcome_class=outcome_class)
        record = {
            "record_kind": "approval_outcome",
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION_V2,
            "approval_id": approval_id,
            "approval_digest": issue["approval_digest"],
            "claim_id": claim_id,
            "claim_digest": _record_fingerprint(claim, "claim_digest"),
            "outcome_class": outcome_class,
            "recorded_at": recorded_at,
            "outcome_evidence": outcome_evidence,
        }
        record["outcome_digest"] = _compute_outcome_digest(record)
        return record, "outcome_digest"
    record = {
        "record_kind": "approval_outcome",
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "approval_id": approval_id,
        "approval_digest": issue["approval_digest"],
        "claim_id": claim_id,
        "claim_digest": _record_fingerprint(claim, "claim_digest"),
        "outcome_class": outcome_class,
        "recorded_at": recorded_at,
    }
    return record, "claim_digest"


def _record_use_outcome_during_session(
    approval_id: str,
    claim_id: str,
    outcome_class: str,
    *,
    outcome_evidence: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist outcome.json while the caller holds ``_approval_use_session`` (Task 25)."""
    if outcome_class not in {OUTCOME_CONSUMED, OUTCOME_AMBIGUOUS}:
        raise ApprovalValidationError(
            f"outcome_class must be {OUTCOME_CONSUMED!r} or {OUTCOME_AMBIGUOUS!r}"
        )
    validate_id(approval_id, "approval")
    _require_active_approval_use_session(approval_id, base_dir)
    bundle = _load_bundle(approval_id, base_dir)
    issue = bundle["issue"]
    if bundle["claim"] is None:
        raise ApprovalStateError("cannot record outcome without claim", approval_id=approval_id)
    if bundle["claim"].get("claim_id") != claim_id:
        raise ApprovalStateError("claim_id does not match existing claim", approval_id=approval_id)
    if bundle["revoke"] is not None:
        raise ApprovalStateError("cannot record outcome for revoked approval", approval_id=approval_id)
    replay_recorded_at = (
        bundle["outcome"]["recorded_at"] if bundle["outcome"] is not None else _utc_now_iso()
    )
    record, digest_field = _build_outcome_record(
        approval_id=approval_id,
        issue=issue,
        claim=bundle["claim"],
        claim_id=claim_id,
        outcome_class=outcome_class,
        recorded_at=replay_recorded_at,
        outcome_evidence=outcome_evidence,
    )
    replay = _readonly_exact_replay(
        bundle["outcome"],
        record,
        run_id=issue["run_id"],
        base_dir=base_dir,
        approval_id=approval_id,
        label="outcome.json",
    )
    if replay is not None:
        return replay
    _mark_control_write_started()
    return _create_immutable_record(
        approval_id,
        "outcome.json",
        record,
        digest_field=digest_field,
        base_dir=base_dir,
    )


def record_use_outcome(
    approval_id: str,
    claim_id: str,
    outcome_class: str,
    *,
    outcome_evidence: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    if outcome_class not in {OUTCOME_CONSUMED, OUTCOME_AMBIGUOUS}:
        raise ApprovalValidationError(
            f"outcome_class must be {OUTCOME_CONSUMED!r} or {OUTCOME_AMBIGUOUS!r}"
        )
    validate_id(approval_id, "approval")
    bundle = _load_bundle(approval_id, base_dir)
    issue = bundle["issue"]
    run_id = issue["run_id"]
    if bundle["claim"] is None:
        raise ApprovalStateError("cannot record outcome without claim", approval_id=approval_id)
    if bundle["claim"].get("claim_id") != claim_id:
        raise ApprovalStateError("claim_id does not match existing claim", approval_id=approval_id)
    if bundle["revoke"] is not None:
        raise ApprovalStateError("cannot record outcome for revoked approval", approval_id=approval_id)

    replay_recorded_at = (
        bundle["outcome"]["recorded_at"] if bundle["outcome"] is not None else _utc_now_iso()
    )
    record, digest_field = _build_outcome_record(
        approval_id=approval_id,
        issue=issue,
        claim=bundle["claim"],
        claim_id=claim_id,
        outcome_class=outcome_class,
        recorded_at=replay_recorded_at,
        outcome_evidence=outcome_evidence,
    )
    replay = _readonly_exact_replay(
        bundle["outcome"],
        record,
        run_id=run_id,
        base_dir=base_dir,
        approval_id=approval_id,
        label="outcome.json",
    )
    if replay is not None:
        return replay

    with _approval_control_barrier(run_id, base_dir):
        fresh = _load_bundle(approval_id, base_dir)
        replay_recorded_at = (
            fresh["outcome"]["recorded_at"] if fresh["outcome"] is not None else _utc_now_iso()
        )
        record, digest_field = _build_outcome_record(
            approval_id=approval_id,
            issue=fresh["issue"],
            claim=fresh["claim"],
            claim_id=claim_id,
            outcome_class=outcome_class,
            recorded_at=replay_recorded_at,
            outcome_evidence=outcome_evidence,
        )
        replay = _readonly_exact_replay(
            fresh["outcome"],
            record,
            run_id=run_id,
            base_dir=base_dir,
            approval_id=approval_id,
            label="outcome.json",
        )
        if replay is not None:
            return replay
        _mark_control_write_started()
        created = _create_immutable_record(
            approval_id,
            "outcome.json",
            record,
            digest_field=digest_field,
            base_dir=base_dir,
        )
    return created


__all__ = [
    "OUTCOME_AMBIGUOUS",
    "OUTCOME_CONSUMED",
    "claim_approval",
    "get_approval",
    "issue_approval",
    "list_approvals",
    "record_use_outcome",
    "revoke_approval",
    "validate_approval",
]
