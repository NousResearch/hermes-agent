"""Stage 33-B — deterministic attempt execute/verify primitive (canary only).

This module is a local isolated-experiment primitive. It is not Goal planning,
not a generic dispatcher, and not a product Task 33 closure.

Write sequence for execute (no cross-file transaction):
  1. precondition checks (no durable canary writes)
  2. O_EXCL create + write + fsync canary bytes through O_NOFOLLOW fds
  3. SHA-256 of the same fd
  4. add_artifact (manifest)
  5. submit_attempt_result

A failure after step 2 may leave a canary file and/or manifest entry.
Those failures are never reported as success and are never auto-repaired.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from htr.artifacts import add_artifact, list_artifacts
from htr.contracts import make_attempt_result, make_verification_result
from htr.events import submit_attempt_result, submit_manual_verification
from htr.execution_lock import begin_run_write, run_mutation_boundary
from htr.ids import validate_id
from htr.io import read_json
from htr.paths import (
    attempt_dir,
    attempt_status_path,
    result_json_path,
    run_manifest_path,
    task_status_path,
)
from htr.state import (
    ATTEMPT_RESULT_SUBMITTED,
    ATTEMPT_RUNNING,
    HTRStateError,
    TASK_RUNNING,
)

CANARY_RELATIVE_PATH = "artifacts/htr_attempt_canary.txt"
CANARY_DIR_NAME = "artifacts"
CANARY_FILE_NAME = "htr_attempt_canary.txt"
CANARY_BYTES = b"HTR_ATTEMPT_EXECUTION_CANARY_V1\n"
CANARY_KIND = "file"
PRODUCED_BY = "htr.attempt_execute"

_O_RDONLY = os.O_RDONLY
_O_RDWR = os.O_RDWR
_O_DIRECTORY = os.O_DIRECTORY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class AttemptExecuteError(HTRStateError):
    """Fail-closed error for the canary execute/verify primitive."""


def _require_posix_nofollow() -> None:
    if os.name != "posix" or not _O_NOFOLLOW or not _O_CLOEXEC:
        raise AttemptExecuteError("attempt canary execute requires POSIX O_NOFOLLOW")


def _open_dir_no_follow(path: Path, *, context: str) -> int:
    try:
        return os.open(
            str(path),
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
    except OSError as exc:
        raise AttemptExecuteError(
            f"refused unsafe directory {context}: {exc}"
        ) from exc


def _openat_dir_no_follow(dir_fd: int, name: str, *, context: str) -> int:
    try:
        return os.open(
            name,
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        raise AttemptExecuteError(
            f"refused unsafe directory {context}: {exc}"
        ) from exc


def _reject_symlink_name(dir_fd: int, name: str, *, context: str) -> None:
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        raise AttemptExecuteError(f"cannot stat {context}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise AttemptExecuteError(f"symlink escape refused at {context}")


def _write_all_fd(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise AttemptExecuteError("short write while persisting canary artifact")
        offset += written


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _read_all_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _assert_regular_file(fd: int, *, context: str) -> None:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise AttemptExecuteError(f"{context} is not a regular file")


def _containment_error(path: Path, attempt_root: Path) -> None:
    raise AttemptExecuteError(
        f"artifact path escapes attempt workspace: {str(path)!r} vs {str(attempt_root)!r}"
    )


def _assert_fd_inside_attempt(fd: int, attempt_root: Path, *, context: str) -> Path:
    try:
        proc_path = Path(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError as exc:
        raise AttemptExecuteError(f"cannot resolve {context}: {exc}") from exc
    try:
        resolved = proc_path.resolve()
        attempt_resolved = attempt_root.resolve()
    except OSError as exc:
        raise AttemptExecuteError(f"cannot resolve {context}: {exc}") from exc
    try:
        resolved.relative_to(attempt_resolved)
    except ValueError:
        _containment_error(resolved, attempt_resolved)
    return resolved


def _open_canary_for_write(attempt_root: Path) -> int:
    """Create the canary file without following symlinks. Caller closes fd."""
    _require_posix_nofollow()
    attempt_fd = _open_dir_no_follow(attempt_root, context="attempt_root")
    artifacts_fd = None
    file_fd = None
    try:
        _reject_symlink_name(attempt_fd, CANARY_DIR_NAME, context=CANARY_DIR_NAME)
        artifacts_fd = _openat_dir_no_follow(
            attempt_fd, CANARY_DIR_NAME, context=CANARY_DIR_NAME
        )
        _assert_fd_inside_attempt(artifacts_fd, attempt_root, context="artifacts_dir")
        try:
            existing = os.stat(
                CANARY_FILE_NAME, dir_fd=artifacts_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        except OSError as exc:
            raise AttemptExecuteError(
                f"cannot inspect canary path: {exc}"
            ) from exc
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise AttemptExecuteError(
                    f"symlink escape refused at {CANARY_RELATIVE_PATH}"
                )
            raise AttemptExecuteError("canary artifact already exists")
        try:
            file_fd = os.open(
                CANARY_FILE_NAME,
                _O_CREAT | _O_EXCL | _O_RDWR | _O_NOFOLLOW | _O_CLOEXEC,
                0o600,
                dir_fd=artifacts_fd,
            )
        except FileExistsError as exc:
            raise AttemptExecuteError("canary artifact already exists") from exc
        except OSError as exc:
            raise AttemptExecuteError(
                f"refused unsafe canary create: {exc}"
            ) from exc
        _assert_regular_file(file_fd, context=CANARY_RELATIVE_PATH)
        _assert_fd_inside_attempt(file_fd, attempt_root, context="canary_file")
        return file_fd
    except Exception:
        if file_fd is not None:
            os.close(file_fd)
        raise
    finally:
        if artifacts_fd is not None:
            os.close(artifacts_fd)
        os.close(attempt_fd)


def _open_canary_for_read(attempt_root: Path) -> int | None:
    """Open the canary file without following symlinks. None if missing."""
    _require_posix_nofollow()
    attempt_fd = _open_dir_no_follow(attempt_root, context="attempt_root")
    artifacts_fd = None
    try:
        try:
            _reject_symlink_name(attempt_fd, CANARY_DIR_NAME, context=CANARY_DIR_NAME)
            artifacts_fd = _openat_dir_no_follow(
                attempt_fd, CANARY_DIR_NAME, context=CANARY_DIR_NAME
            )
        except AttemptExecuteError:
            return None
        _assert_fd_inside_attempt(artifacts_fd, attempt_root, context="artifacts_dir")
        try:
            _reject_symlink_name(
                artifacts_fd, CANARY_FILE_NAME, context=CANARY_RELATIVE_PATH
            )
            file_fd = os.open(
                CANARY_FILE_NAME,
                _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=artifacts_fd,
            )
        except FileNotFoundError:
            return None
        except (OSError, AttemptExecuteError):
            return None
        try:
            _assert_regular_file(file_fd, context=CANARY_RELATIVE_PATH)
            _assert_fd_inside_attempt(file_fd, attempt_root, context="canary_file")
        except AttemptExecuteError:
            os.close(file_fd)
            raise
        return file_fd
    finally:
        if artifacts_fd is not None:
            os.close(artifacts_fd)
        os.close(attempt_fd)


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _require_execute_preconditions(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None,
) -> Path:
    validate_id(run_id, "run")
    validate_id(task_id, "task")
    validate_id(attempt_id, "attempt")

    manifest_path = run_manifest_path(run_id, base_dir)
    if not manifest_path.exists():
        raise AttemptExecuteError(f"run workspace missing for {run_id!r}")

    task_path = task_status_path(run_id, task_id, base_dir)
    if not task_path.exists():
        raise AttemptExecuteError(f"task workspace missing for {task_id!r}")
    task_status = read_json(task_path)
    if task_status.get("status") != TASK_RUNNING:
        raise AttemptExecuteError(
            f"task {task_id!r} is not running; status is {task_status.get('status')!r}"
        )
    if attempt_id not in task_status.get("attempts", []):
        raise AttemptExecuteError(
            f"attempt {attempt_id!r} is not registered on task {task_id!r}"
        )

    attempt_path = attempt_status_path(run_id, task_id, attempt_id, base_dir)
    if not attempt_path.exists():
        raise AttemptExecuteError(f"attempt {attempt_id!r} is not registered")
    attempt_status = read_json(attempt_path)
    if attempt_status.get("status") != ATTEMPT_RUNNING:
        raise AttemptExecuteError(
            f"attempt {attempt_id!r} is not running; "
            f"status is {attempt_status.get('status')!r}"
        )

    return attempt_dir(run_id, task_id, attempt_id, base_dir)


def _require_verify_preconditions(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None,
) -> Path:
    validate_id(run_id, "run")
    validate_id(task_id, "task")
    validate_id(attempt_id, "attempt")

    attempt_path = attempt_status_path(run_id, task_id, attempt_id, base_dir)
    if not attempt_path.exists():
        raise AttemptExecuteError(f"attempt {attempt_id!r} is not registered")
    attempt_status = read_json(attempt_path)
    if attempt_status.get("status") != ATTEMPT_RESULT_SUBMITTED:
        raise AttemptExecuteError(
            f"attempt {attempt_id!r} is not result_submitted; "
            f"status is {attempt_status.get('status')!r}"
        )
    return attempt_dir(run_id, task_id, attempt_id, base_dir)


def _check(
    name: str,
    passed: bool,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "message": message,
        "metadata": metadata if metadata is not None else {},
    }


@run_mutation_boundary
def execute_attempt_canary(
    run_id: str,
    task_id: str,
    attempt_id: str,
    *,
    actor: str = PRODUCED_BY,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Write the fixed canary artifact, register it, and submit attempt_result.

    Does not accept caller result payloads, checksums, or verification outcomes.
    """
    attempt_root = _require_execute_preconditions(
        run_id, task_id, attempt_id, base_dir
    )
    begin_run_write()
    file_fd = _open_canary_for_write(attempt_root)
    digest: str | None = None
    try:
        _write_all_fd(file_fd, CANARY_BYTES)
        os.fsync(file_fd)
        digest = _sha256_fd(file_fd)
        hashed_bytes = _read_all_fd(file_fd)
        if hashed_bytes != CANARY_BYTES:
            raise AttemptExecuteError("canary fd bytes diverged from CANARY_BYTES")
        if hashlib.sha256(hashed_bytes).hexdigest() != digest:
            raise AttemptExecuteError("canary fd digest diverged from SHA-256")
    except Exception:
        os.close(file_fd)
        raise
    os.close(file_fd)

    entry = add_artifact(
        run_id,
        task_id,
        attempt_id,
        path=CANARY_RELATIVE_PATH,
        kind=CANARY_KIND,
        sha256=digest,
        size_bytes=len(CANARY_BYTES),
        metadata={"canary": "htr_attempt_execution_v1"},
        base_dir=base_dir,
    )
    result = make_attempt_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        produced_by=PRODUCED_BY,
        summary="deterministic attempt canary artifact written",
        outputs={
            "canary_relative_path": CANARY_RELATIVE_PATH,
            "sha256": digest,
            "size_bytes": len(CANARY_BYTES),
        },
        artifacts=[
            {
                "path": CANARY_RELATIVE_PATH,
                "kind": CANARY_KIND,
                "sha256": digest,
                "size_bytes": len(CANARY_BYTES),
            }
        ],
        metadata={"primitive": "attempt_execute_canary_v1"},
    )
    event = submit_attempt_result(
        run_id,
        task_id,
        attempt_id,
        result,
        actor=actor,
        base_dir=base_dir,
    )
    return {
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "artifact": entry,
        "sha256": digest,
        "size_bytes": len(CANARY_BYTES),
        "result": result,
        "event": event,
    }


@run_mutation_boundary
def verify_attempt_canary(
    run_id: str,
    task_id: str,
    attempt_id: str,
    *,
    actor: str = PRODUCED_BY,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Re-read the canary artifact and record a measured verification outcome."""
    attempt_root = _require_verify_preconditions(
        run_id, task_id, attempt_id, base_dir
    )
    begin_run_write()

    result_path = result_json_path(run_id, task_id, attempt_id, base_dir)
    stored_result = _load_json_if_present(result_path)
    stored_digest = None
    if isinstance(stored_result, dict):
        outputs = stored_result.get("outputs") or {}
        if isinstance(outputs, dict):
            stored_digest = outputs.get("sha256")

    manifest_entry = None
    try:
        for item in list_artifacts(
            run_id, task_id, attempt_id, base_dir=base_dir
        ):
            if item.get("path") == CANARY_RELATIVE_PATH:
                manifest_entry = item
                break
    except (OSError, ValueError, FileNotFoundError):
        manifest_entry = None

    file_fd = _open_canary_for_read(attempt_root)
    exists = file_fd is not None
    inside_workspace = file_fd is not None
    measured_bytes = b""
    measured_digest = None
    try:
        if file_fd is not None:
            measured_bytes = _read_all_fd(file_fd)
            measured_digest = hashlib.sha256(measured_bytes).hexdigest()
    finally:
        if file_fd is not None:
            os.close(file_fd)

    bytes_match = exists and measured_bytes == CANARY_BYTES
    digest_matches_execute = (
        exists
        and measured_digest is not None
        and stored_digest == measured_digest
        and stored_digest == hashlib.sha256(CANARY_BYTES).hexdigest()
    )
    manifest_matches = (
        exists
        and isinstance(manifest_entry, dict)
        and manifest_entry.get("path") == CANARY_RELATIVE_PATH
        and manifest_entry.get("sha256") == measured_digest
        and manifest_entry.get("size_bytes") == len(measured_bytes)
        and measured_digest is not None
    )

    checks = [
        _check(
            "artifact_exists",
            exists,
            "canary artifact is a regular file" if exists else "canary artifact missing",
        ),
        _check(
            "artifact_inside_workspace",
            inside_workspace,
            "canary opened without following a symlink out of the attempt workspace"
            if inside_workspace
            else "canary path is missing or escaped the attempt workspace",
        ),
        _check(
            "bytes_match_canary",
            bytes_match,
            "measured bytes equal CANARY_BYTES"
            if bytes_match
            else "measured bytes do not equal CANARY_BYTES",
            {"size_bytes": len(measured_bytes)},
        ),
        _check(
            "sha256_matches_execute_record",
            digest_matches_execute,
            "recomputed SHA-256 matches execute record"
            if digest_matches_execute
            else "recomputed SHA-256 does not match execute record",
            {
                "measured_sha256": measured_digest,
                "execute_sha256": stored_digest,
            },
        ),
        _check(
            "manifest_matches_file",
            manifest_matches,
            "artifact manifest matches the measured file"
            if manifest_matches
            else "artifact manifest does not match the measured file",
            {
                "manifest_sha256": (manifest_entry or {}).get("sha256"),
                "measured_sha256": measured_digest,
            },
        ),
    ]
    all_passed = all(item["status"] == "passed" for item in checks)
    outcome = "passed" if all_passed else "failed"
    verification = make_verification_result(
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        outcome=outcome,
        summary=(
            "deterministic canary verification passed"
            if all_passed
            else "deterministic canary verification failed"
        ),
        checks=checks,
        metadata={"primitive": "attempt_execute_canary_v1"},
    )
    event = submit_manual_verification(
        run_id,
        task_id,
        attempt_id,
        verification,
        actor=actor,
        base_dir=base_dir,
    )
    return {
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "outcome": outcome,
        "checks": checks,
        "verification": verification,
        "event": event,
        "measured_sha256": measured_digest,
    }
