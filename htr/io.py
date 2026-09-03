"""Atomic file IO and minimal HTR workspace bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from htr import paths
from htr.execution_lock import begin_run_write, run_mutation_boundary
from htr.ids import validate_id
from htr.schemas import SchemaName, validate

_O_DIRECTORY = os.O_DIRECTORY
_O_RDONLY = os.O_RDONLY
_O_WRONLY = os.O_WRONLY
_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


@dataclass(frozen=True)
class RunRootReservation:
    """Pinned fds for an exclusively reserved successor run root."""

    runs_root_fd: int
    run_root_fd: int
    run_id: str
    created: bool


class RunRootReservationError(Exception):
    """Successor run root could not be reserved safely."""

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        super().__init__(message)
        self.run_id = run_id


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) when missing; return the Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync on *directory* (may be unsupported on some platforms)."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def file_lock(lock_path: Path | str):
    """Minimal per-run advisory file lock via ``fcntl.flock`` (Unix) or best-effort.

    Usage::

        with file_lock(lock_path):
            # critical section
    """
    target = Path(lock_path)
    ensure_dir(target.parent)
    try:
        import fcntl  # Unix-only; import inside to allow Windows fallback
        fd = os.open(str(target), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    except (ImportError, OSError):
        # Fallback: best-effort (no lock on Windows; accept risk for Phase 1)
        yield


def atomic_write_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write JSON atomically via a unique temp file in the target directory.

    Uses ``NamedTemporaryFile`` with a per-target prefix (not a fixed
    ``{name}.tmp`` suffix), flushes and fsyncs payload bytes, replaces into
    place, then best-effort fsyncs the parent directory. Cleans up the temp
    file on failure.
    """
    target = Path(path)
    ensure_dir(target.parent)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        os.replace(tmp_path, target)
        tmp_path = None
        _fsync_dir(target.parent)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def read_json(path: Path | str) -> dict[str, Any]:
    """Read a JSON object from *path*."""
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {target}")
    return data


def append_jsonl(path: Path | str, obj: dict[str, Any]) -> None:
    """Append one JSON object as a single JSONL line with fsync."""
    target = Path(path)
    ensure_dir(target.parent)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object lines in {target}")
        rows.append(row)
    return rows


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of *path*."""
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch_jsonl(path: Path) -> None:
    """Create an empty JSONL file when missing; never truncate existing content."""
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _init_json_if_missing(
    path: Path,
    data: dict[str, Any],
    schema_name: SchemaName,
) -> None:
    """Write *data* to *path* only when the file does not yet exist."""
    if path.exists():
        return
    validate(data, schema_name)
    atomic_write_json(path, data)


@run_mutation_boundary
def create_run_workspace(run_id: str, base_dir: Path | None = None) -> Path:
    """Create run-level directories and bootstrap files when missing.

    Idempotent: existing ``run_manifest.json`` and JSONL files are never
    overwritten or truncated. Repeated calls only ensure directories exist.
    """
    begin_run_write()
    root = paths.run_root(run_id, base_dir)
    ensure_dir(root)
    ensure_dir(paths.reports_dir(run_id, base_dir))
    ensure_dir(paths.tasks_dir(run_id, base_dir))

    manifest_path = paths.run_manifest_path(run_id, base_dir)
    _init_json_if_missing(
        manifest_path,
        {
            "run_id": run_id,
            "created_at": _utc_now_iso(),
            "status": "created",
        },
        "run_manifest",
    )

    _touch_jsonl(paths.task_events_path(run_id, base_dir))
    _touch_jsonl(paths.approvals_path(run_id, base_dir))
    return root


def _open_runs_root_no_follow(base_dir: Path | None) -> int:
    runs_root = paths.runs_root(base_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(
            str(runs_root),
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
    except OSError as exc:
        raise RunRootReservationError(f"unsafe runs root path: {exc}") from exc


def _fsync_dir_fd(dir_fd: int) -> None:
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise RunRootReservationError(f"directory fsync failed: {exc}") from exc


def _mkdirat_name(dir_fd: int, name: str, mode: int) -> bool:
    try:
        os.mkdir(name, mode, dir_fd=dir_fd)
        return True
    except FileExistsError:
        return False
    except OSError as exc:
        raise RunRootReservationError(f"mkdirat failed for {name!r}: {exc}") from exc


def _openat_dir_no_follow(dir_fd: int, name: str) -> int:
    try:
        return os.open(
            name,
            _O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        raise RunRootReservationError(f"unsafe run root path for {name!r}: {exc}") from exc


def _openat_file_exclusive(dir_fd: int, name: str, mode: int = 0o600) -> int:
    try:
        return os.open(
            name,
            _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC,
            mode,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        raise RunRootReservationError(f"exclusive create failed for {name!r}: {exc}") from exc


def _write_all_fd(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise RunRootReservationError("short write while persisting bootstrap file")
        offset += written


def _write_json_exclusive_at(dir_fd: int, name: str, data: dict[str, Any], schema_name: SchemaName) -> None:
    validate(data, schema_name)
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    file_fd = _openat_file_exclusive(dir_fd, name)
    try:
        _write_all_fd(file_fd, payload)
        os.fsync(file_fd)
    finally:
        os.close(file_fd)


def _touch_jsonl_exclusive_at(dir_fd: int, name: str) -> None:
    file_fd = _openat_file_exclusive(dir_fd, name, mode=0o600)
    os.close(file_fd)


def reserve_run_root_exclusive(run_id: str, base_dir: Path | None = None) -> RunRootReservation:
    """Reserve ``{runs_root}/{run_id}/`` with O_EXCL semantics (no execution marker)."""
    if not validate_id(run_id, "run"):
        raise RunRootReservationError(f"invalid run_id: {run_id!r}", run_id=run_id)
    runs_root_fd = _open_runs_root_no_follow(base_dir)
    created = False
    try:
        try:
            created = _mkdirat_name(runs_root_fd, run_id, 0o700)
        except RunRootReservationError:
            raise
        if not created:
            raise RunRootReservationError(
                f"successor run root already exists: {run_id!r}",
                run_id=run_id,
            )
        run_root_fd = _openat_dir_no_follow(runs_root_fd, run_id)
        try:
            _fsync_dir_fd(run_root_fd)
            _fsync_dir_fd(runs_root_fd)
        except RunRootReservationError:
            os.close(run_root_fd)
            if created:
                try:
                    os.unlink(run_id, dir_fd=runs_root_fd)
                except OSError:
                    pass
            raise
        return RunRootReservation(
            runs_root_fd=runs_root_fd,
            run_root_fd=run_root_fd,
            run_id=run_id,
            created=True,
        )
    except Exception:
        os.close(runs_root_fd)
        raise


def release_run_root_reservation(reservation: RunRootReservation) -> None:
    os.close(reservation.run_root_fd)
    os.close(reservation.runs_root_fd)


def bootstrap_reserved_run_workspace(
    run_id: str,
    base_dir: Path | None = None,
    *,
    reservation: RunRootReservation,
) -> Path:
    """Populate a pre-reserved empty run root without Task 23 execution markers."""
    if reservation.run_id != run_id:
        raise RunRootReservationError(
            f"reservation run_id mismatch: {reservation.run_id!r} != {run_id!r}",
            run_id=run_id,
        )
    run_root_fd = reservation.run_root_fd
    runs_root_fd = reservation.runs_root_fd
    for subdir in ("reports", "tasks"):
        _mkdirat_name(run_root_fd, subdir, 0o700)
    manifest = {
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "status": "created",
    }
    _write_json_exclusive_at(run_root_fd, "run_manifest.json", manifest, "run_manifest")
    _touch_jsonl_exclusive_at(run_root_fd, "task_events.jsonl")
    _touch_jsonl_exclusive_at(run_root_fd, "approvals.jsonl")
    _fsync_dir_fd(run_root_fd)
    _fsync_dir_fd(runs_root_fd)
    return paths.run_root(run_id, base_dir)


@run_mutation_boundary
def create_task_workspace(
    run_id: str,
    task_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Create task-level directories and bootstrap ``task_status.json`` when missing.

    Idempotent: does not overwrite an existing ``task_status.json``.

    Reserved path (not created here):
    - ``task_card.yaml`` — created by Task Card schema / task lifecycle (Task 2+).
    """
    manifest_path = paths.run_manifest_path(run_id, base_dir)
    if not manifest_path.exists():
        create_run_workspace(run_id, base_dir)

    begin_run_write()
    task_root = paths.task_dir(run_id, task_id, base_dir)
    ensure_dir(task_root)
    ensure_dir(paths.attempts_dir(run_id, task_id, base_dir))

    _init_json_if_missing(
        paths.task_status_path(run_id, task_id, base_dir),
        {
            "task_id": task_id,
            "run_id": run_id,
            "status": "created",
            "attempts": [],
        },
        "task_status",
    )
    return task_root


@run_mutation_boundary
def create_attempt_workspace(
    run_id: str,
    task_id: str,
    attempt_id: str,
    base_dir: Path | None = None,
) -> Path:
    """Create attempt-level directories and bootstrap status/manifest when missing.

    Idempotent: does not overwrite existing ``attempt_status.json``,
    ``artifact_manifest.json``, or ``tool_calls.jsonl``.

    Reserved paths (not created here):
    - ``output/result.json`` — written by attempt execution, not bootstrap.
    - ``task_card.yaml`` — see :func:`create_task_workspace`.
    """
    task_status_path = paths.task_status_path(run_id, task_id, base_dir)
    if not task_status_path.exists():
        create_task_workspace(run_id, task_id, base_dir)

    begin_run_write()
    attempt_root = paths.attempt_dir(run_id, task_id, attempt_id, base_dir)
    for maker in (
        paths.input_dir,
        paths.working_dir,
        paths.output_dir,
        paths.artifacts_dir,
        paths.logs_dir,
        paths.verification_dir,
        paths.heal_dir,
    ):
        ensure_dir(maker(run_id, task_id, attempt_id, base_dir))

    _init_json_if_missing(
        paths.attempt_status_path(run_id, task_id, attempt_id, base_dir),
        {
            "attempt_id": attempt_id,
            "task_id": task_id,
            "run_id": run_id,
            "status": "created",
        },
        "attempt_status",
    )

    _init_json_if_missing(
        paths.artifact_manifest_path(run_id, task_id, attempt_id, base_dir),
        {
            "schema_version": "1",
            "run_id": run_id,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "artifacts": [],
        },
        "artifact_manifest",
    )

    _touch_jsonl(paths.tool_calls_path(run_id, task_id, attempt_id, base_dir))
    return attempt_root
