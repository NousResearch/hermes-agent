"""Task 30 — multi-project registry and isolation (v1).

Project identity is the registered ``project_id`` bound to a canonical
runs-root path. Display names and the process cwd are never identity.

Storage lives at ``{HERMES_HOME}/.htr/project_registry/``, above any
per-project runs tree, so the default unregistered ``{HERMES_HOME}/runs``
workflow stays unchanged until a project is explicitly registered.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from htr import io, paths
from htr.ids import generate_project_id, validate_id

SCHEMA_NAME = "htr.project_registry.record.v1"
SCHEMA_VERSION = 1
PROJECT_STATUS_ACTIVE = "active"
PROJECT_STATUS_ARCHIVED = "archived"
ALLOWED_STATUSES = frozenset({PROJECT_STATUS_ACTIVE, PROJECT_STATUS_ARCHIVED})
_DISPLAY_NAME_MAX = 200
_RECORD_FILE_MODE = 0o644
_UNSET = object()

ProjectStatus = Literal["active", "archived"]

_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_WRONLY = os.O_WRONLY
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class ProjectRegistryError(Exception):
    """Base fail-closed registry error with a stable ``error_class``."""

    error_class = "registry_error"

    def __init__(self, message: str, *, error_class: str | None = None) -> None:
        super().__init__(message)
        if error_class is not None:
            self.error_class = error_class


class ProjectNotRegistered(ProjectRegistryError):
    error_class = "not_registered"


class ProjectIdentityConflict(ProjectRegistryError):
    error_class = "identity_conflict"


class ProjectPathConflict(ProjectRegistryError):
    error_class = "path_conflict"


class ProjectRegistryCorrupt(ProjectRegistryError):
    error_class = "registry_corrupt"


class ProjectRegistrySchemaUnsupported(ProjectRegistryError):
    error_class = "schema_unsupported"


class ProjectRegistryFilesystemError(ProjectRegistryError):
    error_class = "filesystem_error"


class ProjectRegistryConcurrencyConflict(ProjectRegistryError):
    error_class = "concurrency_conflict"


class ProjectInvalidInput(ProjectRegistryError):
    error_class = "invalid_input"


class ProjectPathEscape(ProjectRegistryError):
    error_class = "path_escape"


@dataclass(frozen=True)
class ProjectRecord:
    """Authoritative registered project identity + bound runs root."""

    project_id: str
    runs_root: Path
    runs_root_digest: str
    project_identity_digest: str
    path_comparison_key: str
    display_name: str | None
    status: str
    schema_version: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_NAME,
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "runs_root": str(self.runs_root),
            "runs_root_digest": self.runs_root_digest,
            "project_identity_digest": self.project_identity_digest,
            "path_comparison_key": self.path_comparison_key,
            "display_name": self.display_name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


@contextmanager
def _registry_lock(hermes_home: Path | None) -> Iterator[None]:
    """Serialize registry check+write. Callers already holding this lock must use unlocked helpers."""
    lock_path = paths.project_registry_lock_path(hermes_home)
    with io.file_lock(lock_path):
        yield


def _fsync_dir(directory: Path) -> None:
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


def path_comparison_key(resolved: Path) -> str:
    """Stable uniqueness key: resolved path + host case-folding rules."""
    return os.path.normcase(str(resolved))


def canonicalize_runs_root(
    value: str | Path,
    *,
    must_exist: bool = True,
) -> Path:
    """Return a resolved absolute directory path. Relative paths are rejected.

    Identity never uses cwd: a non-absolute input fails closed instead of
    being joined to the current working directory.
    """
    if not isinstance(value, (str, Path)):
        raise ProjectInvalidInput("runs_root must be a path string")
    raw = str(value).strip()
    if not raw:
        raise ProjectInvalidInput("runs_root must be a non-empty path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ProjectInvalidInput(
            "runs_root must be an absolute path; relative paths and cwd are not identity"
        )
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise ProjectInvalidInput(
            f"runs_root does not exist: {candidate}",
            error_class="invalid_input",
        ) from exc
    except OSError as exc:
        raise ProjectRegistryFilesystemError(
            f"cannot resolve runs_root: {type(exc).__name__}"
        ) from exc
    if must_exist and not resolved.is_dir():
        raise ProjectInvalidInput("runs_root must be an existing directory")
    return resolved


def _runs_root_digest(resolved: Path) -> str:
    return _sha256_hex({"normalized_path": resolved.as_posix()})


def _identity_digest(project_id: str, comparison_key: str) -> str:
    return _sha256_hex(
        {
            "project_id": project_id,
            "path_comparison_key": comparison_key,
        }
    )


def _normalize_display_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProjectInvalidInput("display_name must be a string or null")
    stripped = value.strip()
    if not stripped:
        return None
    if len(stripped) > _DISPLAY_NAME_MAX:
        raise ProjectInvalidInput(
            f"display_name exceeds {_DISPLAY_NAME_MAX} characters"
        )
    return stripped


def _validate_status(value: str) -> str:
    if value not in ALLOWED_STATUSES:
        raise ProjectInvalidInput(
            f"status must be one of {sorted(ALLOWED_STATUSES)}"
        )
    return value


def ensure_project_registry(*, hermes_home: Path | None = None) -> Path:
    """Create the empty registry directory tree when missing. Does not register a project."""
    root = paths.project_registry_root(hermes_home)
    io.ensure_dir(paths.project_registry_projects_root(hermes_home))
    _fsync_dir(root)
    return root


def _require_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not project_id.strip():
        raise ProjectInvalidInput("project_id is required")
    value = project_id.strip()
    if not validate_id(value, "project"):
        raise ProjectInvalidInput(f"invalid project_id format: {value!r}")
    return value


def _write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    """O_EXCL create + fsync. Existing file raises FileExistsError (no overwrite)."""
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    flags = _O_CREAT | _O_EXCL | _O_WRONLY | _O_NOFOLLOW | _O_CLOEXEC
    fd: int | None = None
    try:
        fd = os.open(str(path), flags, _RECORD_FILE_MODE)
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    except FileExistsError:
        raise
    except OSError as exc:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            fd = None
            try:
                path.unlink()
            except OSError:
                pass
        raise ProjectRegistryFilesystemError(
            f"failed to write project record: {type(exc).__name__}"
        ) from exc
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            fd = None
        try:
            path.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(fd)
        try:
            if hasattr(os, "chmod"):
                os.chmod(path, _RECORD_FILE_MODE, follow_symlinks=False)
        except (OSError, NotImplementedError, TypeError, ValueError):
            try:
                os.chmod(path, _RECORD_FILE_MODE)
            except OSError:
                pass
    _fsync_dir(path.parent)


def _record_from_payload(payload: Any, *, source: Path) -> ProjectRecord:
    if not isinstance(payload, dict):
        raise ProjectRegistryCorrupt(f"project record is not a JSON object: {source.name}")
    schema = payload.get("schema")
    version = payload.get("schema_version")
    if schema != SCHEMA_NAME or not isinstance(version, int):
        raise ProjectRegistrySchemaUnsupported(
            f"unsupported project registry schema: {schema!r} version={version!r}"
        )
    if version != SCHEMA_VERSION:
        raise ProjectRegistrySchemaUnsupported(
            f"unsupported project registry schema version: {version}"
        )
    project_id = payload.get("project_id")
    runs_root_raw = payload.get("runs_root")
    status = payload.get("status")
    created_at = payload.get("created_at")
    updated_at = payload.get("updated_at")
    if not isinstance(project_id, str) or not validate_id(project_id, "project"):
        raise ProjectRegistryCorrupt("project record has invalid project_id")
    if not isinstance(runs_root_raw, str) or not runs_root_raw:
        raise ProjectRegistryCorrupt("project record has invalid runs_root")
    if status not in ALLOWED_STATUSES:
        raise ProjectRegistryCorrupt("project record has invalid status")
    if not isinstance(created_at, str) or not created_at:
        raise ProjectRegistryCorrupt("project record has invalid created_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise ProjectRegistryCorrupt("project record has invalid updated_at")
    stored_key = payload.get("path_comparison_key")
    stored_root_digest = payload.get("runs_root_digest")
    stored_identity = payload.get("project_identity_digest")
    display_name = payload.get("display_name")
    if display_name is not None and not isinstance(display_name, str):
        raise ProjectRegistryCorrupt("project record has invalid display_name")
    if not isinstance(stored_key, str) or not stored_key:
        raise ProjectRegistryCorrupt("project record has invalid path_comparison_key")
    if not isinstance(stored_root_digest, str) or not stored_root_digest:
        raise ProjectRegistryCorrupt("project record has invalid runs_root_digest")
    if not isinstance(stored_identity, str) or not stored_identity:
        raise ProjectRegistryCorrupt("project record has invalid project_identity_digest")

    runs_root = Path(runs_root_raw)
    if not runs_root.is_absolute():
        raise ProjectRegistryCorrupt("project record runs_root is not absolute")
    expected_key = path_comparison_key(runs_root)
    expected_root_digest = _runs_root_digest(runs_root)
    expected_identity = _identity_digest(project_id, stored_key)
    if stored_key != expected_key:
        raise ProjectRegistryCorrupt("project record path_comparison_key does not match runs_root")
    if stored_root_digest != expected_root_digest:
        raise ProjectRegistryCorrupt("project record runs_root_digest does not match runs_root")
    if stored_identity != expected_identity:
        raise ProjectRegistryCorrupt("project record identity digest does not match identity fields")

    return ProjectRecord(
        project_id=project_id,
        runs_root=runs_root,
        runs_root_digest=stored_root_digest,
        project_identity_digest=stored_identity,
        path_comparison_key=stored_key,
        display_name=display_name,
        status=status,
        schema_version=version,
        created_at=created_at,
        updated_at=updated_at,
    )


def _read_record_file(path: Path) -> ProjectRecord:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProjectRegistryFilesystemError(
            f"cannot read project record: {type(exc).__name__}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectRegistryCorrupt("project record is not valid JSON") from exc
    return _record_from_payload(payload, source=path)


def _iter_record_paths(hermes_home: Path | None) -> list[Path]:
    projects_root = paths.project_registry_projects_root(hermes_home)
    if not projects_root.exists():
        return []
    if not projects_root.is_dir():
        raise ProjectRegistryCorrupt("project registry projects path is not a directory")
    found: list[Path] = []
    try:
        entries = sorted(projects_root.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise ProjectRegistryFilesystemError(
            f"cannot list project registry: {type(exc).__name__}"
        ) from exc
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if not entry.is_dir():
            raise ProjectRegistryCorrupt(
                f"unexpected non-directory entry in project registry: {entry.name}"
            )
        if not validate_id(entry.name, "project"):
            raise ProjectRegistryCorrupt(
                f"unexpected project registry directory name: {entry.name}"
            )
        record_path = entry / paths.PROJECT_REGISTRY_RECORD_NAME
        if not record_path.is_file():
            raise ProjectRegistryCorrupt(
                f"project directory is missing {paths.PROJECT_REGISTRY_RECORD_NAME}: {entry.name}"
            )
        found.append(record_path)
    return found


def _load_all_records(hermes_home: Path | None) -> list[ProjectRecord]:
    return [_read_record_file(path) for path in _iter_record_paths(hermes_home)]


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _conflict_with_existing(
    *,
    project_id: str,
    comparison_key: str,
    resolved: Path,
    existing: list[ProjectRecord],
) -> ProjectRecord | None:
    for record in existing:
        same_id = record.project_id == project_id
        same_path = record.path_comparison_key == comparison_key
        if same_id and same_path:
            return record
        if same_id and not same_path:
            raise ProjectIdentityConflict(
                "project_id is already registered to a different runs_root"
            )
        if same_path and not same_id:
            raise ProjectPathConflict(
                "runs_root is already registered to a different project_id"
            )
        if _paths_overlap(record.runs_root, resolved):
            raise ProjectPathConflict(
                "runs_root overlaps an existing project's runs_root"
            )
    return None


def register_project(
    runs_root: str | Path,
    *,
    project_id: str | None = None,
    display_name: str | None = None,
    hermes_home: Path | None = None,
) -> ProjectRecord:
    """Register a project bound to an existing absolute runs-root directory.

    Re-registering the same ``project_id`` + canonical path is idempotent and
    does not rewrite the record (display_name is not applied on replay).
    """
    resolved = canonicalize_runs_root(runs_root, must_exist=True)
    comparison_key = path_comparison_key(resolved)
    name = _normalize_display_name(display_name)
    requested_id = None if project_id is None else _require_project_id(project_id)

    ensure_project_registry(hermes_home=hermes_home)
    lock_path = paths.project_registry_lock_path(hermes_home)
    with io.file_lock(lock_path):
        existing = _load_all_records(hermes_home)
        chosen_id = requested_id or generate_project_id()
        if requested_id is None:
            while any(item.project_id == chosen_id for item in existing):
                chosen_id = generate_project_id()
        matched = _conflict_with_existing(
            project_id=chosen_id,
            comparison_key=comparison_key,
            resolved=resolved,
            existing=existing,
        )
        if matched is not None:
            return matched

        now = _utc_now_iso()
        record = ProjectRecord(
            project_id=chosen_id,
            runs_root=resolved,
            runs_root_digest=_runs_root_digest(resolved),
            project_identity_digest=_identity_digest(chosen_id, comparison_key),
            path_comparison_key=comparison_key,
            display_name=name,
            status=PROJECT_STATUS_ACTIVE,
            schema_version=SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
        )
        record_dir = paths.project_record_dir(chosen_id, hermes_home)
        record_path = paths.project_record_path(chosen_id, hermes_home)
        try:
            io.ensure_dir(record_dir)
            _fsync_dir(record_dir.parent)
        except OSError as exc:
            raise ProjectRegistryFilesystemError(
                f"cannot create project record directory: {type(exc).__name__}"
            ) from exc
        try:
            _write_json_exclusive(record_path, record.to_dict())
        except FileExistsError as exc:
            try:
                replay = _read_record_file(record_path)
            except ProjectRegistryError:
                raise ProjectRegistryConcurrencyConflict(
                    "concurrent register left an unreadable project record"
                ) from exc
            if (
                replay.project_id == record.project_id
                and replay.path_comparison_key == record.path_comparison_key
            ):
                return replay
            raise ProjectRegistryConcurrencyConflict(
                "concurrent register conflicted on project identity"
            ) from exc
        _fsync_dir(paths.project_registry_root(hermes_home))
        return record


def _get_project_unlocked(pid: str, *, hermes_home: Path | None) -> ProjectRecord:
    record_dir = paths.project_record_dir(pid, hermes_home)
    record_path = paths.project_record_path(pid, hermes_home)
    if not record_dir.exists() and not record_path.exists():
        raise ProjectNotRegistered(f"project is not registered: {pid}")
    if record_dir.exists() and not record_dir.is_dir():
        raise ProjectRegistryCorrupt("project record path is not a directory")
    if not record_path.is_file():
        raise ProjectRegistryCorrupt(
            f"project directory is missing {paths.PROJECT_REGISTRY_RECORD_NAME}"
        )
    record = _read_record_file(record_path)
    if record.project_id != pid:
        raise ProjectRegistryCorrupt("project record project_id does not match directory name")
    return record


def get_project(project_id: str, *, hermes_home: Path | None = None) -> ProjectRecord:
    pid = _require_project_id(project_id)
    if not paths.project_registry_root(hermes_home).exists():
        raise ProjectNotRegistered(f"project is not registered: {pid}")
    with _registry_lock(hermes_home):
        return _get_project_unlocked(pid, hermes_home=hermes_home)


def list_projects(
    *,
    hermes_home: Path | None = None,
    include_archived: bool = False,
) -> list[ProjectRecord]:
    if not paths.project_registry_root(hermes_home).exists():
        return []
    with _registry_lock(hermes_home):
        records = _load_all_records(hermes_home)
    if not include_archived:
        records = [item for item in records if item.status == PROJECT_STATUS_ACTIVE]
    return sorted(records, key=lambda item: item.project_id)


def lookup_project_by_runs_root(
    runs_root: str | Path,
    *,
    hermes_home: Path | None = None,
    must_exist: bool = True,
) -> ProjectRecord | None:
    """Return the registered project for *runs_root*, or None if unregistered."""
    resolved = canonicalize_runs_root(runs_root, must_exist=must_exist)
    key = path_comparison_key(resolved)
    if not paths.project_registry_root(hermes_home).exists():
        return None
    with _registry_lock(hermes_home):
        records = _load_all_records(hermes_home)
    for record in records:
        if record.path_comparison_key == key:
            return record
    return None


def update_project_metadata(
    project_id: str,
    *,
    display_name: Any = _UNSET,
    status: str | None = None,
    hermes_home: Path | None = None,
) -> ProjectRecord:
    """Update non-identity fields only. ``runs_root`` and ``project_id`` cannot change.

    Omit ``display_name`` to leave it unchanged. Pass ``None`` or ``""`` to clear it.
    """
    pid = _require_project_id(project_id)
    if display_name is _UNSET and status is None:
        raise ProjectInvalidInput("update_project_metadata requires display_name or status")
    new_status = _validate_status(status) if status is not None else None

    lock_path = paths.project_registry_lock_path(hermes_home)
    with io.file_lock(lock_path):
        current = _get_project_unlocked(pid, hermes_home=hermes_home)
        next_name = (
            current.display_name
            if display_name is _UNSET
            else _normalize_display_name(display_name)
        )
        next_status = current.status if new_status is None else new_status
        if next_name == current.display_name and next_status == current.status:
            return current
        updated = ProjectRecord(
            project_id=current.project_id,
            runs_root=current.runs_root,
            runs_root_digest=current.runs_root_digest,
            project_identity_digest=current.project_identity_digest,
            path_comparison_key=current.path_comparison_key,
            display_name=next_name,
            status=next_status,
            schema_version=current.schema_version,
            created_at=current.created_at,
            updated_at=_utc_now_iso(),
        )
        record_path = paths.project_record_path(pid, hermes_home)
        try:
            io.atomic_write_json(record_path, updated.to_dict())
        except OSError as exc:
            raise ProjectRegistryFilesystemError(
                f"failed to update project record: {type(exc).__name__}"
            ) from exc
        persisted = _read_record_file(record_path)
        if persisted.project_identity_digest != current.project_identity_digest:
            raise ProjectRegistryCorrupt("update mutated project identity")
        if persisted.path_comparison_key != current.path_comparison_key:
            raise ProjectRegistryCorrupt("update mutated runs_root binding")
        return persisted


def resolve_project_runs_root(
    project_id: str,
    *,
    hermes_home: Path | None = None,
) -> Path:
    record = get_project(project_id, hermes_home=hermes_home)
    if record.status == PROJECT_STATUS_ARCHIVED:
        raise ProjectInvalidInput(
            f"project is archived: {record.project_id}",
            error_class="invalid_input",
        )
    return record.runs_root


def assert_path_in_project(
    project_id: str,
    target: str | Path,
    *,
    hermes_home: Path | None = None,
) -> Path:
    """Fail closed when *target* is outside the project's canonical runs_root."""
    record = get_project(project_id, hermes_home=hermes_home)
    if not isinstance(target, (str, Path)) or not str(target).strip():
        raise ProjectInvalidInput("path is required")
    candidate = Path(str(target).strip())
    if not candidate.is_absolute():
        raise ProjectInvalidInput(
            "path must be absolute; relative paths and cwd are not identity"
        )
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise ProjectRegistryFilesystemError(
            f"cannot resolve path: {type(exc).__name__}"
        ) from exc
    try:
        resolved.relative_to(record.runs_root)
    except ValueError as exc:
        raise ProjectPathEscape(
            "path is outside the registered project runs_root"
        ) from exc
    return resolved


def resolve_invocation_runs_root(
    *,
    project_id: str | None = None,
    runs_root: str | Path | None = None,
    hermes_home: Path | None = None,
) -> Path | None:
    """Resolve CLI/API runs_root without breaking unregistered single-project use.

    * both omitted → ``None`` (callers keep default ``HERMES_HOME/runs``)
    * only ``runs_root`` → returned as given Path (legacy observe/plan semantics)
    * only ``project_id`` → registered canonical runs_root
    * both → registered path must match the provided absolute runs_root
    """
    pid = project_id.strip() if isinstance(project_id, str) and project_id.strip() else None
    raw_root = str(runs_root).strip() if runs_root is not None and str(runs_root).strip() else None
    if pid is None and raw_root is None:
        return None
    if pid is not None and raw_root is None:
        return resolve_project_runs_root(pid, hermes_home=hermes_home)
    if pid is None and raw_root is not None:
        return Path(raw_root)
    assert pid is not None and raw_root is not None
    record = get_project(pid, hermes_home=hermes_home)
    provided = canonicalize_runs_root(raw_root, must_exist=True)
    if path_comparison_key(provided) != record.path_comparison_key:
        raise ProjectIdentityConflict(
            "project_id and runs_root do not refer to the same registered project"
        )
    return record.runs_root


def project_registry_error_payload(exc: BaseException) -> dict[str, Any]:
    """Safe JSON error body for CLI — no filesystem owner/secret leakage."""
    error_class = getattr(exc, "error_class", None)
    if not isinstance(error_class, str) or not error_class:
        error_class = "registry_error"
    return {
        "ok": False,
        "error_class": error_class,
        "message": str(exc),
    }
