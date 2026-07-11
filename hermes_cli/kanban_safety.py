"""Persistent safety fencing for SQLite-backed Kanban boards.

This module owns the fail-closed filesystem primitives used by normal Kanban
connections/writes and by offline maintenance.  It deliberately has no
knowledge of gateway lifecycle or repair policy.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_LOCK_TIMEOUT_SECONDS = 1.0
_LOCK_POLL_SECONDS = 0.01


@dataclass
class _MaintenanceLease:
    lock_key: str
    exclusive: bool
    owner_thread: int
    owner_task: int | None
    active: bool = True


_HELD_MAINTENANCE_LOCKS: ContextVar[tuple[_MaintenanceLease, ...]] = ContextVar(
    "kanban_held_maintenance_locks", default=()
)


class KanbanSafetyError(RuntimeError):
    """Base class for fail-closed Kanban safety failures."""


class BoardQuarantinedError(KanbanSafetyError, sqlite3.DatabaseError):
    """The board has an active persistent quarantine marker."""

    def __init__(self, db_path: Path, marker: dict[str, Any]):
        self.db_path = Path(db_path)
        self.marker = marker
        self.reason = str(marker.get("reason") or "unknown reason")
        super().__init__(
            f"Kanban board at {self.db_path} is quarantined: {self.reason}"
        )


class MaintenanceLockError(KanbanSafetyError):
    """A maintenance fence could not be opened or acquired safely."""


class QuarantinePersistenceError(KanbanSafetyError):
    """A discovered corruption could not be durably quarantined."""


class GenerationFencedError(KanbanSafetyError):
    """A caller's expected generation is stale."""


@dataclass(frozen=True)
class Generations:
    service_generation: int = 0
    board_generation: int = 0


def quarantine_marker_path(db_path: Path) -> Path:
    path = Path(db_path)
    return path.with_name(path.name + ".quarantine.json")


def maintenance_lock_path(db_path: Path) -> Path:
    path = Path(db_path)
    return path.with_name(path.name + ".maintenance.lock")


def generations_path(db_path: Path) -> Path:
    path = Path(db_path)
    return path.with_name(path.name + ".generations.json")


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {
            "exists": False,
            "size": None,
            "mtime_ns": None,
            "sha256": None,
        }
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def db_fingerprint(db_path: Path) -> dict[str, Any]:
    """Fingerprint the complete SQLite file set, not only the main DB."""
    path = Path(db_path).expanduser().resolve()
    return {
        "path": str(path),
        "db": _file_fingerprint(path),
        "wal": _file_fingerprint(path.with_name(path.name + "-wal")),
        "shm": _file_fingerprint(path.with_name(path.name + "-shm")),
    }


def _fsync_parent_directory(directory: Path) -> None:
    """Durably flush a directory where the platform exposes that primitive."""
    if sys.platform == "win32":
        # Windows cannot open a directory with os.open() for fsync. os.replace()
        # remains atomic, and the file contents are flushed before replacement.
        return
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path.parent)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise KanbanSafetyError(f"cannot safely read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise KanbanSafetyError(f"invalid safety metadata at {path}")
    return value


def _parse_generation(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise KanbanSafetyError(f"generation metadata is malformed: {field}")
    return value


def read_generations(db_path: Path) -> Generations:
    payload = _read_json(generations_path(db_path))
    if payload is None:
        return Generations()
    try:
        service_value = payload["service_generation"]
        board_value = payload["board_generation"]
    except KeyError as exc:
        raise KanbanSafetyError("generation metadata is malformed: missing field") from exc
    service = _parse_generation(service_value, field="service_generation")
    board = _parse_generation(board_value, field="board_generation")
    return Generations(service, board)


def validate_generations(
    db_path: Path,
    *,
    expected_service_generation: int | None = None,
    expected_board_generation: int | None = None,
) -> Generations:
    current = read_generations(db_path)
    if expected_service_generation is not None:
        expected_service_generation = _parse_generation(
            expected_service_generation, field="expected_service_generation"
        )
    if expected_board_generation is not None:
        expected_board_generation = _parse_generation(
            expected_board_generation, field="expected_board_generation"
        )
    if (
        expected_service_generation is not None
        and expected_service_generation != current.service_generation
    ):
        raise GenerationFencedError(
            "stale service_generation: expected "
            f"{expected_service_generation}, current {current.service_generation}"
        )
    if (
        expected_board_generation is not None
        and expected_board_generation != current.board_generation
    ):
        raise GenerationFencedError(
            "stale board_generation: expected "
            f"{expected_board_generation}, current {current.board_generation}"
        )
    return current


def _open_lock_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a+b")


def _try_lock(handle, *, exclusive: bool) -> None:
    """Platform adapter for one non-blocking lock attempt."""
    if sys.platform == "win32":  # pragma: no cover - exercised via adapter tests
        import msvcrt

        handle.seek(0)
        # Windows' byte-range API has no shared mode. Exclusive for both modes
        # is conservative and preserves fail-closed semantics.
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    if sys.platform == "win32":  # pragma: no cover - exercised via adapter tests
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _current_task_identity() -> int | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return None if task is None else id(task)


@contextlib.contextmanager
def maintenance_lock(
    db_path: Path,
    *,
    exclusive: bool,
    timeout_seconds: float | None = None,
) -> Iterator[None]:
    """Acquire a board's shared normal-operation or exclusive maintenance fence.

    Re-entry for the same board is allowed within one context when it does not
    upgrade a shared lock to exclusive. OS locks still arbitrate across threads
    and processes because ContextVar state is not inherited by new threads.
    """
    lock_path = maintenance_lock_path(db_path)
    lock_key = str(lock_path.expanduser().resolve())
    held_locks = _HELD_MAINTENANCE_LOCKS.get()
    owner_thread = threading.get_ident()
    owner_task = _current_task_identity()
    for held_lease in reversed(held_locks):
        if held_lease.lock_key != lock_key or not held_lease.active:
            continue
        if (
            held_lease.owner_thread != owner_thread
            or held_lease.owner_task != owner_task
        ):
            continue
        if exclusive and not held_lease.exclusive:
            raise MaintenanceLockError(
                f"cannot upgrade shared maintenance lock to exclusive: {lock_path}"
            )
        yield
        return

    try:
        handle = _open_lock_file(lock_path)
    except OSError as exc:
        raise MaintenanceLockError(
            f"failed to open maintenance lock {lock_path}: {exc}"
        ) from exc
    acquired = False
    context_token = None
    lease: _MaintenanceLease | None = None
    body_error: BaseException | None = None
    timeout = (
        DEFAULT_LOCK_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    )
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        while True:
            try:
                _try_lock(handle, exclusive=exclusive)
                acquired = True
                break
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    mode = "exclusive" if exclusive else "shared"
                    raise MaintenanceLockError(
                        f"timed out acquiring {mode} maintenance lock {lock_path}"
                    )
                time.sleep(_LOCK_POLL_SECONDS)
            except OSError as exc:
                raise MaintenanceLockError(
                    f"failed to acquire maintenance lock {lock_path}: {exc}"
                ) from exc
        lease = _MaintenanceLease(
            lock_key=lock_key,
            exclusive=exclusive,
            owner_thread=owner_thread,
            owner_task=owner_task,
        )
        context_token = _HELD_MAINTENANCE_LOCKS.set(held_locks + (lease,))
        yield
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if lease is not None:
            lease.active = False
        if context_token is not None:
            _HELD_MAINTENANCE_LOCKS.reset(context_token)
        release_failures: list[OSError] = []
        if acquired:
            try:
                _unlock(handle)
            except OSError as exc:
                release_failures.append(exc)
        try:
            handle.close()
        except OSError as exc:
            release_failures.append(exc)
        if release_failures:
            detail = "; ".join(str(exc) for exc in release_failures)
            release_error = MaintenanceLockError(
                f"failed to release maintenance lock {lock_path}: {detail}"
            )
            if body_error is not None:
                body_error.add_note(str(release_error))
            else:
                raise release_error from release_failures[0]


def _write_generations(db_path: Path, value: Generations) -> None:
    _atomic_write_json(
        generations_path(db_path),
        {
            "service_generation": value.service_generation,
            "board_generation": value.board_generation,
        },
    )


def _bump_service_generation_locked(db_path: Path) -> int:
    """Bump service generation while caller holds exclusive maintenance."""
    current = read_generations(db_path)
    updated = Generations(current.service_generation + 1, current.board_generation)
    _write_generations(db_path, updated)
    return updated.service_generation


def _bump_board_generation_locked(db_path: Path) -> int:
    """Bump board generation while caller holds exclusive maintenance."""
    current = read_generations(db_path)
    updated = Generations(current.service_generation, current.board_generation + 1)
    _write_generations(db_path, updated)
    return updated.board_generation


def bump_service_generation(db_path: Path) -> int:
    with maintenance_lock(db_path, exclusive=True):
        return _bump_service_generation_locked(db_path)


def bump_board_generation(db_path: Path) -> int:
    with maintenance_lock(db_path, exclusive=True):
        return _bump_board_generation_locked(db_path)


def quarantine_board(db_path: Path, *, reason: str, source: str) -> Path:
    """Atomically persist quarantine from one exclusive generation/file snapshot."""
    path = Path(db_path).expanduser().resolve()
    marker_path = quarantine_marker_path(path)
    with maintenance_lock(path, exclusive=True):
        generation = read_generations(path).board_generation
        payload = {
            "reason": str(reason),
            "source": str(source),
            "timestamp": time.time(),
            "db_fingerprint": db_fingerprint(path),
            "board_generation": generation,
        }
        _atomic_write_json(marker_path, payload)
        if _read_json(marker_path) != payload:
            raise KanbanSafetyError(
                f"quarantine marker read-back verification failed at {marker_path}"
            )
    return marker_path


def active_quarantine(db_path: Path) -> dict[str, Any] | None:
    marker = _read_json(quarantine_marker_path(db_path))
    if marker is None:
        return None
    try:
        marker_generation = _parse_generation(
            marker["board_generation"], field="board_generation"
        )
    except (KeyError, KanbanSafetyError) as exc:
        raise KanbanSafetyError("quarantine marker is malformed") from exc
    if marker_generation != read_generations(db_path).board_generation:
        return None
    return marker


def assert_board_not_quarantined(db_path: Path) -> None:
    marker = active_quarantine(db_path)
    if marker is not None:
        raise BoardQuarantinedError(Path(db_path), marker)


_REQUIRED_TABLE_SHAPES: dict[str, dict[str, tuple[str, int, int]]] = {
    "tasks": {
        "id": ("TEXT", 0, 1),
        "title": ("TEXT", 1, 0),
        "body": ("TEXT", 0, 0),
        "assignee": ("TEXT", 0, 0),
        "status": ("TEXT", 1, 0),
        "priority": ("INTEGER", 0, 0),
        "created_by": ("TEXT", 0, 0),
        "created_at": ("INTEGER", 1, 0),
        "started_at": ("INTEGER", 0, 0),
        "completed_at": ("INTEGER", 0, 0),
        "workspace_kind": ("TEXT", 1, 0),
        "workspace_path": ("TEXT", 0, 0),
        "branch_name": ("TEXT", 0, 0),
        "project_id": ("TEXT", 0, 0),
        "claim_lock": ("TEXT", 0, 0),
        "claim_expires": ("INTEGER", 0, 0),
        "tenant": ("TEXT", 0, 0),
        "result": ("TEXT", 0, 0),
        "idempotency_key": ("TEXT", 0, 0),
        "consecutive_failures": ("INTEGER", 1, 0),
        "worker_pid": ("INTEGER", 0, 0),
        "last_failure_error": ("TEXT", 0, 0),
        "max_runtime_seconds": ("INTEGER", 0, 0),
        "last_heartbeat_at": ("INTEGER", 0, 0),
        "current_run_id": ("INTEGER", 0, 0),
        "workflow_template_id": ("TEXT", 0, 0),
        "current_step_key": ("TEXT", 0, 0),
        "skills": ("TEXT", 0, 0),
        "model_override": ("TEXT", 0, 0),
        "max_retries": ("INTEGER", 0, 0),
        "goal_mode": ("INTEGER", 1, 0),
        "goal_max_turns": ("INTEGER", 0, 0),
        "session_id": ("TEXT", 0, 0),
        "block_kind": ("TEXT", 0, 0),
        "block_recurrences": ("INTEGER", 1, 0),
    },
    "task_links": {
        "parent_id": ("TEXT", 1, 1),
        "child_id": ("TEXT", 1, 2),
    },
    "task_comments": {
        "id": ("INTEGER", 0, 1),
        "task_id": ("TEXT", 1, 0),
        "author": ("TEXT", 1, 0),
        "body": ("TEXT", 1, 0),
        "created_at": ("INTEGER", 1, 0),
    },
    "task_events": {
        "id": ("INTEGER", 0, 1),
        "task_id": ("TEXT", 1, 0),
        "run_id": ("INTEGER", 0, 0),
        "kind": ("TEXT", 1, 0),
        "payload": ("TEXT", 0, 0),
        "created_at": ("INTEGER", 1, 0),
    },
    "task_runs": {
        "id": ("INTEGER", 0, 1),
        "task_id": ("TEXT", 1, 0),
        "profile": ("TEXT", 0, 0),
        "step_key": ("TEXT", 0, 0),
        "status": ("TEXT", 1, 0),
        "claim_lock": ("TEXT", 0, 0),
        "claim_expires": ("INTEGER", 0, 0),
        "worker_pid": ("INTEGER", 0, 0),
        "max_runtime_seconds": ("INTEGER", 0, 0),
        "last_heartbeat_at": ("INTEGER", 0, 0),
        "started_at": ("INTEGER", 1, 0),
        "ended_at": ("INTEGER", 0, 0),
        "outcome": ("TEXT", 0, 0),
        "summary": ("TEXT", 0, 0),
        "metadata": ("TEXT", 0, 0),
        "error": ("TEXT", 0, 0),
    },
    "task_attachments": {
        "id": ("INTEGER", 0, 1),
        "task_id": ("TEXT", 1, 0),
        "filename": ("TEXT", 1, 0),
        "stored_path": ("TEXT", 1, 0),
        "content_type": ("TEXT", 0, 0),
        "size": ("INTEGER", 1, 0),
        "uploaded_by": ("TEXT", 0, 0),
        "created_at": ("INTEGER", 1, 0),
    },
    "kanban_notify_subs": {
        "task_id": ("TEXT", 1, 1),
        "platform": ("TEXT", 1, 2),
        "chat_id": ("TEXT", 1, 3),
        "thread_id": ("TEXT", 1, 4),
        "user_id": ("TEXT", 0, 0),
        "notifier_profile": ("TEXT", 0, 0),
        "created_at": ("INTEGER", 1, 0),
        "last_event_id": ("INTEGER", 1, 0),
    },
    "kanban_writer_requests": {
        "request_id": ("TEXT", 0, 1),
        "board": ("TEXT", 1, 0),
        "actor_profile": ("TEXT", 1, 0),
        "source": ("TEXT", 1, 0),
        "writer_pid": ("INTEGER", 1, 0),
        "operation": ("TEXT", 1, 0),
        "payload_digest": ("TEXT", 1, 0),
        "response": ("TEXT", 1, 0),
        "created_at": ("INTEGER", 1, 0),
    },
}

_REQUIRED_COLUMN_DEFAULTS: dict[tuple[str, str], str] = {
    ("tasks", "priority"): "0",
    ("tasks", "workspace_kind"): "'scratch'",
    ("tasks", "consecutive_failures"): "0",
    ("tasks", "goal_mode"): "0",
    ("tasks", "block_recurrences"): "0",
    ("task_attachments", "size"): "0",
    ("kanban_notify_subs", "thread_id"): "''",
    ("kanban_notify_subs", "last_event_id"): "0",
}

_REQUIRED_INDEX_SHAPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "idx_tasks_assignee_status": ("tasks", ("assignee", "status")),
    "idx_tasks_status": ("tasks", ("status",)),
    "idx_tasks_tenant": ("tasks", ("tenant",)),
    "idx_tasks_idempotency": ("tasks", ("idempotency_key",)),
    "idx_tasks_session_id": ("tasks", ("session_id",)),
    "idx_links_child": ("task_links", ("child_id",)),
    "idx_links_parent": ("task_links", ("parent_id",)),
    "idx_comments_task": ("task_comments", ("task_id", "created_at")),
    "idx_events_task": ("task_events", ("task_id", "created_at")),
    "idx_events_run": ("task_events", ("run_id", "id")),
    "idx_runs_task": ("task_runs", ("task_id", "started_at")),
    "idx_runs_status": ("task_runs", ("status",)),
    "idx_attachments_task": ("task_attachments", ("task_id", "created_at")),
    "idx_notify_task": ("kanban_notify_subs", ("task_id",)),
}


def _validate_required_schema(conn: sqlite3.Connection) -> None:
    problems: list[str] = []
    for table, expected_columns in _REQUIRED_TABLE_SHAPES.items():
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        actual_columns = {
            str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in rows
        }
        actual_defaults = {str(row[1]): row[4] for row in rows}
        for column, expected in expected_columns.items():
            actual = actual_columns.get(column)
            if actual != expected:
                problems.append(
                    f"{table}.{column}: expected={expected!r}, actual={actual!r}"
                )
        for (default_table, column), expected_default in _REQUIRED_COLUMN_DEFAULTS.items():
            if default_table != table:
                continue
            actual_default = actual_defaults.get(column)
            if actual_default != expected_default:
                problems.append(
                    f"{table}.{column} default: expected={expected_default!r}, "
                    f"actual={actual_default!r}"
                )

    for index, (expected_table, expected_columns) in _REQUIRED_INDEX_SHAPES.items():
        row = conn.execute(
            "SELECT tbl_name FROM sqlite_schema WHERE type='index' AND name=?",
            (index,),
        ).fetchone()
        actual_table = str(row[0]) if row is not None else None
        actual_columns = tuple(
            str(info[2])
            for info in conn.execute(f'PRAGMA index_info("{index}")').fetchall()
        )
        index_list = {
            str(info[1]): (int(info[2]), int(info[4]))
            for info in conn.execute(f'PRAGMA index_list("{expected_table}")').fetchall()
        }
        unique_partial = index_list.get(index)
        if (
            actual_table != expected_table
            or actual_columns != expected_columns
            or unique_partial != (0, 0)
        ):
            problems.append(
                f"{index}: expected=({expected_table!r}, {expected_columns!r}, "
                f"unique=0, partial=0), actual=({actual_table!r}, "
                f"{actual_columns!r}, flags={unique_partial!r})"
            )

    if problems:
        raise KanbanSafetyError(
            "database canonical schema validation failed: " + "; ".join(problems)
        )


def _verify_database_health(db_path: Path) -> None:
    """Run the final, non-bypassable health gate used before marker removal."""
    path = Path(db_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise KanbanSafetyError("database health verification failed: DB is missing or empty")
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
        conn.execute("PRAGMA query_only=ON")
        integrity_rows = conn.execute("PRAGMA integrity_check").fetchall()
        if integrity_rows != [("ok",)]:
            raise KanbanSafetyError(
                f"database health verification failed: integrity_check={integrity_rows!r}"
            )
        _validate_required_schema(conn)
        notify_rows = conn.execute(
            "PRAGMA integrity_check('kanban_notify_subs')"
        ).fetchall()
        if notify_rows != [("ok",)]:
            raise KanbanSafetyError(
                "database health verification failed: "
                f"kanban_notify_subs integrity_check={notify_rows!r}"
            )
        foreign_key_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            raise KanbanSafetyError(
                "database health verification failed: "
                f"foreign_key_check={foreign_key_rows!r}"
            )
    except KanbanSafetyError:
        raise
    except (OSError, sqlite3.DatabaseError) as exc:
        raise KanbanSafetyError(f"database health verification failed: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def clear_quarantine(
    db_path: Path, *, expected_fingerprint: dict[str, Any]
) -> None:
    """Clear quarantine only inside an exclusive fence after full DB health checks."""
    path = Path(db_path).expanduser().resolve()
    marker_path = quarantine_marker_path(path)
    with maintenance_lock(path, exclusive=True):
        marker = _read_json(marker_path)
        if marker is None:
            raise KanbanSafetyError(f"no quarantine marker exists at {marker_path}")
        try:
            marker_generation = _parse_generation(
                marker["board_generation"], field="board_generation"
            )
        except (KeyError, KanbanSafetyError) as exc:
            raise KanbanSafetyError("quarantine marker is malformed") from exc
        current_generation = read_generations(path).board_generation
        if marker_generation != current_generation:
            raise GenerationFencedError("quarantine board_generation is stale")
        current_fingerprint = db_fingerprint(path)
        if current_fingerprint != expected_fingerprint:
            raise GenerationFencedError("quarantine fingerprint verification failed")
        _verify_database_health(path)
        try:
            marker_path.unlink()
            _fsync_parent_directory(marker_path.parent)
        except OSError as exc:
            raise KanbanSafetyError(f"failed to clear quarantine marker: {exc}") from exc
