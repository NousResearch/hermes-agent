"""Worker logs for the kanban dispatcher: rotation bounds and the private-mode
open that every spawned worker writes through.

Split out of ``hermes_cli.kanban_db_dispatch`` (per-unit FILE_LINES ratchet).
The origin-resident ``worker_logs_dir`` is reached by a late-bound import
inside ``_open_worker_log`` — same cycle-breaking pattern the dispatcher
module uses for ``_kb`` — so ``kanban_db`` stays importable from both sides
and monkeypatching ``kanban_db.worker_logs_dir`` keeps working.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


# Worker log files larger than this at spawn time are rotated.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.
    Defaults: rotate at 2 MiB, keep one backup (``.log.1``); both overridable
    from ``config.yaml``.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    kanban_cfg = kanban_cfg or {}
    max_bytes = _positive_int(kanban_cfg.get("worker_log_rotate_bytes"), DEFAULT_LOG_ROTATE_BYTES, minimum=1)
    backup_count = _positive_int(kanban_cfg.get("worker_log_backup_count"), DEFAULT_LOG_BACKUP_COUNT, minimum=0)
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``: ``<log>`` → ``<log>.1``,
    older generations shift up to ``backup_count``.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(backup_count, DEFAULT_LOG_BACKUP_COUNT, minimum=0)
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        with contextlib.suppress(OSError):
            if oldest.exists():
                oldest.unlink()
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            with contextlib.suppress(OSError):
                src.rename(_rotated_log_path(log_path, generation + 1))
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _open_worker_log(task: Task, board: Optional[str]):
    """Append-mode per-task log (a re-run on unblock appends, never overwrites),
    rotated first. Anchored at the board root (not the shared kanban root) so
    `hermes kanban log` reads its own file and boards sharing task ids don't
    collide.

    A worker log echoes whatever the worker printed — including live credentials
    that pass through its prompt, tool output, or a file it is authoring — so
    the log and its directory are created private (0o600 / 0o700, the same
    posture as the auth and backup state under HERMES_HOME) instead of at the
    process umask. The mode is set at creation time, never chmod'd after
    (the TOCTOU invariant from ``hermes_cli/auth.py``); logs written by older
    releases are tightened on open. Rotated generations inherit the mode via
    rename. POSIX-only bits: on Windows both calls are best-effort no-ops."""
    from hermes_cli import kanban_db as _kb

    log_dir = _kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        with contextlib.suppress(OSError):
            os.chmod(log_dir, 0o700)  # a dir created by an older release
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    if hasattr(os, "fchmod"):
        with contextlib.suppress(OSError):
            os.fchmod(fd, 0o600)  # an existing log predates the 0o600 creation mode
    return os.fdopen(fd, "ab")
