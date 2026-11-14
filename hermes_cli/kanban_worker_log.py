"""Per-task worker log rotation, split out of ``hermes_cli.kanban_db_dispatch``."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Optional

DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1


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
    from hermes_cli.kanban_db_dispatch import _positive_int
    max_bytes = _positive_int(kanban_cfg.get("worker_log_rotate_bytes"), DEFAULT_LOG_ROTATE_BYTES, minimum=1)
    backup_count = _positive_int(kanban_cfg.get("worker_log_backup_count"), DEFAULT_LOG_BACKUP_COUNT, minimum=0)
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def rotate_worker_log(
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
        from hermes_cli.kanban_db_dispatch import _positive_int
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
        rotated = _rotated_log_path(log_path, 1)
        log_path.rename(rotated)
        # A log written before worker logs were made private would carry its mode into the backup.
        rotated.chmod(rotated.stat().st_mode & 0o7770)
    except OSError:
        pass


def open_worker_log(log_path: Path):
    """Rotate ``log_path``, then open it for binary append as the worker's stdout/stderr."""
    rotate_worker_log(log_path, *worker_log_rotation_config())
    # Worker output is durable and may quote secrets: never world-readable. Only the other bits
    # are stripped (a shared-group home keeps group access), and a log created before this fix
    # is tightened on its next open.
    # A private log owned by another account (shared-group home) is reopened as is: chmod would
    # raise EPERM. One that is still world-readable fails closed, with the descriptor released.
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o660)
    try:
        mode = os.fstat(fd).st_mode
        if mode & 0o007:
            os.chmod(log_path, mode & 0o7770)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "ab")
