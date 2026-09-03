"""Descriptor-pinned SQLite opening and file hardening for Phase 2 stores.

Every durable Phase 2 database (authority ledger, mutation claims) opens
through this module so the no-symlink, no-permissive-create, owner-only-mode
guarantees are enforced at exactly one boundary.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from pathlib import Path

# Serializes writers within this process; SQLite's own locking covers others.
_DB_LOCK = threading.Lock()

# SQLite binds an INTEGER column as a signed 64-bit value. A Python int outside
# this range cannot be bound at all: the INSERT aborts with an untyped
# ``OverflowError`` raised from the driver, before any audit row exists.
_SQLITE_INT_MIN = -(2**63)
_SQLITE_INT_MAX = 2**63 - 1


class _Phase2Connection(sqlite3.Connection):
    """SQLite connection that owns its descriptor-backed database handle."""

    _phase2_fd: int | None = None

    def close(self) -> None:
        fd = self._phase2_fd
        self._phase2_fd = None
        try:
            super().close()
        finally:
            if fd is not None:
                os.close(fd)


def _phase2_descriptor_path(fd: int) -> str:
    for candidate in (f"/proc/self/fd/{fd}", f"/dev/fd/{fd}"):
        if os.path.exists(candidate):
            return candidate
    raise OSError("this POSIX host exposes neither /proc/self/fd nor /dev/fd")


def _check_phase2_db_files(path: Path, *, harden: bool) -> None:
    """Reject substituted SQLite files and optionally enforce mode ``0600``."""

    for suffix in ("", "-wal", "-shm"):
        target = path.parent / (path.name + suffix)
        try:
            opened = os.lstat(target)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(opened.st_mode):
            raise PermissionError(
                f"Phase 2 database path is not a regular file: {target}"
            )
        if harden and os.name == "posix":
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR, follow_symlinks=False)


def _open_phase2_sqlite(
    path: Path, *, readonly: bool = False, timeout: float = 5.0
) -> sqlite3.Connection:
    """Open a Phase 2 database without a permissive-create or symlink window."""

    path = path.expanduser().absolute()
    if not readonly:
        path.parent.mkdir(parents=True, exist_ok=True)
    _check_phase2_db_files(path, harden=not readonly)
    fd: int | None = None
    sqlite_path = path
    if os.name == "posix":
        flags = os.O_RDONLY if readonly else os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise PermissionError(
                    f"Phase 2 database path is not a regular file: {path}"
                )
            if not readonly:
                os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            sqlite_path = Path(_phase2_descriptor_path(fd))
        except BaseException:
            os.close(fd)
            raise
    elif not readonly:
        try:
            created_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(created_fd)
        except FileExistsError:
            pass
    mode = "ro" if readonly else "rw"
    sqlite_uri = f"{sqlite_path.as_uri()}?mode={mode}"
    try:
        conn = sqlite3.connect(
            sqlite_uri,
            uri=True,
            isolation_level=None,
            timeout=timeout,
            factory=_Phase2Connection,
        )
    except BaseException:
        if fd is not None:
            os.close(fd)
        raise
    conn._phase2_fd = fd
    if readonly:
        conn.execute("PRAGMA query_only=ON")
    return conn
