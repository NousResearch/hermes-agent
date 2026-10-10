"""Portable blocking cross-process file lock used by security ledgers."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by import simulation
    _fcntl = None

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None


def secure_file_descriptor_permissions(fd: int) -> None:
    """Tighten a descriptor when the platform exposes ``os.fchmod``.

    Windows Python 3.11/3.12 lacks ``os.fchmod``; secure creation flags and
    the owning user's ACL remain the platform boundary there.
    """

    fchmod = getattr(os, "fchmod", None)
    if fchmod is not None:
        fchmod(fd, 0o600)


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold one OS-backed lock file across all Hermes processes."""

    try:
        if path.is_symlink():
            raise OSError("cross-process lock path must not be a symlink")
    except OSError:
        raise
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        secure_file_descriptor_permissions(fd)
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_EX)
        elif _msvcrt is not None:  # pragma: no cover - Windows
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_LOCK, 1)
        else:
            raise OSError("cross-process file locking is unavailable")
        yield
    finally:
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            elif _msvcrt is not None:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


__all__ = ["exclusive_file_lock", "secure_file_descriptor_permissions"]
