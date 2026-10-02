"""Stdlib-only filesystem primitives PM needs before it can replace the caller's interpreter.

Boot-time dependency selection (``hermes_cli.runtime_state``) imports these, so nothing here
may import a dependency or another PM module.
"""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import time
import random

_LOCK_POLL_SECONDS = 0.05
_CONTENDED_REPLACE_WINERRORS = frozenset({5, 32, 33})
_REPLACE_RETRY_DELAYS_S = (0.02, 0.04, 0.08, 0.1)


def is_junction(path: Path) -> bool:
    """Keep junctions opaque even before Python 3.12's Path.is_junction exists."""
    return os.name == "nt" and path.lstat().st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT


def lock_fd(fd: int, *, wait: bool, timeout: float | None = None) -> bool:
    """Take the byte lock; ``timeout`` bounds the retry loop (None waits forever, 0 tries once)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                return False
            time.sleep(_LOCK_POLL_SECONDS)
    else:
        import fcntl
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                pass
            if not wait or (deadline is not None and time.monotonic() >= deadline):
                return False
            time.sleep(_LOCK_POLL_SECONDS)


def read_bytes_or_none(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def file_digest(path: Path) -> str | None:
    data = read_bytes_or_none(path)
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _is_contended_replace(exc: OSError) -> bool:
    return os.name == "nt" and getattr(exc, "winerror", None) in _CONTENDED_REPLACE_WINERRORS


def _publish_bytes(temporary: str, path: Path, data: bytes) -> None:
    for delay in (0.0, *_REPLACE_RETRY_DELAYS_S):
        if delay:
            time.sleep(delay * (0.5 + random.random()))
        try:
            os.replace(temporary, path)
            return
        except OSError as exc:
            if not _is_contended_replace(exc):
                raise

    # Windows readers commonly deny replacement. Preserve the target inode as a
    # last resort so readers can release it without losing the update.
    fd = os.open(path, os.O_WRONLY | getattr(os, "O_BINARY", 0))
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.ftruncate(fd, len(data))
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` atomically and fsync file and directory so a crash keeps old or new bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_bytes(temporary, path, data)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
