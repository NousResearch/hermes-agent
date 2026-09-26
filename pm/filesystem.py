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

_LOCK_POLL_SECONDS = 0.05


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


# link() failures meaning "this filesystem cannot hard-link", never "the target exists": EPERM
# (FAT/exFAT), EACCES (SELinux on Android app data), ENOTSUP/EOPNOTSUPP/ENOSYS (SMB, FUSE). Windows
# reports FAT and unsupported shares as ERROR_INVALID_FUNCTION (1) / ERROR_NOT_SUPPORTED (50).
_HARD_LINK_REFUSED_ERRNOS = frozenset({errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})
_HARD_LINK_REFUSED_WINERRORS = frozenset({1, 50})


def hard_link_refused(exc: OSError) -> bool:
    return (exc.errno in _HARD_LINK_REFUSED_ERRNOS
            or getattr(exc, "winerror", None) in _HARD_LINK_REFUSED_WINERRORS)


def durable_write_bytes(path: Path, data: bytes) -> None:
    """Replace ``path`` atomically and fsync file and directory so a crash keeps old or new bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
