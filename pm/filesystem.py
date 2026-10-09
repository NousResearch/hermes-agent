"""Stdlib-only filesystem primitives PM needs before it can replace the caller's interpreter.

Boot-time dependency selection (``hermes_cli.runtime_state``) imports these, so nothing here
may import a dependency or another PM module.
"""
from __future__ import annotations

import errno
import hashlib
import ntpath
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time

_LOCK_POLL_SECONDS = 0.05


def is_junction(path: Path) -> bool:
    """Keep junctions opaque even before Python 3.12's Path.is_junction exists."""
    return os.name == "nt" and path.lstat().st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT


_VERBATIM = "\\\\?\\"
_VERBATIM_UNC = "\\\\?\\UNC\\"


def _long_form(text: str) -> str:
    if text.startswith(_VERBATIM):
        return text
    # Windows stops normalising a verbatim path, so it must already be absolute,
    # backslashed and free of `..` before the prefix goes on.
    text = ntpath.abspath(text)
    if text.startswith("\\\\"):
        return _VERBATIM_UNC + text[2:]
    return _VERBATIM + text


def long_root(path: Path) -> Path:
    """Spell a PM-owned root so Windows file calls beneath it are not cut at
    MAX_PATH when LongPathsEnabled is 0 (the default). Every path joined onto the
    result inherits the spelling; relative paths and digests are unchanged."""
    return Path(_long_form(os.fspath(path))) if os.name == "nt" else path


def native(path: str | os.PathLike[str]) -> str:
    """The ordinary spelling, for text that leaves PM's own file calls: argv, cwd,
    environment values, hashed identities, records and messages."""
    text = os.fspath(path)
    if text.startswith(_VERBATIM_UNC):
        return "\\\\" + text[len(_VERBATIM_UNC):]
    return text.removeprefix(_VERBATIM)


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


#: A handle still open inside the tree. Windows reports antivirus and indexer scans of freshly
#: written files this way (WinError 5 and 32 both surface as EACCES) and they clear in moments;
#: a missing source or an occupied destination never does, so those raise at once.
_HELD_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EBUSY})
_HELD_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)


def retry_held(operation):
    """Run a rename-like mutation, riding out a transient Windows hold (~3s) before raising."""
    for delay in (*_HELD_RETRY_DELAYS, None):
        try:
            return operation()
        except OSError as error:
            if delay is None or error.errno not in _HELD_ERRNOS:
                raise
            time.sleep(delay)


def remove_tree(path: Path) -> None:
    """``shutil.rmtree`` that clears a Windows read-only attribute and retries that one unlink.

    Windows refuses to unlink read-only files (PortableGit ships ``etc/hosts`` read-only), and no
    retry clears that bit. Every other failure, and every POSIX failure, raises as plain rmtree
    would: a POSIX permission error is about the parent directory, and chmodding the path there
    would follow a symlink out of the tree. Lives here, not in ``hermes_cli``: PM installs run
    before the rest of the tree exists (the Docker toolchain stage copies only ``pm/``).
    """
    def _retry_writable(func, failed, error):
        error = error[1] if isinstance(error, tuple) else error
        if os.name != "nt" or func not in (os.unlink, os.rmdir) or not isinstance(error, PermissionError):
            raise error
        # Windows >= 3.13 chmod follows links; the attribute to clear is the link's own.
        os.chmod(failed, stat.S_IWRITE, **({"follow_symlinks": False} if os.chmod in os.supports_follow_symlinks else {}))
        func(failed)

    handler = {"onexc": _retry_writable} if sys.version_info >= (3, 12) else {"onerror": _retry_writable}
    shutil.rmtree(path, **handler)
