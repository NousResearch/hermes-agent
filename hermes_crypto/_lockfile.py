"""Cross-platform advisory file locking primitives.

Stdlib-only by design — imported very early by modules that themselves must
stay import-safe (notably :mod:`hermes_crypto.audit`, which runs from
encryption-off code paths during startup). Do not add a dependency on
``cryptography``, ``argon2``, or any other heavy import here.

Both backends release the lock automatically when the file descriptor is
closed or the process dies (normal exit, SIGKILL, segfault, power loss):

- POSIX: ``fcntl.flock(LOCK_EX | LOCK_NB)`` — kernel-tracked, per-process.
- Windows: ``msvcrt.locking(LK_NBLCK, 1)`` — locks a one-byte range at the
  current file offset.

That automatic release is the property that makes ``sweep_sessions`` (and
the analogous audit-log rotation re-entry) able to tell a crashed writer
(lock acquirable now) from a live one (lock still held). Don't add a
Python-level "delete the lockfile if it's old" path — that bypasses the
OS's lock check and breaks the crash-detection invariant.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _open_lockfile(lock_path: Path) -> int:
    """Open or create *lock_path* (0o600) and return its fd.

    Writes a single sentinel byte so the Windows ``msvcrt.locking`` call —
    which locks a *byte range* relative to the file pointer — always has a
    byte to lock. POSIX ``flock`` ignores file contents.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # harden the parent dir to 0o700 on POSIX so the
    # lockfile (and any sibling plaintext session JSONL) isn't world-readable.
    # Best-effort — Windows + cross-FS mounts are silent no-ops. Inlined here
    # rather than imported from .fileio to preserve this module's stdlib-only
    # contract (see module docstring).
    try:
        os.chmod(lock_path.parent, 0o700)
    except OSError:
        pass
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        pass
    return fd


def _acquire_exclusive(fd: int) -> None:
    """Take an exclusive non-blocking advisory lock on *fd*. Raises OSError on conflict."""
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_exclusive(fd: int) -> None:
    """Release the advisory lock on *fd*. Best-effort."""
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
