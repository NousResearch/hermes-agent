"""Cross-process coordination primitive for the verification evidence ledger.

Contract (frozen by PR74986 WAL2B + LOCK_ARTIFACT_CONTRACT_DECISION):

  - ``$HERMES_HOME/.locks/verification_evidence.lock`` is a persistent,
    profile-scoped coordination artifact.
  - MUTABLE writers are authorized to create the lock parent dir + lockfile
    and to materialize the single byte that Windows ``msvcrt.locking``
    requires at offset 0.
  - READ-ONLY callers (``verification_status`` strict accessor) MUST NOT
    create the lockfile, MUST NOT create the parent dir, and MUST NOT
    materialize bytes in an existing lockfile.
  - If the reader finds the lockfile missing, it must fail closed through
    the existing RPC envelope (``unknown``) without falling back to
    ``sqlite3.connect(source)`` or any unlocked read.
  - The reader must leave the lockfile size and SHA256 byte-identical
    across the call.

Two platforms are supported:

  POSIX  : ``fcntl.flock`` on the lockfile inode. Lock is automatically
           released when the fd is closed or the process exits.
  Windows: ``msvcrt.locking`` on a single byte at offset 0 of the lockfile.
           The writer's materialize-byte-zero step is what makes the
           Windows byte-range lock valid against an empty file.

The same module-level functions are used by both writers and readers; the
authority distinction is enforced by ``role=`` parameter on each call site.
"""

from __future__ import annotations

import errno
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from hermes_constants import get_hermes_home

Role = Literal["writer", "reader"]

_LOCKFILE_RELATIVE_PATH = Path(".locks") / "verification_evidence.lock"
_DEFAULT_TIMEOUT_SECONDS = 2.0
_DEFAULT_RETRY_INTERVAL_SECONDS = 0.05
# Single byte at offset 0 is the minimum byte-range the Windows msvcrt
# region-lock primitive can occupy. Writers materialize it on first
# acquisition; readers MUST NOT.
_LOCK_BYTE = b"\x00"


def lockfile_path() -> Path:
    """Resolve the profile-scoped lockfile path under the active HERMES_HOME.

    No filesystem side effects: callers may use this to test for existence
    without creating anything.
    """
    return get_hermes_home() / _LOCKFILE_RELATIVE_PATH


def lockfile_exists() -> bool:
    """True iff the writer-created lockfile is present.

    This is the reader's existence check. It MUST be the only test the
    reader performs before failing closed when the lockfile is absent.
    """
    try:
        return lockfile_path().is_file()
    except OSError:
        return False


@dataclass(frozen=True)
class _LockOutcome:
    fd: int | None
    """POSIX fd or None on Windows. Closed on release."""

    file_handle: object | None
    """Windows file object or None on POSIX. Closed on release."""

    platform: Literal["posix", "windows"]


class LockTimeout(RuntimeError):
    """Raised when a coordinated acquisition could not be obtained in time."""


class LockUnavailable(RuntimeError):
    """Raised when the reader cannot proceed because the lockfile is absent.

    The reader MUST convert this into the existing RPC envelope
    (``verification.status -> 'unknown'``) and MUST NOT fall back to any
    direct source SQLite access.
    """


def _writer_ensure_lockfile(path: Path) -> None:
    """Create the parent dir + lockfile with the minimum Windows byte.

    Called only by writers. Readers MUST NOT invoke this.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Open in r+b after creating so subsequent msvcrt.locking on the
        # fd sees at least one byte at offset 0. The byte's content is
        # irrelevant; only its existence matters for byte-range locks.
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.write(fd, _LOCK_BYTE)
        finally:
            os.close(fd)
    elif path.stat().st_size == 0:
        # The lockfile exists (carried over from a prior writer or a
        # previous acquisition) but is empty. Top it up with the minimum
        # byte so a Windows reader can still byte-range-lock it.
        fd = os.open(str(path), os.O_RDWR)
        try:
            os.write(fd, _LOCK_BYTE)
        finally:
            os.close(fd)


def _acquire_posix(
    path: Path,
    *,
    timeout: float,
    retry_interval: float,
) -> _LockOutcome:
    deadline = time.monotonic() + timeout
    fd = os.open(str(path), os.O_RDWR)
    while True:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _LockOutcome(fd=fd, file_handle=None, platform="posix")
        except (BlockingIOError, OSError) as exc:
            # EWOULDBLOCK means another holder exists; retry until deadline.
            if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                os.close(fd)
                raise
            if time.monotonic() >= deadline:
                os.close(fd)
                raise LockTimeout(
                    f"could not acquire {path} within {timeout:g}s"
                ) from exc
            time.sleep(retry_interval)


def _release_posix(outcome: _LockOutcome) -> None:
    if outcome.fd is None:
        return
    try:
        import fcntl

        fcntl.flock(outcome.fd, fcntl.LOCK_UN)
    except OSError:
        # Lock release failures are best-effort; the descriptor close below
        # will drop the lock anyway on POSIX (kernel releases on close).
        pass
    os.close(outcome.fd)


def _acquire_windows(
    path: Path,
    *,
    timeout: float,
    retry_interval: float,
) -> _LockOutcome:
    import msvcrt  # type: ignore[import-not-found]

    deadline = time.monotonic() + timeout
    # Open in r+b WITHOUT truncate. If the file is empty (writer never
    # materialized), raise LockUnavailable so the reader fails closed.
    handle = open(path, "r+b")
    # Region locking in msvcrt requires a non-empty file at offset 0.
    # If the writer has not yet materialized the byte, the reader must
    # NOT do it; this enforces the no-creation contract for the reader.
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.close()
        raise LockUnavailable(
            f"reader cannot lock empty {path}; writer has not initialized"
        )
    while True:
        try:
            handle.seek(0)
            # LK_NBLCK (mode=2) is the nonblocking acquire.
            # Under genuine byte-range contention on Windows, msvcrt.locking
            # surfaces OSError(errno.EACCES) (PermissionError [Errno 13]),
            # NOT errno.EAGAIN as the candidate's prior comment stated.
            # msvcrt.locking wraps _locking -> LockFileEx, which raises
            # ERROR_LOCK_VIOLATION when the requested nonblocking byte-range
            # lock conflicts with a lock held by another process; CPython's
            # msvcrt module maps that to OSError(errno=EACCES), classified as
            # PermissionError. We therefore re-try on EACCES (and EPERM for
            # hybrid share-mode / opened-via-different-API edge cases) up to
            # the deadline, then raise LockTimeout as the cross-platform
            # contention signal. EAGAIN/EWOULDBLOCK remain accepted for
            # theoretical future Python compatibility paths.
            # All other OSError subclasses (and any non-OSError exception)
            # propagate unchanged so genuine filesystem failures remain
            # surfaced.
            msvcrt.locking(handle.fileno(), 2, 1)
            return _LockOutcome(fd=None, file_handle=handle, platform="windows")
        except OSError as exc:
            if exc.errno not in (
                errno.EWOULDBLOCK,
                errno.EAGAIN,
                errno.EACCES,
                errno.EPERM,
            ):
                handle.close()
                raise
            if time.monotonic() >= deadline:
                handle.close()
                raise LockTimeout(
                    f"could not acquire {path} within {timeout:g}s"
                ) from exc
            time.sleep(retry_interval)


def _release_windows(outcome: _LockOutcome) -> None:
    if outcome.file_handle is None:
        return
    try:
        import msvcrt  # type: ignore[import-not-found]

        outcome.file_handle.seek(0)
        msvcrt.locking(outcome.file_handle.fileno(), 0, 1)  # LK_UNLCK
    except OSError:
        # Best-effort unlock; close below releases the underlying handle
        # and Windows reclaims the region lock on process fd cleanup.
        pass
    outcome.file_handle.close()


def acquire(
    role: Role,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    retry_interval: float = _DEFAULT_RETRY_INTERVAL_SECONDS,
) -> _LockOutcome:
    """Acquire the profile-scoped cross-process lock.

    ``role='writer'`` is authorized to create the lockfile (and parent
    directory) on first use. ``role='reader'`` is strictly read-only and
    will raise ``LockUnavailable`` if the lockfile does not already
    exist.

    The returned outcome must be released via :func:`release` (typically
    via the ``coordinated_lock`` context manager below) before the
    SQLite source family is observed by anyone else.
    """
    path = lockfile_path()
    if role == "writer":
        _writer_ensure_lockfile(path)
    elif not path.is_file():
        # The reader must not create anything. The lockfile's absence is
        # a fail-closed signal: an updated writer has not yet initialized
        # coordination on this profile.
        raise LockUnavailable(
            f"reader found no lockfile at {path}; "
            "mutable writer must initialize it first"
        )

    if sys.platform == "win32":
        return _acquire_windows(
            path, timeout=timeout, retry_interval=retry_interval
        )
    return _acquire_posix(path, timeout=timeout, retry_interval=retry_interval)


def release(outcome: _LockOutcome) -> None:
    """Release a previously-acquired cross-process lock.

    Idempotent and best-effort: descriptor close is the ultimate release
    mechanism on both platforms.
    """
    if outcome.platform == "posix":
        _release_posix(outcome)
    elif outcome.platform == "windows":
        _release_windows(outcome)
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"unknown platform: {outcome.platform!r}")


@contextmanager
def coordinated_lock(
    role: Role,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    retry_interval: float = _DEFAULT_RETRY_INTERVAL_SECONDS,
) -> Iterator[_LockOutcome]:
    """Context manager wrapping :func:`acquire` / :func:`release`.

    Use this from writers around ``_connect()`` (full connection lifetime)
    and from readers around the raw snapshot copy step.
    """
    outcome = acquire(
        role, timeout=timeout, retry_interval=retry_interval
    )
    try:
        yield outcome
    finally:
        release(outcome)


def lockfile_size_and_sha() -> tuple[int, str] | None:
    """Return ``(size, sha256)`` of the lockfile, or None if absent.

    The reader uses this to record the byte-stable witness. The size and
    SHA256 of the lockfile MUST be unchanged across a reader call.
    """
    import hashlib

    path = lockfile_path()
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return len(data), hashlib.sha256(data).hexdigest()
