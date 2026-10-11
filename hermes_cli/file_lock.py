"""Cross-process advisory file locks, stdlib only.

Guards the auth store, profile config edits and PM plugin install metadata.
PM workers import this module directly (``pm/publication.py``), so it must not
pull the application graph: keep every import here on the standard library.
"""

from __future__ import annotations

import errno
import os
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except Exception:
    fcntl = None
try:
    import msvcrt
except Exception:
    msvcrt = None


def _kernel_lock(lock_file: Any, acquire: bool) -> None:
    """Non-blocking exclusive flock (fcntl) or 1-byte msvcrt lock at offset 0; ``acquire=False`` releases."""
    if fcntl:
        fcntl.flock(lock_file.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)
    else:
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)


# Errnos that mean the lock is held rather than broken. POSIX flock contention is a
# ``BlockingIOError`` (EAGAIN); msvcrt reports contention as EACCES — which a real ACL denial
# also uses, so EACCES stays retried rather than aborting every Windows lock. Anything else
# (ENOSYS, EOPNOTSUPP, EIO, ...) cannot clear by retrying and must propagate immediately.
_LOCK_CONTENTION_ERRNOS = frozenset(
    code for code in (errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLK", None)) if code is not None)


def _is_lock_contention(exc: OSError) -> bool:
    if isinstance(exc, BlockingIOError):
        return True
    return exc.errno in _LOCK_CONTENTION_ERRNOS


def _lock_holder_hint(lock_path: Path) -> str:
    """Live-holder hint from the pid a holder stamps into the lock file; empty when there is none.

    Holders stamp their pid on acquire and clear it on release, so a leftover pid from a dead
    process is filtered by a liveness probe — stay silent instead of blaming a ghost."""
    try:
        first_token = lock_path.read_text(encoding="utf-8-sig", errors="replace").split()[0]
        pid = int(first_token)
    except (OSError, ValueError, IndexError):
        return ""
    if pid <= 0 or pid == os.getpid():
        return ""
    if os.name == "posix":
        try:
            os.kill(pid, 0)  # windows-footgun: ok — inside `if os.name == "posix"` gate
        except ProcessLookupError:
            return ""
        except OSError:
            pass  # exists but is not signalable (e.g. EPERM): still a live holder
    return (f"another hermes process (pid {pid}) probably still holds it "
            "(e.g. a dashboard or a slow credential refresh)")


def _stamp_lock_holder_pid(lock_file: Any) -> None:
    """Best-effort pid stamp so a timing-out waiter can name the holder (#124533)."""
    try:
        lock_file.truncate(0)
        lock_file.write(f"{os.getpid()}\n")  # "a+" writes land at the (now empty) end
        lock_file.flush()
    except OSError:
        pass


def _clear_stamped_lock_holder_pid(lock_file: Any) -> None:
    try:
        lock_file.truncate(0)
        if msvcrt:
            lock_file.write(" ")  # msvcrt.locking needs a non-empty file
        lock_file.flush()
    except OSError:
        pass


@contextmanager
def _file_lock(
    lock_path: Path, holder: threading.local, timeout_seconds: float, timeout_message: str):
    """Cross-process advisory flock helper, reentrant per-thread via ``holder.depth``.

    Falls back to a depth-only guard when neither ``fcntl`` nor ``msvcrt`` is available. Callers
    supply their own ``threading.local`` so independent locks (profile store vs global root vs the
    shared Nous store) track reentrancy separately."""
    if getattr(holder, "depth", 0) > 0:
        holder.depth += 1
        try:
            yield
        finally:
            holder.depth -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_file = None
        if fcntl is not None or msvcrt is not None:
            # msvcrt.locking needs a non-empty file with the pointer at 0. This convenience write can
            # race another holder's byte-range lock and raise PermissionError (reproduced with 20
            # concurrent processes on Windows); losing the race just means the file already has
            # content, so swallow it.
            if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
                try:
                    lock_path.write_text(" ", encoding="utf-8")
                except (OSError, PermissionError):
                    pass
            lock_file = stack.enter_context(lock_path.open("r+" if msvcrt else "a+", encoding="utf-8"))
            deadline = time.monotonic() + max(1.0, timeout_seconds)
            while True:
                try:
                    _kernel_lock(lock_file, True)
                    break
                except (BlockingIOError, OSError, PermissionError) as exc:
                    if not _is_lock_contention(exc):
                        # Permanent failure (flock-unsupported filesystem, bad fd, ...): retrying
                        # to the deadline would burn the timeout blaming a holder that does not
                        # exist. Let the original error through instead.
                        raise
                    if time.monotonic() >= deadline:
                        hint = _lock_holder_hint(lock_path)
                        raise TimeoutError(f"{timeout_message}; {hint}" if hint else timeout_message)
                    time.sleep(0.05)

        holder.depth = 1
        try:
            if lock_file is not None:
                _stamp_lock_holder_pid(lock_file)
            yield
        finally:
            holder.depth = 0
            if lock_file is not None:
                _clear_stamped_lock_holder_pid(lock_file)
                try:
                    _kernel_lock(lock_file, False)
                except (OSError, IOError):
                    pass
