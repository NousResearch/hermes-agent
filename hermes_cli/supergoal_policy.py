"""Durable, per-session clarification denial, independent of the goal database.

No shipped pre-supergoal installation needs migration. Development-era active rows
are bootstrapped by load_goal before resuming their loop, never by clarify (which
must work without SQLite). Absence is ordinary; malformed/unreadable markers fail
closed for that session. This is a restriction only, not a second goal store.
"""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path

from hermes_constants import get_hermes_home, mkdir_under_hermes_home
from utils import atomic_write_text, fsync_directory

_CONTENT = "supergoal:deny-clarify:v1\n"


def _marker_path(session_id: str) -> Path:
    # IDs are opaque, not path components (including imported session IDs).
    name = sha256(session_id.encode("utf-8")).hexdigest()
    return get_hermes_home() / "supergoal-policy" / name


@contextmanager
def policy_write_lock(session_id: str):
    """Serialize SG DB+marker transitions, including across processes.

    Nonblocking: contention is a technical error, never an unguarded kickoff.
    The lock inode stays in place; closing the fd releases it even after failure.
    Clarification only reads the marker and never waits for this writer lock.
    """
    path = _marker_path(session_id).with_suffix(".lock")
    mkdir_under_hermes_home(path.parent)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Supergoal policy update unavailable; retry the command") from exc
        yield


def clarification_restricted(session_id: str) -> bool:
    if not session_id:
        return False
    path = _marker_path(session_id)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # A dangling marker is broken policy, not an absent restriction.
        if path.is_symlink():
            raise RuntimeError("Unreadable supergoal policy marker")
        return False
    if content != _CONTENT:
        raise RuntimeError("Invalid supergoal policy marker")
    return True


def restrict_clarification(session_id: str) -> None:
    if not session_id:
        raise RuntimeError("Supergoal session id is required")
    path = _marker_path(session_id)
    atomic_write_text(path, _CONTENT, mode=0o600, fsync_dir=True)
    # Also persist a newly-created policy directory's entry in the profile home.
    fsync_directory(path.parent.parent)
    if not clarification_restricted(session_id):
        raise RuntimeError("Supergoal policy persistence verification failed")


def release_clarification(session_id: str) -> None:
    """Only after the caller has verified a non-active-supergoal DB write."""
    path = _marker_path(session_id)
    path.unlink(missing_ok=True)
    fsync_directory(path.parent)
    if clarification_restricted(session_id):
        raise RuntimeError("Supergoal policy release verification failed")


def carry_clarification_restriction(old_session_id: str, new_session_id: str) -> None:
    """Before publishing a compression child; failures must abort publication."""
    if clarification_restricted(old_session_id):
        restrict_clarification(new_session_id)
