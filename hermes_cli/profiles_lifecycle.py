"""Coordinate profile directory changes made by cooperating Hermes commands."""

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import threading


_THREAD_LOCK = threading.RLock()
_FILE_LOCK_HOLDER = threading.local()


@contextmanager
def profile_lifecycle_lock():
    """Serialize lifecycle writes across threads/processes; external filesystem edits do not participate."""
    from hermes_cli.auth import _file_lock
    from hermes_cli.profiles import _get_profiles_root

    # Keep it under profiles/: default-profile clones/exports exclude that infrastructure.
    # The lock survives renaming/deleting any profile and is independent of HERMES_HOME.
    with _THREAD_LOCK, _file_lock(
        _get_profiles_root() / ".lifecycle.lock", _FILE_LOCK_HOLDER, 30,
        "Timed out waiting for profile lifecycle lock",
    ):
        yield


def directory_identity(path: Path, guards: ExitStack | None = None) -> tuple | None:
    """Pin POSIX directories for a mutation's lifetime so inode reuse cannot pass revalidation.

    Windows stat supplies the file index and creation time without opening a directory.
    Plain snapshots are used for previews and comparisons against a pinned identity.
    """
    try:
        if guards is not None and os.name != "nt":
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            guards.callback(os.close, fd)
            info = os.fstat(fd)
        else:
            info = path.stat()
    except FileNotFoundError:
        return None
    created = getattr(info, "st_birthtime_ns", getattr(info, "st_birthtime", None))
    if os.name == "nt" and created is None:
        # Before Python 3.12, Windows exposed creation time only as st_ctime.
        created = info.st_ctime_ns
    return (str(path.resolve()), info.st_dev, info.st_ino, created)
