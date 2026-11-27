"""The install lock's two locations and the lock that holds a moved lock at both.

Its own module, not ``pm.filesystem`` or ``hermes_cli.runtime_state``: a process started on
v0.23.3 loaded both at boot (``hermes_bootstrap`` activation imports ``runtime_state``) and imports
this lazily after ``hermes update`` moved the checkout (#158723), so it uses only names v0.23.3's
``pm.filesystem`` has.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from pm import filesystem as _filesystem


def install_lock_paths(state: Path) -> tuple[Path, Path]:
    """The install lock where releases before ``locks/`` existed take it, and where it lives now."""
    return state / ".install.lock", state / "locks" / "install.lock"


@contextmanager
def _old_path_lock(old: Path, new: Path, *, timeout: float | None):
    """The lock as v0.23.3 takes it: *old* alone (current processes take *old* first too)."""
    fd = _filesystem.lock_path(old, timeout=timeout, create_dir=True)
    try:
        yield fd is not None
    finally:
        if fd is not None:
            os.close(fd)


moved_lock = getattr(_filesystem, "moved_lock", _old_path_lock)
