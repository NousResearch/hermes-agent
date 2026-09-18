"""Remove checkout trees with read-only files and transient cleanup races."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path


def _make_writable(root: Path, func, path, exc) -> None:
    if isinstance(exc, tuple):
        exc = exc[1]
    if not isinstance(exc, PermissionError):
        raise exc
    # Never chmod the root's parent or follow a link outside the removed tree.
    for target in (Path(path), Path(path).parent):
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            continue
        os.chmod(target, target.stat().st_mode | stat.S_IWUSR)
    func(path)


def rmtree_force(path: str | os.PathLike[str], ignore_errors: bool = False) -> None:
    """Preserve profile cleanup retries while clearing read-only checkout files."""
    path = Path(path).absolute()
    root = path.resolve()

    def onerror(func, failed_path, exc):
        _make_writable(root, func, failed_path, exc)

    for attempt in range(3):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=onerror)
            else:
                shutil.rmtree(path, onerror=onerror)
            return
        except OSError:
            if not path.exists():
                return
            if attempt == 2:
                if not ignore_errors:
                    raise
                return
            time.sleep(0.3 * (attempt + 1))
