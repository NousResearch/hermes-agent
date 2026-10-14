"""Filesystem helpers shared across Hermes CLI subsystems."""

import os
import shutil
import stat
from pathlib import Path


def rmtree_force(path: Path) -> None:
    """``shutil.rmtree`` that clears read-only bits before deleting.

    Windows refuses to delete a tree containing read-only files — git clones
    store objects read-only — raising ``PermissionError`` where POSIX unlinks
    them fine. The ``onerror`` hook chmods the file writable and retries.
    """
    def _onerror(func, p, _exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
        except OSError:
            pass
        func(p)

    shutil.rmtree(path, onerror=_onerror)
