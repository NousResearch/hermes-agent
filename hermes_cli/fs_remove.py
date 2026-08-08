"""Recursive directory removal that survives read-only files.

``shutil.rmtree`` deletes a file with ``os.unlink``, and on Windows
``os.unlink`` refuses a file carrying the read-only attribute with
``PermissionError: [WinError 5] Access is denied``. On POSIX the parent
directory's write bit governs the unlink, so a read-only *file* deletes
fine and this whole class of failure is invisible there.

Git is the common source of such files: objects under ``.git/objects/``
(loose objects, ``pack-*.pack``, ``pack-*.idx``) are written read-only on
purpose, since they're content-addressed and must never be edited in
place. So *any* code path that removes a git checkout (an installed
plugin, a cloned skill, a staged update) fails on Windows unless it
clears the read-only bit first.

Use :func:`rmtree_force` anywhere a tree that might contain a checkout
gets removed.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Union

__all__ = ["rmtree_force"]

_StrPath = Union[str, "os.PathLike[str]"]

# ``onerror`` was removed in 3.12 and ``onexc`` added in the same release.
# Hermes supports >=3.11,<3.14, so both are live.
_USE_ONEXC = sys.version_info >= (3, 12)


def _make_writable(func, path: str, exc) -> None:
    """rmtree error callback: clear the read-only bit, then retry.

    Compatible with both callback APIs:
      ``onexc(func, path, exc_instance)``     on 3.12+
      ``onerror(func, path, exc_info_tuple)`` on 3.11
    """
    if isinstance(exc, tuple):  # exc_info tuple → the exception itself
        exc = exc[1]

    if not isinstance(exc, PermissionError):
        raise exc

    # The path itself may be read-only (mode 0444, or the Windows
    # read-only attribute), and so may its parent. A directory has to be
    # writable for unlink/rmdir of a child to succeed.
    #
    # OR the write bit into the existing mode rather than assigning one:
    # replacing the mode outright strips group and other bits, which breaks
    # group-shared installs (see #67496). On Windows os.chmod only honours
    # the owner-write bit, and clearing it is what drops the read-only
    # attribute that blocks os.unlink in the first place.
    for target in (path, os.path.dirname(path)):
        if not target:
            continue
        try:
            os.chmod(target, os.stat(target).st_mode | stat.S_IWUSR)
        except OSError:
            pass

    func(path)


def rmtree_force(path: _StrPath, ignore_errors: bool = False, attempts: int = 3) -> None:
    """``shutil.rmtree`` that clears read-only files and retries transients.

    Two failure modes are handled:

    * **Read-only files** (the git-objects case above). The error callback
      chmods the path and its parent writable and retries the operation.
    * **Transient locks**. On Windows a file another process just closed,
      or one an antivirus scanner is holding, raises ``PermissionError``
      that a moment later would not. Also ``ENOTEMPTY`` on POSIX when a
      writer lands a file after rmtree walked past its directory. A few
      spaced retries win those races instead of failing the whole delete.

    With ``ignore_errors=True`` this never raises, matching ``shutil``.
    """
    p = Path(path)
    last_exc: OSError | None = None

    for attempt in range(max(1, attempts)):
        try:
            if _USE_ONEXC:
                shutil.rmtree(p, onexc=_make_writable)
            else:
                shutil.rmtree(p, onerror=_make_writable)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last_exc = e
            if not p.exists():
                return
            if attempt < attempts - 1:
                time.sleep(0.3 * (attempt + 1))

    if ignore_errors:
        return
    if last_exc is not None:
        raise last_exc
