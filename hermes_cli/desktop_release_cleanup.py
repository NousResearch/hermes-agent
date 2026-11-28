"""Removing a release tree on Windows when it holds a DOS reserved name.

Windows resolves a file named ``nul`` (or ``con``, ``aux``, ``prn``, ``com1``...) to the DOS
device, not to the file that sits in the directory. Two consequences bite the desktop swapper:

* ``shutil.rmtree`` cannot delete such a file through the normal path-based APIs, and
  ``ignore_errors=True`` swallows that failure, so the directory silently survives.
* ``os.path.abspath`` rewrites a path whose last component is a reserved name into the device
  form (``\\\\.\\nul``), which cannot be deleted either -- so a helper that "normalizes" the
  path first destroys the very handle it needs.

A leftover release tree holding one of those files makes the promotion rename fail with
WinError 183 (ERROR_ALREADY_EXISTS) -- "the file already exists" -- while the real cause is
that the leftover tree could not be removed at all. The device path form (``\\\\?\\`` prefix)
bypasses DOS device-name resolution, so the reserved-name entries can be deleted and the tree
removed.

The removal here is for the desktop release tree only: the aside copy of a packaged app and
the staging directory of a rebuild. Both are Hermes-owned build outputs that a new build
replaces wholesale, so nothing the user authored is removed by it.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

#: Same logger as ``hermes_cli.main_desktop`` so the swap's records stay in one stream.
logger = logging.getLogger("hermes_cli.main")

#: DOS reserved device names. A file whose name (any extension, any case) starts with one of
#: these cannot be created or deleted through the normal Win32 path-based APIs.
_DOS_RESERVED_NAMES = frozenset(
    ["nul", "con", "aux", "prn"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)]
)


def _is_reserved_name(name: str) -> bool:
    """True for a DOS reserved device name ("nul", "NUL", "nul.txt", ...)."""
    return name.split(".", 1)[0].lower() in _DOS_RESERVED_NAMES


def _device_path(path: str) -> str:
    """The ``\\\\?\\`` form of a path, bypassing DOS device-name resolution.

    ``os.path.abspath`` must NOT be used here: Win32's GetFullPathName rewrites a path whose
    last component is a reserved name into ``\\\\.\\nul``, and that device cannot be deleted.
    ``os.path.normpath`` leaves the path alone.
    """
    return "\\\\?\\" + os.path.normpath(str(path))


def remove_reserved_name_entries(root: Path) -> list[str]:
    """Delete every reserved-name FILE under ``root``. Returns the paths that were removed.

    A directory named after a reserved device cannot exist on Windows, so only files are
    considered. Symlinks and junctions are not followed.
    """
    removed: list[str] = []
    root = Path(os.path.normpath(str(root)))
    if not root.is_dir():
        return removed
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            if not _is_reserved_name(name):
                continue
            path = os.path.join(dirpath, name)
            try:
                os.remove(_device_path(path))
            except OSError as exc:
                logger.warning(
                    "could not remove the reserved-name file %s: %s", path, exc
                )
                continue
            removed.append(path)
    return removed


def remove_tree(path: Path) -> str | None:
    """Remove a tree, defeating Windows reserved names. Returns an error text, or None on success.

    The plain removal is tried first. When it fails, the reserved-name entries are deleted
    through the device path and the removal is retried -- that is the case where a leftover
    ``nul`` silently survived ``ignore_errors=True`` and broke the promotion rename with
    WinError 183. A failure that survives both attempts is RETURNED, never swallowed: the
    caller needs the real reason, not "the file already exists".
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        shutil.rmtree(path)
        return None
    except OSError as exc:
        first = exc
    removed = remove_reserved_name_entries(path)
    if removed:
        logger.info(
            "removed %d reserved-name file(s) from %s before retrying the tree removal",
            len(removed),
            path,
        )
    try:
        shutil.rmtree(path)
        return None
    except OSError as exc:
        return f"could not remove {path}: {exc} (first attempt: {first})"
