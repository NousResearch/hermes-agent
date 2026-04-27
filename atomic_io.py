"""Cross-platform atomic writes: temp file + ``fsync`` + ``os.replace``.

See ``atomic_io_replace`` for retry semantics. Ownership restoration can fail
on Termux, Modal, or multi-user hosts without privileges.
"""

from __future__ import annotations

import errno
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import IO, Callable

from atomic_io_replace import (
    capture_target_metadata,
    chmod_chown_with_retry,
    replace_with_retry,
)

logger = logging.getLogger(__name__)

_WIN_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _wsl_drvfs_hint(path: Path) -> bool:
    if sys.platform != "linux":
        return False
    try:
        s = str(path.resolve())
    except OSError:
        s = str(path)
    return s.startswith("/mnt/") or s.startswith("/mnt/wsl/")


def _reject_win_reserved(path: Path) -> None:
    if os.name != "nt":
        return
    base = path.name
    if not base:
        return
    first = base.split(".", 1)[0].upper().rstrip(" \t")
    if first in _WIN_RESERVED:
        raise OSError(errno.EINVAL, f"reserved Win32 device name: {path.name}")


def cross_platform_atomic_writer(
    path: Path,
    writer: Callable[[IO[str]], None] | Callable[[IO[bytes]], None],
    *,
    binary: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Atomically write *path* using *writer* on a temp file, then ``replace``."""
    path = Path(path)
    _reject_win_reserved(path)
    if _wsl_drvfs_hint(path):
        logger.debug("atomic write target may be on WSL drvfs: %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = capture_target_metadata(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        if binary:
            with os.fdopen(fd, "wb") as fp:
                writer(fp)  # type: ignore[arg-type]
                fp.flush()
                os.fsync(fp.fileno())
        else:
            with os.fdopen(fd, "w", encoding=encoding) as fp:
                writer(fp)  # type: ignore[arg-type]
                fp.flush()
                os.fsync(fp.fileno())
        replace_with_retry(tmp_path, path)
        tmp_path = ""
        chmod_chown_with_retry(path, meta)
    except BaseException:
        try:
            if tmp_path:
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
