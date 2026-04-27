"""Internal: ``os.replace`` and metadata restore with transient-error retries."""

from __future__ import annotations

import errno
import logging
import os
import stat
import time
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class TargetFileMeta(NamedTuple):
    mode: int | None
    uid: int | None
    gid: int | None


def capture_target_metadata(path: Path) -> TargetFileMeta:
    if not path.exists():
        return TargetFileMeta(None, None, None)
    try:
        st = path.stat()
    except OSError:
        return TargetFileMeta(None, None, None)
    mode = stat.S_IMODE(st.st_mode)
    uid = int(st.st_uid) if hasattr(st, "st_uid") else None
    gid = int(st.st_gid) if hasattr(st, "st_gid") else None
    if os.name == "nt":
        return TargetFileMeta(mode, None, None)
    return TargetFileMeta(mode, uid, gid)


def _transient_replace_err(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError):
        e = exc.errno
        if e in (errno.EACCES, errno.EPERM, errno.EBUSY, errno.EINTR):
            return True
        if hasattr(errno, "ESTALE") and e == errno.ESTALE:
            return True
        if hasattr(errno, "EAGAIN") and e == errno.EAGAIN:
            return True
        if getattr(exc, "winerror", None) in (32, 33):
            return True
    return False


def _sleep_backoff(attempt: int, base: float = 0.005) -> None:
    time.sleep(base * (2**attempt))


def replace_with_retry(src: str, dst: Path, *, attempts: int = 8) -> None:
    last: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            last = exc
            if not _transient_replace_err(exc) or attempt == attempts - 1:
                raise
            _sleep_backoff(attempt)
    assert last is not None
    raise last


def chmod_chown_with_retry(path: Path, meta: TargetFileMeta, *, attempts: int = 6) -> None:
    for attempt in range(attempts):
        try:
            if meta.mode is not None:
                os.chmod(path, meta.mode)
            if (
                meta.uid is not None
                and meta.gid is not None
                and hasattr(os, "chown")
                and os.name != "nt"
            ):
                os.chown(path, meta.uid, meta.gid)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                logger.warning(
                    "chmod/chown restore incomplete for %s after retries: %s",
                    path,
                    exc,
                )
                return
            if exc.errno not in (
                errno.EBUSY,
                errno.EAGAIN,
                getattr(errno, "ESTALE", -1),
                errno.EPERM,
                errno.EACCES,
            ):
                logger.debug("chmod/chown non-retryable for %s: %s", path, exc)
                return
            _sleep_backoff(attempt)
