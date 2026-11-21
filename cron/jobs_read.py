"""Reads of ``cron/jobs.json``: the tolerant parse, retried across a transient sharing conflict.

Every writer replaces the file through mkstemp+rename, but a reader can still collide with the
swap: on Windows the open lands while the old file is being replaced and answers ``PermissionError``
(``winerror 32`` maps to EACCES), and a network or NFS home can answer EBUSY/ESTALE. A re-read a
few milliseconds later succeeds — measured in #153042, where two jobs started in the same second
and completed while a third died in 1.4 s reading the very same file, and an immediate re-read was
fine. Without a retry the caller turns that collision into ``Failed to read cron database`` and the
run is lost (a cron execution is never re-attempted), so the read retries those errnos a few times
before giving up; the attempt count stays small and bounded so a real permission problem is still
reported as the same failure, 150 ms later.
"""

from __future__ import annotations

import errno
import json
import time
from pathlib import Path
from typing import Any, Callable

# 3 attempts: a rename window is microseconds, so the second read almost always wins; the third
# only exists so a scheduler tick contending for the store does not eat the retry budget.
READ_ATTEMPTS = 3
# Linear backoff (50 ms, 100 ms). It only delays an eventual refusal, never a success by more than
# one collision.
RETRY_BASE_SECONDS = 0.05
# Conflicts a re-read can win. EACCES/EPERM cover Windows' sharing violation; EBUSY/ESTALE cover a
# swap in progress on a network home. A dead disk (EIO) or a missing file is NOT retried.
TRANSIENT_READ_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EBUSY, errno.EAGAIN, errno.ESTALE})


def is_transient_read_error(exc: OSError) -> bool:
    """True for a sharing/permission conflict that a re-read can win (see the module docstring)."""
    return isinstance(exc, PermissionError) or exc.errno in TRANSIENT_READ_ERRNOS


def parse_jobs_file(
    jobs_file: Path, *, sleep: Callable[[float], None] = time.sleep
) -> tuple[Any, bool]:
    """Tolerant jobs.json parse -> ``(data, used_strict_fallback)``.

    utf-8-sig absorbs a BOM; a strict ``json`` failure retries with ``strict=False``; a transient
    sharing conflict retries the read (``READ_ATTEMPTS`` with ``RETRY_BASE_SECONDS`` linear
    backoff). IO/fallback errors propagate after the retries, so the caller keeps deciding between
    repair and bail.
    """
    raw = ""
    for attempt in range(READ_ATTEMPTS):
        try:
            with open(jobs_file, "r", encoding="utf-8-sig") as f:
                raw = f.read()
            break
        except OSError as exc:
            if attempt == READ_ATTEMPTS - 1 or not is_transient_read_error(exc):
                raise
            sleep(RETRY_BASE_SECONDS * (attempt + 1))
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        return json.loads(raw, strict=False), True
