"""Structural disk guards for the gateway housekeeping loop.

Two chores born of the 2026-09-19 incident: cron/verification runs left
abandoned multi-GB SQLite copies (``tmp*.db``, ``statedb_ro*``) in the temp
root — ~107 GB accumulated over three days, the volume hit 100 %, the gateway
died unclean (OOM/SIGKILL) and every board raised
``sqlite3.OperationalError: disk I/O error``.

* :func:`sweep_abandoned_db_copies` — pattern- AND size-bound removal of the
  leaked-copy fingerprints from the temp root. Structural safety net: it works
  no matter which process left the copies behind, independent of prompt
  discipline in any single cron run. Foreign temp files never match: only
  regular, non-symlink files whose name matches ``tmp*.db``/``statedb_ro*``
  AND whose size exceeds 1 GiB are removed. On POSIX an unlink under a live
  sqlite connection is safe (the fd keeps working); on Windows the unlink of
  an open file fails and is skipped at debug level.
* :func:`check_free_disk_warning` — fail-fast early warning in the gateway
  log when free space drops below 5 GiB (ERROR below 1 GiB), so the trend is
  visible hours before SQLite dies. Edge-triggered with a 30-minute re-warn
  while low; recovery is logged at INFO.

Both are best-effort: statvfs/unlink failures degrade to debug logs and never
raise into the housekeeping loop.
"""

from __future__ import annotations

import logging
import shutil
import stat as stat_module
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

_GIB = 1024 * 1024 * 1024

# Leak fingerprints from the incident evidence (16 loose tmp*.db copies at
# 4.4-4.7 GB plus stale read-only verification copies). Pattern-bound on
# purpose: anything outside these globs is foreign temp data and stays.
LEAK_PATTERNS: Tuple[str, ...] = ("tmp*.db", "statedb_ro*")
SWEEP_MIN_BYTES = 1 * _GIB

# Early-warning floors. 5 GiB leaves room to see the trend and act; below
# 1 GiB sqlite journaling/config writes are at imminent risk (the incident's
# terminal state), so that band escalates to ERROR.
FREE_WARN_BYTES = 5 * _GIB
FREE_CRITICAL_BYTES = 1 * _GIB
# Re-warn cadence while low: persistent enough to notice across a long leak,
# quiet enough not to spam the log every 60 s tick.
REWARN_SECONDS = 30 * 60.0

_WARN_STATE: Dict[str, Any] = {"level": "ok", "last_warn_monotonic": 0.0}


def sweep_abandoned_db_copies(
    directory: Optional[Path | str] = None,
    *,
    patterns: Iterable[str] = LEAK_PATTERNS,
    min_bytes: int = SWEEP_MIN_BYTES,
    logger_: Optional[logging.Logger] = None,
) -> int:
    """Remove abandoned >1 GiB DB copies matching *patterns* from *directory*.

    Shallow (non-recursive) scan of the temp root — the incident copies sat
    flat in ``$TMPDIR``. Returns the number of files removed; never raises
    (scan/unlink failures degrade to debug logs). Symlinks and directories
    are never touched, and neither are files at or below ``min_bytes``.
    """
    log = logger_ or logger
    root = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    removed = 0
    for pattern in patterns:
        try:
            candidates = list(root.glob(pattern))
        except OSError as exc:  # unreadable temp root — nothing we can do
            log.debug("Disk guard: cannot scan %s for %r: %s", root, pattern, exc)
            continue
        for victim in candidates:
            try:
                st = victim.lstat()
            except OSError:  # raced away between glob and stat
                continue
            if stat_module.S_ISLNK(st.st_mode) or not stat_module.S_ISREG(st.st_mode):
                continue
            if st.st_size <= min_bytes:
                continue
            try:
                victim.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                # Windows: open file; POSIX: permissions. Skip, retry next tick.
                log.debug("Disk guard: could not remove %s: %s", victim, exc)
                continue
            removed += 1
            log.warning(
                "Disk guard: removed abandoned DB copy %s (%.2f GiB, mtime %s) — "
                "pattern-bound sweep, see incident 2026-09-19.",
                victim, st.st_size / _GIB, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            )
    if removed:
        log.warning("Disk guard: swept %d abandoned DB copy file(s) >1 GiB from %s", removed, root)
    return removed


def check_free_disk_warning(
    paths: Optional[Iterable[Path | str]] = None,
    *,
    warn_bytes: int = FREE_WARN_BYTES,
    critical_bytes: int = FREE_CRITICAL_BYTES,
    state: Optional[Dict[str, Any]] = None,
    _now: Optional[float] = None,
    logger_: Optional[logging.Logger] = None,
) -> bool:
    """Warn in the log when free disk space drops below *warn_bytes*.

    Checks the temp root and the Hermes home (deduped per device), because a
    full volume kills SQLite writes and the gateway itself — the failure must
    surface in the log hours before it does. Edge-triggered: one warning per
    transition into a band, re-warned every ``REWARN_SECONDS`` while low, and
    escalated to ERROR below *critical_bytes*. Recovery logs at INFO.

    Returns True when any checked path is below *warn_bytes* (False includes
    the unreadable-filesystem case: a missing sample must never read as
    "fine", but it also must not cry wolf — it stays silent at debug).
    """
    log = logger_ or logger
    st_ = state if state is not None else _WARN_STATE
    now = time.monotonic() if _now is None else float(_now)

    if paths is None:
        candidates = [Path(tempfile.gettempdir())]
        try:
            from hermes_constants import get_hermes_home

            candidates.append(Path(get_hermes_home()))
        except Exception:  # home resolution must never break the chore
            pass
    else:
        candidates = [Path(p) for p in paths]

    samples = []  # (free_bytes, path)
    seen_devices = set()
    for path in candidates:
        try:
            usage = shutil.disk_usage(path)
            dev = path.stat().st_dev
        except OSError as exc:
            log.debug("Disk guard: cannot sample free space on %s: %s", path, exc)
            continue
        if usage.total <= 0 or dev in seen_devices:
            continue
        seen_devices.add(dev)
        samples.append((usage.free, path))
    if not samples:
        return False

    free, worst_path = min(samples, key=lambda item: item[0])
    level = "critical" if free < critical_bytes else ("warn" if free < warn_bytes else "ok")
    previous = st_.get("level", "ok")

    if level == "ok":
        if previous != "ok":
            log.info(
                "Disk guard: free space recovered to %.2f GiB on %s (warning floor is %.0f GiB).",
                free / _GIB, worst_path, warn_bytes / _GIB,
            )
        st_.update(level="ok", last_warn_monotonic=0.0)
        return False

    escalated = previous != level
    rewarn_due = escalated or (now - float(st_.get("last_warn_monotonic", 0.0)) >= REWARN_SECONDS)
    if rewarn_due:
        if level == "critical":
            log.error(
                "Disk guard: only %.2f GiB free on %s (below %.0f GiB) — SQLite writes and the "
                "gateway are at imminent risk; free space now. Known cause: abandoned tmp*.db / "
                "statedb_ro* copies in the temp root (swept automatically each minute).",
                free / _GIB, worst_path, critical_bytes / _GIB,
            )
        else:
            log.warning(
                "Disk guard: %.2f GiB free on %s is below the %.0f GiB early-warning floor — "
                "disk is filling up; investigate before SQLite writes start failing. Known cause: "
                "abandoned tmp*.db / statedb_ro* copies in the temp root.",
                free / _GIB, worst_path, warn_bytes / _GIB,
            )
        st_.update(level=level, last_warn_monotonic=now)
    return True
