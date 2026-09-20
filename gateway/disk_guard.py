"""Structural disk guards for the gateway housekeeping loop.

Three chores born of the 2026-09-19 and 2026-09-20 incidents: cron/verification
runs left abandoned multi-GB SQLite copies in the temp root — ~107 GB
(``tmp*.db``, ``statedb_ro*``) over three days, then another ~14 GB
(``sess_state*.db``, ``state_check*.db``) after the first guard shipped.
The volume hit 100 %, the gateway died unclean (OOM/SIGKILL) and every board
raised ``sqlite3.OperationalError: disk I/O error``.

* :func:`sweep_abandoned_db_copies` — sweep of abandoned >1 GiB SQLite files
  from the temp root. The second incident proved a hand-maintained leak-pattern
  list cannot win (``sess_state*``/``state_check*`` were missed), so the set is
  closed: ANY >1 GiB ``*.db`` / ``*.db-wal`` / ``*.db-shm`` file (plus the
  extension-less ``statedb_ro*`` fingerprints from the first incident) is sweep
  material — agent/session tooling must not park multi-GB databases in the temp
  root at all. Two guards protect files that are alive:

  - **open-handle guard** — on macOS/Linux the file is only removed when
    :command:`lsof` sees NO process holding it (rc 1). On POSIX an unlink under
    a live sqlite connection would technically be safe (the fd keeps working),
    but "connected" is not "abandoned", so a live DB is left alone on purpose.
    If :command:`lsof` is missing or errors, the mtime guard below is the only
    protection and the file is skipped.
  - **mtime grace** — files modified within ``SWEEP_GRACE_SECONDS`` (10 min)
    are skipped; a process may legitimately hold a >1 GiB temp DB without
    lsof noticing (e.g. mmap without fd, other-user fd, or a transient lsof
    failure). Fresh files are presumed live.

  Symlinks, directories and everything else (other names, other suffixes, other
  sizes) are never touched.
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
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

_GIB = 1024 * 1024 * 1024

# Sweep set. Closing it over the *shape* (a >1 GiB SQLite database) instead of
# a hand-list of leak names is the structural fix: the 2026-09-20 relapse
# (sess_state*.db / state_check*.db) sailed past the first guard because
# nobody had written those names down. "*.db*" alone would be tempting but
# ".db"-suffixed backups (.db.bak) are a common user idiom — enumerate the
# three sqlite spellings explicitly.
SWEEP_PATTERNS: Tuple[str, ...] = ("*.db", "*.db-wal", "*.db-shm", "statedb_ro*")
SWEEP_MIN_BYTES = 1 * _GIB
# A file last modified more recently than this is presumed live even when lsof
# sees no handle (mmap'd without fd, another user's fd, transient lsof error):
# the sweep only takes files that are both handle-free AND stale.
SWEEP_GRACE_SECONDS = 10 * 60.0

# Early-warning floors. 5 GiB leaves room to see the trend and act; below
# 1 GiB sqlite journaling/config writes are at imminent risk (the incident's
# terminal state), so that band escalates to ERROR.
FREE_WARN_BYTES = 5 * _GIB
FREE_CRITICAL_BYTES = 1 * _GIB
# Re-warn cadence while low: persistent enough to notice across a long leak,
# quiet enough not to spam the log every 60 s tick.
REWARN_SECONDS = 30 * 60.0

_WARN_STATE: Dict[str, Any] = {"level": "ok", "last_warn_monotonic": 0.0}


def _has_open_handle(path: Path) -> Optional[bool]:
    """True/False from :command:`lsof`; None when lsof is unavailable/failed.

    rc 0 means at least one process holds the file, rc 1 means no holder.
    Any other outcome (missing binary, timeout, weird path) is None — callers
    treat None as "cannot prove it is abandoned" and fall back to the mtime
    grace guard.
    """
    try:
        proc = subprocess.run(
            ["/usr/sbin/lsof", "-w", "--", str(path)],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def sweep_abandoned_db_copies(
    directory: Optional[Path | str] = None,
    *,
    patterns: Iterable[str] = SWEEP_PATTERNS,
    min_bytes: int = SWEEP_MIN_BYTES,
    grace_seconds: float = SWEEP_GRACE_SECONDS,
    lsof: Optional[Any] = "auto",
    _now: Optional[float] = None,
    logger_: Optional[logging.Logger] = None,
) -> int:
    """Remove abandoned >1 GiB SQLite files from *directory*.

    Shallow (non-recursive) scan of the temp root — both incident copy sets sat
    flat in ``$TMPDIR``. A candidate is removed only when ALL of:

    * name matches *patterns* (closed set of sqlite spellings by default);
    * regular, non-symlink file strictly larger than *min_bytes*;
    * NOT currently held by any process — :command:`lsof` must report no
      open handle (``lsof=None`` disables the check);
    * older than *grace_seconds* (mtime) — the backstop for live files lsof
      cannot see.

    Returns the number of files removed; never raises (scan/lsof/unlink
    failures degrade to debug logs and are retried next tick).
    """
    log = logger_ or logger
    root = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    lsof_check = _has_open_handle if lsof == "auto" else lsof
    now = time.time() if _now is None else float(_now)

    seen: set[Path] = set()
    removed = 0
    for pattern in patterns:
        try:
            candidates = list(root.glob(pattern))
        except OSError as exc:  # unreadable temp root — nothing we can do
            log.debug("Disk guard: cannot scan %s for %r: %s", root, pattern, exc)
            continue
        for victim in candidates:
            if victim in seen:
                continue  # *.db glob also lists the .db-wal/.db-shm siblings
            seen.add(victim)
            try:
                st = victim.lstat()
            except OSError:  # raced away between glob and stat
                continue
            if stat_module.S_ISLNK(st.st_mode) or not stat_module.S_ISREG(st.st_mode):
                continue
            if st.st_size <= min_bytes:
                continue
            if now - st.st_mtime < grace_seconds:
                continue  # fresh: presume a live writer, retry next tick
            if lsof_check is not None:
                held = lsof_check(victim)
                if held is None:
                    log.debug(
                        "Disk guard: lsof inconclusive on %s — skipping this tick", victim
                    )
                    continue
                if held:
                    log.debug(
                        "Disk guard: %s is held open by a live process — not sweeping", victim
                    )
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
                "closed-set sweep of stale handle-free *.db files, incidents "
                "2026-09-19/2026-09-20.",
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
                "gateway are at imminent risk; free space now. Known cause: abandoned *.db "
                "copies in the temp root (swept automatically each minute).",
                free / _GIB, worst_path, critical_bytes / _GIB,
            )
        else:
            log.warning(
                "Disk guard: %.2f GiB free on %s is below the %.0f GiB early-warning floor — "
                "disk is filling up; investigate before SQLite writes start failing. Known "
                "cause: abandoned *.db copies in the temp root.",
                free / _GIB, worst_path, warn_bytes / _GIB,
            )
        st_.update(level=level, last_warn_monotonic=now)
    return True
