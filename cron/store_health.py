"""Degraded state for an unwritable cron store (ENOSPC, EROFS, EACCES).

A full disk or read-only mount fails every store write on every 60s tick. Instead of one warning
per failing call site, each store directory gets ONE in-memory record: set by the first failed
write, updated by later ones, cleared by the first jobs.json save that lands. While it is set,
the tick skips the advance/claim work that can only fail and re-probes the store at most once a
minute. ``probe_store`` is also what ``hermes cron status`` and ``hermes doctor`` use: they run in
another process, so they cannot read this record or trust markers in a directory nobody can write.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("cron.jobs")

PROBE_INTERVAL_SECONDS = 60.0
# Below this the probe reports ENOSPC: an empty temp file can still be created on a full disk.
_PROBE_MIN_FREE_BYTES = 1 << 20
# `hermes doctor` warns below this, before the store actually starts failing.
LOW_FREE_BYTES = 100 << 20
FIX_HINT = "free disk space, remount it read-write, or fix permissions on {store}"


def describe_error(exc: OSError) -> str:
    code = errno.errorcode.get(exc.errno or 0, "")
    text = exc.strerror or str(exc)
    return f"{code}: {text}" if code else text


@dataclass
class StoreDegraded:
    store: str
    since: float  # epoch seconds of the first failed write
    error: str
    sites: set = field(default_factory=set)
    skipped: set = field(default_factory=set)  # distinct (job id, scheduled instant) not run
    # Monotonic time of the last failed dispatch write or re-probe. None until a dispatch write
    # fails, so a due one-shot reaches its claim and settles its ``failed`` row once per probe.
    last_probe: Optional[float] = None

    @property
    def skipped_runs(self) -> int:
        return len(self.skipped)

    def notice_fields(self) -> dict:
        since = datetime.fromtimestamp(self.since).astimezone().isoformat(timespec="seconds")
        return {"store": self.store, "error": self.error, "since": since, "skipped": self.skipped_runs,
                "fix": FIX_HINT.format(store=self.store)}


_DISPATCH_SITES = frozenset({"advance", "claim"})
_degraded: dict[str, StoreDegraded] = {}
_lock = threading.Lock()
# The gateway consumes transitions; ``fn(event, record)`` with event "unwritable" or "recovered".
_listener: Optional[Callable[[str, StoreDegraded], None]] = None


def set_transition_listener(fn: Optional[Callable[[str, StoreDegraded], None]]) -> None:
    global _listener
    _listener = fn


def _notify(event: str, record: StoreDegraded) -> None:
    listener = _listener
    if listener is None:
        return
    try:  # a notice must never break the cron tick that reported the transition
        listener(event, record)
    except Exception:
        logger.debug("Cron store %s transition listener failed", event, exc_info=True)


def _active_cron_dir() -> Path:
    from cron.jobs import _current_cron_store
    return _current_cron_store().cron_dir


def _run_keys(jobs: Iterable[dict]) -> set:
    return {(job.get("id"), job.get("next_run_at")) for job in jobs}


def note_unwritable(cron_dir: Path, exc: OSError, consequence: str, site: str,
                    skipped_jobs: Iterable[dict] = ()) -> None:
    """Record a failed store write; WARN once, when the store enters the degraded state."""
    store = str(cron_dir)
    with _lock:
        record = _degraded.get(store)
        entered = record is None
        if entered:
            record = _degraded[store] = StoreDegraded(store, time.time(), describe_error(exc))
        new_site = site not in record.sites
        record.sites.add(site)
        if site in _DISPATCH_SITES:  # a dispatch write failed: throttle the next attempt from here
            record.last_probe = time.monotonic()
        record.error = describe_error(exc)
        record.skipped |= _run_keys(skipped_jobs)
    if entered:
        logger.warning(
            "Cron store %s is unwritable (%s); %s. Scheduled jobs are skipped until it accepts writes "
            "again, then each due job fires once. Fix: %s.", store, exc, consequence,
            FIX_HINT.format(store=store))
        _notify("unwritable", record)
    elif new_site:
        logger.info("Cron store %s still unwritable (%s); %s", store, exc, consequence)


def note_writable(cron_dir: Path) -> None:
    """A jobs.json save landed: clear the degraded state for that store."""
    if not _degraded:
        return
    with _lock:
        record = _degraded.pop(str(cron_dir), None)
    if record is None:
        return
    logger.warning("Cron store %s is writable again; %d skipped run(s), catching up once per job",
                   record.store, record.skipped_runs)
    _notify("recovered", record)


def degraded_record(cron_dir: Optional[Path] = None) -> Optional[StoreDegraded]:
    return _degraded.get(str(cron_dir if cron_dir is not None else _active_cron_dir()))


def degraded_records() -> list:
    with _lock:
        return list(_degraded.values())


def degraded_since(cron_dir: Path) -> Optional[float]:
    record = _degraded.get(str(cron_dir))
    return record.since if record is not None else None


def dispatch_blocked(due_jobs: list) -> bool:
    """Whether this tick skips advance/claim for ``due_jobs``: the active store is known
    unwritable and the minute-throttled re-probe has not seen it accept a write. Skipped runs are
    recorded; once the probe passes, the dispatch's own save confirms recovery (or re-degrades)."""
    cron_dir = _active_cron_dir()
    record = _degraded.get(str(cron_dir))
    if record is None:
        return False
    now = time.monotonic()
    if record.last_probe is None:  # entered in load/scan: dispatch has not been tried yet
        return False
    error = None
    if now - record.last_probe < PROBE_INTERVAL_SECONDS or (error := probe_store(cron_dir)) is not None:
        with _lock:
            if error is not None:
                record.last_probe, record.error = now, describe_error(error)
            record.skipped |= _run_keys(due_jobs)
        return True
    record.last_probe = now
    return False


def probe_report(cron_dir: Path) -> Optional[dict]:
    """Cross-process view for `cron status` / `doctor`: ``None`` when the store accepts writes,
    else the notice fields. ``since`` is jobs.json's mtime (the last write that landed);
    ``skipped`` is left to the caller, which counts the due jobs that have not fired."""
    error = probe_store(cron_dir)
    if error is None:
        return None
    try:
        since = datetime.fromtimestamp((cron_dir / "jobs.json").stat().st_mtime).astimezone()
        since_text = since.isoformat(timespec="seconds")
    except OSError:
        since_text = "unknown"
    store = str(cron_dir)
    return {"store": store, "error": describe_error(error), "since": since_text,
            "fix": FIX_HINT.format(store=store)}


def free_bytes(path: Path) -> Optional[int]:
    """Bytes this process can still write. Root may also fill the reserved blocks that
    ``disk_usage().free`` leaves out, so a 'full' disk is not full for root."""
    try:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            stats = os.statvfs(path)
            return stats.f_bfree * stats.f_frsize
        return shutil.disk_usage(path).free
    except OSError:
        return None


def probe_store(cron_dir: Path) -> Optional[OSError]:
    """Cheap write probe: ``None`` when ``cron_dir`` accepts a new file (or does not exist yet),
    else the OSError a store write would hit."""
    if not cron_dir.is_dir():
        return None
    free = free_bytes(cron_dir)
    if free is not None and free < _PROBE_MIN_FREE_BYTES:
        return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(cron_dir))
    try:
        fd, tmp = tempfile.mkstemp(dir=cron_dir, prefix=".probe_")
        os.close(fd)
        os.unlink(tmp)
    except OSError as exc:
        return exc
    return None
