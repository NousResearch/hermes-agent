"""Degraded state for an unwritable cron store (ENOSPC, EROFS, EACCES).

A full disk or read-only mount fails every store write on every 60s tick. Instead of one warning
per failing call site, each store directory gets ONE in-memory record: set by the first failed
write, updated by later ones, cleared by the first jobs.json save that lands. While it is set,
the tick skips the advance/claim work that can only fail and re-probes the store at most once a
minute. ``probe_store`` is also what ``hermes cron status`` and ``hermes doctor`` use: they run in
another process, so they cannot read this record or trust markers in a directory nobody can write.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("cron.jobs")

PROBE_INTERVAL_SECONDS = 60.0
# Store-write failures that degrade the store (the tick keeps running) instead of failing the tick.
UNWRITABLE_ERRNOS = frozenset({errno.EROFS, errno.EACCES, errno.EPERM, errno.ENOSPC, errno.EDQUOT})
# `hermes doctor` warns below this, before the store actually starts failing.
LOW_FREE_BYTES = 100 << 20
FIX_HINT = "free disk space, remount it read-write, or fix permissions on {store}"


def describe_error(exc: OSError) -> str:
    code = errno.errorcode.get(exc.errno or 0, "")
    text = exc.strerror or str(exc)
    return f"{code}: {text}" if code else text


def _report_fields(store: str, error: str, since: Optional[float]) -> dict:
    """The fields `cron status`, doctor and the home-channel notices print for an unwritable store."""
    since_text = "unknown" if since is None else datetime.fromtimestamp(since).astimezone().isoformat(timespec="seconds")
    return {"store": store, "error": error, "since": since_text, "fix": FIX_HINT.format(store=store)}


@dataclass
class StoreDegraded:
    store: str
    since: float  # epoch seconds of the first failed write
    error: str
    recovered_at: Optional[float] = None  # epoch seconds of the first write that landed again
    sites: set = field(default_factory=set)
    skipped: set = field(default_factory=set)  # distinct (job id, scheduled instant) not run
    # Monotonic time of the last failed dispatch write or re-probe. None until a dispatch write
    # fails, so a due one-shot reaches its claim and settles its ``failed`` row once per probe.
    last_probe: Optional[float] = None

    @property
    def skipped_runs(self) -> int:
        return len(self.skipped)

    def notice_fields(self) -> dict:
        return dict(_report_fields(self.store, self.error, self.since), skipped=self.skipped_runs)


_DISPATCH_SITES = frozenset({"advance", "claim"})
_degraded: dict[str, StoreDegraded] = {}
# Each store's just-ended outage, kept until a due scan's save lands (``outage_covers``): the save that
# ends an outage is often another job's heartbeat, landing before the scan that fires the
# one-shots the outage skipped.
_recovered: dict[str, StoreDegraded] = {}
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


def _key(cron_dir: Optional[Path]) -> str:
    """Record key for a store: its resolved path (default: the active store). Callers pass both
    resolved (cron store) and unresolved (tick-lock dir, profile homes) paths for one store."""
    return os.path.realpath(cron_dir if cron_dir is not None else _active_cron_dir())


def _run_keys(jobs: Iterable[dict]) -> set:
    return {(job.get("id"), job.get("next_run_at")) for job in jobs}


def note_unwritable(exc: OSError, consequence: str, site: str, skipped_jobs: Iterable[dict] = (),
                    cron_dir: Optional[Path] = None) -> None:
    """Record a failed write to ``cron_dir`` (default: the active store); WARN once, when the store
    enters the degraded state. Callers skip the dispatch that needed the write: no job runs
    without a durable advance/fire claim."""
    store = _key(cron_dir)
    with _lock:
        record = _degraded.get(store)
        entered = record is None
        if entered:  # an ended outage not yet seen by a saved scan carries over: keep its start
            ended = _recovered.pop(store, None)
            since = ended.since if ended is not None else time.time()
            record = _degraded[store] = StoreDegraded(store, since, describe_error(exc))
        new_site = site not in record.sites
        record.sites.add(site)
        # A failed dispatch write throttles the next attempt; once dispatch has been tried, a failed
        # scan save is this tick's write attempt too, so the re-probe does not add a second one.
        if site in _DISPATCH_SITES or record.last_probe is not None:
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
        record = _degraded.pop(_key(cron_dir), None)
        if record is not None:
            record.recovered_at = time.time()
            _recovered[record.store] = record
    if record is None:
        return
    logger.warning("Cron store %s is writable again; %d skipped run(s), catching up once per job",
                   record.store, record.skipped_runs)
    _notify("recovered", record)


def degraded_record(cron_dir: Optional[Path] = None) -> Optional[StoreDegraded]:
    return _degraded.get(_key(cron_dir))


def outage_covers(cron_dir: Path, due_at: float, grace: float) -> bool:
    """Whether a run due at ``due_at`` fell inside this store's current or just-ended outage."""
    return any(r is not None and r.since <= due_at + grace and (r.recovered_at is None or due_at <= r.recovered_at)
               for r in (_degraded.get(_key(cron_dir)), _recovered.get(_key(cron_dir))))


def store_key(home) -> str:
    """Record key of a profile home's cron store (a symlinked cron/ resolves elsewhere)."""
    return _key(Path(home) / "cron")


def forget_homes(stores) -> None:
    """Drop the state of ``stores`` (``store_key`` values, resolved while each home existed) whose
    profile this process no longer ticks, so a profile that left this gateway cannot keep the
    host-wide gauges at writable=0."""
    stores = set(stores)
    with _lock:
        for records in (_degraded, _recovered):
            for store in [s for s in records if s in stores]:
                del records[store]


def degraded_records() -> list:
    with _lock:
        return list(_degraded.values())


def dispatch_blocked(due_jobs: list) -> bool:
    """Whether this tick skips advance/claim for ``due_jobs``: the active store is known
    unwritable and the minute-throttled re-probe has not seen it accept a write. Skipped runs are
    recorded; once the probe passes, the dispatch's own save confirms recovery (or re-degrades)."""
    cron_dir = _active_cron_dir()
    record = _degraded.get(_key(cron_dir))
    # Entered in load/scan (last_probe None): dispatch has not been tried yet, so let it try.
    if record is None or record.last_probe is None:
        return False
    now = time.monotonic()
    throttled = now - record.last_probe < PROBE_INTERVAL_SECONDS
    error = None if throttled else probe_store(cron_dir)  # file I/O stays outside the lock
    with _lock:
        if not throttled:
            record.last_probe = now
        if error is not None:
            record.error = describe_error(error)
        blocked = throttled or error is not None
        if blocked:
            record.skipped |= _run_keys(due_jobs)
    return blocked


def end_recovery_window(cron_dir: Path) -> None:
    """A due scan's save landed after the outage ended: normal grace applies again."""
    if _recovered:
        with _lock:
            _recovered.pop(_key(cron_dir), None)


def recheck_idle() -> None:
    """Idle tick (nothing due, so no save will land): re-probe a degraded store at most once a
    minute. The probe only creates an empty file, so a passing probe is confirmed by a real
    jobs.json save, whose ``note_writable`` ends the outage; metrics, notices and the one-shot
    grace gate then stop treating a recovered store as degraded."""
    if not _degraded:
        return
    cron_dir = _active_cron_dir()
    record = _degraded.get(_key(cron_dir))
    now = time.monotonic()
    if record is None or (record.last_probe is not None and now - record.last_probe < PROBE_INTERVAL_SECONDS):
        return
    record.last_probe = now
    error = probe_store(cron_dir)
    if error is None:
        from cron.jobs import _jobs_lock, load_jobs, save_jobs
        try:
            with _jobs_lock():
                save_jobs(load_jobs())
            return
        except OSError as exc:  # e.g. EDQUOT or a read-only jobs.json in a writable dir
            error = exc
    with _lock:
        record.error = describe_error(error)


def probe_report(cron_dir: Path) -> Optional[dict]:
    """Cross-process view for `cron status` / `doctor`: ``None`` when the store accepts writes,
    else the notice fields. ``since`` is jobs.json's mtime (the last write that landed);
    ``skipped`` is left to the caller, which counts the due jobs that have not fired."""
    error = probe_store(cron_dir)
    if error is None:
        return None
    try:
        since = (cron_dir / "jobs.json").stat().st_mtime
    except OSError:
        since = None
    return _report_fields(str(cron_dir), describe_error(error), since)


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
    """Write probe mirroring a jobs.json save: ``None`` when it would land (or the store does not
    exist yet), else the OSError it would hit. Stages a payload the size of the current jobs.json
    (plus a page) where the save stages it (beside a symlinked file's real target), so a full,
    over-quota or read-only target fails here the way the save would."""
    if not cron_dir.is_dir():
        return None
    from utils import mkstemp_beside
    jobs_file = cron_dir / "jobs.json"
    try:
        size = jobs_file.stat().st_size + 4096
    except OSError:
        size = 4096
    # A read-only jobs.json symlink target, or a tick lock the ticker cannot open (root-owned in a
    # writable dir: the tick skips every run while the dir itself still accepts writes).
    for target in (os.path.realpath(jobs_file), str(cron_dir / ".tick.lock")):
        if os.path.exists(target) and not os.access(target, os.W_OK):
            return OSError(errno.EACCES, os.strerror(errno.EACCES), target)
    tmp = None
    chunk = b"\0" * 65536
    try:
        fd, tmp = mkstemp_beside(jobs_file, prefix=".probe_")
        with os.fdopen(fd, "wb") as f:
            for offset in range(0, size, len(chunk)):  # bounded memory for a large jobs.json
                f.write(chunk[:size - offset])
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        return exc
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    return None
