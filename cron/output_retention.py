"""F-008 — Rotation + size cap for per-job cron output dirs.

Without rotation, `~/.hermes/cron/output/{job_id}/` grows unbounded:
a 10-minute-interval job producing ~10 KB output consumes ~500 MB/year;
five of them exhaust the typical laptop ~/ partition in a few years of
unattended operation.

The retention policy is: for each per-job output dir, keep files whose
mtime is within `MAX_AGE_DAYS` AND whose total size is within
`MAX_SIZE_BYTES_PER_JOB`. When either cap is exceeded, oldest files are
deleted first until both caps are satisfied.

This module is intentionally self-contained and side-effect-minimal:
importing it does nothing; call `rotate_all()` or `rotate_job(job_id)`
on a cadence from the scheduler, or via `hermes cron gc`.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_AGE_DAYS = 30
MAX_SIZE_BYTES_PER_JOB = 1 * 1024 * 1024 * 1024  # 1 GB per job
# Minimum interval between rotations for a given job, in seconds. Called
# opportunistically after each successful run; this damper prevents hot-firing.
MIN_INTERVAL_SECONDS = 3600


def _output_root() -> Path:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(home) / "cron" / "output"


def _job_dir(job_id: str) -> Path:
    return _output_root() / job_id


# Per-job last-rotation timestamps, in memory. Not persisted — on process
# restart we re-rotate, which is safe (rotation is idempotent).
_last_rotation: dict[str, float] = {}


def _list_files_oldest_first(dir_: Path) -> List[Tuple[Path, float, int]]:
    """Return (path, mtime, size) triples, oldest first."""
    entries: List[Tuple[Path, float, int]] = []
    if not dir_.exists():
        return entries
    for child in dir_.iterdir():
        if not child.is_file():
            continue
        try:
            st = child.stat()
        except OSError:
            continue
        entries.append((child, st.st_mtime, st.st_size))
    entries.sort(key=lambda t: t[1])
    return entries


def rotate_job(
    job_id: str,
    *,
    max_age_days: int = MAX_AGE_DAYS,
    max_size_bytes: int = MAX_SIZE_BYTES_PER_JOB,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Rotate one job's output dir. Returns (files_deleted, bytes_reclaimed)."""
    dir_ = _job_dir(job_id)
    files = _list_files_oldest_first(dir_)
    if not files:
        return (0, 0)

    now = time.time()
    age_cutoff = now - (max_age_days * 86400)

    deleted = 0
    reclaimed = 0

    # Pass 1: age cutoff. Keep files newer than cutoff; drop older ones.
    survivors: List[Tuple[Path, float, int]] = []
    for path, mtime, size in files:
        if mtime < age_cutoff:
            if not dry_run:
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("cron retention: unlink failed for %s: %s", path, exc)
                    survivors.append((path, mtime, size))
                    continue
            deleted += 1
            reclaimed += size
        else:
            survivors.append((path, mtime, size))

    # Pass 2: size cutoff. If survivors still exceed the cap, delete oldest
    # until under cap.
    total_size = sum(s for _, _, s in survivors)
    i = 0
    while total_size > max_size_bytes and i < len(survivors):
        path, mtime, size = survivors[i]
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("cron retention: unlink failed for %s: %s", path, exc)
                i += 1
                continue
        total_size -= size
        deleted += 1
        reclaimed += size
        i += 1

    if deleted:
        logger.info(
            "cron retention: job=%s deleted=%d reclaimed=%d bytes (dry_run=%s)",
            job_id, deleted, reclaimed, dry_run,
        )
    return (deleted, reclaimed)


def rotate_all(*, dry_run: bool = False) -> Tuple[int, int]:
    """Rotate every job dir under ~/.hermes/cron/output/. Returns totals."""
    root = _output_root()
    if not root.exists():
        return (0, 0)
    total_deleted = 0
    total_reclaimed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        d, r = rotate_job(child.name, dry_run=dry_run)
        total_deleted += d
        total_reclaimed += r
    return (total_deleted, total_reclaimed)


def maybe_rotate_after_run(job_id: str) -> None:
    """Opportunistic rotation called after a successful job run.

    Damped so we rotate at most once per MIN_INTERVAL_SECONDS per job.
    Swallows all errors — retention must never affect job execution.
    """
    now = time.time()
    last = _last_rotation.get(job_id, 0.0)
    if now - last < MIN_INTERVAL_SECONDS:
        return
    _last_rotation[job_id] = now
    try:
        rotate_job(job_id)
    except Exception as exc:  # pragma: no cover — best-effort
        logger.warning("maybe_rotate_after_run failed for %s: %s", job_id, exc)
