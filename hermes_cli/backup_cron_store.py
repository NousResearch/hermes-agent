"""Cron job-store helpers for backups: the jobs.json + runtime.db pair and the post-update restore.

One profile's cron store is split across two files (``cron/runtime_state.py``). Backups, quick
snapshots and the post-update safety net copy or restore them together under that store's
``_jobs_lock()`` so they never form a pair that did not exist on disk at one moment.
"""
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from hermes_cli.backup_restore import _count_cron_jobs

logger = logging.getLogger("hermes_cli.backup")

_CRON_JOBS_REL = "cron/jobs.json"
# What reading a cron store can raise: an unreadable or corrupted file (OSError, a JSON or
# encoding ValueError), a corrupted runtime row or journal (RuntimeError), a failing database.
_CRON_STORE_READ_ERRORS = (OSError, ValueError, RuntimeError, sqlite3.Error)


@contextmanager
def _cron_store_lock(home: Path):
    """``home``'s cron job-store lock (cron.jobs._jobs_lock, scoped to that profile's store)."""
    from cron.jobs import _jobs_lock, use_cron_store

    with use_cron_store(home), _jobs_lock():
        yield


# One cron job store split in two files (cron/runtime_state.py): copied apart, they can form a pair
# that never existed together — a definition whose runtime row a removal already deleted.
_CRON_STORE_PAIR = (_CRON_JOBS_REL, "cron/runtime.db")


def _cron_pair_home(root: Path, rel: Any) -> Optional[Path]:
    """The home whose cron store pair ``rel`` (relative to ``root``) belongs to: ``root`` itself, or
    a named profile under ``root/profiles/`` (a full backup of the hermes root holds every
    profile's store); ``None`` for every other file."""
    parts = Path(rel).parts
    if "/".join(parts[-2:]) not in _CRON_STORE_PAIR:
        return None
    prefix = parts[:-2]
    if not prefix or (len(prefix) == 2 and prefix[0] == "profiles"):
        return root.joinpath(*prefix)
    return None


def _copy_with_cron_pairs_locked(root: Path, entries, copy) -> None:
    """``copy(abs_path, rel)`` for every ``(abs_path, rel)`` entry, in order, except that each cron
    store's pair (cron/jobs.json, cron/runtime.db) is copied back to back, at its first member's
    position, under that store's ``_jobs_lock()``. Everything else — state.db,
    cron/executions.db, cron/output/ — is copied without the lock, so it is held for two small
    files, never across a large one: a hold longer than ``_jobs_lock``'s 30 s wait would let
    another process's removal proceed degraded between the two copies.

    Pair membership is decided under the lock, from the disk, not from the caller's earlier scan:
    a concurrent first load can migrate a pre-split store in between, stripping jobs.json and
    creating a runtime.db the scan never listed. A pair file present on disk is copied even if
    the scan missed it; one that has disappeared is skipped. A store that is still pre-split
    (no runtime.db: jobs.json carries all state) is copied exactly as before."""
    pairs: dict[Path, list] = {}
    for entry in entries:
        home = _cron_pair_home(root, entry[1])
        if home is not None:
            pairs.setdefault(home, []).append(entry)
    for abs_path, rel in entries:
        home = _cron_pair_home(root, rel)
        if home is None:
            copy(abs_path, rel)
            continue
        members = pairs.pop(home, None)
        if members is None:  # this store's pair was already copied
            continue
        prefix = Path(members[0][1]).parent.parent
        with _cron_store_lock(home):
            for pair_rel in _CRON_STORE_PAIR:
                member_path = home / pair_rel
                if member_path.is_file():
                    copy(member_path, (prefix / pair_rel).as_posix()
                         if isinstance(members[0][1], str) else prefix / pair_rel)


def _restore_cron_jobs_with_runtime(home: Path, snap_path: Path, live_path: Path) -> None:
    """Restore ``live_path`` (cron/jobs.json) from a snapshot, under the store's lock, together
    with the scheduler state of the jobs it brings back.

    After the definition/runtime split a snapshot's jobs.json carries no run state: copied alone,
    a lost job would return with its repeat progress, claims, next run and run history forgotten
    whenever its live runtime row was deleted with it. So every snapshot job with no row in the
    live runtime.db gets its snapshot row folded back into its record
    as legacy scheduler fields (``cron.jobs._merge_job``), which the next ``load_jobs()`` migrates
    into runtime.db. A job with a live row keeps it: live state is newer than the snapshot's. A
    pre-split snapshot (no runtime.db) already carries the state in its jobs.json, and an
    unreadable runtime.db falls back to the plain copy with a warning.

    Every path, the plain copy included, writes jobs.json while holding the lock that the live
    rows were read under: a removal slipping in between would otherwise have its definition
    brought back without its runtime row."""
    import copy

    from cron.jobs import _merge_job, _parse_jobs_file, _write_definitions_file
    from cron.runtime_state import RUNTIME_DB_NAME, read_store

    snap_rows: dict[str, Any] = {}
    records: Any = None
    if (snap_path.parent / RUNTIME_DB_NAME).is_file():
        try:
            snap_rows = read_store(snap_path.parent)[0]
            data, _ = _parse_jobs_file(snap_path)
            records = data.get("jobs") if isinstance(data, dict) else data
        except _CRON_STORE_READ_ERRORS as exc:
            logger.warning(
                "Restoring cron/jobs.json without its run state: the snapshot's %s is unreadable (%s)",
                RUNTIME_DB_NAME, exc)
            snap_rows = {}
    folded = 0
    with _cron_store_lock(home):
        restored = []
        if snap_rows and isinstance(records, list):
            try:
                live_rows = read_store(live_path.parent)[0]
            except _CRON_STORE_READ_ERRORS as exc:
                logger.warning(
                    "Restoring cron/jobs.json without its run state: the live %s is unreadable (%s)",
                    RUNTIME_DB_NAME, exc)
            else:
                for record in records:
                    job_id = str(record.get("id") or "") if isinstance(record, dict) else ""
                    if job_id and job_id not in live_rows and job_id in snap_rows:
                        record = _merge_job(copy.deepcopy(record), copy.deepcopy(snap_rows[job_id]))
                        folded += 1
                    restored.append(record)
        if folded:
            _write_definitions_file(live_path, restored)
        else:
            shutil.copy2(snap_path, live_path)
    if folded:
        logger.warning("Restored the run state of %d cron job(s) from the snapshot's %s",
                       folded, RUNTIME_DB_NAME)


def restore_cron_jobs_if_emptied(
    snapshot_id: str,
    hermes_home: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Safety net for silent cron-job loss across ``hermes update``.

    Config-version migrations have been observed to leave ``cron/jobs.json``
    valid-but-empty after an update, silently dropping every scheduled job
    (issue #34600). The desktop scheduler can also overwrite the file with its
    own small set of internally-tracked crons, causing partial loss (issue
    #52144).

    This compares the *current* job count against the pre-update snapshot. If
    the live file now has **fewer** jobs than the snapshot, the snapshot copy
    of ``cron/jobs.json`` is restored in place, with the snapshot's runtime.db
    state for the jobs it brings back (``_restore_cron_jobs_with_runtime``).

    The check is deliberately conservative — it only ever restores when there
    is unambiguous evidence of loss (snapshot had more jobs than live file),
    so a user who genuinely deleted jobs during/after the update is never
    second-guessed, and an unreadable live file (count ``None``) is left
    untouched so real corruption still surfaces.

    Args:
        snapshot_id: The pre-update quick-snapshot id (from
            :func:`create_quick_snapshot`).
        hermes_home: Override for the Hermes home directory (tests).

    Returns:
        ``None`` when no action was taken (the common, healthy path). On a
        successful restore, a dict ``{"restored": True, "job_count": N,
        "snapshot_id": ...}`` so the caller can warn the user.
    """
    if not snapshot_id:
        return None
    from hermes_cli.backup import _quick_snapshot_root

    home = hermes_home or get_hermes_home()
    live_path = home / _CRON_JOBS_REL

    live_count = _count_cron_jobs(live_path)
    # ``None`` (missing or unparseable) is intentionally left alone — that's a
    # different failure mode the user should see rather than have papered over.
    if live_count is None:
        return None

    snap_path = _quick_snapshot_root(home) / snapshot_id / _CRON_JOBS_REL
    snap_count = _count_cron_jobs(snap_path)
    if not snap_count:  # None or 0 — nothing worth restoring
        return None

    # Restore when live has FEWER jobs than the pre-update snapshot.
    # Catches both total loss (0 vs N) and partial loss (1 vs 19) — the
    # desktop scheduler can overwrite jobs.json with its own small set of
    # internally-tracked crons after an update/restart.
    if live_count >= snap_count:
        return None

    try:
        live_path.parent.mkdir(parents=True, exist_ok=True)
        _restore_cron_jobs_with_runtime(home, snap_path, live_path)
    except (OSError, PermissionError) as exc:
        logger.error(
            "Cron jobs were emptied during update but auto-restore failed: %s", exc
        )
        return None

    logger.warning(
        "Restored %d cron job(s) from pre-update snapshot %s "
        "(live file had %d job(s), snapshot had %d — jobs were lost during migration)",
        snap_count,
        snapshot_id,
        live_count,
        snap_count,
    )
    return {"restored": True, "job_count": snap_count, "snapshot_id": snapshot_id}
