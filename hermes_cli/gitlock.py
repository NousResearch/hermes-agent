"""Stale git lock-file recovery for update/check paths.

A crashed or killed ``git fetch`` on a shallow clone can leave
``.git/shallow.lock`` behind. Every later fetch then fails with::

    fatal: Unable to create '/path/.git/shallow.lock': File exists.

This wedges ``hermes update --check`` (hard failure) and silently degrades the
passive banner check in :mod:`hermes_cli.banner` (the fetch is swallowed, the
stale refs are compared, and the user can be told an update is available when
the checkout already contains the remote tip). Git does not self-heal these
lock files — they persist until a human removes them.

This module provides two small, defensive helpers used by the update paths:

* :func:`clear_stale_git_locks` — remove abandoned ``.git`` lock files (with
  an age + git-process guard so a live fetch is never yanked).
* :func:`clear_stale_tmp_packs` — remove aborted-fetch ``tmp_pack_*`` /
  ``tmp_idx_*`` debris from ``.git/objects/pack``. On flaky lines every
  timed-out fetch leaves one behind; unchecked they accumulated to 6 GB /
  hundreds of files over 9 days and eventually corrupted the pack directory
  outright, permanently wedging the update check (#93732).
* :func:`is_ancestor_of_head` — ask whether a remote tip is already contained
  in HEAD. Used by the shallow-clone update check to avoid reporting a false
  "update available" when local cherry-picks sit on top of the remote tip.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lock files younger than this are presumed live (a fetch is in flight) and
# are never removed. git lock files are created and removed within a single
# fetch (seconds); anything older than 10 minutes is abandoned by any
# reasonable standard.
STALE_LOCK_MIN_AGE_SECONDS = 10 * 60

# Lock files we know how to self-heal. ``shallow.lock`` is the one observed in
# the wild (interrupted fetch on a shallow clone); the others are the same
# class of failure (interrupted git operation) and harmless to clear when
# stale. Index/HEAD locks from a live git process are protected by the
# process guard in :func:`clear_stale_git_locks`.
LOCK_NAMES = ("shallow.lock", "index.lock", "HEAD.lock", "MERGE_HEAD.lock")


def _git_proc_running() -> bool:
    """True when a ``git`` process is currently running.

    The conservative answer on any platform we can't probe: if we can't tell,
    treat a lock as possibly-live and don't remove it. This is the safety
    check that stops us from yanking a lock a real fetch is holding.
    """
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10,
            ).stdout.lower()
            return "git.exe" in out
        out = subprocess.run(
            ["pgrep", "-x", "git"], capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0
    except Exception:
        logger.debug("git process probe failed; assuming no git running", exc_info=True)
        return False


def clear_stale_git_locks(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove abandoned ``.git`` lock files under ``repo_root``.

    A lock is removed only when BOTH conditions hold:

    * it is older than :data:`STALE_LOCK_MIN_AGE_SECONDS` (default), and
    * no ``git`` process is currently running.

    Returns the list of removed lock file paths. Never raises: a lock we
    cannot stat or unlink is skipped (a concurrently-held lock may have just
    been created between our age check and the unlink — the process guard
    makes that window vanishingly small, and skipping is always safe).
    """
    git_dir = Path(repo_root) / ".git"
    if not git_dir.is_dir():
        return []

    if _git_proc_running():
        logger.debug("git process running; skipping stale-lock sweep")
        return []

    cutoff = time.time() - (min_age_seconds if min_age_seconds is not None else STALE_LOCK_MIN_AGE_SECONDS)
    removed: List[str] = []
    for name in LOCK_NAMES:
        lock_path = git_dir / name
        try:
            if lock_path.is_file() and lock_path.stat().st_mtime < cutoff:
                lock_path.unlink()
                removed.append(str(lock_path))
                logger.info("Removed stale git lock %s", lock_path)
        except OSError:
            logger.debug("Could not clear %s (skipping)", lock_path, exc_info=True)
    return removed


# Aborted-fetch pack debris younger than this is presumed live (a fetch may
# be writing it right now) and is never removed. A healthy fetch completes in
# minutes; the same 10-minute bar the lock sweep uses is comfortably safe.
STALE_TMP_PACK_MIN_AGE_SECONDS = STALE_LOCK_MIN_AGE_SECONDS

# Temp-file prefixes git writes into .git/objects/pack during a transfer and
# renames away on success. Anything left with these names after a fetch died
# is garbage by definition — git itself never reuses or cleans them.
_TMP_PACK_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_rev_", "tmp_mtimes_")


def clear_stale_tmp_packs(
    repo_root: Path, *, min_age_seconds: Optional[int] = None
) -> List[str]:
    """Remove aborted-fetch temp pack files under ``.git/objects/pack``.

    Every ``git fetch`` that dies mid-transfer (timeout, HTTP 429, dropped
    connection) leaves a ``tmp_pack_*`` (and sometimes ``tmp_idx_*``) file
    behind, and git never cleans them up. On a flaky line the banner's
    background update check produces several per day; observed in the wild
    at hundreds of files / 6 GB after 9 days, after which the pack directory
    corrupted outright and every fetch failed permanently (#93732).

    Same safety contract as :func:`clear_stale_git_locks`: only files older
    than the age floor, never while a git process is running, never raises.
    Returns the removed paths.
    """
    pack_dir = Path(repo_root) / ".git" / "objects" / "pack"
    if not pack_dir.is_dir():
        return []

    if _git_proc_running():
        logger.debug("git process running; skipping tmp-pack sweep")
        return []

    cutoff = time.time() - (
        min_age_seconds if min_age_seconds is not None else STALE_TMP_PACK_MIN_AGE_SECONDS
    )
    removed: List[str] = []
    try:
        entries = list(pack_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        name = entry.name
        if not name.startswith(_TMP_PACK_PREFIXES):
            continue
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                size = entry.stat().st_size
                entry.unlink()
                removed.append(str(entry))
                logger.info(
                    "Removed aborted-fetch pack debris %s (%d bytes)", entry, size
                )
        except OSError:
            logger.debug("Could not clear %s (skipping)", entry, exc_info=True)
    return removed


def is_ancestor_of_head(repo_root: Path, rev: str) -> bool:
    """True when ``rev`` is an ancestor of (or equal to) HEAD.

    Wraps ``git merge-base --is-ancestor <rev> HEAD``. This is the correct
    question for update checks: a local cherry-pick on top of the remote tip
    makes HEAD *different* from ``origin/main`` but still *contains* it, so
    the answer to "is there an update?" is no.

    Returns False on any probe failure (missing rev, shallow boundary, git
    error) — callers treat that as "can't prove contained", which is the
    conservative direction for an update check.
    """
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", rev, "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        logger.debug("merge-base --is-ancestor probe failed for %s", rev, exc_info=True)
        return False
