"""Stale git lock-file and aborted-fetch pack-debris recovery for update/check paths.

A killed ``git fetch`` can leave ``.git/shallow.lock`` behind (every later fetch fails with "Unable to
create '.../shallow.lock': File exists") and ``tmp_pack_*`` files git itself never cleans up."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Files younger than this are presumed live (a fetch may be in flight) and are never removed. Lock
# files live for seconds and a healthy fetch completes in minutes; 10 minutes is abandoned.
STALE_LOCK_MIN_AGE_SECONDS = 10 * 60
STALE_TMP_PACK_MIN_AGE_SECONDS = STALE_LOCK_MIN_AGE_SECONDS
# ``shallow.lock`` is the one observed in the wild; the others are the same class of failure
# (interrupted git operation). Locks held by a live git process are protected by the process guard.
LOCK_NAMES = ("shallow.lock", "index.lock", "HEAD.lock", "MERGE_HEAD.lock")
# Temp-file prefixes git writes into .git/objects/pack during a transfer and renames away on
# success; anything left with these names after a fetch died is garbage by definition.
_TMP_PACK_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_rev_", "tmp_mtimes_")


def _git_proc_running() -> bool:
    """True when a ``git`` process is running — the check that stops us yanking a lock a live fetch holds.

    A failed probe logs and returns False; the age floor in the sweep still applies.
    """
    try:
        if os.name == "nt":
            proc = subprocess.run(["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"],
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            return "git.exe" in proc.stdout.lower()
        proc = subprocess.run(["pgrep", "-x", "git"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=10)
        return proc.returncode == 0
    except Exception:
        logger.debug("git process probe failed; assuming no git running", exc_info=True)
        return False


def _sweep_stale(directory: Path, candidates: Callable[[], Iterable[Path]], *, min_age_seconds: Optional[int],
                 default_age: int, skip_msg: str, log_removed: Callable[[Path, int], None]) -> List[str]:
    """Shared guard + age-floor sweep. Never raises; skips anything it cannot stat/unlink."""
    if not directory.is_dir():
        return []
    if _git_proc_running():
        logger.debug(skip_msg)
        return []
    cutoff = time.time() - (min_age_seconds if min_age_seconds is not None else default_age)
    removed: List[str] = []
    for entry in candidates():
        try:
            if entry.is_file() and (st := entry.stat()).st_mtime < cutoff:
                entry.unlink()
                removed.append(str(entry))
                log_removed(entry, st.st_size)
        except OSError:
            logger.debug("Could not clear %s (skipping)", entry, exc_info=True)
    return removed


def clear_stale_git_locks(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove abandoned ``.git`` lock files under ``repo_root``; returns the removed paths.

    Removes only when older than the age floor AND no git process is running. Never raises: a lock we cannot
    stat/unlink is skipped (it may have been re-created between the age check and the unlink; skipping is safe).
    """
    git_dir = Path(repo_root) / ".git"
    return _sweep_stale(
        git_dir, lambda: [git_dir / name for name in LOCK_NAMES],
        min_age_seconds=min_age_seconds, default_age=STALE_LOCK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping stale-lock sweep",
        log_removed=lambda p, _size: logger.info("Removed stale git lock %s", p),
    )


def clear_stale_tmp_packs(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> List[str]:
    """Remove aborted-fetch temp pack files under ``.git/objects/pack``; same contract as clear_stale_git_locks."""
    pack_dir = Path(repo_root) / ".git" / "objects" / "pack"

    def _candidates():
        try:
            return [e for e in pack_dir.iterdir() if e.name.startswith(_TMP_PACK_PREFIXES)]
        except OSError:
            return []

    return _sweep_stale(
        pack_dir, _candidates,
        min_age_seconds=min_age_seconds, default_age=STALE_TMP_PACK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping tmp-pack sweep",
        log_removed=lambda p, size: logger.info("Removed aborted-fetch pack debris %s (%d bytes)", p, size),
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----


def _git_stdout_lines(repo_root: Path, args: List[str]) -> List[str]:
    """Run a read-only git query in ``repo_root``; [] on any failure."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(repo_root),
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        logger.debug("git query failed: %s", args, exc_info=True)
        return []


def _batch_missing_parents(repo_root: Path, candidates: List[str]) -> set[str]:
    """Return local commit objects whose parent objects are missing."""
    if not candidates:
        return set()
    try:
        parents_by_commit = {}
        parents = set()
        request = "\n".join(candidates) + "\n"
        result = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=str(repo_root),
            input=request.encode(),
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return set()
        data = result.stdout
        cursor = 0
        for candidate in candidates:
            header_end = data.find(b"\n", cursor)
            if header_end < 0:
                return set()
            header = data[cursor:header_end].split()
            cursor = header_end + 1
            if len(header) >= 3 and header[1] == b"commit":
                size = int(header[2])
                body = data[cursor:cursor + size]
                cursor += size
                if data[cursor:cursor + 1] != b"\n":
                    return set()
                cursor += 1
                decoded = body.decode(errors="replace")
                commit_parents = {
                    fields[1]
                    for line in decoded.splitlines()
                    for fields in [line.split()]
                    if len(fields) >= 2 and fields[0] == "parent"
                }
                parents_by_commit[candidate] = commit_parents
                parents.update(commit_parents)
            elif len(header) < 2 or header[1] != b"missing":
                return set()
        if not parents:
            return set()
        check = subprocess.run(
            ["git", "cat-file", "--batch-check"],
            cwd=str(repo_root),
            input=("\n".join(sorted(parents)) + "\n").encode(),
            capture_output=True,
            timeout=10,
        )
        missing = {
            line.split()[0]
            for line in check.stdout.decode(errors="replace").splitlines()
            if line.endswith(" missing")
        }
        return {commit for commit, commit_parents in parents_by_commit.items() if commit_parents & missing}
    except Exception:
        logger.debug("parent-object probe failed for %s", repo_root, exc_info=True)
        return set()


def repair_broken_shallow_boundaries(repo_root: Path) -> int:
    """Repair reflog-only shallow boundaries dropped by #108286's prune bug.

    This complements the prevention fix in PR #108290: existing corrupted installs need
    boundaries reconstructed because their broken history prevents the reflogs from expiring.
    """
    try:
        shallow_rel = _git_stdout_lines(repo_root, ["rev-parse", "--git-path", "shallow"])
        if not shallow_rel:
            return 0
        shallow_path = Path(shallow_rel[0])
        if not shallow_path.is_absolute():
            shallow_path = Path(repo_root) / shallow_path
        if not shallow_path.is_file():
            return 0
        original = shallow_path.read_text(encoding="utf-8")
        existing = {line for line in original.splitlines() if line}
        if not existing:
            return 0
        candidates = _git_stdout_lines(repo_root, ["cat-file", "--batch-all-objects", "--batch-check"])
        candidates = sorted({line.split()[0] for line in candidates
                             if len(line.split()) >= 2 and line.split()[1] == "commit"})
        candidates = sorted(set(candidates + _git_stdout_lines(
            repo_root, ["reflog", "show", "--all", "--format=%H"])))
        broken = _batch_missing_parents(repo_root, candidates)
        repaired = broken - existing
        if not repaired:
            return 0
        tmp_path = shallow_path.with_name(shallow_path.name + ".hermes-repair")
        tmp_path.write_text("\n".join(sorted(existing | repaired)) + "\n", encoding="utf-8")
        os.replace(tmp_path, shallow_path)
        if not _git_stdout_lines(repo_root, ["rev-list", "--count", "--all", "--reflog"]):
            shallow_path.write_text(original, encoding="utf-8")
            return 0
        logger.info("Restored %d broken shallow boundary(ies) in %s", len(repaired), repo_root)
        return len(repaired)
    except Exception:
        logger.debug("shallow boundary repair failed for %s", repo_root, exc_info=True)
        return 0


def prune_stale_shallow_grafts(repo_root: Path) -> int:
    """Drop ``.git/shallow`` graft lines no live ref still points at (#105951).

    Every ``git fetch --depth 1`` appends the fetched tip to ``.git/shallow`` as a new
    graft and never removes the previous one, so a long-lived shallow installer checkout
    accumulates one graft per update check (57 observed in the wild). The stale grafts
    break ``merge-base`` and push ``hermes update`` into the orphan-divergence reset path
    on every run. Keep only the boundaries that still protect referenced tips (HEAD,
    FETCH_HEAD, and every ref tip): the dropped commits are already unreachable and their
    objects are left for ``git gc``. Returns the number of graft lines removed; never
    raises, and restores the original file if the trimmed set breaks history walking.
    """
    try:
        shallow_rel = _git_stdout_lines(repo_root, ["rev-parse", "--git-path", "shallow"])
        if not shallow_rel:
            return 0
        shallow_path = Path(shallow_rel[0])
        if not shallow_path.is_absolute():
            shallow_path = Path(repo_root) / shallow_path
        if not shallow_path.is_file():
            return 0
        lines = [line for line in shallow_path.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            return 0
        keep = set(lines) & {
            *(_git_stdout_lines(repo_root, ["rev-parse", "HEAD"]) or []),
            *(_git_stdout_lines(repo_root, ["rev-parse", "--verify", "--quiet", "FETCH_HEAD"]) or []),
            *_git_stdout_lines(repo_root, ["for-each-ref", "--format=%(objectname)"]),
        }
        if len(keep) == len(lines):
            return 0
        original = shallow_path.read_text(encoding="utf-8")
        tmp_path = shallow_path.with_name(shallow_path.name + ".hermes-prune")
        tmp_path.write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
        os.replace(tmp_path, shallow_path)
        # Fail-safe: if any reachable walk now crosses a boundary we wrongly removed,
        # put the grafts back — a growing file beats a broken repo.
        still_walks = _git_stdout_lines(repo_root, ["rev-list", "--count", "HEAD"]) and \
            _git_stdout_lines(repo_root, ["rev-list", "--count", "--all"])
        if not still_walks:
            shallow_path.write_text(original, encoding="utf-8")
            logger.debug("shallow prune self-check failed; grafts restored")
            return 0
        logger.info("Pruned %d stale shallow graft(s) in %s", len(lines) - len(keep), repo_root)
        return len(lines) - len(keep)
    except Exception:
        logger.debug("shallow graft prune failed for %s", repo_root, exc_info=True)
        return 0
