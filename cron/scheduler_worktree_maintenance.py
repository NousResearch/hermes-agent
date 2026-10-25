"""Per-repo claim for the cron worktree prune.

Served profiles mostly resolve the same repos (the hermes checkout at least), and every prune
pass ends in a footprint probe of the whole ``.worktrees/``, so a per-profile pass multiplied that
disk walk by the profile count at the same second. One process claims each repo once per interval.
"""

import os
import threading
import time

_last_worktree_prune_at: dict[str, float] = {}  # resolved repo root -> monotonic stamp
_lock = threading.Lock()


def claim_due_worktree_repos(repos: list[str], interval_s: float) -> list[str]:
    """The subset of *repos* no served profile pruned within *interval_s*, stamped as claimed."""
    now = time.monotonic()
    due = []
    with _lock:
        for repo in repos:
            key = os.path.realpath(repo)
            last = _last_worktree_prune_at.get(key)
            if last is None or now - last >= interval_s:
                _last_worktree_prune_at[key] = now
                due.append(repo)
    return due
