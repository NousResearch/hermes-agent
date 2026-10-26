"""Process-wide coordination for the cron tick's worktree prune (``cron.scheduler`` runs in here).

One process ticks every served profile (``cron/AGENTS.md``), so the per-profile throttle in
``cron.scheduler`` legitimately lets each profile decide it is due, and each due profile gets its
own ``cron-worktree-prune`` thread. Those threads used to fork ``git`` side by side, which starved a
338-profile gateway's event loop until the liveness watchdog exited 75 (#143255). Two guards:

- **One sweep at a time.** The tick is sequential, the sweeps it dispatches are not; serialising
  them (discovery included) keeps the number of concurrent ``git`` children at one regardless of
  profile count. A waiting sweep holds no resources, so the wait is capped rather than unbounded.
- **One prune per repo per interval.** Served profiles mostly resolve the same repos (the hermes
  checkout at least), and every prune pass ends in a footprint probe of the whole ``.worktrees/``,
  so the second profile to reach a shared repo within the interval has nothing to do.
"""

import logging
import os
import threading
import time
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Bounded so a wedged sweep cannot park a profile's thread forever; a normal sweep is one git pass
# over a handful of repos.
SWEEP_WAIT_SECONDS = 120.0

_sweep_slot = threading.Semaphore(1)
_last_worktree_prune_at: dict[str, float] = {}  # canonical repo path -> monotonic stamp
_lock = threading.Lock()


def claim_sweep_slot(wait_seconds: Optional[float] = None) -> bool:
    """Take the process-wide prune slot, waiting for the sweep ahead of us (bounded)."""
    return _sweep_slot.acquire(timeout=SWEEP_WAIT_SECONDS if wait_seconds is None else wait_seconds)


def release_sweep_slot() -> None:
    _sweep_slot.release()


def claim_due_worktree_repos(repos: Iterable[str], interval_s: float) -> list[str]:
    """The subset of *repos* no served profile pruned within *interval_s*, stamped as claimed.

    Keyed on the path ``git`` resolves to, case-folded where the filesystem is (Windows).
    """
    now = time.monotonic()
    due = []
    with _lock:
        for repo in repos:
            key = os.path.normcase(os.path.realpath(repo))
            last = _last_worktree_prune_at.get(key)
            if last is None or now - last >= interval_s:
                _last_worktree_prune_at[key] = now
                due.append(repo)
    return due


def run_sweep(discover: Callable[[], Iterable[str]], *, interval_seconds: float,
              wait_seconds: Optional[float] = None) -> None:
    """Prune the repos ``discover()`` reports that are due, from the cron tick's daemon thread.

    ``discover`` stays in the caller so a per-profile repo lookup (and anything a test patches) is
    read where it is defined. Errors never propagate: GC is hygiene, not scheduling.
    """
    if not claim_sweep_slot(wait_seconds):
        logger.debug("Cron worktree maintenance skipped: another sweep is still running")
        return
    try:
        repos = claim_due_worktree_repos(discover() or [], interval_seconds)
        if not repos:
            return
        from hermes_cli.worktree_ops import _prune_stale_worktrees

        for repo in repos:
            try:
                _prune_stale_worktrees(repo)
            except Exception:
                logger.debug("Cron worktree maintenance failed for %s", repo, exc_info=True)
    except Exception:
        logger.debug("Cron worktree maintenance skipped", exc_info=True)
    finally:
        release_sweep_slot()
