"""The cron tick's worktree prune must not fan its git work out with the profile count (#143255).

A multiplexed gateway ticks every served profile from one process and each due profile gets its own
``cron-worktree-prune`` thread; hundreds of them forked ``git`` side by side until the liveness
watchdog gave up.
"""

import threading
import time

from cron import scheduler
from cron import scheduler_worktree_maintenance as maintenance


def _join_prune_threads(timeout: float = 30.0) -> None:
    """Join the sweeps this test dispatched (they run on daemon threads)."""
    deadline = time.monotonic() + timeout
    for thread in threading.enumerate():
        if thread.name == "cron-worktree-prune":
            thread.join(max(0.0, deadline - time.monotonic()))


def _new_prune_threads(known):
    return [t for t in threading.enumerate()
            if t.name == "cron-worktree-prune" and t.ident not in known]


def test_a_sweep_waits_instead_of_forking_git_beside_the_running_one(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    discoveries = []

    def discover():
        discoveries.append(len(discoveries) + 1)
        entered.set()
        release.wait(30)
        return []

    monkeypatch.setattr(scheduler, "_last_worktree_maintenance_at", None)
    monkeypatch.setattr(scheduler, "_worktree_maintenance_repos", discover)
    # The queue-ahead sweep stays blocked; the one behind it must give up, not run beside it.
    monkeypatch.setattr(maintenance, "SWEEP_WAIT_SECONDS", 0.05, raising=False)

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "first")
    try:
        scheduler._maybe_run_worktree_maintenance()
        assert entered.wait(30), "the first sweep never started"

        known = {t.ident for t in threading.enumerate() if t.name == "cron-worktree-prune"}
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "second")
        scheduler._maybe_run_worktree_maintenance()
        for thread in _new_prune_threads(known):
            thread.join(30)

        assert len(discoveries) == 1, "a second sweep forked git beside the running one"
    finally:
        release.set()
        _join_prune_threads()


def test_a_waiting_sweep_runs_once_the_one_ahead_fails(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    second_ran = threading.Event()
    calls = []

    def discover():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            release.wait(30)
            raise RuntimeError("discovery failed")
        second_ran.set()
        return []

    monkeypatch.setattr(scheduler, "_last_worktree_maintenance_at", None)
    monkeypatch.setattr(scheduler, "_worktree_maintenance_repos", discover)
    monkeypatch.setattr(maintenance, "SWEEP_WAIT_SECONDS", 30.0, raising=False)

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "first")
    try:
        scheduler._maybe_run_worktree_maintenance()
        assert entered.wait(30), "the first sweep never started"
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "second")
        scheduler._maybe_run_worktree_maintenance()
        release.set()
        # Serialising must queue the sweep behind, not lose it when the one ahead raises.
        assert second_ran.wait(30), "the waiting sweep never ran after the first one failed"
    finally:
        release.set()
        _join_prune_threads()

