"""The cron tick's worktree prune must not fan its git work out with the profile count (#143255).

A multiplexed gateway ticks every served profile from one process, and each of those ticks pruned
the SAME repos (the Hermes checkout is in the list for every profile) — hundreds of concurrent
``cron-worktree-prune`` threads forking ``git`` against one repo until the liveness watchdog gave up.
"""

import threading
import time

import pytest

from cron import scheduler

try:  # the base revision has no process-wide coordination layer
    from cron import scheduler_worktree_maintenance as maintenance
except ImportError:  # pragma: no cover - only on the base revision
    maintenance = None


def _join_prune_threads(timeout: float = 30.0) -> None:
    """Join the sweeps this test dispatched (they run on daemon threads)."""
    deadline = time.monotonic() + timeout
    for thread in threading.enumerate():
        if thread.name == "cron-worktree-prune":
            thread.join(max(0.0, deadline - time.monotonic()))


def _new_prune_threads(known):
    return [t for t in threading.enumerate()
            if t.name == "cron-worktree-prune" and t.ident not in known]


def test_a_shared_repo_is_pruned_once_however_many_profiles_tick(tmp_path, monkeypatch):
    repo = tmp_path / "shared-repo"
    (repo / ".worktrees").mkdir(parents=True)
    pruned = []

    monkeypatch.setattr(scheduler, "_last_worktree_maintenance_at", None)
    monkeypatch.setattr(scheduler, "_worktree_maintenance_repos", lambda: [str(repo)])
    monkeypatch.setattr("hermes_cli.worktree_ops._prune_stale_worktrees", pruned.append)

    for index in range(3):
        home = tmp_path / f"profile-{index}"
        home.mkdir()
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda home=home: home)
        scheduler._maybe_run_worktree_maintenance()

    _join_prune_threads()

    assert pruned == [str(repo)], "every profile's tick pruned the same shared repo"


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
    if maintenance is not None:
        # The queue-ahead sweep stays blocked; the one behind it must give up, not run beside it.
        monkeypatch.setattr(maintenance, "SWEEP_WAIT_SECONDS", 0.05)

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


def test_a_profiles_own_repo_is_still_pruned(tmp_path, monkeypatch):
    first = tmp_path / "first-repo"
    second = tmp_path / "second-repo"
    for repo in (first, second):
        (repo / ".worktrees").mkdir(parents=True)
    pruned = []
    current = {"repos": [str(first)]}

    monkeypatch.setattr(scheduler, "_last_worktree_maintenance_at", None)
    monkeypatch.setattr(scheduler, "_worktree_maintenance_repos", lambda: current["repos"])
    monkeypatch.setattr("hermes_cli.worktree_ops._prune_stale_worktrees", pruned.append)

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "a")
    scheduler._maybe_run_worktree_maintenance()
    _join_prune_threads()

    current["repos"] = [str(second)]
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "b")
    scheduler._maybe_run_worktree_maintenance()
    _join_prune_threads()

    assert pruned == [str(first), str(second)]


def test_the_sweep_slot_is_released_when_discovery_raises(tmp_path, monkeypatch):
    module = maintenance
    if module is None:
        pytest.skip("no process-wide sweep slot on this revision")
        return

    def discover():
        raise RuntimeError("repo lookup blew up")

    monkeypatch.setattr(scheduler, "_last_worktree_maintenance_at", None)
    monkeypatch.setattr(scheduler, "_worktree_maintenance_repos", discover)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "a")

    scheduler._maybe_run_worktree_maintenance()
    _join_prune_threads()

    assert maintenance.claim_sweep_slot(0.1), "a failed sweep parked the process-wide slot"
    maintenance.release_sweep_slot()
