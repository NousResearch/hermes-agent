"""The cron store must refuse a TEST-context write to a NON-TEMP (real) store.

Regression for the 2026-07-15 recurrence: a test/e2e harness (a real-agent
blackbox session or a kanban worker booted from a worktree that shares the live
``~/.hermes``) that runs the cron test suite OUTSIDE pytest's hermetic conftest
imports ``cron.jobs`` with ``HERMES_HOME`` pointing at the real home, then calls
``create_job`` with fixture data — leaking ``brief``/``claim job``/``paused job``
jobs into the LIVE ``cron/jobs.json`` and paging cron-health.

The guard discriminates by TEMP-vs-real, not by matching a "production" path:
a correctly-isolated test always writes under ``tempfile.gettempdir()``; the
leak writes to a non-temp home. This is env-independent, so it never false-fires
on a well-behaved test that legitimately monkeypatches ``Path.home()``.
"""
import os
import tempfile
from pathlib import Path

import pytest

from cron import jobs as jobs_mod


def _fake_nontemp_home(tmp_path):
    """A path that is NOT under the system temp dir (simulates the real ~/.hermes),
    built so the test itself never touches a real store."""
    # tmp_path IS under gettempdir; construct a sibling non-temp path we never write.
    return Path("/nonexistent-fake-home/.hermes/cron/jobs.json")


def test_nontemp_write_under_pytest_is_refused(monkeypatch):
    """PYTEST_CURRENT_TEST + a non-temp target => RuntimeError (the leak scenario)."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
    with pytest.raises(RuntimeError, match="NON-TEMP home"):
        jobs_mod._guard_against_test_write_to_live_store(
            Path("/Users/someone/.hermes/cron/jobs.json")
        )


def test_tempdir_write_under_pytest_is_allowed(tmp_path, monkeypatch):
    """A correctly-isolated test (store under the system temp dir) writes fine."""
    (tmp_path / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
    jobs_mod.save_jobs([{"id": "ok", "prompt": "x", "name": "t"}])
    assert (tmp_path / "cron" / "jobs.json").exists()


def test_tempdir_write_survives_monkeypatched_path_home(tmp_path, monkeypatch):
    """Even a test that monkeypatches Path.home() to a tmp home is allowed —
    the guard keys on temp-vs-real, not on Path.home() (the v1 false-fire)."""
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
    jobs_mod.save_jobs([{"id": "ok", "prompt": "x", "name": "t"}])
    assert (home / "cron" / "jobs.json").exists()


def test_explicit_use_cron_store_override_bypasses_the_guard(tmp_path, monkeypatch):
    """An explicit use_cron_store() scope is a deliberate target — never guarded."""
    (tmp_path / "cron").mkdir(parents=True)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
    with jobs_mod.use_cron_store(str(tmp_path)):
        jobs_mod.save_jobs([{"id": "ok", "prompt": "x", "name": "t"}])
    assert (tmp_path / "cron" / "jobs.json").exists()


def test_production_write_no_pytest_context_is_a_noop(monkeypatch):
    """The guard must NOT fire for a real production write (no PYTEST_CURRENT_TEST)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # Even a non-temp target is fine when there is no pytest signal.
    jobs_mod._guard_against_test_write_to_live_store(
        Path("/Users/someone/.hermes/cron/jobs.json")
    )


def test_blocked_leak_leaves_no_filesystem_side_effect(tmp_path, monkeypatch):
    """A blocked save_jobs must NOT create cron/ or the advisory lock in the
    target home — the guard fires BEFORE _jobs_lock()/ensure_dirs()."""
    # Simulate a non-temp home by monkeypatching the store to a non-temp path.
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
    monkeypatch.setattr(
        jobs_mod, "_current_cron_store",
        lambda: jobs_mod._CronStorePaths(
            Path("/Users/nobody/.hermes/cron"),
            Path("/Users/nobody/.hermes/cron/jobs.json"),
            Path("/Users/nobody/.hermes/cron/output"),
        ),
    )
    with pytest.raises(RuntimeError, match="NON-TEMP home"):
        jobs_mod.save_jobs([{"id": "leak", "prompt": "brief", "name": "brief"}])
    # The non-temp home must not have been created as a side effect.
    assert not Path("/Users/nobody/.hermes").exists()

