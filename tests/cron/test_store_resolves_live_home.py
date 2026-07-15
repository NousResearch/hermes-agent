"""The cron store must resolve HERMES_HOME LIVE, not freeze it at import time.

Regression for a silent live-store corruption (2026-07-15): ``cron/jobs.py``
computed ``CRON_DIR`` / ``JOBS_FILE`` as module-level constants at *import*
time, and ``_current_cron_store()`` returned those frozen paths whenever no
explicit ``use_cron_store()`` override was active. So if the module was imported
while ``HERMES_HOME`` pointed at the real ``~/.hermes`` (e.g. a real agent
booting from a worktree that shares the live home), a *later* ``HERMES_HOME``
redirect — the pytest hermetic fixture, or an e2e harness pointing at a
tempdir — was ignored, and ``create_job`` wrote fixture jobs straight into the
LIVE ``cron/jobs.json``.

These assert the store honours a HERMES_HOME change made AFTER the module was
imported. Both fail against the frozen-constant implementation.
"""
import os

from cron import jobs as jobs_mod


def test_current_store_follows_home_changed_after_import(tmp_path, monkeypatch):
    """A HERMES_HOME set after import must be reflected by _current_cron_store()."""
    late_home = tmp_path / "late_home"
    (late_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(late_home))

    store = jobs_mod._current_cron_store()

    assert store.jobs_file == late_home / "cron" / "jobs.json"
    assert store.cron_dir == late_home / "cron"


def test_create_job_writes_to_live_home_not_import_frozen(tmp_path, monkeypatch):
    """create_job must persist under the CURRENT HERMES_HOME, never a stale one."""
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    for h in (home_a, home_b):
        (h / "cron").mkdir(parents=True)

    # First "home" — stands in for the home present when the module imported.
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    # Redirect to a second home, as a test/e2e harness would after import.
    monkeypatch.setenv("HERMES_HOME", str(home_b))

    job = jobs_mod.create_job(prompt="probe", schedule="every 1h", name="probe")

    assert (home_b / "cron" / "jobs.json").exists(), "job must land in the current home"
    assert not (home_a / "cron" / "jobs.json").exists(), "must NOT leak into the stale home"
    assert jobs_mod.get_job(job["id"]) is not None
