"""Cron profile pinning tests.

Profile-pinned jobs should only be considered due by the gateway running that
profile. Unpinned jobs keep the historical any-profile race behaviour.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import get_due_jobs, save_jobs


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Use one shared cron store while varying the active HERMES_HOME profile."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    return tmp_path


@pytest.fixture()
def fixed_now(monkeypatch):
    now = datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    return now


_UNSET = object()


def _due_job(now: datetime, *, profile=_UNSET):
    run_at = (now - timedelta(seconds=10)).isoformat()
    job = {
        "id": "job1",
        "name": "profile pinning test",
        "prompt": "run",
        "skills": [],
        "skill": None,
        "schedule": {"kind": "once", "run_at": run_at},
        "schedule_display": run_at,
        "repeat": {"times": 1, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "created_at": now.isoformat(),
        "next_run_at": run_at,
        "last_run_at": None,
        "last_status": None,
        "deliver": "local",
        "origin": None,
    }
    if profile is not _UNSET:
        job["profile"] = profile
    return job


def _due_ids():
    return {job["id"] for job in get_due_jobs()}


def test_no_profile_field_runs_anywhere(cron_store, fixed_now, monkeypatch):
    save_jobs([_due_job(fixed_now)])

    monkeypatch.setenv("HERMES_HOME", str(cron_store))
    assert "job1" in _due_ids()

    monkeypatch.setenv("HERMES_HOME", str(cron_store / "profiles" / "dev"))
    assert "job1" in _due_ids()


def test_profile_match_runs(cron_store, fixed_now, monkeypatch):
    save_jobs([_due_job(fixed_now, profile="dev")])

    monkeypatch.setenv("HERMES_HOME", str(cron_store / "profiles" / "dev"))

    assert "job1" in _due_ids()


def test_profile_mismatch_skips(cron_store, fixed_now, monkeypatch):
    save_jobs([_due_job(fixed_now, profile="dev")])

    monkeypatch.setenv("HERMES_HOME", str(cron_store))

    assert "job1" not in _due_ids()


def test_lock_still_prevents_double_fire(tmp_path, monkeypatch):
    """A held tick.lock still makes a second scheduler tick skip immediately."""
    hermes_home = tmp_path / "profiles" / "dev"
    lock_dir = hermes_home / "cron"
    lock_dir.mkdir(parents=True)
    lock_file = lock_dir / ".tick.lock"

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys, time\n"
                f"path = pathlib.Path({str(lock_file)!r})\n"
                "path.parent.mkdir(parents=True, exist_ok=True)\n"
                "fd = path.open('w', encoding='utf-8')\n"
                "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                "print('locked', flush=True)\n"
                "time.sleep(10)\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        import cron.scheduler as scheduler

        called = False

        def fail_if_called():
            nonlocal called
            called = True
            raise AssertionError("get_due_jobs should not run while tick.lock is held")

        monkeypatch.setattr(scheduler, "get_due_jobs", fail_if_called)

        assert scheduler.tick(verbose=False, sync=True) == 0
        assert called is False
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)
