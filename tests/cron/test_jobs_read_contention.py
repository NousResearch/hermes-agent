"""Regression for #144166: a transient Windows sharing violation while jobs.json is being saved
must not fail the whole read.

Each concurrent job finish saves jobs.json through ``utils.atomic_replace``, and while that save is
in flight ``open()`` fails with ``PermissionError``/errno 13 — the C runtime drops ``winerror``, so
the writer's own contended-error classifier cannot see it. The write side had a retry budget and
the read side had none, so ``load_jobs()`` raised ``RuntimeError`` and the tick lost every job.

A real no-sharing kernel handle is used (``open()`` alone shares read/write by default, so it would
not reproduce the contention).
"""

import ctypes
import threading
from pathlib import Path

import pytest

from cron.jobs import create_job, load_jobs, use_cron_store

pytestmark = pytest.mark.platforms("windows")

GENERIC_READ = 0x80000000
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80


def _hold_without_sharing(path: Path, started: threading.Event, release: threading.Event) -> None:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateFileW(
        str(path), GENERIC_READ, 0, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    try:
        started.set()
        release.wait(10)
    finally:
        kernel32.CloseHandle(handle)


@pytest.fixture()
def contended(tmp_path, monkeypatch):
    """A store with one job, a deterministic retry budget, and the jobs file held exclusively."""
    monkeypatch.setattr("cron.jobs._READ_RETRY_BASE_DELAY_S", 0.1, raising=False)
    monkeypatch.setattr("cron.jobs._READ_RETRY_MAX_DELAY_S", 0.1, raising=False)
    with use_cron_store(tmp_path):
        job = create_job(prompt="nightly report", schedule="every 1h")
        yield tmp_path / "cron" / "jobs.json", job


@pytest.fixture()
def holding(contended):
    """Start the exclusive holder and hand back ``(jobs_file, release, join)``."""

    def start(hold_seconds: float | None):
        jobs_file, _job = contended
        started, release = threading.Event(), threading.Event()
        holder = threading.Thread(
            target=_hold_without_sharing, args=(jobs_file, started, release), daemon=True)
        holder.start()
        assert started.wait(10), "holder never opened the jobs file"
        if hold_seconds is not None:
            threading.Timer(hold_seconds, release.set).start()

        def finish():
            release.set()
            holder.join(10)

        return release, finish

    return start


def test_transient_hold_is_retried(contended, holding):
    _jobs_file, job = contended
    _release, finish = holding(0.25)
    try:
        assert [j["id"] for j in load_jobs()] == [job["id"]]
    finally:
        finish()


def test_hold_beyond_the_budget_still_raises(contended, holding):
    _jobs_file, _job = contended
    _release, finish = holding(None)
    try:
        with pytest.raises(RuntimeError, match="Failed to read cron database"):
            load_jobs()
    finally:
        finish()
