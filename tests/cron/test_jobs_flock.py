"""F-M5 regression: jobs.json read-modify-write is inter-process atomic.

Without flock, concurrent create_job/remove_job calls racing on load→mutate→save
can drop jobs: A reads, B reads (sees the same state), A writes, B writes,
A's job is gone.

These tests use `multiprocessing` (not threads) to exercise the real
fcntl.flock path. Threads would pass even without the lock because Python
threads can't interleave inside C-level fcntl syscalls.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest


def _worker_create(hermes_home: str, prompt: str, idx: int) -> None:
    """Child process: create a job. Run under flock via create_job."""
    os.environ["HERMES_HOME"] = hermes_home
    # Clear the cached import so subprocess picks up the right HERMES_HOME
    # (each child starts fresh, but be explicit for clarity).
    from cron import jobs
    jobs.create_job(
        prompt=f"{prompt}-{idx}",
        schedule="every 60m",
    )


def _worker_remove(hermes_home: str, job_id: str) -> None:
    os.environ["HERMES_HOME"] = hermes_home
    from cron import jobs
    jobs.remove_job(job_id)


@pytest.fixture
def hermes_tmp_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Reload cron.jobs so module-level CRON_DIR / JOBS_FILE pick up the new home.
    for mod in ("cron.jobs",):
        if mod in sys.modules:
            del sys.modules[mod]
    return home


def test_concurrent_creates_do_not_lose_jobs(hermes_tmp_home):
    """20 concurrent create_job calls must all land in jobs.json."""
    N = 20
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker_create, args=(str(hermes_tmp_home), "concurrent", i))
        for i in range(N)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"

    jobs_file = hermes_tmp_home / "cron" / "jobs.json"
    assert jobs_file.exists()
    data = json.loads(jobs_file.read_text())
    names = {j["prompt"] for j in data["jobs"]}
    expected = {f"concurrent-{i}" for i in range(N)}
    assert names == expected, f"lost jobs: {expected - names}"


def test_update_jobs_context_manager_is_reentrant_safe(hermes_tmp_home):
    """update_jobs() nested inside save_jobs path must not deadlock."""
    from cron import jobs
    # Seed two jobs
    j1 = jobs.create_job(prompt="a", schedule="every 60m")
    j2 = jobs.create_job(prompt="b", schedule="every 60m")
    # remove_job uses update_jobs() which internally calls _save_jobs_unlocked
    # (NOT save_jobs) — this test pins that split so a future refactor can't
    # reintroduce the deadlock.
    assert jobs.remove_job(j1["id"]) is True
    remaining = jobs.load_jobs()
    assert len(remaining) == 1
    assert remaining[0]["id"] == j2["id"]
