"""Regression for #80624 — concurrent creates must not be clobbered on save.

``no_agent`` watchdog jobs (and any CLI/tool create while the gateway ticker
or a ``cron remove`` is live) were vanishing from ``jobs.json`` when a writer
persisted a stale/smaller in-memory snapshot. The save path now merges
unexpected on-disk ids back unless the caller passes ``removed_ids``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    (home / "scripts" / "watch.sh").write_text("#!/bin/bash\necho alert\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    import cron.jobs

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    return home


def test_stale_empty_save_preserves_concurrent_no_agent_create(hermes_env):
    """Gateway-style stale writer with [] must not wipe a concurrent create."""
    from cron.jobs import create_job, load_jobs, save_jobs

    job = create_job(
        prompt=None,
        schedule="every 2m",
        script="watch.sh",
        no_agent=True,
        deliver="local",
        name="watchdog",
        repeat=0,
    )
    assert [j["id"] for j in load_jobs()] == [job["id"]]

    # Simulate a degraded-lock writer that still holds an empty snapshot from
    # before the create (the filed incident: jobs.json rewritten empty).
    save_jobs([])

    remaining = load_jobs()
    assert [j["id"] for j in remaining] == [job["id"]]
    assert remaining[0].get("no_agent") is True
    assert remaining[0].get("script") == "watch.sh"


def test_remove_other_job_preserves_concurrent_create(hermes_env):
    """``cron remove`` of job A must not drop job B created mid-flight."""
    from cron.jobs import create_job, load_jobs, remove_job, save_jobs

    agent = create_job(
        prompt="hello",
        schedule="every 5m",
        name="agent",
        deliver="local",
    )
    # Stale remove payload: only knew about `agent`, never saw the watchdog.
    stale_after_remove = []
    save_jobs(stale_after_remove, removed_ids={agent["id"]})

    watchdog = create_job(
        prompt=None,
        schedule="every 2m",
        script="watch.sh",
        no_agent=True,
        deliver="local",
        name="watchdog",
        repeat=0,
    )
    # A second stale remove of `agent` (already gone) with empty payload —
    # must keep the watchdog that landed on disk in between.
    save_jobs([], removed_ids={agent["id"]})

    ids = {j["id"] for j in load_jobs()}
    assert ids == {watchdog["id"]}


def test_intentional_remove_still_deletes(hermes_env):
    from cron.jobs import create_job, get_job, remove_job

    job = create_job(
        prompt=None,
        schedule="every 2m",
        script="watch.sh",
        no_agent=True,
        deliver="local",
        name="watchdog",
        repeat=0,
    )
    assert remove_job(job["id"]) is True
    assert get_job(job["id"]) is None


def test_replace_flag_allows_wholesale_rewrite(hermes_env):
    from cron.jobs import create_job, load_jobs, save_jobs

    create_job(
        prompt=None,
        schedule="every 2m",
        script="watch.sh",
        no_agent=True,
        deliver="local",
        name="watchdog",
        repeat=0,
    )
    save_jobs([], replace=True)
    assert load_jobs() == []


def test_jobs_json_on_disk_matches_merge(hermes_env):
    from cron.jobs import create_job, save_jobs

    job = create_job(
        prompt=None,
        schedule="every 2m",
        script="watch.sh",
        no_agent=True,
        deliver="local",
        name="watchdog",
        repeat=0,
    )
    save_jobs([])
    payload = json.loads((Path(hermes_env) / "cron" / "jobs.json").read_text())
    assert [j["id"] for j in payload["jobs"]] == [job["id"]]
