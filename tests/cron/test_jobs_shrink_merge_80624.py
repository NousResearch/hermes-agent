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


def test_stamp_fast_path_skips_merge_when_file_unchanged(hermes_env, monkeypatch):
    """Inside a critical section whose load stamp still matches, the save
    must not re-read jobs.json at all (#80703's single-stat fast path)."""
    import cron.jobs as jobs
    from cron.jobs import create_job

    job = create_job(
        prompt=None,
        schedule="every 2m",
        script="watch.sh",
        no_agent=True,
        deliver="local",
        name="watchdog",
        repeat=0,
    )

    peeks = {"count": 0}
    real_peek = jobs._peek_jobs_unlocked

    def _counting_peek():
        peeks["count"] += 1
        return real_peek()

    monkeypatch.setattr(jobs, "_peek_jobs_unlocked", _counting_peek)

    # load -> save inside ONE critical section, no sibling write in between:
    # the stamp matches, so the merge (and its peek) must be skipped.
    with jobs._jobs_lock():
        current = jobs.load_jobs()
        jobs._save_jobs_unlocked(current)
    assert peeks["count"] == 0, "healthy same-section save should not re-parse"

    # A sibling write invalidates the stamp -> the merge runs again.
    with jobs._jobs_lock():
        current = jobs.load_jobs()
        (jobs._current_cron_store().jobs_file).write_text(
            json.dumps({"jobs": [dict(job), {"id": "bbbbbbbbbbbb", "name": "b"}]}),
            encoding="utf-8",
        )
        jobs._save_jobs_unlocked(current)
    assert peeks["count"] > 0, "changed stamp must re-trigger the merge"
    ids = {j["id"] for j in jobs.load_jobs()}
    assert ids == {job["id"], "bbbbbbbbbbbb"}


def test_merge_does_not_mutate_caller_list(hermes_env):
    """The shrink-merge returns a new list; the caller's payload object must
    not grow as a side effect of save_jobs()."""
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
    my_payload = []  # stale snapshot from before the create
    save_jobs(my_payload)
    assert my_payload == [], "caller's list was mutated in place by the merge"


def test_corrupt_disk_file_does_not_break_save(hermes_env):
    """A corrupt jobs.json under a save must not recurse or crash: the
    non-repairing peek returns None and the save overwrites cleanly."""
    import cron.jobs as jobs
    from cron.jobs import load_jobs, save_jobs

    jobs.ensure_dirs()
    jobs_file = jobs._current_cron_store().jobs_file
    jobs_file.write_text('{"jobs": [{"id": "ccc', encoding="utf-8")
    save_jobs([{"id": "aaaaaaaaaaaa", "name": "a"}])
    assert [j["id"] for j in load_jobs()] == ["aaaaaaaaaaaa"]


def test_nested_create_survives_outer_stale_save(hermes_env):
    """A save inside a critical section invalidates the section's stamp, so
    an outer caller's later save with a pre-create payload must re-merge and
    keep the nested create (stamp refresh here would deterministically
    clobber it)."""
    import cron.jobs as jobs
    from cron.jobs import create_job, load_jobs, save_jobs

    seed = {"id": "aaaaaaaaaaaa", "name": "a"}
    save_jobs([seed], replace=True)

    with jobs._jobs_lock():
        stale = jobs.load_jobs()  # records the stamp
        created = create_job(
            prompt="x", schedule="every 5m", name="nested", deliver="local"
        )
        jobs._save_jobs_unlocked(stale)  # outer save with pre-create payload

    ids = {j["id"] for j in load_jobs()}
    assert created["id"] in ids, "nested create was clobbered by outer stale save"
    assert seed["id"] in ids
