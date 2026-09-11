"""Regression for #82990 — an agent-type job's prompt silently collapsing
to its own name field.

restore_cron_jobs_if_emptied() (hermes_cli/backup.py) only compares job
COUNT against the pre-update snapshot, so it correctly does not fire when
the same number of jobs survive an update but one job's ``prompt`` gets
overwritten with its own ``name`` -- field-level degradation, not job
loss. save_jobs() now guards against this specific shape at the central
write path.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    import cron.jobs

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    return home


def test_prompt_collapsed_to_name_is_rejected_and_prior_prompt_restored(hermes_env):
    """The exact reported shape: a long, real prompt gets overwritten with
    the job's own short name. The write must be rejected and the prior
    prompt restored rather than persisted."""
    from cron.jobs import create_job, load_jobs, save_jobs

    long_prompt = (
        "Generate the daily research brief. Format: markdown with three "
        "sections (Overview, Key Findings, Sources). Source from the "
        "configured RSS feeds and the last 24h of tracked repos. " * 3
    )
    job = create_job(
        prompt=long_prompt,
        schedule="every 1d",
        name="daily research brief",
        deliver="local",
        repeat=0,
    )
    assert load_jobs()[0]["prompt"] == long_prompt

    # Simulate the reported corruption: some other writer (e.g. the
    # desktop scheduler's own internally-tracked-crons sync, #52144)
    # persists the same job with prompt collapsed to its own name.
    corrupted = dict(job)
    corrupted["prompt"] = corrupted["name"]
    save_jobs([corrupted])

    restored = load_jobs()
    assert len(restored) == 1
    assert restored[0]["prompt"] == long_prompt, (
        "the prior, real prompt must survive -- a name is not a prompt"
    )


def test_no_agent_jobs_are_not_guarded(hermes_env):
    """Script (no_agent) jobs legitimately keep prompt == "" -- the guard
    must not interfere with that class at all."""
    from cron.jobs import create_job, load_jobs, save_jobs

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="check.sh",
        no_agent=True,
        name="check",
        deliver="local",
        repeat=0,
    )
    assert load_jobs()[0]["prompt"] == ""

    unchanged = dict(job)
    save_jobs([unchanged])
    assert load_jobs()[0]["prompt"] == ""


def test_short_prompt_already_matching_name_is_a_legitimate_edit(hermes_env):
    """A user intentionally setting a SHORT prompt that happens to equal
    the name (or editing an already-short prompt) is a legitimate,
    unusual configuration, not a collapse -- the guard only protects
    against shrinking from a MATERIALLY LONGER prior prompt."""
    from cron.jobs import create_job, load_jobs, save_jobs

    job = create_job(
        prompt="ping",
        schedule="every 1h",
        name="ping",
        deliver="local",
        repeat=0,
    )
    assert load_jobs()[0]["prompt"] == "ping"

    unchanged = dict(job)
    save_jobs([unchanged])
    assert load_jobs()[0]["prompt"] == "ping"


def test_genuine_prompt_edit_to_a_different_longer_text_is_allowed(hermes_env):
    """The guard must not block a real, intentional prompt edit -- only
    the specific collapse-to-bare-name shape."""
    from cron.jobs import create_job, load_jobs, update_job

    job = create_job(
        prompt="Check disk usage and report if over 80%.",
        schedule="every 1h",
        name="disk check",
        deliver="local",
        repeat=0,
    )
    updated = update_job(job["id"], {"prompt": "Check disk usage and report if over 90%, with a breakdown by mount point."})
    assert updated is not None
    assert load_jobs()[0]["prompt"].endswith("mount point.")


def test_replace_mode_bypasses_the_guard(hermes_env):
    """save_jobs(..., replace=True) is disaster recovery / test rewrite --
    matching the existing shrink-merge guard's own bypass, the prompt
    guard must not block a deliberate wholesale rewrite either."""
    from cron.jobs import create_job, load_jobs, save_jobs

    job = create_job(
        prompt="A genuinely long and detailed prompt for the daily job.",
        schedule="every 1d",
        name="daily job",
        deliver="local",
        repeat=0,
    )
    corrupted = dict(job)
    corrupted["prompt"] = corrupted["name"]
    save_jobs([corrupted], replace=True)

    assert load_jobs()[0]["prompt"] == "daily job"
