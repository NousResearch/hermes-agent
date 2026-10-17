"""Regression test for #97905 / carrier PR #92489.

A multiplex ticker (desktop dashboard backend, multiplex gateway) resolves
``hermes_time`` under the process's own startup profile, then ticks OTHER
profiles' cron stores via ``set_hermes_home_override()`` + ``use_cron_store()``.
Before the profile-keyed timezone cache, the first profile's resolved zone
was process-global, so ``compute_next_run`` / ``create_job`` / ``mark_job_run``
persisted ``next_run_at`` into the ticked profile's jobs.json with the FOREIGN
process's UTC offset — e.g. ``0 14 * * *`` for an America/New_York profile
stored as ``14:00+00:00``, which becomes due at 10:00 ET and fires four hours
early (#97905; single-profile sibling report #88220).

The invariant pinned here: ``next_run_at`` written during a foreign-process
tick carries the JOB-OWNING profile's configured UTC offset.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import hermes_time


@pytest.fixture(autouse=True)
def _fresh_tz_cache(monkeypatch):
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def test_foreign_process_tick_persists_owning_profile_offset(
    tmp_path, monkeypatch
):
    """next_run_at persisted during a multiplex tick uses the ticked
    profile's timezone, not the backend process's."""
    backend_home = tmp_path / "backend"  # desktop backend's own profile: UTC
    profile_a = tmp_path / "north-caribbean"  # job-owning profile: Eastern
    backend_home.mkdir()
    profile_a.mkdir()
    (backend_home / "config.yaml").write_text(
        "timezone: UTC\n", encoding="utf-8"
    )
    (profile_a / "config.yaml").write_text(
        "timezone: America/New_York\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(backend_home))

    # Backend process resolves its own timezone first (process startup).
    assert hermes_time.now().utcoffset() == datetime.now(
        ZoneInfo("UTC")
    ).utcoffset()

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from cron.jobs import create_job, load_jobs, use_cron_store

    # Exactly the scoping the multiplex ticker applies per profile
    # (cron/scheduler_provider.py::_tick_profiles).
    token = set_hermes_home_override(str(profile_a))
    try:
        with use_cron_store(profile_a):
            create_job(name="daily-2pm", prompt="x", schedule="0 14 * * *")
            job = load_jobs()[0]
    finally:
        reset_hermes_home_override(token)

    next_run = datetime.fromisoformat(job["next_run_at"])
    expected_offset = datetime.now(ZoneInfo("America/New_York")).utcoffset()
    assert next_run.utcoffset() == expected_offset, (
        f"next_run_at {job['next_run_at']} persisted with foreign offset "
        f"{next_run.utcoffset()} instead of the owning profile's "
        f"{expected_offset} — fires (offset delta) early (#97905)"
    )
    # And the wall clock honors the cron expression in the profile's zone.
    assert (next_run.hour, next_run.minute) == (14, 0)
