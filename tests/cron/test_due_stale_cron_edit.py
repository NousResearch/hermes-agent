"""Stale-schedule guard on the cron fire path (#93049).

A direct ``jobs.json`` edit that changes ``schedule.expr`` leaves the stored
``next_run_at`` computed under the *old* expression — e.g. editing a daily
``0 7 * * *`` job down to ``0 7 * * 1-5`` keeps a Saturday 07:00
``next_run_at``. The old due check only compared ``next_run_at <= now``, so
the job fired on the day the new expression excludes. The guard re-anchors
``next_run_at`` from the current expression and skips the fire.

These exercise the real store against a temp ``HERMES_HOME`` (no mocks) per
the E2EE-over-mocks discipline for file-touching code.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _write_cron_job(schedule_expr: str, next_run_at: datetime) -> str:
    """Persist a cron job with a pinned next_run_at (the stale-edit shape)."""
    from cron.jobs import create_job, save_jobs, load_jobs

    job = create_job(prompt="x", schedule="every 5m", name="t")
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            j["schedule"] = {"kind": "cron", "expr": schedule_expr}
            j["next_run_at"] = next_run_at.isoformat()
    save_jobs(jobs)
    return job["id"]


# 2026-08-22 is a Saturday; 2026-08-24 is the following Monday.
_SATURDAY_0700 = datetime.fromisoformat("2026-08-22T07:00:00+00:00")
_MONDAY_0700 = datetime.fromisoformat("2026-08-24T07:00:00+00:00")


def test_stale_next_run_on_excluded_dow_does_not_fire(temp_home, monkeypatch):
    """next_run_at computed under the old daily expr must not fire the new
    Mon-Fri schedule on a Saturday — the job re-anchors to Monday instead."""
    from cron.jobs import get_due_jobs, get_job

    monkeypatch.setattr(
        "cron.jobs._hermes_now", lambda: _SATURDAY_0700 + timedelta(seconds=30)
    )
    jid = _write_cron_job("0 7 * * 1-5", _SATURDAY_0700)

    due = get_due_jobs()

    assert [j["id"] for j in due if j["id"] == jid] == []
    stored = get_job(jid)
    assert stored["next_run_at"].startswith(_MONDAY_0700.isoformat())


def test_matching_next_run_still_fires(temp_home, monkeypatch):
    """Control: the same stored instant under an expression it matches (daily)
    still fires — the guard only blocks schedule-drifted instants."""
    from cron.jobs import get_due_jobs

    monkeypatch.setattr(
        "cron.jobs._hermes_now", lambda: _SATURDAY_0700 + timedelta(seconds=30)
    )
    jid = _write_cron_job("0 7 * * *", _SATURDAY_0700)

    due = get_due_jobs()

    assert jid in [j["id"] for j in due]


def test_stale_next_run_skips_even_inside_catchup_window(temp_home, monkeypatch):
    """The catch-up 'run once now' policy must not resurrect a stale instant:
    hours after the stored time, the drifted job re-anchors without firing."""
    from cron.jobs import get_due_jobs, get_job

    monkeypatch.setattr(
        "cron.jobs._hermes_now", lambda: _SATURDAY_0700 + timedelta(hours=5)
    )
    jid = _write_cron_job("0 7 * * 1-5", _SATURDAY_0700)

    due = get_due_jobs()

    assert [j["id"] for j in due if j["id"] == jid] == []
    assert get_job(jid)["next_run_at"].startswith(_MONDAY_0700.isoformat())


def test_helper_reports_match_for_non_cron_and_unvalidatable(temp_home):
    """Best-effort validation: non-cron kinds, missing expr, and croniter
    unavailable all report a match so the fire path is unchanged there."""
    from cron.jobs import _cron_next_run_matches_expr

    assert _cron_next_run_matches_expr({"kind": "interval"}, _SATURDAY_0700) is True
    assert _cron_next_run_matches_expr({"kind": "cron"}, _SATURDAY_0700) is True
    # A croniter-matching instant reports a match; an excluded one does not.
    assert (
        _cron_next_run_matches_expr(
            {"kind": "cron", "expr": "0 7 * * *"}, _SATURDAY_0700
        )
        is True
    )
    assert (
        _cron_next_run_matches_expr(
            {"kind": "cron", "expr": "0 7 * * 1-5"}, _SATURDAY_0700
        )
        is False
    )
