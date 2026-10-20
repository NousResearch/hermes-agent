"""One-shots due while no cron ticker was running.

A one-shot due during ticker downtime (host powered off overnight, gateway stopped) was never
missed — nothing was alive to fire it. The due scan's grace gate already makes that distinction
for a slot skipped by an unwritable store; these tests pin the same release for a recorded ticker
downtime window, and pin that it stays OFF unless the operator opts in.
"""

from datetime import datetime, timedelta, UTC

import pytest

from cron import jobs, ticker_downtime

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clear_windows():
    ticker_downtime.forget_stores(list(ticker_downtime._windows))
    yield
    ticker_downtime.forget_stores(list(ticker_downtime._windows))


@pytest.fixture()
def cron_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _oneshot(run_at_dt):
    return {
        "id": "overnight",
        "name": "overnight",
        "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at_dt.isoformat()},
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1, "completed": 0},
        "deliver": "local",
    }


def _store_oneshot(home, due_at, *, opt_in):
    (home / "config.yaml").write_text(
        f"cron:\n  catch_up_missed_oneshots: {str(opt_in).lower()}\n", encoding="utf-8")
    with jobs.use_cron_store(home):
        jobs.save_jobs([_oneshot(due_at)])


@pytest.mark.parametrize("opt_in,expected_due", [(True, ["overnight"]), (False, [])])
def test_oneshot_due_during_ticker_downtime_follows_opt_in(cron_home, opt_in, expected_due):
    """A one-shot due inside a recorded downtime window fires once when opted in, and is retired
    as before when not — the default must not silently change late-firing behaviour."""
    due_at = FIXED_NOW - timedelta(hours=8)
    _store_oneshot(cron_home, due_at, opt_in=opt_in)
    with jobs.use_cron_store(cron_home):
        # Ticker was last alive before the job came due and only started again just now.
        ticker_downtime.record_startup_gap(
            jobs._current_cron_store().cron_dir, due_at.timestamp() - 3600,
            FIXED_NOW.timestamp(), min_gap=120)
        assert [row["id"] for row in jobs.get_due_jobs()] == expected_due


def test_oneshot_outside_any_downtime_window_still_retired(cron_home):
    """The window is evidence, not a blanket amnesty: a one-shot due while the ticker WAS running
    (it just failed to fire) stays retired even with the opt-in on."""
    due_at = FIXED_NOW - timedelta(hours=8)
    _store_oneshot(cron_home, due_at, opt_in=True)
    with jobs.use_cron_store(cron_home):
        # Downtime ended an hour before the job came due, so the ticker was up when it was due.
        ticker_downtime.record_startup_gap(
            jobs._current_cron_store().cron_dir, due_at.timestamp() - 7200,
            due_at.timestamp() - 3600, min_gap=120)
        assert [row["id"] for row in jobs.get_due_jobs()] == []


def test_short_restart_is_not_downtime(tmp_path):
    """A heartbeat gap within the tick cadence is ordinary restart slip, so no window is recorded
    and nothing is released."""
    now = FIXED_NOW.timestamp()
    assert ticker_downtime.record_startup_gap(
        tmp_path / "cron", now - 30, now, min_gap=120) is None
    assert not ticker_downtime.covers(tmp_path / "cron", now - 30, 120)


def test_never_ticked_store_records_no_window(tmp_path):
    """No heartbeat marker means "never ticked", which is not evidence of downtime."""
    now = FIXED_NOW.timestamp()
    assert ticker_downtime.record_startup_gap(tmp_path / "cron", None, now, min_gap=120) is None
    assert not ticker_downtime.covers(tmp_path / "cron", now - 99999, 120)
