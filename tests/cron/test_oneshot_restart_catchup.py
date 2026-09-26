"""Restart catch-up for cron jobs whose fire time fell inside a gateway restart.

2026-09-25: two one-shots ('in 70m' / 'in 2h') had their run time fall inside
routine deploy restarts. On boot the due-scan removed each one ("more than the
120s grace window in the past") and wrote only a success-looking note to
cron/output, so an armed deploy step and an armed measurement never ran.

Pins:
  - one-shot 3 min past due        -> fires, prompt carries a late-fire note
  - one-shot 7 h past due          -> removed, LOUD MISSED notice delivered
  - cron.oneshot_catchup_s         -> config.yaml knob bounds the window
  - recurring missed < interval    -> one run
  - recurring missed > interval    -> one run, not N (no burst)
  - tick carries the late stamp across claim_job_for_fire
  - external-provider misfire backstop uses the same window + stamp
"""

from datetime import datetime, timedelta, timezone

import pytest

import cron.jobs as jobs_mod
from cron.jobs import (
    LATE_FIRE_KEY,
    claim_job_for_fire,
    drain_missed_oneshot_notices,
    get_due_jobs,
    load_jobs,
    save_jobs,
)

FIXED_NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    drain_missed_oneshot_notices()
    yield tmp_path
    drain_missed_oneshot_notices()


def _set_cron_cfg(monkeypatch, cron_cfg):
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: {"cron": cron_cfg}
    )


def _oneshot(jid, run_at_dt):
    return {
        "id": jid,
        "name": jid,
        "prompt": "run the armed deploy step",
        "schedule": {"kind": "once", "run_at": run_at_dt.isoformat()},
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1, "completed": 0},
        "deliver": "local",
    }


def _interval(jid, minutes, next_run_dt):
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "interval", "minutes": minutes},
        "next_run_at": next_run_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
    }


class TestOneShotRestartCatchup:
    def test_three_min_past_due_fires_with_late_note(self, cron_store, monkeypatch):
        _set_cron_cfg(monkeypatch, {})  # default window (6h)
        save_jobs([_oneshot("late", FIXED_NOW - timedelta(minutes=3))])

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["late"]
        assert due[0][LATE_FIRE_KEY] == 180
        # The stamp is transient: never written to jobs.json.
        stored = load_jobs()
        assert [j["id"] for j in stored] == ["late"]
        assert LATE_FIRE_KEY not in stored[0]
        assert drain_missed_oneshot_notices() == []

        from cron.scheduler import _build_job_prompt

        prompt = _build_job_prompt(due[0])
        note = "[Note: this scheduled task fired late by 3 min after a gateway restart"
        assert note in prompt
        # The note leads the user's prompt (after the system cron preamble).
        assert prompt.index(note) < prompt.index("run the armed deploy step")

    def test_on_time_oneshot_prompt_has_no_note(self, cron_store, monkeypatch):
        _set_cron_cfg(monkeypatch, {})
        save_jobs([_oneshot("ontime", FIXED_NOW - timedelta(seconds=30))])
        due = get_due_jobs()
        assert [d["id"] for d in due] == ["ontime"]
        assert LATE_FIRE_KEY not in due[0]
        from cron.scheduler import _build_job_prompt

        assert "fired late" not in _build_job_prompt(due[0])

    def test_seven_hours_past_due_removed_with_loud_notice(
        self, cron_store, monkeypatch
    ):
        _set_cron_cfg(monkeypatch, {})
        save_jobs([_oneshot("gone", FIXED_NOW - timedelta(hours=7))])

        due = get_due_jobs()

        assert due == []
        assert load_jobs() == []
        diag = list((cron_store / "cron" / "output" / "gone").glob("*.md"))
        assert diag
        text = diag[0].read_text(encoding="utf-8")
        assert text.startswith("# ⚠️ MISSED")
        assert "did NOT run" in text and "NEVER RAN" in text
        assert "past due by: 420 min" in text

        # Queued for delivery; the tick delivers it framed as a FAILURE.
        import cron.scheduler as sched

        calls = []

        def _fake_deliver(job, content, adapters=None, loop=None, *, for_failure=False):
            calls.append((job.get("id"), content, not for_failure))
            return None

        monkeypatch.setattr(sched, "_deliver_result", _fake_deliver)
        assert sched._deliver_missed_oneshot_notices() == 1
        assert calls and calls[0][0] == "gone"
        assert calls[0][2] is False
        assert "MISSED" in calls[0][1]
        # Drained: never delivered twice.
        assert sched._deliver_missed_oneshot_notices() == 0

    def test_catchup_window_is_a_config_knob(self, cron_store, monkeypatch):
        _set_cron_cfg(monkeypatch, {"oneshot_catchup_s": 600})
        save_jobs(
            [
                _oneshot("inside", FIXED_NOW - timedelta(minutes=9)),
                _oneshot("outside", FIXED_NOW - timedelta(minutes=15)),
            ]
        )
        due = get_due_jobs()
        assert [d["id"] for d in due] == ["inside"]
        assert [j["id"] for j in load_jobs()] == ["inside"]

    def test_zero_restores_120s_grace_only(self, cron_store, monkeypatch):
        _set_cron_cfg(monkeypatch, {"oneshot_catchup_s": 0})
        save_jobs([_oneshot("late", FIXED_NOW - timedelta(minutes=3))])
        assert get_due_jobs() == []
        assert load_jobs() == []

    def test_catchup_seconds_default(self, monkeypatch):
        _set_cron_cfg(monkeypatch, {})
        assert jobs_mod._oneshot_catchup_seconds() == 21600


class TestRecurringRestartCatchup:
    def test_missed_by_less_than_interval_runs_once(self, cron_store, monkeypatch):
        _set_cron_cfg(monkeypatch, {})
        save_jobs([_interval("hourly", 60, FIXED_NOW - timedelta(minutes=40))])

        due = get_due_jobs()
        assert [d["id"] for d in due] == ["hourly"]
        # Once the tick claims that one run, the next scan (same instant) must not fire it again.
        assert claim_job_for_fire("hourly")
        nxt = datetime.fromisoformat(load_jobs()[0]["next_run_at"])
        assert nxt > FIXED_NOW
        assert get_due_jobs() == []

    def test_missed_by_many_intervals_runs_once_not_n(self, cron_store, monkeypatch):
        _set_cron_cfg(monkeypatch, {})
        # 5 hourly ticks missed while the gateway was down.
        save_jobs([_interval("hourly", 60, FIXED_NOW - timedelta(hours=5))])

        first = get_due_jobs()
        assert [d["id"] for d in first] == ["hourly"]
        assert claim_job_for_fire("hourly")
        nxt = datetime.fromisoformat(load_jobs()[0]["next_run_at"])
        assert nxt > FIXED_NOW
        # No backlog replay: repeated scans at the same instant fire nothing.
        for _ in range(4):
            assert get_due_jobs() == []


class TestTickCarriesLateStamp:
    def test_claimed_snapshot_keeps_late_stamp(self, cron_store, monkeypatch):
        """The tick re-reads the persisted record via claim_job_for_fire; the
        transient late-fire stamp must survive that swap into run_one_job."""
        import cron.scheduler as sched

        _set_cron_cfg(monkeypatch, {})
        save_jobs([_oneshot("late", FIXED_NOW - timedelta(minutes=3))])

        seen = []

        def _fake_run_one_job(job, **kwargs):
            seen.append(dict(job))
            return True

        monkeypatch.setattr(sched, "run_one_job", _fake_run_one_job)
        monkeypatch.setattr(sched, "_get_lock_paths", lambda: (cron_store / "lk", cron_store / "lk" / ".tick.lock"))
        sched.tick(verbose=False, sync=True)

        assert [j["id"] for j in seen] == ["late"]
        assert seen[0].get(LATE_FIRE_KEY) == 180


class TestMisfireBackstopUsesCatchupWindow:
    class _Provider:
        def __init__(self):
            self.claimed = []
            self.snapshots = []

        def claim_fire(self, job_id):
            self.claimed.append(job_id)
            snap = {"id": job_id}
            self.snapshots.append(snap)
            return snap

        def fire_claimed(self, *a, **k):
            return True

    def _run(self, monkeypatch, jobs, cron_cfg):
        from cron.scheduler_provider import fire_overdue_jobs

        provider = self._Provider()
        _set_cron_cfg(monkeypatch, cron_cfg)
        monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
        monkeypatch.setattr("cron.jobs.is_job_runnable", lambda j: True)
        monkeypatch.setattr("cron.scheduler_provider._misfire_grace_minutes", lambda: 5.0)
        fire_overdue_jobs(provider, now=FIXED_NOW)
        return provider

    def test_oneshot_two_hours_late_fires_with_stamp(self, monkeypatch):
        p = self._run(
            monkeypatch, [_oneshot("late", FIXED_NOW - timedelta(hours=2))], {}
        )
        assert p.claimed == ["late"]
        assert p.snapshots[0][LATE_FIRE_KEY] == 7200

    def test_oneshot_beyond_window_not_fired(self, monkeypatch):
        p = self._run(
            monkeypatch, [_oneshot("gone", FIXED_NOW - timedelta(hours=7))], {}
        )
        assert p.claimed == []
