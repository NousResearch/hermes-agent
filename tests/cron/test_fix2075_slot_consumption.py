"""Regression tests for FIX-2075 — a cron slot consumed without any trace.

Incident (2026-10-25, ops profile, job 299e2e35eb46 'wiki-灵魂审计', cron
``40 * * * *``): the 21:40:53 fire ran 58 minutes and finished at 22:40:18 —
18s AFTER the pre-advanced 22:40:00 slot, 35s BEFORE the 22:40:53 tick.
``mark_job_run`` recomputed ``next_run_at`` from the completion time
(22:40:18 -> 23:40:00), overwriting the 22:40:00 slot; the 22:40:53 due scan
saw a future ``next_run_at`` and silently never fired it — no run, no ledger
row, no log line. A one-hour monitoring blind spot in the self-audit.

Two fixes are covered here:

1. ``cron.jobs._advance_after_run`` keeps the stored (pre-advanced) slot when
   it is still on/behind the cron lattice within the catch-up grace window, so
   an overran run cannot consume the next occurrence (restoring the
   ``get_due_jobs`` catch-up promise for overran runs, #33315).
2. ``cron.scheduler._submit_with_guard`` books a terminal ``failed`` ledger
   row when the in-flight dedup guard refuses dispatch — the pre-advanced
   slot is consumed by the refusal, and previously only an INFO log line
   remained (still no ledger row).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.jobs as jobs_mod

LOCAL_TZ = datetime.now().astimezone().tzinfo
CRON_SCHEDULE = {"kind": "cron", "expr": "40 * * * *"}  # hourly :40 -> grace = 1800s (period//2, clamped [120s, 2h])


def _iso(dt: datetime) -> str:
    return dt.astimezone(LOCAL_TZ).isoformat()


def _at(hh: int, mm: int, ss: int = 0) -> str:
    return _iso(datetime(2026, 10, 25, hh, mm, ss, tzinfo=LOCAL_TZ))


def _instant(value: str) -> datetime:
    """Parse for comparison: compute_next_run normalizes its output to UTC,
    while the fixtures are written in the local zone — compare instants, not
    raw strings."""
    return datetime.fromisoformat(value)


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    yield hermes_home
    # Teardown: a tick in this fixture's home leaves its persistent dispatch
    # pool in the process-global registry; drop it so later tests that assert
    # on the registry (pool-reaping) see a clean state.
    from cron import scheduler as sched
    from hermes_constants import hermes_home_key

    key = hermes_home_key(str(hermes_home))
    pool = sched._parallel_pools.pop(key, None)
    sched._parallel_pool_max_workers.pop(key, None)
    if pool is not None:
        pool.shutdown(wait=True)


def _ledger_rows(job_id: str):
    from cron import executions as exec_mod

    with exec_mod._transaction() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM executions WHERE job_id=? ORDER BY claimed_at",
                (job_id,),
            ).fetchall()
        ]


class TestAdvanceAfterRunSlotGuard:
    """mark_job_run's completion re-anchor must not jump past an unclaimed,
    still catch-up-eligible slot (the 22:40 incident)."""

    def test_overrun_completion_keeps_in_grace_slot(self):
        # The incident: slot 22:40:00 pre-advanced, run finished 22:40:18.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _at(22, 40, 0)}
        jobs_mod._advance_after_run(job, _at(22, 40, 18))
        assert job["next_run_at"] == _at(22, 40, 0), (
            "an overran run must not consume the next occurrence; the stored "
            f"slot must survive to the next tick's catch-up, got {job['next_run_at']}")

    def test_normal_completion_before_slot_unchanged(self):
        # Run finished 21:50:19, slot 22:40:00 still ahead: recompute and
        # stored agree — behaviour identical to the pre-fix code.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _at(22, 40, 0)}
        jobs_mod._advance_after_run(job, _at(21, 50, 19))
        assert job["next_run_at"] == _at(22, 40, 0)

    def test_completion_exactly_at_slot_boundary_keeps_slot(self):
        # finished_at == slot instant: recompute would jump to 23:40; the slot
        # is not yet past, so it must stay.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _at(22, 40, 0)}
        jobs_mod._advance_after_run(job, _at(22, 40, 0))
        assert job["next_run_at"] == _at(22, 40, 0)

    def test_stale_slot_beyond_grace_is_recomputed(self):
        # Slot an hour and ten minutes past (grace for hourly cron is 1800s):
        # fast-forward policy applies, recompute wins.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _at(21, 10, 0)}
        jobs_mod._advance_after_run(job, _at(22, 40, 18))
        assert _instant(job["next_run_at"]) == _instant(_at(23, 40, 0))

    def test_slot_at_exact_grace_boundary_is_kept(self):
        # elapsed == grace (1800s): the window is inclusive, keep the slot.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _at(21, 10, 0)}
        jobs_mod._advance_after_run(job, _at(21, 40, 0))
        assert job["next_run_at"] == _at(21, 10, 0)

    def test_slot_one_second_past_grace_is_recomputed(self):
        # elapsed == grace + 1s: outside the window, recompute wins.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _at(21, 9, 59)}
        jobs_mod._advance_after_run(job, _at(21, 40, 0))
        assert _instant(job["next_run_at"]) == _instant(_at(22, 40, 0))

    def test_far_future_stored_slot_is_recomputed(self):
        # Stored slot AHEAD of the natural next lattice point (drift / direct
        # edit): recompute converges to the lattice, stored is NOT kept.
        job = {"schedule": CRON_SCHEDULE, "next_run_at": _iso(
            datetime(2026, 10, 25, 23, 59, 0, tzinfo=LOCAL_TZ))}
        jobs_mod._advance_after_run(job, _at(22, 40, 18))
        assert _instant(job["next_run_at"]) == _instant(_at(23, 40, 0))

    def test_interval_job_keeps_completion_based_reanchor(self):
        # Interval schedules re-anchor from the completion time by design
        # (rolling lattice drifts with run duration) — the guard must not
        # change their semantics.
        job = {"schedule": {"kind": "interval", "minutes": 1},
               "next_run_at": _at(22, 41, 0)}
        jobs_mod._advance_after_run(job, _at(22, 40, 18))
        assert _instant(job["next_run_at"]) == _instant(_at(22, 41, 18))

    def test_missing_stored_slot_still_recomputes(self):
        job = {"schedule": CRON_SCHEDULE, "next_run_at": None}
        jobs_mod._advance_after_run(job, _at(22, 40, 18))
        assert _instant(job["next_run_at"]) == _instant(_at(23, 40, 0))


class TestOverrunThenDueScanFiresCatchup:
    """End to end (store level): after an overran completion the NEXT tick's
    due scan must still see the job as due — the slot is not lost."""

    def test_job_stays_due_after_overrun_completion(self, cron_store, monkeypatch):
        job = jobs_mod.create_job(prompt="audit", schedule="40 * * * *")
        job_id = job["id"]
        slot = _at(22, 40, 0)
        # The tick pre-advanced the slot before dispatch:
        jobs = jobs_mod.load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["next_run_at"] = slot
        jobs_mod.save_jobs(jobs)

        # 22:40:18: the overran run completes (mark_job_run re-anchors).
        monkeypatch.setattr(jobs_mod, "_hermes_now", lambda: datetime.fromisoformat(_at(22, 40, 18)))
        assert jobs_mod.mark_job_run(job_id, True) is True

        reloaded = next(j for j in jobs_mod.load_jobs() if j["id"] == job_id)
        assert reloaded["next_run_at"] == slot, (
            "mark_job_run consumed the pre-advanced slot (FIX-2075 regression)")

        # 22:40:53: the next tick's due scan — the job must be due (catch-up).
        monkeypatch.setattr(jobs_mod, "_hermes_now", lambda: datetime.fromisoformat(_at(22, 40, 53)))
        due = jobs_mod.get_due_jobs()
        assert [j["id"] for j in due] == [job_id], (
            "the catch-up fire for the overran slot was lost in the due scan")


class TestInflightSkipLedgerRow:
    """The in-flight dedup guard refuses dispatch AFTER the due scan pre-
    advanced the slot — the refusal consumes it and must leave a 'failed'
    ledger row naming the slot (was: an INFO log line and nothing else)."""

    def _tick_with_due(self, sched, job):
        with patch.object(sched, "get_due_jobs", return_value=[dict(job)]):
            sched.tick(verbose=False, sync=True)

    def test_guard_refusal_books_failed_ledger_row(self, cron_store):
        from cron import scheduler as sched

        job = jobs_mod.create_job(prompt="long run", schedule="40 * * * *")
        job_id = job["id"]
        slot = _at(22, 40, 0)
        jobs = jobs_mod.load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["next_run_at"] = slot
        jobs_mod.save_jobs(jobs)
        job["next_run_at"] = slot
        job["_scheduled_instant"] = slot

        # A prior fire is still in flight when this tick arrives:
        assert sched.try_register_running_job(job_id) is True
        try:
            self._tick_with_due(sched, job)
        finally:
            sched.release_running_job(job_id)

        rows = _ledger_rows(job_id)
        assert len(rows) == 1, f"exactly one ledger row for the skipped fire, got {rows}"
        row = rows[0]
        assert row["status"] == "failed"
        assert row["error"] is not None and "Fire skipped" in row["error"]
        assert slot in row["error"], "the consumed slot must be named in the diagnostic"
        # scheduled_instant is normalized to UTC in the ledger: compare instants.
        assert datetime.fromisoformat(row["scheduled_instant"]) == datetime.fromisoformat(slot)
        assert row["source"] == "builtin"

    def test_guard_refusal_does_not_touch_job_status(self, cron_store):
        from cron import scheduler as sched

        job = jobs_mod.create_job(prompt="long run", schedule="40 * * * *")
        job_id = job["id"]
        slot = _at(22, 40, 0)
        jobs = jobs_mod.load_jobs()
        for j in jobs:
            if j["id"] == job_id:
                j["next_run_at"] = slot
        jobs_mod.save_jobs(jobs)
        job["next_run_at"] = slot
        job["_scheduled_instant"] = slot

        assert sched.try_register_running_job(job_id) is True
        try:
            self._tick_with_due(sched, job)
        finally:
            sched.release_running_job(job_id)

        reloaded = next(j for j in jobs_mod.load_jobs() if j["id"] == job_id)
        # jobs.json bookkeeping (what L3 reads) must be untouched: no error,
        # no last_status flip, no failure streak.
        assert reloaded.get("last_status") is None
        assert reloaded.get("last_error") in (None, "")
        assert reloaded.get("failure_streak") in (None, 0)

    def test_no_row_when_guard_passes(self, cron_store):
        from cron import scheduler as sched

        job = jobs_mod.create_job(prompt="ok", schedule="40 * * * *")
        job_id = job["id"]
        # Nothing in flight: dispatch proceeds normally (no-agent-free agent job
        # is refused headless? No — headless=False here). Block the real run
        # body so the test stays store-level; the guard itself passed.
        with patch.object(sched, "run_one_job", return_value=True):
            self._tick_with_due(sched, job)
        rows = _ledger_rows(job_id)
        assert all(r["error"] is None or "Fire skipped" not in r["error"] for r in rows)
        assert job_id not in sched.get_running_job_ids()

    def test_skip_row_does_not_shadow_live_run(self, cron_store):
        """Consumer regression: a terminal skip trace is newer than a live
        'running' row of the same job — latest-row consumers (ownership,
        delivery settlement, list display) must still see the live attempt."""
        from cron import executions as exec_mod

        job = jobs_mod.create_job(prompt="long run", schedule="40 * * * *")
        job_id = job["id"]
        live = exec_mod.create_execution(job_id, source="builtin")
        assert exec_mod.mark_execution_running(live["id"]) is not None
        slot = _at(23, 40, 0)
        skip = exec_mod.create_execution(job_id, source="builtin", scheduled_instant=slot)
        exec_mod.finish_execution(
            skip["id"], success=False,
            error=f"{exec_mod.FIRE_SKIP_ERROR_PREFIX} previous run still in flight "
                  f"(in-flight dedup guard); scheduled slot {slot} was consumed without a run.")
        try:
            assert exec_mod.latest_execution(job_id, exclude_fire_skips=True)["id"] == live["id"]
            inflight = exec_mod.live_inflight_execution(job_id)
            assert inflight is not None and inflight["id"] == live["id"], (
                "ownership query shadowed by the skip trace (FIX-2075 regression)")
            assert exec_mod.latest_executions(
                [job_id], exclude_fire_skips=True)[job_id]["id"] == live["id"]
            # History consumers keep the full picture — the trace stays visible:
            all_rows = exec_mod.list_executions(job_id=job_id)
            assert {r["id"] for r in all_rows} == {live["id"], skip["id"]}
        finally:
            exec_mod.finish_execution(live["id"], success=True)
