"""Refused occurrences are recorded, never silently consumed (RED/GREEN).

A recurring job whose prior run is still in flight at its next scheduled occurrence is
refused by the same-job exclusion guard on purpose: a second run must not start while a
prior run may still hold the job. But the tick has already advanced ``next_run_at`` past
the occurrence, so the refusal used to consume it with a bare INFO line — no execution
row — and the ``pending_slot`` recovery stamp of the earliest unclaimed occurrence was
silently replaced by the next occurrence's stamp. While a run stayed non-terminal, every
later occurrence disappeared the same way, violating the scheduler contract
(cron/AGENTS.md): "Never drop a slot silently".

Fixture shape requested with the report: consecutive occurrences against a still-live
prior Future, a late completion, and an indefinitely blocked Future.

Contract pinned here:
  * every refused occurrence leaves exactly one terminal execution row (never started,
    no side effects) naming its scheduled instant;
  * the pending-slot stamp of the earliest refused occurrence survives later scans
    instead of being clobbered;
  * single-job exclusion holds: no second run starts while the prior Future is live,
    and the stale sweep never releases a claim whose Future is still executing;
  * a late completion lets the job run again — and the refused occurrences are not
    replayed.
"""
from __future__ import annotations

import concurrent.futures
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class _Clock:
    """Controllable ``_hermes_now`` replacement for consecutive-occurrence ticks."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron env + a recurring no_agent interval job, due NOW."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    job = jobs_mod.create_job(
        prompt="probe",
        schedule="every 10m",
        no_agent=True,
        script="probe.py",
    )
    now = datetime.now(timezone.utc)
    jobs_mod.update_job(job["id"], {"next_run_at": (now - timedelta(minutes=1)).isoformat()})

    script = hermes_home / "scripts" / "probe.py"
    script.write_text("print('ok')\n")

    return {"home": hermes_home, "job_id": job["id"]}


class TestRefusedOccurrenceRecords:
    def _setup(self, cron_env, monkeypatch):
        from cron import executions as E
        from cron import scheduler as S

        env = cron_env
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])
        import cron.jobs as J

        clock = _Clock(datetime.now(timezone.utc))
        monkeypatch.setattr(J, "_hermes_now", clock)
        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()
        return S, E, J, env, clock

    def _simulate_prior_run(self, S, job_id):
        """A prior run of the job that is in flight and never terminalizes (blocked Future)."""
        assert S.try_register_running_job(job_id), "simulated prior run must register"
        key = S._inflight_key(job_id)
        fut = concurrent.futures.Future()  # never completes
        S._running_futures[key] = fut
        return fut

    def _tick_at(self, S, J, env, clock, when: datetime):
        clock.now = when
        return S.tick(verbose=False, sync=True)

    def _tick_next_occurrence(self, S, J, env, clock):
        """Advance the clock to the job's stored next occurrence and tick there."""
        next_run = J.get_job(env["job_id"])["next_run_at"]
        return self._tick_at(S, J, env, clock, datetime.fromisoformat(next_run))

    def test_refused_occurrence_leaves_a_record(self, cron_env, monkeypatch):
        """The refused occurrence is recorded as a terminal, never-started execution row."""
        S, E, J, env, clock = self._setup(cron_env, monkeypatch)
        from cron.occurrences import scheduled_instant

        job_id = env["job_id"]
        occurrence = J.get_job(job_id)["next_run_at"]
        self._simulate_prior_run(S, job_id)

        assert self._tick_next_occurrence(S, J, env, clock) == 0, "nothing may dispatch"

        rows = E.list_executions(job_id=job_id)
        assert len(rows) == 1, f"exactly one record for the refused occurrence, got {rows}"
        row = rows[0]
        assert row["scheduled_instant"] == scheduled_instant(occurrence)
        assert row["status"] == "failed"
        assert row["started_at"] is None, "the refused occurrence must never start"
        assert "still in flight" in (row["error"] or "")

    def test_each_refused_occurrence_keeps_its_own_record_and_slot(self, cron_env, monkeypatch):
        """Consecutive occurrences: one record each, and the earliest pending slot survives."""
        S, E, J, env, clock = self._setup(cron_env, monkeypatch)
        from cron.occurrences import scheduled_instant, stored_pending_slot

        job_id = env["job_id"]
        self._simulate_prior_run(S, job_id)

        instants = []
        for _ in range(3):
            record = J.get_job(job_id)
            instants.append(scheduled_instant(record["next_run_at"]))
            assert self._tick_next_occurrence(S, J, env, clock) == 0

        rows = E.list_executions(job_id=job_id)
        assert len(rows) == 3, f"one record per refused occurrence, got {len(rows)}"
        assert sorted(r["scheduled_instant"] for r in rows) == sorted(instants)
        assert all(r["started_at"] is None for r in rows)

        # The earliest unclaimed stamp must survive the later scans (it is the one the
        # restore path has to recover), not be silently replaced by a newer instant.
        assert stored_pending_slot(J.get_job(job_id)) == instants[0]

    def test_blocked_future_never_doubles_the_run(self, cron_env, monkeypatch):
        """An indefinitely blocked prior run keeps its claim; no second run ever starts."""
        S, E, J, env, clock = self._setup(cron_env, monkeypatch)

        job_id = env["job_id"]
        fut = self._simulate_prior_run(S, job_id)

        for _ in range(3):
            assert self._tick_next_occurrence(S, J, env, clock) == 0

        assert not fut.done()
        assert job_id in S.get_running_job_ids(), (
            "the stale sweep must not release a claim whose Future is still executing")
        rows = E.list_executions(job_id=job_id)
        assert len(rows) == 3, "every refused occurrence is still recorded"
        assert all(r["started_at"] is None for r in rows), (
            "single-job exclusion: the refused occurrences never start")

    def test_late_completion_recovers_without_replaying(self, cron_env, monkeypatch):
        """A prior run that completes late lets the job run again — refused ones don't replay."""
        S, E, J, env, clock = self._setup(cron_env, monkeypatch)
        from cron.occurrences import scheduled_instant

        job_id = env["job_id"]
        fut = self._simulate_prior_run(S, job_id)

        refused = []
        for _ in range(2):
            refused.append(scheduled_instant(J.get_job(job_id)["next_run_at"]))
            assert self._tick_next_occurrence(S, J, env, clock) == 0

        # Late completion of the prior run: the worker releases its claim and its outcome
        # lands (mark_job_run — the real completion path, which also consumes the job's
        # pending stamp).
        fut.set_result(True)
        S.release_running_job(job_id)
        assert J.mark_job_run(job_id, True)
        assert job_id not in S.get_running_job_ids()

        # Consecutive occurrences after the release: the job must run again.
        for _ in range(4):
            self._tick_next_occurrence(S, J, env, clock)
            rows = E.list_executions(job_id=job_id)
            if any(r["status"] == "completed" for r in rows):
                break

        rows = E.list_executions(job_id=job_id)
        completed = [r for r in rows if r["status"] == "completed"]
        assert completed, "the job must run again once the prior run completes"
        assert {r["scheduled_instant"] for r in completed}.isdisjoint(refused), (
            "a refused occurrence is recorded, never re-executed after a completed run")
        for instant in refused:
            assert sum(1 for r in rows if r["scheduled_instant"] == instant) == 1, (
                "exactly one record per refused occurrence")

    def test_lost_completion_restores_the_earliest_slot_once(self, cron_env, monkeypatch):
        """A run whose completion never lands leaves the earliest refused occurrence claimable
        through the recovery stamp — restored once, never N replays."""
        S, E, J, env, clock = self._setup(cron_env, monkeypatch)
        from cron.occurrences import scheduled_instant, stored_pending_slot

        job_id = env["job_id"]
        fut = self._simulate_prior_run(S, job_id)

        refused = []
        for _ in range(2):
            refused.append(scheduled_instant(J.get_job(job_id)["next_run_at"]))
            assert self._tick_next_occurrence(S, J, env, clock) == 0

        # The worker vanishes without its outcome landing (process death mid-run): claim
        # released, no mark_job_run, so the pending stamp survives.
        fut.set_result(True)
        S.release_running_job(job_id)
        assert stored_pending_slot(J.get_job(job_id)) == refused[0], (
            "the earliest refused occurrence keeps the recovery stamp")

        self._tick_next_occurrence(S, J, env, clock)
        rows = E.list_executions(job_id=job_id)
        completed = [r for r in rows if r["status"] == "completed"]
        assert len(completed) <= 1, "the restored slot runs at most once, never N replays"
        if completed:
            assert completed[0]["scheduled_instant"] == refused[0], (
                "the recovery stamp belongs to the earliest refused occurrence")
        # The later refused occurrence is record-only: it is never replayed behind the first.
        assert sum(1 for r in rows if r["scheduled_instant"] == refused[1]) == 1
        assert all(r["started_at"] is None for r in rows if r["scheduled_instant"] == refused[1])
