"""Persisted-state stale-error recovery for recurring cron jobs (t_8b5480b3).

The 2026-08-14 incident (t_20e23f84): 4 recurring no_agent interval jobs
EAGAIN-failed at 12:50 and then recorded ZERO executions for ~1h47m (~9 missed
cycles) even after the substrate recovered, and the wedge SURVIVED a gateway
restart.  The in-memory stale-claim sweep (cron/scheduler.py, t_3778a491)
heals a leaked `_running_job_ids` claim in-process, but that is only the
in-process half.  A recurring job whose *persisted* state shows
`last_status == "error"` and whose `next_run_at` was re-armed into the future
by `mark_job_run` is invisible to the in-memory sweep: it is not in the
running set (nothing force-releases it) and it is not due (so `get_due_jobs`
never returns it).  It just sits — exactly the restart-surviving symptom
documented in the incident.

This recovery (cron/jobs.py `_get_due_jobs_locked`) re-arms `next_run_at` to
now for a recurring job that:
  * is in a persisted `last_status == "error"` state, AND
  * has NOT re-fired within a full cadence (`last_run_at` older than
    cadence + grace — so it is NOT a normal transient-error retry that will
    fire on its own soon), AND
  * is not currently running in this process.

The scheduler then re-dispatches it on the next tick WITHOUT force-run/resume
— the built-in equivalent of mech_red_guard's `cron resume`.

Deterministic RED/GREEN: on unfixed code the wedged job is left untouched (no
execution row, no re-dispatch); with the recovery it re-fires and completes.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron env + a recurring no_agent interval job."""
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
    script = hermes_home / "scripts" / "probe.py"
    script.write_text("print('ok')\n")
    return {"home": hermes_home, "job_id": job["id"]}


def _setup(cron_env, monkeypatch):
    from cron import scheduler as S
    from cron import executions as E
    import cron.jobs as J

    env = cron_env
    monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
    monkeypatch.setattr(S, "_hermes_home", env["home"])
    return S, E, J, env


def _persist_stale_error(J, job_id, *, error_age_minutes=110):
    """Persist the incident wedge state: last_status=error, last_run stale by
    more than a full cadence, next_run_at re-armed into the future."""
    now = datetime.now(timezone.utc)
    J.update_job(
        job_id,
        {
            "next_run_at": (now + timedelta(minutes=5)).isoformat(),
            "last_status": "error",
            "last_error": "EAGAIN [Errno 11] Resource temporarily unavailable",
            "last_run_at": (now - timedelta(minutes=error_age_minutes)).isoformat(),
        },
    )


class TestPersistedStaleErrorRecovery:
    def test_stale_error_recurring_job_redispatches_without_force_run(
        self, cron_env, monkeypatch
    ):
        """GREEN: a recurring job wedged in a persisted stale-error state with
        next_run_at parked in the future is re-armed and re-dispatches on the
        next tick — no force-run, no resume."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        _persist_stale_error(J, job_id, error_age_minutes=110)

        job = J.get_job(job_id)
        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            n = S.tick(verbose=False, sync=True)

        latest = E.latest_execution(job_id)
        assert latest is not None, (
            "wedged stale-error job must be re-dispatched without force-run"
        )
        assert latest["status"] == "completed"
        # The recovery must be countable / probe-visible.
        from cron import jobs as Jmod
        stats = Jmod.get_persisted_error_recovery_stats()
        assert stats["persisted_error_recoveries"] >= 1

    def test_two_consecutive_auto_fires_after_persisted_recovery(
        self, cron_env, monkeypatch
    ):
        """GREEN: after the persisted recovery re-arms the job, it fires on
        consecutive ticks (no manual intervention)."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        _persist_stale_error(J, job_id, error_age_minutes=110)
        job = J.get_job(job_id)
        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            S.tick(verbose=False, sync=True)
        latest1 = E.latest_execution(job_id)
        assert latest1["status"] == "completed"

        # Re-arm due normally and tick again: fire #2.
        now = datetime.now(timezone.utc)
        J.update_job(job_id, {"next_run_at": (now - timedelta(minutes=1)).isoformat()})
        with mock.patch("cron.jobs.load_jobs", return_value=[J.get_job(job_id)]):
            S.tick(verbose=False, sync=True)
        latest2 = E.latest_execution(job_id)
        assert latest2["status"] == "completed"
        assert latest2["id"] != latest1["id"], "two distinct executions"

    def test_recent_error_within_cadence_not_force_rearmed(self, cron_env, monkeypatch):
        """The discriminator must NOT re-arm a normal transient error whose
        last_run is within a full cadence — that job retries on its own
        schedule and must be left alone (otherwise we'd over-fire healthy
        erroring jobs)."""
        S, E, J, env = _setup(cron_env, monkeypatch)
        job_id = env["job_id"]

        _persist_stale_error(J, job_id, error_age_minutes=5)  # within cadence

        job = J.get_job(job_id)
        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            S.tick(verbose=False, sync=True)

        latest = E.latest_execution(job_id)
        assert latest is None, (
            "a within-cadence error is a normal retry — must not be "
            "force-re-armed onto an immediate fire"
        )
