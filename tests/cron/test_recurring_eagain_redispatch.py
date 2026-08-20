"""Deterministic reproduction of the recurring-cron EAGAIN in-memory wedge (t_8b5480b3).

Scenario modelled on the 2026-08-14 incident: a recurring no_agent interval job
whose script subprocess raises EAGAIN ([Errno 11] Resource temporarily
unavailable) during a substrate thread-exhaustion spike. After the failure is
recorded (terminal 'failed' execution row), the job must be re-dispatched on
the NEXT tick once the substrate recovers — with no force-run.

SCOPE NOTE (honest): this file exercises the in-memory EAGAIN-release class —
the claim taken by ``_submit_with_guard`` is released when the submit/init
path fails, so the NEXT tick re-dispatches. It passes on unfixed base because
a terminal ``failed`` execution row does not itself suppress a recurring job's
next due fire; that is pre-existing behavior, not the fix. The persisted-state
(restart-surviving) half of the incident — a recurring job whose persisted
``last_status=error`` and ``next_run_at`` parked in the future is never
re-dispatched without force-run/resume — is covered by
``test_recurring_persisted_error_recovery.py``, which provides the clean
behavioral RED (unfixed: no execution) / GREEN (fixed: re-dispatched) for the
new recovery path.

This file drives the REAL `tick()` end-to-end against a throwaway HERMES_HOME:
  tick 1 -> script EAGAINs (subprocess.run raises OSError 11) -> failed exec row
  tick 2 -> substrate recovered (script runs clean) -> job MUST fire again
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def wedge_env(tmp_path, monkeypatch):
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

    script = hermes_home / "scripts" / "probe.py"
    script.write_text("print('ok')\n")

    # Create a recurring no_agent interval job.
    job = jobs_mod.create_job(
        prompt="probe",
        schedule="every 10m",
        no_agent=True,
        script="probe.py",
    )
    # Force it due now.
    now = datetime.now(timezone.utc)
    jobs_mod.update_job(job["id"], {"next_run_at": (now - timedelta(minutes=1)).isoformat()})

    return {"home": hermes_home, "job_id": job["id"]}


class TestEAGAINRecurringRedispatches:
    def _make_script_eagain(self, env, monkeypatch):
        """Make the next subprocess.Popen raise EAGAIN once, then pass.

        The script runner spawns via Popen (polling loop for cancel/timeout),
        so the substrate-failure injection point is the Popen constructor.
        """
        import cron.scheduler as sched_mod
        state = {"n": 0}

        class _OkProc:
            def __init__(self, argv, **kwargs):
                self.returncode = 0

            def poll(self):
                return self.returncode

            def communicate(self, timeout=None):
                return ("ok\n", "")

            def wait(self, timeout=None):
                return 0

        def fake_popen(argv, **kwargs):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError(11, "Resource temporarily unavailable")
            return _OkProc(argv, **kwargs)

        monkeypatch.setattr(sched_mod.subprocess, "Popen", fake_popen)
        return state

    def test_eagain_then_redispatched_on_next_tick(self, wedge_env, monkeypatch, tmp_path):
        """Tick 1 records a failed execution (EAGAIN); tick 2 must re-fire."""
        from cron import scheduler as S
        from cron import executions as E

        env = wedge_env
        # Point the executions ledger at the throwaway home.
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])
        monkeypatch.setattr(S, "get_due_jobs", S.get_due_jobs)  # no-op, keep real

        state = self._make_script_eagain(env, monkeypatch)

        # Tick 1: EAGAIN failure.
        n1 = S.tick(verbose=False, sync=True)
        # Assert the failure was recorded.
        latest = E.latest_execution(env["job_id"])
        assert latest is not None, "tick 1 must create an execution"
        assert latest["status"] == "failed", f"expected failed, got {latest['status']}"

        # The job must still be scheduled (recurring), next_run_at advanced.
        import cron.jobs as J
        job = J.get_job(env["job_id"])
        assert job["enabled"] is True
        assert job["state"] == "scheduled"
        assert job["next_run_at"] is not None

        # Force next_run_at due again (simulate the substrate recovery tick).
        now = datetime.now(timezone.utc)
        J.update_job(env["job_id"], {"next_run_at": (now - timedelta(minutes=1)).isoformat()})

        # Tick 2: script passes -> job must fire (completed execution).
        n2 = S.tick(verbose=False, sync=True)
        latest2 = E.latest_execution(env["job_id"])
        assert latest2 is not None
        assert latest2["status"] == "completed", (
            f"job must be re-dispatched after EAGAIN recovery, got {latest2['status']}"
        )
        assert state["n"] >= 2

    def test_trigger_job_unwedges_persisted_state(self, wedge_env, monkeypatch, tmp_path):
        """The incident force-run (`cron run <id>` -> trigger_job) resets the
        persisted due state so the next tick fires the job. This is the
        operator escape that cleared each wedge."""
        from cron import scheduler as S
        from cron import executions as E
        from cron.jobs import trigger_job, update_job

        env = wedge_env
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])

        self._make_script_eagain(env, monkeypatch)
        n1 = S.tick(verbose=False, sync=True)

        # Simulate the persisted non-dispatch state: next_run_at far in the
        # future (job not due) but still enabled/scheduled — the observed
        # wedge where get_due_jobs never returns it.
        from datetime import timezone as tz
        far = datetime.now(tz.utc) + timedelta(days=1)
        update_job(env["job_id"], {"next_run_at": far.isoformat()})
        n2 = S.tick(verbose=False, sync=True)  # not due -> no dispatch

        # Force-run (trigger_job) sets next_run_at = now -> due again.
        triggered = trigger_job(env["job_id"])
        assert triggered is not None
        n3 = S.tick(verbose=False, sync=True)
        latest = E.latest_execution(env["job_id"])
        assert latest["status"] == "completed", "force-run must clear the wedge"
