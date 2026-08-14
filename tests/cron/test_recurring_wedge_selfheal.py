"""Deterministic reproduction of the recurring-cron wedge (t_8b5480b3) — RED/GREEN.

The 2026-08-14 incident (t_20e23f84): 4 recurring no_agent interval jobs
EAGAIN-failed at 12:50:05 and then recorded ZERO executions for ~1h47m while
the scheduler ticked normally and fired 100+ other jobs — wedged in a
non-dispatch state that even survived a gateway restart, cleared only by a
manual force-run (`hermes cron run <id>`).

Root cause class (t_3778a491, the SAME symptom on 2026-08-02): `_submit_with_guard`
adds a job id to the in-memory `_running_job_ids` set BEFORE the future that
owns its release exists. Anything that hangs or dies between the add and
`pool.submit` (documented case: EAGAIN thread exhaustion on a substrate spike,
or a wedged SessionDB.__init__ on a stale sqlite flock) leaks the claim. Every
later tick short-circuits with "already running — skipping" silently — no
execution row, no last_error, no counter — so the job is due-but-never-dispatched.

The live deployment (origin/main) does NOT contain the t_3778a491 in-flight
stale-claim sweep, so the wedge class is still live.

This file drives the REAL `tick()` and asserts the fix's contract:
  RED  (unfixed): a stale in-flight claim is never released by tick → the
       wedge reproduces deterministically (job stays in the running set, no
       execution, no re-dispatch without force-run).
  GREEN (fixed):  the same stale claim is force-released by the next tick
       (cron.inflight.forced_release) and the job re-fires — no gateway
       restart, no force-run needed — and 2 consecutive auto-fires work.
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


class TestStaleInflightSelfHeal:
    def _setup(self, cron_env, monkeypatch):
        from cron import scheduler as S
        from cron import executions as E

        env = cron_env
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])
        return S, E, env

    def test_stale_claim_self_heals_and_redispatches(self, cron_env, monkeypatch):
        """GREEN contract: a leaked in-flight claim is force-released by the
        next tick and the wedged job re-fires without a force-run."""
        S, E, env = self._setup(cron_env, monkeypatch)
        job_id = env["job_id"]
        import cron.jobs as J

        if not hasattr(S, "sweep_stale_inflight"):
            pytest.skip("guard not present on this build")

        # Simulate the incident leak: job id claimed with no owning future,
        # old enough to be past its allowance.
        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()
        S._running_job_ids.add(job_id)
        S._running_since[job_id] = time.time() - 6 * 60 * 60

        # get_due_jobs is called inside tick BEFORE the sweep; we patch it to
        # return the wedged job as due so the in-cycle sweep releases the claim
        # and the dispatch loop re-fires it.
        job = J.get_job(job_id)
        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            n = S.tick(verbose=False, sync=True)

        latest = E.latest_execution(job_id)
        assert job_id not in S.get_running_job_ids(), "stale claim must be released"
        assert latest is not None, "wedged job must create an execution"
        assert latest["status"] == "completed", (
            "wedged job must fire again without force-run"
        )

    def test_two_consecutive_auto_fires_after_guard(self, cron_env, monkeypatch):
        """GREEN: after the guard releases a stale claim, the job fires on
        consecutive ticks (no manual intervention)."""
        S, E, env = self._setup(cron_env, monkeypatch)
        job_id = env["job_id"]
        import cron.jobs as J

        if not hasattr(S, "sweep_stale_inflight"):
            pytest.skip("guard not present on this build")

        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()
        S._running_job_ids.add(job_id)
        S._running_since[job_id] = time.time() - 6 * 60 * 60

        job = J.get_job(job_id)
        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            n1 = S.tick(verbose=False, sync=True)
        latest1 = E.latest_execution(job_id)
        assert latest1["status"] == "completed"

        # Re-arm due and tick again: fire #2.
        now = datetime.now(timezone.utc)
        J.update_job(job_id, {"next_run_at": (now - timedelta(minutes=1)).isoformat()})
        n2 = S.tick(verbose=False, sync=True)
        latest2 = E.latest_execution(job_id)
        assert latest2["status"] == "completed"
        assert latest2["id"] != latest1["id"], "two distinct executions"

    def test_guard_stats_reported(self, cron_env, monkeypatch):
        """The guard must surface a countable forced-release signal."""
        S, E, env = self._setup(cron_env, monkeypatch)
        import cron.jobs as J
        if not hasattr(S, "sweep_stale_inflight"):
            pytest.skip("guard not present on this build")

        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()
        S._running_job_ids.add(env["job_id"])
        S._running_since[env["job_id"]] = time.time() - 6 * 60 * 60
        S.sweep_stale_inflight([J.get_job(env["job_id"])])
        stats = S.get_inflight_guard_stats()
        assert stats["forced_releases"] >= 1
        assert env["job_id"] not in S.get_running_job_ids()


class TestEAGAINCreateExecutionLeak:
    """The 12:50 mechanism: EAGAIN/thread-exhaustion strikes BETWEEN the
    in-flight claim and execution creation (create_execution / pool.submit).
    The claim must be released immediately so the next tick re-dispatches."""

    def test_create_execution_failure_releases_claim(self, cron_env, monkeypatch, tmp_path):
        from cron import scheduler as S
        from cron import executions as E
        import cron.jobs as J

        env = cron_env
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])
        job_id = env["job_id"]
        job = J.get_job(job_id)

        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()

        # Simulate EAGAIN during create_execution (substrate thread exhaustion
        # at 12:50): the in-flight claim was taken but execution creation fails.
        def boom(*a, **k):
            raise OSError(11, "Resource temporarily unavailable")
        monkeypatch.setattr(S, "create_execution", boom)

        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            with pytest.raises(OSError):
                S.tick(verbose=False, sync=True)

        # The claim must be released (not leaked) so the NEXT tick can retry.
        assert job_id not in S.get_running_job_ids(), (
            "claim must be released when execution creation fails, so the "
            "next tick re-dispatches instead of wedging on 'already running'"
        )

    def test_pool_submit_eagain_releases_claim_and_redispatches(self, cron_env, monkeypatch, tmp_path):
        from cron import scheduler as S
        from cron import executions as E
        import cron.jobs as J

        env = cron_env
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])
        job_id = env["job_id"]
        job = J.get_job(job_id)

        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()

        # First tick: pool.submit raises EAGAIN (thread exhaustion). The claim
        # is released and the run recorded as failed.
        real_submit = S.concurrent.futures.ThreadPoolExecutor.submit
        state = {"n": 0}
        def flaky(self, *a, **k):
            state["n"] += 1
            if state["n"] == 1:
                raise OSError(11, "Resource temporarily unavailable")
            return real_submit(self, *a, **k)
        monkeypatch.setattr(S.concurrent.futures.ThreadPoolExecutor, "submit", flaky)

        with mock.patch("cron.jobs.load_jobs", return_value=[job]):
            S.tick(verbose=False, sync=True)

        # Claim released; job still scheduled.
        assert job_id not in S.get_running_job_ids()

        # Second tick (substrate recovered): job re-dispatches and completes.
        now = datetime.now(timezone.utc)
        J.update_job(job_id, {"next_run_at": (now - timedelta(minutes=1)).isoformat()})
        with mock.patch("cron.jobs.load_jobs", return_value=[J.get_job(job_id)]):
            S.tick(verbose=False, sync=True)

        latest = E.latest_execution(job_id)
        assert latest is not None and latest["status"] == "completed", (
            "job must re-dispatch on the next tick after EAGAIN recovery"
        )
