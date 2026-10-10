"""An occurrence skipped as already-running must leave an execution row (#127723).

Tick order on origin/main: ``advance_next_runs`` runs before ``_submit_with_guard``
(cron/scheduler_tick.py), and the guard skips with "already running" when a prior
tick's run is still in flight (cron/scheduler.py) — while the pending-slot restore
stays suppressed for running jobs (cron/occurrences.py). A stuck execution therefore
silently consumes later occurrences: next_run_at moved on, no execution row.

Contract: the skip path writes an explicit terminal row bound to the consumed
occurrence (create + finish without ever starting). ``record_cron_finish`` reports
such never-started rows as ``skipped``, and ``completed_occurrence`` only honours
``completed`` rows, so the row is visible without changing dedupe.
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _guard_job():
    return {
        "id": "skip-row-job",
        "name": "skip-row-test",
        "prompt": "test",
        "schedule": "every 5m",
        "enabled": True,
        "next_run_at": "2020-01-01T00:00:00",
        "deliver": "local",
        "_scheduled_instant": "2020-01-01T00:00:00+00:00",
    }


class TestAlreadyRunningSkipRow:
    def test_skip_writes_terminal_occurrence_row(self, monkeypatch):
        """RED (#127723): the skip path must create + finish a row for the occurrence."""
        import cron.scheduler as sched

        sched._running_job_ids.clear()
        job = _guard_job()
        sched._running_job_ids.add(sched._inflight_key(job["id"]))

        created = []
        finished = []
        monkeypatch.setattr(
            sched, "create_execution",
            lambda job_id, **kw: created.append((job_id, kw)) or {"id": "exec-skip-1"},
        )
        monkeypatch.setattr(
            sched, "finish_execution",
            lambda execution_id, **kw: finished.append((execution_id, kw))
            or {"id": execution_id, "status": "failed"},
        )

        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                assert sched._submit_with_guard(job, pool, lambda j: True) is None
            finally:
                pool.shutdown(wait=False)
        finally:
            sched._running_job_ids.discard(sched._inflight_key(job["id"]))

        assert created == [
            (job["id"], {"source": "builtin",
                         "scheduled_instant": job["_scheduled_instant"]})
        ], "skipped occurrence must create an execution row bound to its instant"
        assert len(finished) == 1
        execution_id, kw = finished[0]
        assert execution_id == "exec-skip-1"
        assert kw.get("success") is False
        assert "already running" in str(kw.get("error", ""))

    def test_skip_row_failure_never_breaks_tick(self, monkeypatch):
        """A degraded ledger must not turn a skip into a tick crash."""
        import cron.scheduler as sched

        sched._running_job_ids.clear()
        job = _guard_job()
        sched._running_job_ids.add(sched._inflight_key(job["id"]))

        def _boom(*_a, **_kw):
            raise OSError("ledger unavailable")

        monkeypatch.setattr(sched, "create_execution", _boom)

        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                assert sched._submit_with_guard(job, pool, lambda j: True) is None
            finally:
                pool.shutdown(wait=False)
        finally:
            sched._running_job_ids.discard(sched._inflight_key(job["id"]))


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


class TestStuckRunConsumesOccurrenceVisibly:
    def test_real_tick_leaves_row_for_skipped_occurrence(self, cron_env, monkeypatch):
        """End-to-end #127723: a stuck (fresh, same-process) in-flight run makes the
        next tick consume the occurrence with no execution row. After the fix the
        tick leaves a terminal row bound to the consumed instant."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from cron import scheduler as S
        from cron import executions as E
        import cron.jobs as J

        env = cron_env
        monkeypatch.setattr(E, "EXECUTIONS_FILE", env["home"] / "cron" / "executions.db")
        monkeypatch.setattr(S, "_hermes_home", env["home"])
        S._running_job_ids.clear()
        S._running_since.clear()
        S._running_futures.clear()
        try:
            job_id = env["job_id"]
            stored = J.get_job(job_id)
            assert stored is not None
            slot = stored["next_run_at"]
            # Stuck run: fresh claim, owned by THIS process (no worker thread behind it).
            S._running_job_ids.add(S._inflight_key(job_id))
            S._running_since[S._inflight_key(job_id)] = time.time()

            assert E.latest_execution(job_id) is None
            assert S.tick(verbose=False, sync=True) == 0

            row = E.latest_execution(job_id)
            assert row is not None, "skipped occurrence must leave an execution row"
            from cron.occurrences import scheduled_instant
            assert row["scheduled_instant"] == scheduled_instant(slot)
            assert row["status"] == "failed"
            assert "already running" in str(row.get("error", ""))
            # A non-completed row must never prove completion (no dedupe side effects).
            from cron.occurrences import completed_occurrence
            refreshed = J.get_job(job_id)
            assert refreshed is not None
            assert completed_occurrence(refreshed, slot) is False
        finally:
            S._running_job_ids.clear()
            S._running_since.clear()
            S._running_futures.clear()
