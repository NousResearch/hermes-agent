"""Regression tests for #86522 — one-shot run_claim cleared on dispatch failure.

``get_due_jobs`` stamps a ``run_claim`` on one-shot jobs before returning them
as due (#59229); ``mark_job_run`` clears it on completion.  When dispatch
itself fails (interpreter shutdown, execution-creation error, executor submit
error) the job never reaches ``mark_job_run`` and the stale claim blocked
re-dispatch until the TTL expired (default 30 min) — a precisely-timed
one-shot reminder arrived up to 30 minutes late with no error surfaced.

The fix (salvaged from PR #87591 by @RelaxJonh) adds
``cron.jobs.clear_run_claim`` and calls it on all three ``_submit_with_guard``
early-exit paths, wrapped best-effort so a failing store can never crash the
tick these paths exist to protect.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.jobs as jobs_mod
from cron.jobs import clear_run_claim


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _make_oneshot(claimed: bool = True) -> dict:
    job = jobs_mod.create_job(prompt="remind me", schedule="30m")
    if claimed:
        jobs = jobs_mod.load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["run_claim"] = {"at": "2026-08-17T10:00:00+00:00", "by": "test:1"}
        jobs_mod.save_jobs(jobs)
    return job


class TestClearRunClaim:
    def test_clears_claim_on_oneshot(self, cron_store):
        job = _make_oneshot(claimed=True)
        assert clear_run_claim(job["id"]) is True
        reloaded = [j for j in jobs_mod.load_jobs() if j["id"] == job["id"]][0]
        assert reloaded.get("run_claim") is None

    def test_noop_when_already_clear(self, cron_store):
        job = _make_oneshot(claimed=False)
        assert clear_run_claim(job["id"]) is False

    def test_never_touches_recurring_jobs(self, cron_store):
        job = jobs_mod.create_job(prompt="tick", schedule="every 10m")
        jobs = jobs_mod.load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["run_claim"] = {"at": "2026-08-17T10:00:00+00:00", "by": "test:1"}
        jobs_mod.save_jobs(jobs)
        assert clear_run_claim(job["id"]) is False
        reloaded = [j for j in jobs_mod.load_jobs() if j["id"] == job["id"]][0]
        assert reloaded.get("run_claim") is not None  # untouched

    def test_unknown_job_id_returns_false(self, cron_store):
        assert clear_run_claim("no-such-job") is False


class TestDispatchFailurePathsClearClaim:
    """Each _submit_with_guard early-exit must clear the one-shot claim so the
    next healthy tick re-dispatches instead of waiting out the 30-min TTL."""

    def _tick_one(self, job):
        from cron import scheduler as sched
        with patch.object(sched, "get_due_jobs", return_value=[dict(job)]):
            return sched.tick(verbose=False, sync=True)

    def test_interpreter_shutdown_path_clears_claim(self, cron_store):
        from cron import scheduler as sched
        job = _make_oneshot(claimed=True)
        with patch.object(sched, "_interpreter_shutting_down", return_value=True):
            self._tick_one(job)
        reloaded = [j for j in jobs_mod.load_jobs() if j["id"] == job["id"]][0]
        assert reloaded.get("run_claim") is None, (
            "shutdown-path dispatch failure must clear run_claim (#86522)"
        )

    def test_execution_creation_failure_clears_claim(self, cron_store):
        from cron import scheduler as sched
        job = _make_oneshot(claimed=True)
        with patch.object(sched, "create_execution", side_effect=RuntimeError("db gone")):
            self._tick_one(job)
        reloaded = [j for j in jobs_mod.load_jobs() if j["id"] == job["id"]][0]
        assert reloaded.get("run_claim") is None
        assert job["id"] not in sched.get_running_job_ids()

    def test_submit_failure_clears_claim(self, cron_store):
        from cron import scheduler as sched
        job = _make_oneshot(claimed=True)

        class _ExplodingPool:
            def submit(self, *a, **k):
                raise RuntimeError("cannot schedule new futures")

        pool = _ExplodingPool()
        with patch.object(sched, "_get_parallel_pool", return_value=pool), \
             patch.object(sched, "_get_sequential_pool", return_value=pool):
            self._tick_one(job)
        reloaded = [j for j in jobs_mod.load_jobs() if j["id"] == job["id"]][0]
        assert reloaded.get("run_claim") is None
        assert job["id"] not in sched.get_running_job_ids()

    def test_clear_failure_is_best_effort_not_fatal(self, cron_store):
        """A raising clear_run_claim (corrupt store, teardown I/O error) must
        not crash the tick — these early-exit paths exist to skip cleanly; the
        claim then simply expires at the TTL."""
        from cron import scheduler as sched
        job = _make_oneshot(claimed=True)
        with patch.object(sched, "_interpreter_shutting_down", return_value=True), \
             patch.object(sched, "clear_run_claim", side_effect=OSError(24, "Too many open files")):
            n = self._tick_one(job)  # must not raise
        assert n == 0
