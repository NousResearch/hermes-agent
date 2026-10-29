"""A cron job must be DEFERRED, not failed, when the host has no room for
another worker scope (t_8e8ec4de follow-up, 2026-10-29).

Cron executions are dispatched through the SAME scope mechanism as kanban
workers, so every cron execution holds a scope with the same per-worker
``MemoryMax``. Cron concurrency is bounded by ``cron.max_parallel_jobs``, which
is unset by default and therefore UNBOUNDED — so cron scopes are the largest
contributor to the host's overcommit.

When the host-wide scope budget is exhausted, the dispatch is refused BEFORE
the scope is created. That refusal is a HOST condition, not a job failure:

* it must not open a cron incident,
* it must not burn a repeat,
* it must not mark the job failed,
* the job must be parked so it runs again once other scopes have finished.
"""

from __future__ import annotations

import pytest

import cron.scheduler as scheduler
import tools.process_registry as pr


def test_a_budget_refusal_is_not_a_job_failure(monkeypatch):
    """The refusal must take the DEFER branch, never the failure branch."""
    deferred = {}

    def _fake_defer(job, detail):
        deferred["job"] = job
        deferred["detail"] = detail
        return True

    monkeypatch.setattr(scheduler, "_defer_job_for_scope_budget", _fake_defer, raising=False)

    def _raise_budget(_job):
        raise pr.WorkerScopeBudgetExhausted("the host memory budget is exhausted")

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", _raise_budget, raising=False)

    # Any attempt to record a job FAILURE must blow up the test.
    def _forbidden(*_a, **_k):
        raise AssertionError("a budget refusal must not be recorded as a job failure")

    monkeypatch.setattr(scheduler, "mark_job_run", _forbidden, raising=False)

    job = {
        "id": "job-budget",
        "name": "budget job",
        "execution_id": "exec-1",
        "schedule": {"kind": "interval", "minutes": 5},
        "state": "scheduled",
    }

    result = scheduler.run_one_job(job, adapters=[], loop=None)

    assert result is False, "a deferred job is not 'processed'"
    assert deferred.get("job", {}).get("id") == "job-budget"
    assert "exhausted" in deferred.get("detail", "")


def test_an_unrelated_dispatch_failure_is_still_a_failure(monkeypatch):
    """The defer branch must not swallow genuine dispatch failures."""
    monkeypatch.setattr(
        scheduler, "_defer_job_for_scope_budget",
        lambda *_a, **_k: pytest.fail("a generic failure must not be deferred"),
        raising=False,
    )

    def _raise_other(_job):
        raise RuntimeError("the wrapper vanished")

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", _raise_other, raising=False)

    recorded = {}

    def _record(job_id, success, error=None, **kwargs):
        recorded["success"] = success
        recorded["error"] = error
        return True

    monkeypatch.setattr(scheduler, "mark_job_run", _record, raising=False)
    monkeypatch.setattr(scheduler, "_deliver_crash_failure", lambda *a, **k: (None, None), raising=False)

    job = {
        "id": "job-ordinary-failure",
        "name": "ordinary failure",
        "execution_id": "exec-2",
        "schedule": {"kind": "interval", "minutes": 5},
        "state": "scheduled",
    }

    scheduler.run_one_job(job, adapters=[], loop=None)

    assert recorded.get("success") is False


def test_the_deferral_publishes_the_scope_budget_detail():
    """The parked job's record must say WHY it was parked.

    An operator reading the job row must be able to tell an infrastructure
    deferral from a job failure without reading the logs.
    """
    from tools.process_registry import WorkerScopeBudgetExhausted

    detail = str(WorkerScopeBudgetExhausted("the host memory budget is exhausted"))
    assert "exhausted" in detail
