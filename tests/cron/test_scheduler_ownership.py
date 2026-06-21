"""Scheduler-ownership guard tests.

A non-authoritative cron ticker (e.g. the Desktop dashboard backend) must
defer to a gateway that already owns the scheduler for a profile, so jobs only
ever execute from the authorised process ancestry. The ``.tick.lock`` alone
gives at-most-once execution but not deterministic provenance, which macOS
TCC / Full Disk Access depends on.
"""

import logging

import pytest

import cron.scheduler as sched


def _make_job(job_id="job1"):
    return {
        "id": job_id, "name": job_id, "prompt": "test",
        "schedule": "every 5m", "enabled": True,
        "next_run_at": "2020-01-01T00:00:00", "deliver": "local",
    }


def _patch_run_path(monkeypatch, due):
    """Patch the execution path so a due job runs end-to-end without I/O."""
    monkeypatch.setattr(sched, "get_due_jobs", lambda: due)
    monkeypatch.setattr(sched, "advance_next_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "save_job_output", lambda *_a, **_kw: "/tmp/out")
    monkeypatch.setattr(sched, "mark_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "_deliver_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(sched, "run_job", lambda j: (True, "out", "resp", None))


@pytest.fixture
def reset_pools():
    sched._parallel_pool = None
    sched._parallel_pool_max_workers = None
    sched._sequential_pool = None
    sched._running_job_ids.clear()
    yield
    sched._shutdown_parallel_pool()
    sched._running_job_ids.clear()


class TestSchedulerOwnershipGuard:
    def test_defer_when_gateway_owner_active_skips_execution(self, monkeypatch):
        """defer_to_gateway_owner=True + a gateway owner → no jobs, no lock."""
        monkeypatch.setattr(sched, "_gateway_scheduler_owner_active", lambda: True)

        due_jobs_called = []
        monkeypatch.setattr(
            sched, "get_due_jobs",
            lambda: due_jobs_called.append(True) or [_make_job()],
        )

        ran = sched.tick(verbose=False, defer_to_gateway_owner=True)

        assert ran == 0
        assert not due_jobs_called, "deferred tick must not inspect/run jobs"

    def test_runs_when_no_gateway_owner(self, monkeypatch, reset_pools):
        """defer_to_gateway_owner=True but no owner → ticker runs the job."""
        monkeypatch.setattr(sched, "_gateway_scheduler_owner_active", lambda: False)
        _patch_run_path(monkeypatch, [_make_job()])

        ran = sched.tick(verbose=False, defer_to_gateway_owner=True)

        assert ran == 1

    def test_gateway_ticker_runs_even_when_owner_active(self, monkeypatch, reset_pools):
        """The gateway's own ticker (defer flag False) ignores the owner guard."""
        # Owner "active" here would be the gateway itself; the guard must not
        # apply to the default (gateway-ticker) call path.
        monkeypatch.setattr(sched, "_gateway_scheduler_owner_active", lambda: True)
        _patch_run_path(monkeypatch, [_make_job()])

        ran = sched.tick(verbose=False)  # defer_to_gateway_owner defaults False

        assert ran == 1


class TestGatewaySchedulerOwnerActive:
    def test_delegates_to_runtime_lock(self, monkeypatch):
        import gateway.status as status

        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda: True)
        assert sched._gateway_scheduler_owner_active() is True

        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda: False)
        assert sched._gateway_scheduler_owner_active() is False

    def test_fails_open_on_error(self, monkeypatch, caplog):
        import gateway.status as status

        def _boom():
            raise RuntimeError("lock backend unavailable")

        monkeypatch.setattr(status, "is_gateway_runtime_lock_active", _boom)
        with caplog.at_level(logging.DEBUG, logger=sched.logger.name):
            assert sched._gateway_scheduler_owner_active() is False
