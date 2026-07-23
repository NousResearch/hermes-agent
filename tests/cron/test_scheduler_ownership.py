import logging


def test_gateway_scheduler_owner_helper_delegates_to_runtime_lock(monkeypatch):
    import cron.scheduler as sched

    monkeypatch.setattr(
        "gateway.status.is_gateway_runtime_lock_active",
        lambda: True,
    )

    assert sched._gateway_scheduler_owner_active() is True


def test_gateway_scheduler_owner_helper_fails_open(monkeypatch, caplog):
    import cron.scheduler as sched

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "gateway.status.is_gateway_runtime_lock_active",
        _boom,
    )

    with caplog.at_level(logging.DEBUG):
        assert sched._gateway_scheduler_owner_active() is False

    assert "gateway scheduler-owner check failed" in caplog.text


def test_tick_defers_before_lock_or_job_lookup(monkeypatch):
    import cron.scheduler as sched

    monkeypatch.setattr(sched, "_gateway_scheduler_owner_active", lambda: True)
    monkeypatch.setattr(
        sched,
        "get_due_jobs",
        lambda: (_ for _ in ()).throw(AssertionError("should not inspect due jobs")),
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("should not acquire the cron tick lock")
        ),
    )

    assert sched.tick(verbose=False, defer_to_gateway_owner=True) == 0


def test_tick_runs_normally_when_no_gateway_owner(tmp_path, monkeypatch):
    import cron.scheduler as sched

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sched, "_gateway_scheduler_owner_active", lambda: False)

    seen = {"due_jobs": 0}

    def _get_due_jobs():
        seen["due_jobs"] += 1
        return []

    monkeypatch.setattr(sched, "get_due_jobs", _get_due_jobs)

    assert sched.tick(verbose=False, defer_to_gateway_owner=True) == 0
    assert seen["due_jobs"] == 1


def test_tick_ignores_gateway_owner_without_opt_in(tmp_path, monkeypatch):
    import cron.scheduler as sched

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sched, "_gateway_scheduler_owner_active", lambda: True)

    seen = {"due_jobs": 0}

    def _get_due_jobs():
        seen["due_jobs"] += 1
        return []

    monkeypatch.setattr(sched, "get_due_jobs", _get_due_jobs)

    assert sched.tick(verbose=False) == 0
    assert seen["due_jobs"] == 1
