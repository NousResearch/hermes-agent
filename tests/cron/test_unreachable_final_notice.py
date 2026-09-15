"""A suppressed failure must have another run left to deliver a result."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("repeat", [1, 2, None])
def test_retry_notice_matches_remaining_runs(tmp_path, monkeypatch, repeat):
    from cron.jobs import create_job, get_job, mark_job_run
    from cron import scheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    job = create_job("daily report", "every 24h", repeat=repeat)
    job["_model_unreachable"] = True
    delivered = []
    monkeypatch.setattr(
        scheduler, "_deliver_result", lambda *args, **kw: delivered.append(args[1])
    )
    outcome = scheduler._RunDelivery(job, False, "ConnectError: unavailable")
    scheduler._save_compose_deliver(
        outcome,
        scheduler._FireOwnership(job, None),
        "",
        "Connection failed",
        adapters=None,
        loop=None,
        verbose=False,
        execution_token=object(),
    )
    assert mark_job_run(job["id"], False, outcome.error, model_unreachable=True)
    remaining = get_job(job["id"])
    another_run = remaining["next_run_at"] is not None
    assert bool(delivered) is not another_run
    if repeat == 1:
        assert remaining["repeat"]["completed"] == repeat
        assert not another_run
