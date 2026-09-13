"""The post-delivery loss path records what the attempt actually did.

Regression for the 2026-09-13 phantom rows: a run that had already delivered, whose ownership latch
fired post hoc (a heartbeat tick that could not take the per-job fence while the run held it), was
recorded ``failed`` with "Interrupted by shutdown before terminal completion." and flipped its job's
``last_status`` to ``error``.

These exercise the real job store and the real execution ledger against a temp HERMES_HOME — no
mocks — because the bug lived exactly in the seam between those two.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real jobs.json + real executions.db under one temp HERMES_HOME."""
    import cron.executions as executions_mod
    from cron.jobs import use_cron_store

    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    with use_cron_store(tmp_path / "cron"):
        yield tmp_path


def _claimed_job_and_attempt(name: str):
    """A claimed job (real fire_claim) plus the ledger attempt recorded for that fire."""
    from cron.executions import create_execution, mark_execution_running
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="x", schedule="every 5m", name=name)
    claimed = claim_job_for_fire(job["id"], return_job=True)
    assert isinstance(claimed, dict)
    claim = claimed["fire_claim"]
    attempt = create_execution(
        job["id"], source="builtin", scheduled_instant=claimed.get("_scheduled_instant"))
    mark_execution_running(attempt["id"])
    return job["id"], claimed, str(claim["by"]), attempt["id"]


def _delivery(claimed: dict, *, delivered: bool):
    from cron.scheduler import _RunDelivery

    return _RunDelivery(
        job=claimed, success=True, error=None,
        delivery_attempted=delivered, delivery_error=None,
        should_deliver=delivered)


def test_a_delivered_run_under_a_post_hoc_latch_completes(store):
    from cron.executions import get_execution
    from cron.jobs import get_job
    from cron.scheduler import _finish_overtaken_run

    job_id, claimed, owner, attempt_id = _claimed_job_and_attempt("phantom")

    assert _finish_overtaken_run(
        _delivery(claimed, delivered=True), owner, attempt_id, delivered=True) is True

    assert get_execution(attempt_id)["status"] == "completed"
    record = get_job(job_id)
    assert record["last_status"] == "ok"
    assert record["last_error"] is None


def test_an_overtaken_delivered_run_leaves_the_job_record_alone(store):
    from cron.executions import create_execution, get_execution
    from cron.jobs import get_job
    from cron.scheduler import _finish_overtaken_run

    job_id, claimed, owner, attempt_id = _claimed_job_and_attempt("overtaken")
    before = get_job(job_id)
    # A later fire for the same job owns the job record now.
    create_execution(
        job_id, source="builtin", scheduled_instant="2026-09-14T00:00:00+00:00")

    assert _finish_overtaken_run(
        _delivery(claimed, delivered=True), owner, attempt_id, delivered=True) is True

    row = get_execution(attempt_id)
    assert row["status"] == "superseded"
    after = get_job(job_id)
    assert after["last_status"] == before["last_status"]
    assert after["last_run_at"] == before["last_run_at"]


def test_an_undelivered_interruption_still_reaches_the_job_record(store):
    """Negative control: a run that delivered nothing keeps the pre-existing interruption record."""
    from cron.executions import get_execution
    from cron.jobs import get_job
    from cron.scheduler import _finish_overtaken_run

    job_id, claimed, owner, attempt_id = _claimed_job_and_attempt("interrupted")

    assert _finish_overtaken_run(
        _delivery(claimed, delivered=False), owner, attempt_id, delivered=False) is True

    record = get_job(job_id)
    assert record["last_status"] == "error"
    assert "Interrupted by shutdown" in record["last_error"]
    row = get_execution(attempt_id)
    assert row["status"] == "failed"
    assert "Interrupted by shutdown" in row["error"]
