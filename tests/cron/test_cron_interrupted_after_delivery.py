"""An interrupt latch consumed AFTER delivery must not fabricate a failed row.

Regression for the second half of the 2026-09-13 phantom-row class (WH-CREATED-23D2B8E1FBE6,
reviewing PR #110152 head 68e742a9): ``_finish_interrupted_run`` unconditionally wrote
``finish_execution(success=False, "Interrupted by gateway shutdown before terminal completion.")``
from a call site that sits AFTER ``_save_compose_deliver`` — so a shutdown flag set one line after
the notice left the process still produced a "已交付 → row 寫 failed" phantom.

The sibling fix (``test_cron_phantom_row_post_delivery.py``) covered the post-hoc ownership latch;
this file covers the shutdown latch, and asserts the negative control (an interrupt that really did
deliver nothing keeps the shutdown narrative).

Real job store + real execution ledger under a temp HERMES_HOME — the bug lived in the seam.
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


def test_an_interrupted_run_that_already_delivered_completes(store):
    """Delivered one line earlier: the ledger decides, so the row is completed, never failed."""
    from cron.executions import get_execution
    from cron.jobs import get_job
    from cron.scheduler import _finish_interrupted_run

    job_id, claimed, _owner, attempt_id = _claimed_job_and_attempt("latch-after-delivery")
    before = get_job(job_id)

    _finish_interrupted_run(claimed, attempt_id, None, delivered=True)

    row = get_execution(attempt_id)
    assert row["status"] == "completed", row
    assert "Interrupted by gateway shutdown" not in (row.get("error") or "")
    # The shutdown path already wrote last_status; this helper must not rewrite the job record.
    after = get_job(job_id)
    assert after["last_status"] == before["last_status"]
    assert after["last_run_at"] == before["last_run_at"]


def test_an_interrupted_run_that_delivered_nothing_keeps_the_shutdown_failure(store):
    """Negative control: no delivery means the pre-existing interruption record stands."""
    from cron.executions import get_execution
    from cron.scheduler import _finish_interrupted_run

    _job_id, claimed, _owner, attempt_id = _claimed_job_and_attempt("latch-no-delivery")

    _finish_interrupted_run(claimed, attempt_id, None, delivered=False)

    row = get_execution(attempt_id)
    assert row["status"] == "failed", row
    assert "Interrupted by gateway shutdown before terminal completion." in (row.get("error") or "")


def test_an_interrupted_delivered_run_overtaken_by_a_later_fire_is_superseded(store):
    """A later fire owns the job record: the delivered attempt records superseded, not a failure."""
    from cron.executions import create_execution, get_execution
    from cron.jobs import get_job
    from cron.scheduler import _finish_interrupted_run

    job_id, claimed, _owner, attempt_id = _claimed_job_and_attempt("latch-overtaken")
    before = get_job(job_id)
    create_execution(
        job_id, source="builtin", scheduled_instant="2026-09-14T00:00:00+00:00")

    _finish_interrupted_run(claimed, attempt_id, None, delivered=True)

    row = get_execution(attempt_id)
    assert row["status"] == "superseded", row
    assert "Interrupted by gateway shutdown" not in (row.get("error") or "")
    after = get_job(job_id)
    assert after["last_status"] == before["last_status"]
    assert after["last_run_at"] == before["last_run_at"]


def test_an_interrupted_run_with_a_delivery_error_still_records_the_defect(store):
    """A notice that never left the process is still a delivery defect on the job record."""
    from cron.executions import get_execution
    from cron.jobs import get_job
    from cron.scheduler import _finish_interrupted_run

    job_id, claimed, _owner, attempt_id = _claimed_job_and_attempt("latch-delivery-error")

    _finish_interrupted_run(claimed, attempt_id, "telegram: 502", delivered=False)

    assert get_job(job_id)["last_delivery_error"] == "telegram: 502"
    row = get_execution(attempt_id)
    assert row["status"] == "failed", row


def _silent_success(claimed: dict):
    """A successful run whose contract was "no message" ([SILENT] suppression)."""
    from cron.scheduler import _RunDelivery

    return _RunDelivery(
        job=claimed, success=True, error=None, delivery_attempted=False, delivery_error=None,
        should_deliver=False, silence_suppressed=True)


def test_a_silenced_success_is_a_terminal_outcome_not_a_lost_delivery(store):
    """Only a delivered run or an intentional [SILENT] counts; everything else stays interrupted."""
    from cron.scheduler import _RunDelivery, _attempt_reached_terminal_delivery

    _job_id, claimed, _owner, _attempt_id = _claimed_job_and_attempt("helper-shapes")
    assert _attempt_reached_terminal_delivery(_silent_success(claimed)) is True
    # A failure that suppressed its own notice is NOT a terminal delivery.
    assert _attempt_reached_terminal_delivery(
        _RunDelivery(job=claimed, success=False, error="boom")) is False
    # Nor is a successful run that was cut off before it could deliver (latched before delivery).
    assert _attempt_reached_terminal_delivery(
        _RunDelivery(job=claimed, success=True, error=None)) is False


def test_a_silenced_success_under_a_post_hoc_latch_is_not_recorded_as_error(store):
    """The [SILENT] contract is not an interruption: last_status must read ok, not error.

    Base behaviour (68e742a9ad + the gap-1 fix alone): ``delivered`` was computed only from
    ``delivery_attempted``, so this shape classified INTERRUPTED and ``mark_job_run(False,
    INTERRUPTED_ERROR)`` flipped ``last_status`` to error for a run that succeeded by design.
    """
    from cron.executions import get_execution
    from cron.jobs import get_job
    from cron.scheduler import _attempt_reached_terminal_delivery, _finish_overtaken_run

    job_id, claimed, owner, attempt_id = _claimed_job_and_attempt("silent-success")
    d = _silent_success(claimed)

    assert _finish_overtaken_run(
        d, owner, attempt_id, delivered=_attempt_reached_terminal_delivery(d)) is True

    assert get_execution(attempt_id)["status"] == "completed"
    record = get_job(job_id)
    assert record["last_status"] == "ok"
    assert record["last_error"] is None
