"""`superseded` gets its own error_class instead of a silent ``None`` (WH-CREATED-5A4D2A184BBA AC1).

`superseded` is a terminal, NON-alarming outcome — the attempt's work was not lost, a newer fire
owns the job record (`cron/attempt_outcome.py`) — so the telemetry projection must neither drop it
to ``None`` (which made the new terminal state invisible in counts/dashboards) nor run the failure
patterns over the row's bookkeeping text. Contract pinned here:

  1. status ``superseded`` -> error_class ``superseded``, whatever the error text says;
  2. ``failed`` / ``unknown`` keep the pattern classification (no regression);
  3. non-terminal statuses stay ``None`` (no invented error semantics).
"""

from __future__ import annotations

from agent.monitoring.cron_health import (
    SUPERSEDED_ERROR_CLASS,
    classify_cron_status,
    project_execution_event,
)


def test_superseded_is_classified_by_status_not_by_its_error_text():
    assert classify_cron_status("superseded", "") == SUPERSEDED_ERROR_CLASS
    # The bookkeeping text explains why the claim moved; it must NOT be read as a failure class.
    assert classify_cron_status(
        "superseded", "fire claim ownership lost"
    ) == SUPERSEDED_ERROR_CLASS
    assert classify_cron_status(
        "superseded", "Interrupted by shutdown before terminal completion."
    ) == SUPERSEDED_ERROR_CLASS


def test_failed_and_unknown_keep_pattern_classification():
    assert classify_cron_status("failed", "Request timed out after 30s") == "timeout"
    assert classify_cron_status("unknown", "authentication failed") == "auth_failed"
    assert classify_cron_status("failed", "no rule matches this") == "unknown"


def test_non_terminal_statuses_carry_no_error_class():
    assert classify_cron_status("running", "boom") is None
    assert classify_cron_status("claimed", "boom") is None
    assert classify_cron_status("completed", "boom") is None


def test_projection_carries_the_class_for_a_superseded_row():
    event = project_execution_event(
        {
            "status": "superseded",
            "job_id": "job-superseded",
            "claimed_at": "2026-09-13T20:00:00+00:00",
            "started_at": "2026-09-13T20:00:01+00:00",
            "finished_at": "2026-09-13T20:05:00+00:00",
            "error": "fire claim ownership lost",
        }
    )
    assert event.status == "superseded"
    assert event.error_class == SUPERSEDED_ERROR_CLASS
