from agent.job_lifecycle import (
    JobLifecycleState,
    cron_lifecycle_state,
    default_queue_metadata,
    make_attempt_record,
    normalize_queue_metadata,
    process_lifecycle_state,
)


def test_process_running_maps_to_running() -> None:
    assert process_lifecycle_state(exited=False, exit_code=None) == JobLifecycleState.running


def test_process_zero_exit_maps_to_completed() -> None:
    assert process_lifecycle_state(exited=True, exit_code=0) == JobLifecycleState.completed


def test_process_sigterm_maps_to_cancelled() -> None:
    assert process_lifecycle_state(exited=True, exit_code=-15) == JobLifecycleState.cancelled


def test_cron_scheduled_job_maps_to_queued() -> None:
    state = cron_lifecycle_state({"state": "scheduled", "enabled": True, "next_run_at": "2030-01-01T00:00:00+00:00"})
    assert state == JobLifecycleState.queued


def test_cron_failed_recurring_job_maps_to_retrying() -> None:
    state = cron_lifecycle_state(
        {
            "state": "scheduled",
            "enabled": True,
            "next_run_at": "2030-01-01T00:00:00+00:00",
            "last_status": "error",
        }
    )
    assert state == JobLifecycleState.retrying


def test_cron_paused_job_maps_to_paused() -> None:
    state = cron_lifecycle_state({"state": "paused", "enabled": False})
    assert state == JobLifecycleState.paused


def test_cron_terminal_error_maps_to_failed() -> None:
    state = cron_lifecycle_state({"state": "completed", "enabled": False, "last_status": "error"})
    assert state == JobLifecycleState.failed


def test_default_queue_metadata_starts_empty() -> None:
    queue = default_queue_metadata(queued_at="2030-01-01T00:00:00")
    assert queue == {
        "retry_count": 0,
        "retry_backoff_seconds": 0,
        "next_retry_at": None,
        "current_attempt": None,
        "last_attempt": None,
        "attempt_history": [],
        "last_queued_at": "2030-01-01T00:00:00",
        "runner": {
            "kind": "cron",
            "queue_name": "default",
            "priority": 0,
            "active": False,
            "claimed_at": None,
            "lease_expires_at": None,
            "worker_id": None,
            "last_started_at": None,
            "last_finished_at": None,
        },
    }


def test_normalize_queue_metadata_preserves_attempt_records() -> None:
    queue = normalize_queue_metadata(
        {
            "retry_count": "2",
            "current_attempt": make_attempt_record(
                attempt=3,
                state=JobLifecycleState.running,
                started_at="2030-01-01T00:00:00",
                retry_count=2,
            ),
        },
        queued_at="2030-01-01T00:00:00",
    )
    assert queue["retry_count"] == 2
    assert queue["current_attempt"]["attempt"] == 3
    assert queue["current_attempt"]["lifecycle_state"] == JobLifecycleState.running.value
    assert queue["attempt_history"] == []
    assert queue["next_retry_at"] is None
    assert queue["retry_backoff_seconds"] == 0
    assert queue["runner"]["kind"] == "cron"
    assert queue["runner"]["queue_name"] == "default"
    assert queue["runner"]["lease_expires_at"] is None
