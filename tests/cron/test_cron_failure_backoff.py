"""Failure-alert backoff and recovery notifications for recurring cron jobs."""

from cron import scheduler


def test_first_failure_alerts_and_repeated_failure_is_suppressed(monkeypatch):
    job = {"id": "job-1", "name": "Example"}
    persisted = []
    monkeypatch.setattr(scheduler, "update_job", lambda _id, values: persisted.append(values))
    monkeypatch.setattr(scheduler.time, "time", lambda: 1000.0)

    first = scheduler._prepare_cron_failure_alert(job, "provider timeout request 12345678")
    assert first is not None
    scheduler._confirm_cron_failure_alert(job)

    repeated = scheduler._prepare_cron_failure_alert(job, "provider timeout request 87654321")
    assert repeated is None
    assert persisted[-1]["failure_alert"]["suppressed"] == 1


def test_repeated_failure_gets_daily_reminder(monkeypatch):
    job = {"id": "job-1", "name": "Example"}
    persisted = []
    monkeypatch.setattr(scheduler, "update_job", lambda _id, values: persisted.append(values))
    now = [1000.0]
    monkeypatch.setattr(scheduler.time, "time", lambda: now[0])

    assert scheduler._prepare_cron_failure_alert(job, "provider timeout") is not None
    scheduler._confirm_cron_failure_alert(job)
    assert scheduler._prepare_cron_failure_alert(job, "provider timeout") is None
    job["failure_alert"] = persisted[-1]["failure_alert"]

    now[0] += 24 * 60 * 60
    reminder = scheduler._prepare_cron_failure_alert(job, "provider timeout")
    assert reminder is not None
    assert "1 repeated failure(s) suppressed" in reminder


def test_failed_delivery_is_retried_on_the_next_failure(monkeypatch):
    job = {"id": "job-1", "name": "Example"}
    persisted = []
    monkeypatch.setattr(scheduler, "update_job", lambda _id, values: persisted.append(values))
    monkeypatch.setattr(scheduler.time, "time", lambda: 1000.0)

    assert scheduler._prepare_cron_failure_alert(job, "provider timeout") is not None
    # No _confirm_cron_failure_alert call: the transport failed.
    retry = scheduler._prepare_cron_failure_alert(job, "provider timeout")
    assert retry is not None
    assert persisted[-1]["failure_alert"]["pending"] is True


def test_recovery_notice_reports_and_clears_after_success(monkeypatch):
    job = {
        "id": "job-1",
        "name": "Example",
        "failure_alert": {"consecutive": 3, "suppressed": 2},
    }
    notice = scheduler._cron_recovery_notice(job)
    assert notice is not None
    assert "recovered after 3 consecutive failure(s)" in notice
