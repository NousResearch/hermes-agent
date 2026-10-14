"""Interval cron jobs must expose their real cadence, not legacy cron syntax."""

from cron.jobs import _normalize_job_record


def test_interval_display_ignores_legacy_cron_expression():
    job = _normalize_job_record({
        "id": "session-contract-extract",
        "schedule": {"kind": "interval", "minutes": 1440, "display": "0 18 * * *"},
        "schedule_display": "0 18 * * *",
    })

    assert job["schedule_display"] == "every 1440m"


def test_cron_display_is_preserved():
    job = _normalize_job_record({
        "id": "daily",
        "schedule": {"kind": "cron", "expr": "0 18 * * *", "display": "0 18 * * *"},
        "schedule_display": "0 18 * * *",
    })

    assert job["schedule_display"] == "0 18 * * *"
