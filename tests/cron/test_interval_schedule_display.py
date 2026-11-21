from cron.jobs import _normalize_job_record


def test_cron_prefers_expr_over_stale_display_snapshot():
    job = _normalize_job_record({
        "id": "daily",
        "schedule": {"kind": "cron", "expr": "0 12 * * *", "display": "0 10 * * *"},
        "schedule_display": "0 10 * * *",
    })

    assert job["schedule_display"] == "0 12 * * *"
