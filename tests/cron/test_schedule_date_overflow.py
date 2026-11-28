"""Cron rejects durations whose resulting date cannot be represented."""

import pytest

from cron.jobs import create_job, get_job, list_jobs, pause_job, update_job, use_cron_store


@pytest.mark.parametrize("schedule", ["in 3000000d", "every 3000000d", "3000000d"])
def test_create_rejects_schedule_date_overflow(tmp_path, schedule):
    # The duration fits in a timedelta, but the resulting date is after year 9999.
    with use_cron_store(tmp_path):
        with pytest.raises(ValueError, match="execution date is out of range"):
            create_job("test job", schedule)

        assert list_jobs() == []


@pytest.mark.parametrize("paused", [False, True])
def test_edit_rejects_schedule_date_overflow(tmp_path, paused):
    with use_cron_store(tmp_path):
        job_id = create_job("test job", "every 30m")["id"]
        if paused:
            pause_job(job_id)
        before = get_job(job_id)

        with pytest.raises(ValueError, match="execution date is out of range"):
            update_job(job_id, {"schedule": "every 3000000d"})

        assert get_job(job_id) == before
