from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("timezone: America/New_York\n", encoding="utf-8")
    return home


def test_public_schedule_edits_preserve_pinned_timezone_and_dst(cron_home, monkeypatch):
    pytest.importorskip("croniter")
    from cron import jobs
    from cron.jobs import create_job, get_job, update_job
    from tools.cronjob_tools import cronjob
    import json

    monkeypatch.setattr(
        jobs,
        "_hermes_now",
        lambda: datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc),
    )
    job = create_job(
        prompt="Pinned wall clock",
        schedule="0 9 * * *",
        timezone_name="America/Los_Angeles",
    )

    direct = update_job(job["id"], {"schedule": "0 10 * * *"})
    assert direct is not None
    assert direct["schedule"]["timezone"] == "America/Los_Angeles"

    response = json.loads(cronjob(action="update", job_id=job["id"], schedule="0 11 * * *"))
    assert response["success"] is True
    stored = get_job(job["id"])
    assert stored is not None
    assert stored["schedule"]["timezone"] == "America/Los_Angeles"
    next_run = datetime.fromisoformat(stored["next_run_at"])
    local_next = next_run.astimezone(ZoneInfo("America/Los_Angeles"))
    assert local_next.hour == 11
    assert local_next.utcoffset() == timedelta(hours=-7)  # DST began earlier that morning.


def test_schedule_edit_can_explicitly_change_or_clear_timezone(cron_home):
    pytest.importorskip("croniter")
    from cron.jobs import create_job, parse_schedule, update_job

    job = create_job(
        prompt="Pinned wall clock",
        schedule="0 9 * * *",
        timezone_name="America/Los_Angeles",
    )
    changed = update_job(
        job["id"],
        {"schedule": parse_schedule("0 10 * * *", timezone_name="Europe/London")},
    )
    assert changed is not None
    assert changed["schedule"]["timezone"] == "Europe/London"

    cleared_schedule = parse_schedule("0 11 * * *")
    cleared_schedule["timezone"] = None
    cleared = update_job(job["id"], {"schedule": cleared_schedule})
    assert cleared is not None
    assert "timezone" not in cleared["schedule"] or cleared["schedule"]["timezone"] is None
