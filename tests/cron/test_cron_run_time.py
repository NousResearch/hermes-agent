"""A cron fire's prompt carries its own wall-clock run time (the system prompt is date-only)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from cron.scheduler import _build_job_prompt

_FIRE = datetime(2026, 10, 31, 19, 5, tzinfo=timezone(timedelta(hours=-7), "PDT"))


def _prompt(job):
    with patch("cron.scheduler_prompt._hermes_now", return_value=_FIRE), patch(
        "tools.skills_tool.skill_view", return_value='{"success": false, "error": "not found"}'
    ):
        return _build_job_prompt(job)


def test_run_time_follows_the_hint_and_precedes_the_job_prompt():
    result = _prompt({"prompt": "Morning brief"})
    line = "[CURRENT RUN TIME: Saturday, 2026-10-31 19:05 PDT (UTC-07:00)]"
    assert line in result
    assert result.index("You are running as a scheduled cron job") < result.index(line)
    assert result.index(line) < result.index("Morning brief")


def test_run_time_survives_the_skills_assembly_path():
    result = _prompt({"skills": ["ghost-skill"], "prompt": "go"})
    assert "[CURRENT RUN TIME: Saturday, 2026-10-31 19:05 PDT (UTC-07:00)]" in result


def test_a_pinned_job_is_told_its_own_zones_time():
    """A 09:00 New York job fired from a Los Angeles profile is told 09:00 New York, not 06:00:
    the expression, the fire and the line the model reads must agree on one wall clock."""
    nine_am_ny = _FIRE.replace(hour=6)  # 06:05 PDT == 09:05 EDT
    with patch("cron.scheduler_prompt._hermes_now", return_value=nine_am_ny):
        result = _build_job_prompt({"prompt": "Pre-market", "timezone": "America/New_York"})
    assert "[CURRENT RUN TIME: Saturday, 2026-10-31 09:05 EDT (UTC-04:00)]" in result


def test_a_corrupt_pin_falls_back_to_the_profile_zone():
    result = _prompt({"prompt": "x", "timezone": "Mars/Olympus_Mons"})
    assert "[CURRENT RUN TIME: Saturday, 2026-10-31 19:05 PDT (UTC-07:00)]" in result
