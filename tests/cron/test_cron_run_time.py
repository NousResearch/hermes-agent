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
