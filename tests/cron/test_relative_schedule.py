"""End-to-end coverage for model-authored relative cron schedules."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


def test_cronjob_relative_duration_uses_configured_clock(monkeypatch):
    """The public tool path must preserve a relative delay across timezones."""
    from tools.cronjob_tools import cronjob

    configured_now = datetime(
        2026,
        8,
        16,
        11,
        51,
        0,
        tzinfo=timezone(timedelta(hours=-4)),
    )
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: configured_now)

    result = json.loads(
        cronjob(
            action="create",
            prompt="Send this reminder: take a bath",
            schedule="2m",
        )
    )

    next_run_at = datetime.fromisoformat(result["next_run_at"])
    assert result["success"] is True
    assert result["schedule"] == "once in 2m"
    assert next_run_at == configured_now + timedelta(minutes=2)
    assert next_run_at.utcoffset() == configured_now.utcoffset()
