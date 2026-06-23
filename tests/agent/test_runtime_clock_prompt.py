from datetime import datetime
from zoneinfo import ZoneInfo


def test_runtime_clock_prompt_includes_fresh_time_and_relative_date_guidance(monkeypatch):
    import hermes_time
    from agent.system_prompt import build_runtime_clock_prompt

    monkeypatch.setattr(
        hermes_time,
        "now",
        lambda: datetime(2026, 6, 1, 13, 4, 5, tzinfo=ZoneInfo("America/Chicago")),
    )

    prompt = build_runtime_clock_prompt()

    assert "## Current Date and Time" in prompt
    assert "Monday, June 01, 2026 at 01:04 PM" in prompt
    assert "UTC-05:00" in prompt
    assert "2026-06-01T13:04:05-05:00" in prompt
    assert "this week" in prompt
    assert "do not assume 'this week' means 'weekend'" in prompt
