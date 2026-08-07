from datetime import datetime
from zoneinfo import ZoneInfo

from cron.scheduler import _resolve_cron_max_iterations


TZ = ZoneInfo("America/Argentina/Buenos_Aires")


def test_per_job_limit_wins_over_cron_and_interactive_limits():
    cfg = {
        "agent": {"max_turns": 60},
        "cron": {"max_iterations": 20},
    }

    assert _resolve_cron_max_iterations({"max_iterations": 11}, cfg) == 11


def test_cron_limit_is_used_when_job_has_no_valid_limit():
    cfg = {
        "agent": {"max_turns": 60},
        "cron": {"max_iterations": 20},
    }

    assert _resolve_cron_max_iterations({"max_iterations": 0}, cfg) == 20


def test_final_day_limit_applies_only_to_matching_provider_before_reset():
    cfg = {
        "agent": {"max_turns": 60},
        "cron": {
            "weekly_final_day": {
                "enabled": True,
                "provider": "zai",
                "reset_weekday": 5,
                "reset_time": "10:06:40",
                "window_hours": 24,
                "max_iterations": 15,
            }
        },
    }
    friday = datetime(2026, 8, 7, 11, 10, tzinfo=TZ)

    assert _resolve_cron_max_iterations(
        {"max_iterations": 11}, cfg, provider="zai", now=friday
    ) == 15
    assert _resolve_cron_max_iterations(
        {"max_iterations": 11}, cfg, provider="custom", now=friday
    ) == 11


def test_final_day_limit_stops_after_reset():
    cfg = {
        "cron": {
            "weekly_final_day": {
                "enabled": True,
                "provider": "zai",
                "reset_weekday": 5,
                "reset_time": "10:06:40",
                "window_hours": 24,
                "max_iterations": 15,
            }
        }
    }
    after_reset = datetime(2026, 8, 8, 10, 10, tzinfo=TZ)

    assert _resolve_cron_max_iterations(
        {"max_iterations": 11}, cfg, provider="zai", now=after_reset
    ) == 11
