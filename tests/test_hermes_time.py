from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import hermes_time


def test_format_display_timestamp_preserves_configured_strftime_format():
    instant = datetime(2026, 8, 8, 19, 4, 5, tzinfo=timezone.utc)

    rendered = hermes_time.format_display_timestamp(
        instant,
        enabled=True,
        format_string="%Y-%m-%d %H:%M:%S",
        tz=ZoneInfo("America/New_York"),
    )

    assert rendered == "2026-08-08 15:04:05"


def test_format_display_timestamp_is_empty_when_disabled():
    instant = datetime(2026, 8, 8, 19, 4, 5, tzinfo=timezone.utc)

    assert (
        hermes_time.format_display_timestamp(
            instant,
            enabled=False,
            format_string="%Y-%m-%d %H:%M:%S",
        )
        == ""
    )


def test_format_display_timestamp_converts_epoch_in_requested_timezone():
    instant = datetime(2026, 8, 8, 19, 4, 5, tzinfo=timezone.utc)

    rendered = hermes_time.format_display_timestamp(
        instant.timestamp(),
        enabled=True,
        format_string="%Y-%m-%d %H:%M:%S %Z",
        tz=ZoneInfo("Europe/Berlin"),
    )

    assert rendered == "2026-08-08 21:04:05 CEST"