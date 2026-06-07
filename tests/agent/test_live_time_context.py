from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from agent.live_time_context import (
    add_sent_timestamp_prefix,
    format_live_time_context,
    sent_timestamp_prefix,
    strip_sent_timestamp_prefix,
)


def test_sent_timestamp_prefix_uses_compact_iso_with_offset():
    now = datetime(2026, 6, 7, 17, 42, tzinfo=ZoneInfo("Europe/Berlin"))

    assert sent_timestamp_prefix(now) == "[sent: 2026-06-07T17:42+02:00]"


def test_add_sent_timestamp_prefix_is_idempotent():
    now = datetime(2026, 6, 7, 17, 42, tzinfo=ZoneInfo("Europe/Berlin"))
    content = "hello"

    stamped = add_sent_timestamp_prefix(content, now)

    assert stamped == "[sent: 2026-06-07T17:42+02:00]\nhello"
    assert add_sent_timestamp_prefix(stamped, now) == stamped


def test_add_sent_timestamp_prefix_does_not_invent_now_without_timestamp():
    assert add_sent_timestamp_prefix("old message") == "old message"
    assert sent_timestamp_prefix(None) == ""


def test_strip_sent_timestamp_prefix_removes_internal_marker():
    assert (
        strip_sent_timestamp_prefix("[sent: 2026-06-07T17:42+02:00]\nhello")
        == "hello"
    )
    assert strip_sent_timestamp_prefix("hello") == "hello"


def test_strip_sent_timestamp_prefix_removes_leaked_inline_marker():
    assert (
        strip_sent_timestamp_prefix("Sure. [sent: 2026-06-07T17:42+02:00] hello")
        == "Sure. hello"
    )


def test_strip_sent_timestamp_prefix_accepts_seconds_and_zulu():
    assert (
        strip_sent_timestamp_prefix("[sent: 2026-06-07T15:42:33Z] hello")
        == "hello"
    )


def test_iso_string_timestamp_is_preserved_compactly():
    assert (
        sent_timestamp_prefix("2026-06-07T17:42:33+02:00")
        == "[sent: 2026-06-07T17:42+02:00]"
    )


def test_utc_timestamp_formats_in_configured_timezone(monkeypatch):
    monkeypatch.setattr(
        "hermes_time.get_timezone",
        lambda: ZoneInfo("Europe/Berlin"),
    )

    now = datetime(2026, 6, 7, 19, 38, tzinfo=timezone.utc)

    assert sent_timestamp_prefix(now) == "[sent: 2026-06-07T21:38+02:00]"


def test_legacy_format_live_time_context_returns_same_compact_prefix():
    now = datetime(2026, 6, 7, 17, 42, tzinfo=ZoneInfo("Europe/Berlin"))

    assert format_live_time_context(now) == "[sent: 2026-06-07T17:42+02:00]"
