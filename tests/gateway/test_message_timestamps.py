from datetime import datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo

from gateway.message_timestamps import (
    coerce_message_timestamp,
    render_user_content_with_timestamp,
    strip_leading_message_timestamps,
)
from run_agent import AIAgent


BERLIN = ZoneInfo("Europe/Berlin")


class _WindowsStyleTZ(tzinfo):
    """A tzinfo whose name contains spaces, mirroring how Windows reports the
    local zone through ``datetime.strftime('%Z')`` (e.g. ``Pacific Daylight
    Time``).  POSIX systems return short abbreviations like ``PDT``, but Windows
    returns the full descriptive name, which is what the gateway emits when no
    IANA timezone is configured (``tz`` is ``None`` and the code falls back to
    ``astimezone()``)."""

    def utcoffset(self, dt):
        return timedelta(hours=-7)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "Pacific Daylight Time"


_WINDOWS_TZ = _WindowsStyleTZ()


def _epoch(year, month, day, hour, minute, second):
    return datetime(year, month, day, hour, minute, second, tzinfo=BERLIN).timestamp()


def test_render_user_content_adds_single_context_timestamp():
    ts = _epoch(2026, 4, 28, 13, 40, 53)

    rendered = render_user_content_with_timestamp(
        "[Example User] Timestamp should be in context",
        ts,
        tz=BERLIN,
    )

    assert rendered == (
        "[Tue 2026-04-28 13:40:53 CEST] "
        "[Example User] Timestamp should be in context"
    )


def test_render_user_content_deduplicates_existing_timestamp_and_preserves_embedded_time():
    db_processing_ts = _epoch(2026, 4, 27, 15, 55, 36)
    stored_content = (
        "[Mon 2026-04-27 15:54:44 CEST] "
        "[Example User] This should go on our todo list"
    )

    rendered = render_user_content_with_timestamp(
        stored_content,
        db_processing_ts,
        tz=BERLIN,
    )

    assert rendered == stored_content
    assert rendered.count("2026-04-27") == 1


def test_strip_leading_message_timestamps_removes_multiple_prefixes_and_prefers_inner_time():
    content = (
        "[Mon 2026-04-27 15:55:36 CEST] "
        "[Mon 2026-04-27 15:54:44 CEST] "
        "[Example User] This should go on our todo list"
    )

    stripped, embedded_ts = strip_leading_message_timestamps(content, tz=BERLIN)

    assert stripped == "[Example User] This should go on our todo list"
    assert embedded_ts == _epoch(2026, 4, 27, 15, 54, 44)


def test_strip_removes_multiword_timezone_prefix():
    # Windows ``strftime('%Z')`` yields multi-word zone names. The strip pass
    # (which runs on every inbound gateway message to keep storage clean) must
    # still recognise such a prefix, otherwise contaminated rows are persisted
    # and never cleaned.
    content = (
        "[Tue 2026-04-28 13:40:53 Pacific Daylight Time] "
        "[Example User] hello"
    )

    stripped, _embedded = strip_leading_message_timestamps(content)

    assert stripped == "[Example User] hello"


def test_render_is_idempotent_with_multiword_timezone():
    # Re-rendering a row that already carries a multi-word-timezone prefix must
    # not stack a second prefix. On affected systems this is the
    # "[timestamp] [timestamp] ..." accumulation the module exists to prevent.
    ts = datetime(2026, 4, 28, 13, 40, 53, tzinfo=_WINDOWS_TZ).timestamp()

    once = render_user_content_with_timestamp(
        "[Example User] hi", ts, tz=_WINDOWS_TZ
    )
    twice = render_user_content_with_timestamp(once, ts, tz=_WINDOWS_TZ)

    assert once == twice
    assert once.count("2026-04-28") == 1


def test_strip_still_handles_abbreviation_timezones():
    # Guard against over-widening: short abbreviations and numeric offsets must
    # keep stripping exactly as before the multi-word fix.
    for tz_token in ("PDT", "CEST", "UTC", "GMT+5", "+05:30"):
        content = f"[Tue 2026-04-28 13:40:53 {tz_token}] body"
        stripped, _embedded = strip_leading_message_timestamps(content)
        assert stripped == "body", tz_token


def test_coerce_message_timestamp_accepts_datetime_and_epoch():
    dt = datetime(2026, 4, 28, 13, 40, 53, tzinfo=BERLIN)

    assert coerce_message_timestamp(dt, tz=BERLIN) == dt.timestamp()
    assert coerce_message_timestamp(dt.timestamp(), tz=BERLIN) == dt.timestamp()


def test_persist_user_message_override_keeps_clean_content_and_timestamp_metadata():
    agent = AIAgent.__new__(AIAgent)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "[Example User] Clean content"
    agent._persist_user_message_timestamp = _epoch(2026, 4, 28, 13, 40, 53)
    messages = [
        {
            "role": "user",
            "content": "[Tue 2026-04-28 13:40:53 CEST] [Example User] Clean content",
        }
    ]

    agent._apply_persist_user_message_override(messages)

    assert messages == [
        {
            "role": "user",
            "content": "[Example User] Clean content",
            "timestamp": _epoch(2026, 4, 28, 13, 40, 53),
        }
    ]


# ---------------------------------------------------------------------------
# Opt-in gate: gateway.message_timestamps.enabled (default OFF)
# ---------------------------------------------------------------------------


def test_message_timestamps_enabled_defaults_off():
    from gateway.run import _message_timestamps_enabled

    assert _message_timestamps_enabled(None) is False
    assert _message_timestamps_enabled({}) is False
    assert _message_timestamps_enabled({"gateway": {}}) is False
    assert (
        _message_timestamps_enabled({"gateway": {"message_timestamps": {}}}) is False
    )


def test_message_timestamps_enabled_when_opted_in():
    from gateway.run import _message_timestamps_enabled

    assert _message_timestamps_enabled(
        {"gateway": {"message_timestamps": {"enabled": True}}}
    ) is True
    # Bare shorthand also accepted.
    assert _message_timestamps_enabled({"gateway": {"message_timestamps": True}}) is True


def test_build_history_injects_only_when_enabled():
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "hello", "timestamp": _epoch(2026, 4, 28, 13, 40, 53)},
        {"role": "assistant", "content": "hi"},
    ]

    # Default (off): user content stays clean, no timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history)
    assert agent_history[0]["content"] == "hello"

    # Enabled: user content gets exactly one timestamp prefix.
    agent_history, _ = _build_gateway_agent_history(history, inject_timestamps=True)
    assert agent_history[0]["content"].startswith("[")
    assert agent_history[0]["content"].endswith("hello")
    # Assistant message is never timestamped.
    assert agent_history[1]["content"] == "hi"
