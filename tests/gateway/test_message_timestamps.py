from datetime import datetime
from zoneinfo import ZoneInfo

from gateway.message_timestamps import (
    coerce_message_timestamp,
    render_user_content_with_timestamp,
    strip_leading_message_timestamps,
)
from run_agent import AIAgent


BERLIN = ZoneInfo("Europe/Berlin")


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
