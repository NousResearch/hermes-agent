from datetime import datetime
from unittest.mock import MagicMock
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


def test_persist_user_message_override_applies_only_on_db_flush():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._session_db = MagicMock()
    agent.session_id = "session-123"
    agent._last_flushed_db_idx = 0
    agent._session_db_created = True
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "[Example User] Clean content"
    agent._persist_user_message_timestamp = _epoch(2026, 4, 28, 13, 40, 53)
    augmented = "[Tue 2026-04-28 13:40:53 CEST] [Example User] Clean content"
    messages = [{"role": "user", "content": augmented}]

    agent._apply_persist_user_message_override(messages)
    assert messages == [{"role": "user", "content": augmented}]

    agent._flush_messages_to_session_db(messages, [])

    assert messages[0]["content"] == augmented
    assert messages[0].get("_db_persisted") is True
    first_db_write = agent._session_db.append_message.call_args_list[0].kwargs
    assert first_db_write["content"] == "[Example User] Clean content"
    assert first_db_write["timestamp"] == _epoch(2026, 4, 28, 13, 40, 53)


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
