"""Age of a reply-context anchor must reach the model.

``[Replying to: "..."]`` alone cannot distinguish a reply to the previous
message from a reply to something sent hours and several turns ago — the
anchor renders identically either way, so the agent reads a stale frame as
the live thread. The anchor's send time is available at every adapter that
has a real reply target, and the prefix must surface it as an age bucket.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _now():
    return datetime.now(tz=timezone.utc)


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")


def _event(reply_to_ts=None, text="following up on that", reply_to_text="the referenced message", **kwargs):
    return MessageEvent(
        text=text,
        reply_to_message_id="42",
        reply_to_text=reply_to_text,
        reply_to_timestamp=reply_to_ts,
        **kwargs,
    )


def _render(event):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    return runner._prepend_inbound_reply_context(event, _source(), event.text)


def test_fresh_reply_mentions_just_now():
    rendered = _render(_event(reply_to_ts=_now() - timedelta(seconds=30)))
    assert rendered.startswith('[Replying to a message from just now: "the referenced message"]')


def test_minutes_ago_bucket():
    rendered = _render(_event(reply_to_ts=_now() - timedelta(minutes=5, seconds=5)))
    assert rendered.startswith('[Replying to a message from 5 minutes ago: "the referenced message"]')


def test_hours_ago_bucket():
    rendered = _render(_event(reply_to_ts=_now() - timedelta(hours=2, minutes=30)))
    assert rendered.startswith('[Replying to a message from 2 hours ago: "the referenced message"]')


def test_hours_ago_bucket_singular():
    rendered = _render(_event(reply_to_ts=_now() - timedelta(hours=1, minutes=5)))
    assert rendered.startswith('[Replying to a message from 1 hour ago: "the referenced message"]')


def test_yesterday_bucket():
    rendered = _render(_event(reply_to_ts=_now() - timedelta(hours=30)))
    assert rendered.startswith('[Replying to a message from yesterday: "the referenced message"]')


def test_older_than_yesterday_shows_date():
    ts = _now() - timedelta(days=5)
    rendered = _render(_event(reply_to_ts=ts))
    assert rendered.startswith(
        f'[Replying to a message from {ts.astimezone(timezone.utc).strftime("%Y-%m-%d")}: "the referenced message"]'
    )


def test_own_message_wording_keeps_age():
    rendered = _render(
        _event(
            reply_to_ts=_now() - timedelta(minutes=45),
            reply_to_is_own_message=True,
            reply_to_text="here you go",
            text="thanks!",
        )
    )
    assert rendered.startswith('[Replying to your previous message from 45 minutes ago: "here you go"]')


def test_missing_timestamp_degrades_to_legacy_prefix():
    rendered = _render(_event(reply_to_ts=None))
    assert rendered.startswith('[Replying to: "the referenced message"]')


def test_naive_timestamp_degrades_to_legacy_prefix():
    rendered = _render(_event(reply_to_ts=datetime.now().replace(tzinfo=None) - timedelta(hours=2)))
    assert rendered.startswith('[Replying to: "the referenced message"]')


def test_future_timestamp_clamped_to_just_now():
    rendered = _render(_event(reply_to_ts=_now() + timedelta(hours=1)))
    assert rendered.startswith('[Replying to a message from just now: "the referenced message"]')


@pytest.mark.asyncio
async def test_prepared_inbound_text_carries_age_through_full_path():
    """End-to-end through the public preparation path, not just the renderer: the anchor
    that reaches the model names the age bucket."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    source = _source()

    result = await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="following up on that",
            source=source,
            reply_to_message_id="42",
            reply_to_text="the referenced message",
            reply_to_timestamp=_now() - timedelta(minutes=20, seconds=10),
        ),
        source=source,
        history=[],
    )
    assert result is not None
    assert result.startswith('[Replying to a message from 20 minutes ago: "the referenced message"]')
    assert result.endswith("following up on that")


def test_telegram_adapter_carries_reply_target_date():
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    target_ts = _now() - timedelta(hours=3)
    msg = SimpleNamespace(
        chat=SimpleNamespace(id=111, type="private", title=None, full_name="Alice"),
        from_user=SimpleNamespace(id=42, full_name="Alice"),
        text="following up",
        message_thread_id=None,
        message_id=1001,
        reply_to_message=SimpleNamespace(message_id=42, text="the referenced message", date=target_ts),
        quote=None,
        date=_now(),
        forum_topic_created=None,
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)
    assert event.reply_to_timestamp == target_ts


def test_telegram_adapter_tolerates_target_without_date():
    """PTB always sends ``date``; test doubles may not. The event must still build."""
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    msg = SimpleNamespace(
        chat=SimpleNamespace(id=111, type="private", title=None, full_name="Alice"),
        from_user=SimpleNamespace(id=42, full_name="Alice"),
        text="following up",
        message_thread_id=None,
        message_id=1001,
        reply_to_message=SimpleNamespace(message_id=42, text="the referenced message"),
        quote=None,
        date=_now(),
        forum_topic_created=None,
    )
    event = adapter._build_message_event(msg, MessageType.TEXT)
    assert event.reply_to_text == "the referenced message"
    assert event.reply_to_timestamp is None
