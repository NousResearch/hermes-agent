"""Regression test for the internal-event skip in ``GatewayRunner._prefix_inbound_sender_context``.

Bug (#139697): in a shared multi-user session (a forum topic under the default
``thread_sessions_per_user: false``), the sender prefix ``[<name>] `` was prepended to
internal events too — plugin injections from ``ctx.inject_message()`` reached the model
as a message from the human named in the restored source. That both broke control-line
protocols plugins use to recognise their own injected text and misattributed the text to
that person. Internal events (plugin injections, heartbeat/goal wakeups, automation) are
not participant messages, so they carry no sender prefix; the channel-context backfill
below the prefix is unaffected.
"""
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    return runner


def _topic_source() -> SessionSource:
    # A forum topic: shared across participants under the default thread_sessions_per_user=False.
    return SessionSource(
        platform=Platform.TELEGRAM, chat_id="-100123", chat_type="group", thread_id="42",
        user_id="111", user_name="Owner Name",
    )


def test_human_message_keeps_sender_prefix():
    runner = _make_runner()
    src = _topic_source()
    ev = MessageEvent(text="hello there", message_type=MessageType.TEXT, source=src, internal=False)
    assert runner._prefix_inbound_sender_context(ev, src, ev.text) == "[Owner Name] hello there"


def test_internal_event_gets_no_sender_prefix():
    runner = _make_runner()
    src = _topic_source()
    ev = MessageEvent(text="PLUGIN TEXT", message_type=MessageType.TEXT, source=src, internal=True)
    assert runner._prefix_inbound_sender_context(ev, src, ev.text) == "PLUGIN TEXT"


def test_internal_event_keeps_channel_backfill():
    # Only the participant attribution is skipped; the channel-context backfill stays.
    runner = _make_runner()
    src = _topic_source()
    ev = MessageEvent(text="PLUGIN TEXT", message_type=MessageType.TEXT, source=src, internal=True,
                      channel_context="CHANNEL CONTEXT")
    assert runner._prefix_inbound_sender_context(
        ev, src, ev.text) == "CHANNEL CONTEXT\n\n[New message]\nPLUGIN TEXT"
