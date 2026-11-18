"""show_reasoning on a streamed reply (#7251, revised).

Originally (#7251) a streamed reply showed its reasoning as a trailing message *after* the answer,
because the answer streamed live and the reasoning block was only known at turn end. That put the
reasoning below the answer in the chat. The behaviour now: when show_reasoning is on for a
messaging platform, the final answer is NOT streamed live, so the gateway's normal final send
delivers reasoning and answer as ONE message with the 💭 block on top (``_hmwa_prepend_reasoning``).
Local viewers are unaffected (they read reasoning off their event stream).
"""

import pytest
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner


async def _deliver(monkeypatch, *, show_reasoning: bool, already_sent: bool, is_bot: bool = False,
                   platform: Platform = Platform.TELEGRAM):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": show_reasoning}})
    adapter = ProgressCaptureAdapter(platform=platform)
    runner = _make_runner(adapter)
    source = SessionSource(platform=platform, chat_id="42", chat_type="dm", is_bot=is_bot)
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")
    result = {"final_response": "the answer is 42", "last_reasoning": "thinking hard about it",
              "already_sent": already_sent}
    response = runner._hmwa_prepend_reasoning(result, "the answer is 42", source, False)

    class _Entry:
        session_id = "s1"

    returned = await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), "agent:main:telegram:dm:42", None, result, [], response, None, False)
    return runner, adapter, returned


@pytest.mark.asyncio
async def test_reasoning_leads_the_answer_in_one_message(monkeypatch):
    """show_reasoning on + not streamed: the normal send carries reasoning ABOVE the answer, as one
    message. Nothing trails."""
    _runner, adapter, returned = await _deliver(monkeypatch, show_reasoning=True, already_sent=False)

    # The gateway's own final send carries the combined message; nothing is sent separately here.
    assert adapter.sent == []
    assert returned.startswith("💭") and "thinking hard about it" in returned
    assert returned.endswith("the answer is 42")


@pytest.mark.asyncio
async def test_show_reasoning_suppresses_live_answer_streaming(monkeypatch):
    """The gate that keeps reasoning on top: a messaging platform with show_reasoning on must not
    stream the final answer's deltas, so the combined send fires instead of a trailing block."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": True}})
    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    tg = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    local = SessionSource(platform=Platform.LOCAL, chat_id="42", chat_type="dm")

    assert runner._show_reasoning_for_source(tg) is True
    # Local viewers read reasoning off their event stream, never the projected block: never gated.
    assert runner._show_reasoning_for_source(local) is False


@pytest.mark.asyncio
async def test_webhook_never_prepends_reasoning_even_globally_enabled(monkeypatch):
    """A global display.show_reasoning must not move a webhook final's first line: the
    urgent-only response policy matches the escalation header on line 0, and a prepended
    reasoning fence would suppress a real page with a successful admission."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": True}})
    adapter = ProgressCaptureAdapter(platform=Platform.WEBHOOK)
    runner = _make_runner(adapter)
    wh = SessionSource(platform=Platform.WEBHOOK, chat_id="webhook:incident:x", chat_type="webhook")

    assert runner._show_reasoning_for_source(wh) is False
    result = {"final_response": "[NEEDS-APPROVAL] disk full", "last_reasoning": "thinking hard",
              "already_sent": False}
    body = "[NEEDS-APPROVAL] disk full\nAction required: purge the WAL archive"
    assert runner._hmwa_prepend_reasoning(result, body, wh, False) == body


class _Consumer:
    """Captures stream_deltas_enabled the way the real consumer carries it."""

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.stream_deltas_enabled = True


def _stream_runner(adapter, show_reasoning):
    """A runner wired to drive the real _setup_stream_consumer: streaming on globally, with the
    live _show_reasoning_for_source resolver bound from the mixin over this show_reasoning default."""
    from gateway.config import GatewayConfig, StreamingConfig
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

    cfg = GatewayConfig(streaming=StreamingConfig(enabled=True))
    cfg.multiplex_profiles = False
    runner = SimpleNamespace(
        config=cfg, _primary_profile_name="default", _profile_configs={},
        _show_reasoning=show_reasoning,
        _delivery_adapter_for=lambda _s: adapter,
        _build_stream_consumer_config=lambda _s, c, _a, **_k: (SimpleNamespace(transport=c.transport), None),
    )
    runner._show_reasoning_for_source = GatewayTurnPersistenceMixin._show_reasoning_for_source.__get__(runner)
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("show_reasoning, platform, expect_stream", [
    (True, Platform.TELEGRAM, False),   # reasoning leads → the answer is NOT streamed live
    (False, Platform.TELEGRAM, True),   # no reasoning → the answer streams as before
    (True, Platform.LOCAL, True),       # LOCAL always streams (reasoning rides the event stream)
])
async def test_setup_stream_consumer_gates_on_reasoning(monkeypatch, show_reasoning, platform, expect_stream):
    """Drive the REAL _setup_stream_consumer: dropping the ``not _reasoning_leads`` term would make
    a Telegram turn stream the answer with no reasoning, so this asserts the gate's effect directly.
    """
    import gateway.run as gateway_run
    from gateway.run_turn_runner import TurnRunner

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": show_reasoning}})
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", _Consumer)
    adapter = ProgressCaptureAdapter(platform=platform)
    runner = _stream_runner(adapter, show_reasoning)
    ctx = SimpleNamespace(
        mute_notification_reply=False, streaming_tts_consumer_holder=[None], user_config={},
        resolve_display_setting=lambda *_a: None, scheduled_heartbeat=False,
        interim_assistant_messages_enabled=False,
        source=SessionSource(platform=platform, chat_id="42", chat_type="dm"),
        _status_thread_metadata=None, progress_queue=None, event_message_id=None,
        _run_still_current=lambda: True, stream_consumer_holder=[None],
    )
    consumer, callback, _interim, _want = TurnRunner(runner, ctx)._setup_stream_consumer(
        "telegram" if platform is Platform.TELEGRAM else "cli")
    streamed = consumer is not None and consumer.stream_deltas_enabled and callback is not None
    assert streamed is expect_stream


@pytest.mark.asyncio
async def test_no_reasoning_block_when_hidden(monkeypatch):
    """show_reasoning off: the normal send carries the bare answer, no 💭 block."""
    _runner, adapter, returned = await _deliver(monkeypatch, show_reasoning=False, already_sent=False)

    assert adapter.sent == []
    assert returned == "the answer is 42"


@pytest.mark.asyncio
async def test_bot_peer_reasoning_matches_the_non_streamed_path(monkeypatch):
    """Streamed and non-streamed turns now behave the same: the normal combined send carries
    whatever ``_hmwa_prepend_reasoning`` produced for this source, with nothing sent separately."""
    runner, adapter, returned = await _deliver(monkeypatch, show_reasoning=True, already_sent=False,
                                               is_bot=True)
    assert adapter.sent == []
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", is_bot=True)
    expected = runner._hmwa_prepend_reasoning(
        {"last_reasoning": "thinking hard about it"}, "the answer is 42", source, False)
    assert returned == expected
