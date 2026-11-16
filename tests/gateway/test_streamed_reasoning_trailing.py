"""show_reasoning on a streamed reply (#7251, revised).

Originally (#7251) a streamed reply showed its reasoning as a trailing message *after* the answer,
because the answer streamed live and the reasoning block was only known at turn end. That put the
reasoning below the answer in the chat. The behaviour now: when show_reasoning is on for a
messaging platform, the final answer is NOT streamed live, so the gateway's normal final send
delivers reasoning and answer as ONE message with the 💭 block on top (``_hmwa_prepend_reasoning``).
Local viewers are unaffected (they read reasoning off their event stream).
"""

import pytest

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
