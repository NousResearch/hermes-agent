"""Tests for #109002: the drain-window refusal must be debounced per (profile, chat).

During a restart drain, the idle-path ``_draining`` guard used to answer EVERY inbound
message with the identical "not accepting new work" notice. A drain window can last
~30 minutes (force-drain cap), so a user who kept typing saw the same line repeated
and reasonably concluded the bot was stuck in a loop. The notice now goes through the
existing per-(profile, chat) cooldown helper (same pattern as the Telegram lobby
reminder): first message answered, messages inside the cooldown window stay silent.
"""
import time
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_source(chat_id: str = "c1") -> SessionSource:
    return SessionSource(
        platform=Platform.WECOM,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


def _make_event(source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _make_runner() -> "GatewayRunner":
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._draining = True
    runner._restart_requested = True  # → "restarting" in the notice text
    return runner


async def _dispatch(runner, source):
    return await runner._hm_dispatch_quick_and_plugin_commands(
        _make_event(source), source, None
    )


@pytest.mark.asyncio
async def test_first_drain_message_gets_the_notice():
    runner = _make_runner()
    source = _make_source()

    handled, result, command = await _dispatch(runner, source)

    assert handled is True
    assert result is not None and "not accepting new work" in result
    assert command is None


@pytest.mark.asyncio
async def test_repeat_messages_inside_cooldown_window_stay_silent():
    runner = _make_runner()
    source = _make_source()

    _handled, first, _cmd = await _dispatch(runner, source)
    assert first is not None

    # Burst (0.9 s apart in the report) and minutes-later follow-ups: all inside the window.
    for _ in range(3):
        handled, result, command = await _dispatch(runner, source)
        assert handled is True
        assert result is None  # already told this chat; swallow instead of repeating
        assert command is None


@pytest.mark.asyncio
async def test_notice_returns_after_the_cooldown_window_elapses():
    runner = _make_runner()
    source = _make_source()

    _handled, first, _cmd = await _dispatch(runner, source)
    assert first is not None

    # Age the stamp past the window (the drain itself can last ~30 min).
    key = next(iter(runner._drain_notice_ts))
    runner._drain_notice_ts[key] = time.monotonic() - runner._DRAIN_NOTICE_COOLDOWN_S - 1.0

    handled, result, _cmd = await _dispatch(runner, source)
    assert handled is True
    assert result is not None and "not accepting new work" in result


@pytest.mark.asyncio
async def test_cooldown_is_per_chat():
    runner = _make_runner()

    _handled, first_a, _cmd = await _dispatch(runner, _make_source(chat_id="chat-a"))
    _handled, first_b, _cmd = await _dispatch(runner, _make_source(chat_id="chat-b"))
    assert first_a is not None
    assert first_b is not None  # a different chat is not suppressed by chat-a's notice

    _handled, second_a, _cmd = await _dispatch(runner, _make_source(chat_id="chat-a"))
    assert second_a is None  # ...but chat-a itself stays silent inside the window


@pytest.mark.asyncio
async def test_no_drain_means_no_gate():
    runner = _make_runner()
    runner._draining = False
    runner.config = {}  # no quick commands configured
    source = _make_source()

    handled, result, command = await _dispatch(runner, source)

    assert handled is False  # falls through to normal idle processing
    assert result is None
