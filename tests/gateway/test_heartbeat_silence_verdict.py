"""Heartbeat-injected turns keep their silence; the human-turn guard stays intact.

#113031: a NO_REPLY marker returned on a heartbeat turn was rejected by the
human-silence guard and replaced with the user-facing warning every interval.
Heartbeat events stay non-internal (authorization and the emergency stop still
apply), so the silence verdict honors the heartbeat-injection marker instead.
"""

import asyncio  # noqa: F401
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn import _UNEXPECTED_SILENCE_REPLY
from gateway.session import SessionSource


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")


def _silence_result():
    return {"final_response": "NO_REPLY", "failed": False, "api_calls": 0, "messages": []}


async def _shape(runner, event, source, agent_result, persist_user_display_kind=None):
    session_entry = SimpleNamespace(session_id="heartbeat-session")
    return await runner._hmwa_shape_agent_response(
        agent_result, source, [], session_entry, None, "quick-key", 1,
        "heartbeat-session", "telegram", 0.0,
        persist_user_display_kind=persist_user_display_kind,
        event=event,
    )


@pytest.mark.asyncio
async def test_heartbeat_injected_turn_may_stay_silent():
    runner = object.__new__(GatewayRunner)
    source = _source()
    event = MessageEvent(text="check status", message_type=MessageType.TEXT, source=source, internal=False)
    assert not event.internal  # authorization and emergency-stop still apply
    event._heartbeat_session_id = "heartbeat-session"  # what _heartbeat_poll_watch stamps
    response, intentional_silence, _ = await _shape(runner, event, source, _silence_result())
    assert intentional_silence is True
    assert response == "NO_REPLY"


@pytest.mark.asyncio
async def test_human_turn_silence_still_rejected():
    runner = object.__new__(GatewayRunner)
    source = _source()
    event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source, internal=False)
    response, intentional_silence, _ = await _shape(runner, event, source, _silence_result())
    assert intentional_silence is False
    assert response == _UNEXPECTED_SILENCE_REPLY
