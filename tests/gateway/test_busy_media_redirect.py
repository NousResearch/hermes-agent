"""Busy media retains its attachment context without cancelling foreground work."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy", "priority"])
@pytest.mark.parametrize("kind,caption,mime,transcript", [
    (MessageType.DOCUMENT, "Read this too", "text/plain", None),
    (MessageType.PHOTO, "", "image/png", None),
    (MessageType.TEXT, "See attachment", "application/pdf", None),
    (MessageType.VOICE, "", "audio/ogg", "voice correction"),
    (MessageType.VOICE, "Use the cached voice", "audio/ogg", None),
])
async def test_interrupt_media_redirects_without_interrupting(route, kind, caption, mime, transcript, monkeypatch):
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester", thread_id="thread")
    event = MessageEvent(text=caption, source=source, message_type=kind,
                         media_urls=["/cached/attachment"], media_types=[mime], message_id="followup")
    received = []

    class Agent:
        _supports_active_turn_redirect = True

        def redirect(self, text):
            received.append(text)
            return True

        def interrupt(self, text):
            pytest.fail("an attachment must not cancel the active command")

    agent = Agent()
    runner._session_state("key").turn.agent = agent
    monkeypatch.setattr(runner, "_transcribe_and_echo_pending_voice",
                        AsyncMock(return_value=(transcript or "", [transcript] if transcript else [])))
    if route == "priority":
        await runner._hm_busy_interrupt(event, source, agent, "key")
    else:
        outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
        assert outcome.redirected
    assert len(received) == 1
    assert (transcript or "/cached/attachment") in received[0]
    assert caption in received[0]


@pytest.mark.asyncio
async def test_command_is_not_reinterpreted_as_a_media_redirect():
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester")
    event = MessageEvent(text="/stop", source=source, message_type=MessageType.COMMAND,
                         media_urls=["/cached/attachment"], media_types=["text/plain"])

    class Agent:
        _supports_active_turn_redirect = True

        def redirect(self, text):
            pytest.fail("commands retain their control semantics")

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", Agent())
    assert not outcome.redirected
