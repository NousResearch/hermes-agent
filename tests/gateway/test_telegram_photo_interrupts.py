from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.run import GatewayRunner


class _PendingAdapter:
    def __init__(self):
        self._pending_messages = {}


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: _PendingAdapter()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    return runner


@pytest.mark.asyncio
async def test_handle_message_does_not_priority_interrupt_photo_followup():
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")
    session_key = build_session_key(source)
    running_agent = MagicMock()
    runner._running_agents[session_key] = running_agent

    event = MessageEvent(
        text="caption",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo-a.jpg"],
        media_types=["image/jpeg"],
    )

    result = await runner._handle_message(event)

    assert result is None
    running_agent.interrupt.assert_not_called()
    assert runner.adapters[Platform.TELEGRAM]._pending_messages[session_key] is event


@pytest.mark.asyncio
async def test_handle_message_priority_path_steers_photo_followup():
    runner = _make_runner()
    runner._busy_input_mode = "steer"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="u1",
    )
    session_key = build_session_key(source)
    running_agent = MagicMock()
    running_agent.steer.return_value = True
    runner._running_agents[session_key] = running_agent

    event = MessageEvent(
        text="(attachment)",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/photo-a.heic"],
        media_types=["image/heic"],
    )

    result = await runner._handle_message(event)

    assert result is None
    steer_text = running_agent.steer.call_args.args[0]
    assert "vision_analyze" in steer_text
    assert "/tmp/photo-a.heic" in steer_text
    running_agent.interrupt.assert_not_called()
    assert session_key not in runner.adapters[Platform.TELEGRAM]._pending_messages
