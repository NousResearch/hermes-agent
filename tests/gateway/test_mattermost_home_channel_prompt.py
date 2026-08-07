import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _StopProcessing(RuntimeError):
    pass


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.MATTERMOST: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.MATTERMOST: SimpleNamespace(send=AsyncMock())}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id="session-1",
        session_key="agent:main:mattermost:channel:chan_456",
        created_at=1,
        updated_at=2,
        was_auto_reset=False,
        last_prompt_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._set_session_env = MagicMock(return_value=[])
    return runner


def _make_event(thread_id=None):
    source = SessionSource(
        platform=Platform.MATTERMOST,
        chat_type="channel",
        chat_id="chan_456",
        chat_name="Town Square",
        user_id="user_123",
        user_name="alan",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="post_123",
    )


@pytest.mark.asyncio
async def test_skips_home_channel_prompt_for_thread_messages(monkeypatch):
    runner = _make_runner()
    event = _make_event(thread_id="root_post_123")

    monkeypatch.delenv("MATTERMOST_HOME_CHANNEL", raising=False)

    with patch(
        "gateway.run.build_session_context_prompt", return_value="ctx"
    ), patch.object(
        runner,
        "_prepare_inbound_message_text",
        AsyncMock(side_effect=_StopProcessing("stop after onboarding check")),
    ):
        with pytest.raises(_StopProcessing):
            await runner._handle_message_with_agent(event, event.source, "quick-key", 1)

    runner.adapters[Platform.MATTERMOST].send.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompts_for_home_channel_on_new_channel_session(monkeypatch):
    runner = _make_runner()
    event = _make_event()

    monkeypatch.delenv("MATTERMOST_HOME_CHANNEL", raising=False)

    with patch(
        "gateway.run.build_session_context_prompt", return_value="ctx"
    ), patch.object(
        runner,
        "_prepare_inbound_message_text",
        AsyncMock(side_effect=_StopProcessing("stop after onboarding check")),
    ):
        with pytest.raises(_StopProcessing):
            await runner._handle_message_with_agent(event, event.source, "quick-key", 1)

    runner.adapters[Platform.MATTERMOST].send.assert_awaited_once()
    args = runner.adapters[Platform.MATTERMOST].send.await_args.args
    assert args[0] == "chan_456"
    assert "No home channel is set for Mattermost" in args[1]
