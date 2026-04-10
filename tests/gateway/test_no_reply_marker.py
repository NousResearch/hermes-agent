"""Tests for the gateway's silent reply marker contract."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_runner(platform):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=f"agent:main:{platform.value}:group:chat-1:user-1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._update_prompt_pending = {}
    runner.pairing_store = MagicMock()
    runner._set_session_env = lambda _context: None
    runner._is_user_authorized = lambda _source: True
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._deliver_media_from_response = AsyncMock()
    runner._run_process_watcher = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "[[NO_REPLY]]",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


@pytest.mark.asyncio
async def test_no_reply_marker_suppresses_outbound_message_but_still_persists_session():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    event = MessageEvent(
        text="只是路过说一句",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m1",
    )

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_placeholder_suppresses_outbound_message_but_still_persists_session():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="发张图",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="group",
        ),
        message_id="m2",
    )

    result = await runner._handle_message(event)

    assert result is None
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_placeholder_in_dm_returns_fallback_reply():
    platform = getattr(Platform, "QQ_NAPCAT")
    runner = _make_runner(platform)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "(empty)",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    event = MessageEvent(
        text="你在吗",
        source=SessionSource(
            platform=platform,
            user_id="123456",
            chat_id="987654",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m3",
    )

    result = await runner._handle_message(event)

    assert result == "刚才接口抽了，没吐出正文。你再发一条，或者我继续接着刚才的话题说。"
    assert runner.session_store.append_to_transcript.called
    runner._send_voice_reply.assert_not_awaited()
    runner._deliver_media_from_response.assert_not_awaited()
