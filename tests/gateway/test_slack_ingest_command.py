from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="U123",
        chat_id="C123",
        user_name="tester",
        chat_type="channel",
        thread_id="171111.2222",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.SLACK: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._check_slash_access = lambda *_args, **_kwargs: None
    runner._is_telegram_topic_root_lobby = lambda *_args, **_kwargs: False
    runner._thread_metadata_for_source = lambda _source: {}
    runner._claim_active_session_slot = lambda *_args, **_kwargs: (None, None)
    runner._begin_session_run_generation = lambda *_args, **_kwargs: 1
    runner._release_running_agent_state = lambda *_args, **_kwargs: None
    runner._post_turn_goal_continuation = AsyncMock()
    runner._handle_message_with_agent = AsyncMock(return_value="ok")
    return runner


@pytest.mark.asyncio
async def test_slack_ingest_rewrites_to_agent_prompt(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    event = _make_event(
        "/slack-ingest 2026-06-24 기준으로 <#C0B7QVCLQF9>, <#C0B833C0C2H>, "
        "<#C05PG78UK19> 채널의 업무만 요약해서 LLM-Wiki에 1일 단위로 인제스트해줘"
    )
    result = await runner._handle_message(event)

    assert result == "ok"
    assert runner._handle_message_with_agent.await_count == 1
    rewritten_event = runner._handle_message_with_agent.await_args.args[0]
    assert "Slack channel history" in rewritten_event.text
    assert "LLM-Wiki" in rewritten_event.text
    runner.adapters[Platform.SLACK].send.assert_awaited()


@pytest.mark.asyncio
async def test_slack_ingest_without_args_returns_usage(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("blank /slack-ingest should not reach the agent")
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/slack-ingest"))

    assert "Usage: /slack-ingest" in result
