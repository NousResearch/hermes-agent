from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_gateway_passes_parent_trace_id_to_run_agent(monkeypatch, tmp_path):
    adapter = MagicMock()
    adapter.send = AsyncMock()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
        sessions_dir=tmp_path / "sessions",
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._pending_approvals = {}
    runner._pending_messages = {}
    runner._running_agents = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._prefill_messages = None
    runner._provider_routing = {}
    runner._fallback_model = None

    runner.session_store = MagicMock()
    session_entry = SessionEntry(
        session_key="agent:main:telegram:dm",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        langfuse_parent_observation_id="obs-parent-1",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.ensure_langfuse_parent_trace_id.return_value = "trace-parent-1"
    runner.session_store.update_langfuse_parent_observation_id = MagicMock()
    runner.session_store.load_transcript.return_value = []

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "langfuse_current_observation_id": "obs-new-2",
        }
    )

    monkeypatch.setenv("HERMES_LANGFUSE_ENABLED", "true")

    event = MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        message_id="m1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    runner.session_store.ensure_langfuse_parent_trace_id.assert_called_once()

    call_kwargs = runner._run_agent.await_args.kwargs
    assert call_kwargs["parent_trace_id"] == "trace-parent-1"
    assert call_kwargs["parent_observation_id"] == "obs-parent-1"

    runner.session_store.update_langfuse_parent_observation_id.assert_called_once_with(
        session_entry, "obs-new-2"
    )


@pytest.mark.asyncio
async def test_gateway_does_not_touch_trace_context_when_disabled(monkeypatch, tmp_path):
    adapter = MagicMock()
    adapter.send = AsyncMock()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
        sessions_dir=tmp_path / "sessions",
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._pending_approvals = {}
    runner._pending_messages = {}
    runner._running_agents = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._prefill_messages = None
    runner._provider_routing = {}
    runner._fallback_model = None

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.ensure_langfuse_parent_trace_id = MagicMock()
    runner.session_store.update_langfuse_parent_observation_id = MagicMock()
    runner.session_store.load_transcript.return_value = []

    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.setenv("HERMES_LANGFUSE_ENABLED", "false")

    event = MessageEvent(
        text="hello",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        message_id="m1",
    )

    result = await runner._handle_message(event)

    assert result == "ok"
    assert "parent_trace_id" in runner._run_agent.await_args.kwargs
    assert runner._run_agent.await_args.kwargs["parent_trace_id"] is None
    assert runner._run_agent.await_args.kwargs.get("parent_observation_id") is None
    runner.session_store.ensure_langfuse_parent_trace_id.assert_not_called()
    runner.session_store.update_langfuse_parent_observation_id.assert_not_called()
