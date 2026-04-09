"""Gateway-flow tests for Nostr DM handling."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.NOSTR,
        user_id="abcdef1234",
        chat_id="abcdef1234",
        user_name="npub1sender",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="gift-1",
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.NOSTR: PlatformConfig(
                enabled=True,
                token="nsec1test",
                extra={"relays": ["wss://relay.example"]},
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.NOSTR: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._model = "openai/test-model"
    runner._base_url = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner.delivery_router = MagicMock()
    runner.pairing_store = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_handle_message_persists_timeout_reply_for_nostr_dm(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    timeout_reply = (
        "⏱️ Agent inactive for 3 min — no tool calls or API responses.\n"
        "Last activity: waiting for provider response (180s ago, iteration 1/90). "
        "The agent may have been waiting on an API response."
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="nostr-sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.NOSTR,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": timeout_reply,
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "failed": True,
        }
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("Hi Hermes"))

    assert result == timeout_reply
    persisted_entries = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert persisted_entries[0]["role"] == "session_meta"
    assert persisted_entries[0]["platform"] == "nostr"
    assert persisted_entries[1]["role"] == "user"
    assert persisted_entries[1]["content"] == "Hi Hermes"
    assert persisted_entries[2]["role"] == "assistant"
    assert persisted_entries[2]["content"] == timeout_reply
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        last_prompt_tokens=0,
    )
