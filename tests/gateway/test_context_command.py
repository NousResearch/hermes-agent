from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, coerce_plaintext_gateway_command
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(chat_type: str = "dm", *, chat_id: str = "c1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type=chat_type,
    )


def _make_event(text: str, *, chat_type: str = "dm", chat_id: str = "c1") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(chat_type, chat_id=chat_id), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    def _session_for(source):
        return SessionEntry(
            session_key=build_session_key(source),
            session_id=f"sess-{getattr(source, 'chat_id', '1')}",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.side_effect = _session_for
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
    runner._draining = False
    return runner


@pytest.mark.asyncio
async def test_gateway_context_new_and_status(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CONTEXT_CARDS_DIR", str(tmp_path))
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("/context leaked to agent"))

    created = await runner._handle_message(_make_event("/context new hermes-agent B001-hermes-chat-context-control"))
    status = await runner._handle_message(_make_event("/ctx status"))

    assert "새 Context 고정" in created
    assert "Project: hermes-agent" in status
    assert "Batch: B001-hermes-chat-context-control" in status
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_context_active_card_is_scoped_by_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CONTEXT_CARDS_DIR", str(tmp_path))
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("/context leaked to agent"))

    await runner._handle_message(_make_event("/context new hermes-agent B001", chat_id="c1"))
    await runner._handle_message(_make_event("/context new MIM B002", chat_id="c2"))

    first_status = await runner._handle_message(_make_event("/context status", chat_id="c1"))
    second_status = await runner._handle_message(_make_event("/context status", chat_id="c2"))

    assert "Project: hermes-agent" in first_status
    assert "Batch: B001" in first_status
    assert "Project: MIM" in second_status
    assert "Batch: B002" in second_status
    runner._run_agent.assert_not_called()


def test_plaintext_context_command_is_dm_only():
    dm_event = _make_event("현재 맥락 보여줘", chat_type="dm")
    group_event = _make_event("현재 맥락 보여줘", chat_type="group")

    coerce_plaintext_gateway_command(dm_event)
    coerce_plaintext_gateway_command(group_event)

    assert dm_event.text == "/context status"
    assert group_event.text == "현재 맥락 보여줘"
