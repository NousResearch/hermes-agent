from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource, SendResult
from gateway.session import SessionEntry, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_make_source(), message_id="m1")


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
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
    runner._gateway_run_states = {}
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
    runner._MAX_INTERRUPT_DEPTH = 3
    runner.pairing_store = MagicMock()
    runner.pairing_store._is_rate_limited = MagicMock(return_value=False)
    runner.pairing_store.generate_code = MagicMock(return_value="ABC123")
    return runner


@pytest.mark.asyncio
async def test_update_gateway_run_state_tracks_timeout_state():
    now = datetime.now()
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    event = _make_event("hello")

    await runner._update_gateway_run_state(
        "timed_out",
        source=event.source,
        session_key=build_session_key(event.source),
        event=event,
        reason="agent_timeout:300s",
    )

    state = runner._gateway_run_states[event.run_id]
    assert state["state"] == "timed_out"
    assert state["reason"] == "agent_timeout:300s"


@pytest.mark.asyncio
async def test_update_gateway_run_state_tracks_partial_failure_state():
    now = datetime.now()
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    event = _make_event("hello")

    await runner._update_gateway_run_state(
        "partial_failure",
        source=event.source,
        session_key=build_session_key(event.source),
        event=event,
        reason="post_stream_media_partial_failure",
    )

    state = runner._gateway_run_states[event.run_id]
    assert state["state"] == "partial_failure"
    assert state["reason"] == "post_stream_media_partial_failure"


@pytest.mark.asyncio
async def test_update_gateway_run_state_tracks_awaiting_approval_state():
    now = datetime.now()
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    event = _make_event("hello")

    await runner._update_gateway_run_state(
        "awaiting_approval",
        source=event.source,
        session_key=build_session_key(event.source),
        event=event,
        reason="dangerous_command_requires_approval",
    )

    state = runner._gateway_run_states[event.run_id]
    assert state["state"] == "awaiting_approval"
    assert state["reason"] == "dangerous_command_requires_approval"


@pytest.mark.asyncio
async def test_deliver_media_from_response_tracks_partial_failure_state():
    now = datetime.now()
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    event = _make_event("hello")

    class _Adapter:
        name = "Telegram"

        def extract_media(self, response):
            return [], response

        def extract_images(self, response):
            return [], response

        def extract_local_files(self, response):
            return ["/tmp/test.png"], response

        async def send_image_file(self, **kwargs):
            return SendResult(success=False, error="boom")

        async def send_document(self, **kwargs):
            return SendResult(success=True, message_id="d1")

        async def send_voice(self, **kwargs):
            return SendResult(success=True, message_id="v1")

        async def send_video(self, **kwargs):
            return SendResult(success=True, message_id="vid1")

    await runner._deliver_media_from_response("hello", event, _Adapter())

    state = runner._gateway_run_states[event.run_id]
    assert state["state"] == "partial_failure"
    assert state["reason"] == "post_stream_media_partial_failure"
