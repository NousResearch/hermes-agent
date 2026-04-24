from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="1632061707",
        chat_id="1632061707",
        user_name="Thiago",
        chat_type="dm",
    )


def _make_event(pdf_path: str = "/tmp/doc.pdf") -> MessageEvent:
    return MessageEvent(
        text="",
        source=_make_source(),
        message_id="m1",
        message_type=MessageType.DOCUMENT,
        media_urls=[pdf_path],
        media_types=["application/pdf"],
    )


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
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
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
    runner.pairing_store = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_pdf_dm_triggers_auto_ingest_and_skips_agent(monkeypatch):
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("agent should not run for auto-ingest pdf drops"))
    runner._auto_ingest_pdf_drop = AsyncMock(return_value="PDF received and ingested.\nrecord_id: src_pdf_123")

    result = await runner._handle_message(_make_event())

    assert result is not None
    assert "record_id: src_pdf_123" in result
    runner._run_agent.assert_not_called()
    runner._auto_ingest_pdf_drop.assert_awaited_once()


@pytest.mark.asyncio
async def test_pdf_drop_sends_receipt_message(monkeypatch):
    runner = _make_runner()

    async def _fake_send(chat_id, content, metadata=None):
        return None

    runner.adapters[Platform.TELEGRAM].send = AsyncMock(side_effect=_fake_send)

    async def _fake_run_in_executor(_executor, func):
        return {
            "status": "success",
            "record_id": "src_pdf_deadbeef",
            "parser_selected": "docling",
            "route_reason": "default Docling path",
            "staged_path": "/tmp/staged.pdf",
        }

    loop = MagicMock()
    loop.run_in_executor = AsyncMock(side_effect=_fake_run_in_executor)
    monkeypatch.setattr("gateway.run.asyncio.get_running_loop", lambda: loop)

    result = await runner._auto_ingest_pdf_drop(_make_event(), _make_source())

    runner.adapters[Platform.TELEGRAM].send.assert_awaited_once()
    sent_args = runner.adapters[Platform.TELEGRAM].send.await_args.args
    assert sent_args[1] == "Received PDF. Staging and parsing now."
    assert "record_id: src_pdf_deadbeef" in result


@pytest.mark.asyncio
async def test_non_pdf_document_does_not_auto_ingest():
    from gateway.run import GatewayRunner

    event = MessageEvent(
        text="",
        source=_make_source(),
        message_id="m1",
        message_type=MessageType.DOCUMENT,
        media_urls=["/tmp/file.docx"],
        media_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    )
    assert GatewayRunner._is_auto_ingest_pdf_drop(event, _make_source()) is False
