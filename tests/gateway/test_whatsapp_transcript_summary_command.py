from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_source(
    *,
    platform: Platform = Platform.WHATSAPP,
    user_id: str = "15550000001",
    chat_id: str = "15551230000@s.whatsapp.net",
    chat_type: str = "dm",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="Operator",
        chat_type=chat_type,
    )


def _make_event(
    text: str,
    *,
    participant_role: str | None = "owner_operator",
    command_authority_scope: str | None = "owner_only",
    platform: Platform = Platform.WHATSAPP,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform=platform),
        message_id="m1",
        participant_role=participant_role,
        message_classification=(
            "command_capable"
            if participant_role == "owner_operator"
            else "conversational_only"
        ),
        command_authority_scope=command_authority_scope,
    )


def _make_runner(tmp_path):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True, extra={"allow_admin_from": ["15550000001"]}
            )
        }
    )
    runner.adapters = {Platform.WHATSAPP: MagicMock()}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:whatsapp:dm:15551230000",
        session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        platform=Platform.WHATSAPP,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._session_sources = {}
    runner._session_run_generation = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_wsummary_command_returns_readable_summary_for_authorized_operator(
    tmp_path, monkeypatch
):
    base_dir = tmp_path / ".hermes" / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from gateway.whatsapp_message_store import append_whatsapp_record

    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": "r1",
            "conversation_key": "whatsapp:dm:15551230000",
            "destination_key": "whatsapp:dm:15551230000",
            "destination_context_type": "direct_message",
            "destination_chat_id": "15551230000@s.whatsapp.net",
            "destination_target_id": "15551230000",
            "group_chat_id": None,
            "dm_counterparty_id": "15551230000",
            "direction": "inbound",
            "effective_event_at_utc": "2024-06-02T09:01:00Z",
            "record_sequence": 1,
            "participant_role": "external_party",
            "message_classification": "conversational_only",
            "command_authority_scope": "none",
            "sender_id": "15551230000",
            "sender_name": "Vendor",
            "text": "We can deliver 200 units by Friday.",
        },
        effective_event_at=datetime(2024, 6, 2, 9, 1, 0, tzinfo=timezone.utc),
        base_dir=base_dir,
    )

    runner = _make_runner(tmp_path)
    result = await runner._handle_message(
        _make_event(
            "/wsummary destination_key=whatsapp:dm:15551230000 "
            "range_start_utc=2024-06-02T09:00:00Z "
            "range_end_utc=2024-06-02T10:00:00Z"
        )
    )

    assert "WhatsApp transcript summary" in result
    assert "Status: ready" in result
    assert "Conversation recap:" in result
    assert "Vendor: We can deliver 200 units by Friday." in result


@pytest.mark.asyncio
async def test_wsummary_command_fails_closed_for_external_party(tmp_path):
    runner = _make_runner(tmp_path)

    result = await runner._handle_whatsapp_summary_command(
        _make_event(
            "/wsummary destination_key=whatsapp:dm:15551230000 "
            "range_start_utc=2024-06-02T09:00:00Z "
            "range_end_utc=2024-06-02T10:00:00Z",
            participant_role="external_party",
            command_authority_scope="none",
        )
    )

    assert "Status: forbidden" in result


@pytest.mark.asyncio
async def test_wsummary_command_rejects_invalid_scope_shape(tmp_path):
    runner = _make_runner(tmp_path)

    result = await runner._handle_message(
        _make_event(
            "/wsummary range_start_utc=2024-06-02T10:00:00Z "
            "range_end_utc=2024-06-02T09:00:00Z"
        )
    )

    assert "Status: invalid_request" in result


@pytest.mark.asyncio
async def test_wsummary_command_is_not_available_from_non_whatsapp_surface(tmp_path):
    runner = _make_runner(tmp_path)

    result = await runner._handle_whatsapp_summary_command(
        _make_event(
            "/wsummary destination_key=whatsapp:dm:15551230000 "
            "range_start_utc=2024-06-02T09:00:00Z "
            "range_end_utc=2024-06-02T10:00:00Z",
            platform=Platform.DISCORD,
        )
    )

    assert "Status: forbidden" in result
