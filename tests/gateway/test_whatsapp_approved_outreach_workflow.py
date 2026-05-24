from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionEntry, SessionSource
from gateway.whatsapp_message_store import append_whatsapp_record
from gateway.whatsapp_approved_outreach import load_whatsapp_outreach_run_records


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
        user_name="Founder",
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
        message_id="evt-1",
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
                enabled=True,
                extra={"allow_admin_from": ["15550000001"]},
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


def _append_record(
    base_dir,
    *,
    record_id: str,
    text: str,
    participant_role: str,
    message_id: str,
    effective_event_at: str,
) -> None:
    append_whatsapp_record(
        {
            "record_kind": "conversation_record",
            "record_id": record_id,
            "conversation_key": "whatsapp:dm:15551230000",
            "destination_key": "whatsapp:dm:15551230000",
            "destination_context_type": "direct_message",
            "destination_chat_id": "15551230000@s.whatsapp.net",
            "destination_target_id": "15551230000",
            "group_chat_id": None,
            "dm_counterparty_id": "15551230000",
            "direction": "outbound" if participant_role == "agent" else "inbound",
            "effective_event_at_utc": effective_event_at,
            "record_sequence": int(record_id.split("-")[-1]),
            "participant_role": participant_role,
            "message_classification": (
                "command_capable"
                if participant_role == "owner_operator"
                else "conversational_only"
            ),
            "command_authority_scope": (
                "owner_only" if participant_role == "owner_operator" else "none"
            ),
            "sender_id": "agent" if participant_role == "agent" else "15551230000",
            "sender_name": "Hermes" if participant_role == "agent" else "Vendor",
            "text": text,
            "message_id": message_id,
            "media_types": [],
        },
        effective_event_at=datetime.fromisoformat(
            effective_event_at.replace("Z", "+00:00")
        ),
        base_dir=base_dir,
    )


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_sends_one_bounded_follow_up(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(
            success=True,
            message_id="bridge-msg-9",
            raw_response={
                "dispatch_group_id": "dispatch-123",
                "messageId": "bridge-msg-9",
            },
        )
    )

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_awaited_once_with(
        "15551230000@s.whatsapp.net",
        "Following up on the revised quote.",
    )
    assert "WhatsApp approved outreach" in result
    assert "Status: ready" in result
    assert "trigger_source: owner_instruction" in result
    assert "run_status: completed" in result
    assert "execution_status: sent" in result
    assert "dispatch_group_id: dispatch-123" in result
    assert "message_id: bridge-msg-9" in result

    run_records = load_whatsapp_outreach_run_records()
    assert len(run_records) == 1
    persisted = run_records[0]
    assert (
        persisted["plan"]["approved_targets"][0]["max_outbound_messages_per_run"] == 1
    )
    assert persisted["run"]["trigger_source"] == "owner_instruction"
    assert persisted["run"]["target_executions"][0]["execution_status"] == "sent"


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_can_complete_without_send(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock()

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="check whether a follow-up is needed"'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_not_awaited()
    assert "Status: ready" in result
    assert "run_status: completed" in result
    assert "execution_status: no_send_required" in result
    assert "history_record_count: 1" in result


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_fails_closed_for_ambiguous_target(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock()

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            "dm_counterparty_id=15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    runner.adapters[Platform.WHATSAPP].send.assert_not_awaited()
    assert "Status: invalid_request" in result


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_is_forbidden_for_external_party(tmp_path):
    runner = _make_runner(tmp_path)

    result = await runner._handle_whatsapp_approved_outreach_instruction(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."',
            participant_role="external_party",
            command_authority_scope="none",
        )
    )

    assert "Status: forbidden" in result


@pytest.mark.asyncio
async def test_instruction_triggered_outreach_surfaces_send_failure(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    base_dir = hermes_home / "gateway" / "whatsapp-records"
    base_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _append_record(
        base_dir,
        record_id="record-1",
        text="Prior vendor thread context.",
        participant_role="external_party",
        message_id="msg-1",
        effective_event_at="2024-06-02T09:01:00Z",
    )

    runner = _make_runner(tmp_path)
    runner.adapters[Platform.WHATSAPP].send = AsyncMock(
        return_value=SendResult(success=False, error="bridge down")
    )

    result = await runner._handle_message(
        _make_event(
            "whatsapp outreach destination_key=whatsapp:dm:15551230000 "
            'operator_objective="request the revised quote" '
            'message_text="Following up on the revised quote."'
        )
    )

    assert "Status: ready" in result
    assert "run_status: failed" in result
    assert "execution_status: send_failed" in result
    assert "last_error: bridge down" in result
