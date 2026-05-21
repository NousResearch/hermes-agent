from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource
from gateway.whatsapp_message_store import query_whatsapp_records
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_whatsapp_plugin = load_plugin_adapter("whatsapp")


def _config(extra: dict | None = None):
    return SimpleNamespace(
        extra=extra or {},
        enabled=True,
        token=None,
        api_key=None,
        home_channel=None,
        reply_to_mode="first",
    )


def _source(*, chat_id: str, user_id: str, chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name="Sender",
    )


@pytest.mark.asyncio
async def test_external_whatsapp_slash_message_stays_conversational_and_is_logged(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = _whatsapp_plugin.WhatsAppPluginAdapter(
        _config(extra={"allow_admin_from": ["15551234567@s.whatsapp.net"]})
    )
    raw_event = MessageEvent(
        text="/status please",
        source=_source(chat_id="18885550000@s.whatsapp.net", user_id="18885550000@lid"),
        raw_message={"messageId": "inbound-1"},
        message_id="inbound-1",
    )

    with patch.object(
        _whatsapp_plugin.BuiltinWhatsAppAdapter,
        "_build_message_event",
        AsyncMock(return_value=raw_event),
    ):
        event = await adapter._build_message_event(
            {
                "chatId": "18885550000@s.whatsapp.net",
                "senderId": "18885550000@lid",
                "senderName": "Lead",
                "body": "/status please",
                "messageId": "inbound-1",
                "timestamp": 1_717_324_800,
            }
        )

    assert event is raw_event
    assert event.participant_role == "external_party"
    assert event.message_classification == "conversational_only"
    assert event.command_authority_scope == "none"
    assert event.get_command() is None
    assert event.get_command_args() == "/status please"

    records = query_whatsapp_records(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert len(records) == 1
    record = records[0]
    assert record["direction"] == "inbound"
    assert record["participant_role"] == "external_party"
    assert record["message_classification"] == "conversational_only"
    assert record["command_authority_scope"] == "none"
    assert record["destination_key"] == "whatsapp:dm:18885550000"
    assert record["destination_context_type"] == "direct_message"
    assert record["destination_chat_id"] == "18885550000@s.whatsapp.net"
    assert record["destination_target_id"] == "18885550000"
    assert record["dm_counterparty_id"] == "18885550000"
    assert record["text"] == "/status please"


@pytest.mark.asyncio
async def test_owner_whatsapp_message_remains_command_capable_via_canonical_identifier(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    session_dir = hermes_home / "whatsapp" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / "lid-mapping-15551234567.json").write_text('"999999999999999"', encoding="utf-8")
    (session_dir / "lid-mapping-999999999999999_reverse.json").write_text('"15551234567"', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = _whatsapp_plugin.WhatsAppPluginAdapter(
        _config(extra={"allow_admin_from": ["15551234567@s.whatsapp.net"]})
    )
    raw_event = MessageEvent(
        text="/new",
        source=_source(chat_id="999999999999999@lid", user_id="999999999999999@lid"),
        raw_message={"messageId": "owner-1"},
        message_id="owner-1",
    )

    with patch.object(
        _whatsapp_plugin.BuiltinWhatsAppAdapter,
        "_build_message_event",
        AsyncMock(return_value=raw_event),
    ):
        event = await adapter._build_message_event(
            {
                "chatId": "999999999999999@lid",
                "senderId": "999999999999999@lid",
                "body": "/new",
                "messageId": "owner-1",
                "timestamp": 1_717_324_801,
            }
        )

    assert event.participant_role == "owner_operator"
    assert event.message_classification == "command_capable"
    assert event.command_authority_scope == "owner_only"
    assert event.get_command() == "new"

    records = query_whatsapp_records(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert records[0]["participant_role"] == "owner_operator"
    assert records[0]["destination_key"] == "whatsapp:dm:15551234567"
    assert records[0]["dm_counterparty_id"] == "15551234567"


@pytest.mark.asyncio
async def test_outbound_send_appends_agent_record_with_dispatch_group_and_destination(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = _whatsapp_plugin.WhatsAppPluginAdapter(_config())
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter.truncate_message = lambda content, limit: ["first chunk", "second chunk"]
    adapter.format_message = lambda content: content

    responses = [
        {"success": True, "messageId": "msg-1", "messageIds": ["msg-1"]},
        {"success": True, "messageId": "msg-2", "messageIds": ["msg-2"]},
    ]

    class _Response:
        def __init__(self, payload):
            self.status = 200
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return self._payload

        async def text(self):
            return str(self._payload)

    class _Session:
        closed = False

        def post(self, *_args, **_kwargs):
            return _Response(responses.pop(0))

    adapter._http_session = _Session()

    result = await adapter.send("18885550000@s.whatsapp.net", "hello there")

    assert isinstance(result, SendResult)
    assert result.success is True
    assert result.message_id == "msg-2"

    records = query_whatsapp_records(
        datetime.now(timezone.utc) - timedelta(days=1),
        datetime.now(timezone.utc) + timedelta(days=1),
    )
    assert len(records) == 2
    assert {record["dispatch_group_sequence"] for record in records} == {1, 2}
    assert len({record["dispatch_group_id"] for record in records}) == 1
    for record in records:
        assert record["direction"] == "outbound"
        assert record["participant_role"] == "agent"
        assert record["message_classification"] == "conversational_only"
        assert record["command_authority_scope"] == "none"
        assert record["destination_key"] == "whatsapp:dm:18885550000"
        assert record["destination_target_id"] == "18885550000"


def test_query_whatsapp_records_scans_overlapping_daily_partitions_only(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    start = datetime(2024, 6, 1, 23, 59, tzinfo=timezone.utc)
    day_one = hermes_home / "gateway" / "whatsapp-records" / "2024-06-01.jsonl"
    day_two = hermes_home / "gateway" / "whatsapp-records" / "2024-06-02.jsonl"
    day_three = hermes_home / "gateway" / "whatsapp-records" / "2024-06-03.jsonl"
    day_one.parent.mkdir(parents=True)
    day_one.write_text(
        '{"effective_event_at_utc":"2024-06-01T23:59:30Z","record_sequence":2,"destination_key":"whatsapp:dm:1"}\n',
        encoding="utf-8",
    )
    day_two.write_text(
        '{"effective_event_at_utc":"2024-06-02T00:00:01Z","record_sequence":1,"destination_key":"whatsapp:dm:1"}\n',
        encoding="utf-8",
    )
    day_three.write_text(
        '{"effective_event_at_utc":"2024-06-03T00:00:01Z","record_sequence":1,"destination_key":"whatsapp:dm:1"}\n',
        encoding="utf-8",
    )

    records = query_whatsapp_records(
        start,
        datetime(2024, 6, 2, 0, 1, tzinfo=timezone.utc),
    )

    assert [record["effective_event_at_utc"] for record in records] == [
        "2024-06-01T23:59:30Z",
        "2024-06-02T00:00:01Z",
    ]
    assert day_three.exists()
