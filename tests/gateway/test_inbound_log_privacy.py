"""Privacy regressions for gateway inbound-message logging."""

import logging
import sys
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import (
    _log_safe_gateway_identity,
    _log_inbound_message,
    _log_leftover_steer,
    _log_response_ready,
    _log_startup_restore_message,
)
from gateway.session import SessionSource, build_session_key, session_key_for_log
from hermes_cli.debug import _redact_log_text
from gateway.platforms.whatsapp_cloud import (
    WhatsAppCloudAdapter,
    _log_whatsapp_identifier,
)
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _inbound_record(caplog):
    records = [
        record
        for record in caplog.records
        if record.name == "gateway.run" and record.message.startswith("inbound message:")
    ]
    assert len(records) == 1
    return records[0].message


@pytest.mark.asyncio
async def test_whatsapp_chat_info_exception_log_omits_raw_jid(caplog, monkeypatch):
    chat_id = "15551234567@s.whatsapp.net"
    exception = RuntimeError(
        f"GET http://127.0.0.1:3000/chat/{chat_id} failed for {chat_id}"
    )
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._http_session = MagicMock()
    adapter._bridge_port = 3000
    adapter._bridge_process = None
    adapter._http_session.get = MagicMock(side_effect=exception)
    monkeypatch.setitem(
        sys.modules,
        "aiohttp",
        SimpleNamespace(ClientTimeout=lambda **_kwargs: None),
    )

    with caplog.at_level(
        logging.DEBUG, logger="plugins.platforms.whatsapp.adapter"
    ):
        result = await adapter.get_chat_info(chat_id)

    assert result == {"name": chat_id, "type": "dm"}
    records = [
        record.message
        for record in caplog.records
        if record.name == "plugins.platforms.whatsapp.adapter"
        and record.message.startswith("Could not get WhatsApp chat info")
    ]
    assert len(records) == 1
    record = records[0]
    assert chat_id not in record
    assert "RuntimeError" in record
    assert chat_id not in _redact_log_text(record + "\n")


def test_whatsapp_authorization_exception_log_omits_raw_traceback_values(caplog):
    user_id = "15551234567@s.whatsapp.net"
    private_name = "Private Contact"
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP_CLOUD
    adapter._authorization_check = MagicMock(
        side_effect=RuntimeError(
            f"authorization failed for {user_id} {private_name}"
        )
    )

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
        assert adapter._is_sender_authorized(user_id, "dm", user_id) is None

    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.base"
        and "Authorization check raised" in record.message
    ]
    assert len(records) == 1
    record = records[0]
    assert user_id not in record.message
    assert private_name not in record.message
    assert "RuntimeError" in record.message
    assert record.exc_info is None


@pytest.mark.asyncio
async def test_whatsapp_cloud_media_exception_log_omits_raw_url_and_traceback(
    caplog,
):
    media_id = "wamid.SECRET-987654321012345"
    private_name = "Private Contact"
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._http_client = MagicMock()
    adapter._http_client.get = AsyncMock(
        side_effect=RuntimeError(
            f"GET https://graph.facebook.com/v20.0/{media_id} failed "
            f"for {private_name}"
        )
    )
    adapter._access_token = "test-token"
    adapter._api_version = "v20.0"

    with caplog.at_level(
        logging.WARNING, logger="gateway.platforms.whatsapp_cloud"
    ):
        result = await adapter._download_media_to_cache(media_id)

    assert result == (None, None)
    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.whatsapp_cloud"
        and "media metadata fetch raised" in record.message
    ]
    assert len(records) == 1
    record = records[0]
    assert media_id not in record.message
    assert private_name not in record.message
    assert "https://graph.facebook.com" not in record.message
    assert "RuntimeError" in record.message
    assert record.exc_info is None


@pytest.mark.asyncio
async def test_whatsapp_cloud_graph_error_log_omits_raw_body(caplog):
    """Graph error bodies can echo WhatsApp text and must stay out of logs."""
    private_marker = "15551234567-private-graph-body-marker"
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(
        return_value=MagicMock(
            status_code=400,
            json=MagicMock(
                return_value={
                    "error": {
                        "code": 131026,
                        "message": f"invalid recipient {private_marker}",
                    }
                }
            ),
        )
    )
    adapter._phone_number_id = "1234567890"
    adapter._api_version = "v20.0"
    adapter._access_token = "test-token"
    adapter._reply_prefix = None

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.whatsapp_cloud"):
        result = await adapter.send("15551234567", "hello")

    assert not result.success
    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.whatsapp_cloud"
        and "send rejected" in record.message
    ]
    assert len(records) == 1
    assert private_marker not in records[0].message
    assert "GraphAPIError" in records[0].message
    assert private_marker not in _redact_log_text(records[0].message + "\n")


@pytest.mark.asyncio
async def test_whatsapp_clarify_error_log_omits_bridge_body(caplog, monkeypatch):
    private_marker = "15551234567-private-poll-question-marker"
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.send_poll = AsyncMock(
        return_value=SimpleNamespace(
            success=False,
            error=private_marker,
        )
    )
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    monkeypatch.setattr(
        BasePlatformAdapter,
        "send_clarify",
        AsyncMock(return_value=SendResult(success=False, error=private_marker)),
    )

    with caplog.at_level(
        logging.WARNING, logger="plugins.platforms.whatsapp.adapter"
    ):
        await adapter.send_clarify(
            "15551234567",
            "Pick one",
            ["A", "B"],
            "clarify-id",
            "session",
        )

    records = [
        record
        for record in caplog.records
        if record.name == "plugins.platforms.whatsapp.adapter"
        and "Native WhatsApp clarify poll failed" in record.message
    ]
    assert len(records) == 1
    assert private_marker not in records[0].message
    assert "error_detail_present=True" in records[0].message


@pytest.mark.asyncio
async def test_whatsapp_document_media_logs_omit_bridge_filename(
    tmp_path, monkeypatch, capsys
):
    """Inbound bridge paths and read errors never enter adapter stdout."""
    private_name = "PRIVATE_INBOUND_FILENAME_MARKER.txt"
    document = tmp_path / private_name
    document.write_text("private document body")

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._should_process_message = MagicMock(return_value=True)
    adapter.build_source = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path",
        lambda _path: True,
    )

    event = await adapter._build_message_event(
        {
            "chatId": "15551234567",
            "senderId": "15551234567",
            "mediaType": "document",
            "hasMedia": True,
            "mime": "text/plain",
            "mediaUrls": [str(document)],
            "body": "",
        }
    )

    assert event is not None
    output = capsys.readouterr().out
    assert private_name not in output
    assert str(document) not in output
    assert private_name not in _redact_log_text(output + "\n")


@pytest.mark.asyncio
async def test_whatsapp_cloud_wamid_dispatch_exception_log_is_type_only(caplog):
    wamid = "wamid.SECRET-987654321012345"
    private_name = "Private Contact"
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._seen_wamids = OrderedDict()
    adapter._duplicate_count = 0
    adapter._accepted_count = 0
    adapter._build_message_event_from_cloud = AsyncMock(
        side_effect=RuntimeError(f"build failed for {wamid} {private_name}")
    )
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "changes": [{
                "field": "messages",
                "value": {
                    "messages": [{"id": wamid}],
                    "contacts": [],
                    "metadata": {},
                },
            }],
        }],
    }

    with caplog.at_level(
        logging.WARNING, logger="gateway.platforms.whatsapp_cloud"
    ):
        await adapter._dispatch_payload(payload)

    records = [
        record
        for record in caplog.records
        if record.name == "gateway.platforms.whatsapp_cloud"
        and "failed to build event for wamid" in record.message
    ]
    assert len(records) == 1
    record = records[0]
    assert wamid not in record.message
    assert private_name not in record.message
    assert "RuntimeError" in record.message
    assert record.exc_info is None


def test_whatsapp_cloud_log_omits_body_and_identity_values(caplog):
    wa_id = "15551234567"
    body = f"please call {wa_id} about private details"
    reply = "previous private message"
    source = SimpleNamespace(
        platform=Platform.WHATSAPP_CLOUD,
        user_name="",
        user_id=wa_id,
        chat_id=wa_id,
    )
    event = SimpleNamespace(
        text=body,
        reply_to_message_id="wamid.synthetic",
        reply_to_text=reply,
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_inbound_message(event, source)

    record = _inbound_record(caplog)
    assert wa_id not in record
    assert body not in record
    assert reply not in record
    assert "wamid.synthetic" not in record
    assert "platform=whatsapp_cloud" in record
    assert "user_present=True" in record
    assert "chat_present=True" in record
    assert f"msg_len={len(body)}" in record
    assert "reply_to_id_present=True" in record
    assert f"reply_to_text_len={len(reply)}" in record


def test_non_phone_identity_is_omitted_while_metadata_remains_diagnostic(caplog):
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        user_name="Alice",
        user_id="user-42",
        chat_id="room-7",
    )
    event = SimpleNamespace(
        text="hello world",
        reply_to_message_id=None,
        reply_to_text=None,
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_inbound_message(event, source)

    record = _inbound_record(caplog)
    assert "Alice" not in record
    assert "user-42" not in record
    assert "room-7" not in record
    assert "user_present=True" in record
    assert "chat_present=True" in record
    assert "msg_len=11" in record
    assert "reply_to_id_present=False" in record
    assert "reply_to_text_len=0" in record
    assert "hello world" not in record


def test_whatsapp_response_ready_log_omits_chat_identity(caplog):
    wa_id = "15551234567"
    source = SimpleNamespace(chat_id=wa_id)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_response_ready("whatsapp_cloud", source, 1.25, 2, 47)

    record = caplog.records[-1].message
    assert wa_id not in record
    assert "platform=whatsapp_cloud" in record
    assert "chat_present=True" in record
    assert "time=1.2s" in record
    assert "api_calls=2" in record
    assert "response=47 chars" in record


def test_whatsapp_startup_restore_log_omits_chat_identity(caplog):
    wa_id = "15551234567"
    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.WHATSAPP_CLOUD, chat_id=wa_id)
    )

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        _log_startup_restore_message(event)

    record = caplog.records[-1].message
    assert wa_id not in record
    assert "platform=whatsapp_cloud" in record
    assert "chat_present=True" in record


def test_leftover_steer_log_omits_body(caplog):
    body = "call 15551234567 about private health details"

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        _log_leftover_steer(body)

    record = caplog.records[-1].message
    assert body not in record
    assert "15551234567" not in record
    assert f"msg_len={len(body)}" in record


def test_whatsapp_session_key_log_view_masks_phone_but_preserves_runtime_key():
    wa_id = "15551234567"
    source = SessionSource(
        platform=Platform.WHATSAPP_CLOUD,
        chat_id=wa_id,
        user_id=wa_id,
        chat_type="dm",
    )
    session_key = build_session_key(source)

    safe_key = session_key_for_log(session_key)

    assert safe_key != session_key
    assert wa_id not in safe_key
    assert "agent:main:whatsapp_cloud:dm:" in safe_key
    assert session_key == f"agent:main:whatsapp_cloud:dm:{wa_id}"


def test_non_whatsapp_numeric_session_key_remains_diagnostic():
    session_key = "agent:main:discord:dm:123456789012345"

    assert session_key_for_log(session_key) == session_key


@pytest.mark.parametrize(
    "platform",
    [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, "whatsapp", "whatsapp_cloud"],
)
def test_log_safe_gateway_identity_masks_phone_only_for_whatsapp(platform):
    wa_id = "15551234567"

    safe_identity = _log_safe_gateway_identity(platform, wa_id)

    assert wa_id not in safe_identity
    assert safe_identity == "15****67"


def test_log_safe_gateway_identity_preserves_non_whatsapp_numeric_id():
    discord_id = "123456789012345"

    assert (
        _log_safe_gateway_identity(Platform.DISCORD, discord_id)
        == discord_id
    )


def test_log_safe_gateway_identity_treats_missing_whatsapp_identity_as_absent():
    assert _log_safe_gateway_identity(Platform.WHATSAPP_CLOUD, None) == "absent"
    assert _log_safe_gateway_identity(Platform.WHATSAPP_CLOUD, "") == "absent"


@pytest.mark.parametrize(
    "record",
    [
        "Redelivered recovered final response to whatsapp_cloud:15551234567 "
        "(obligation o1, attempt 1)",
        "Ignoring /start platform ping for active session "
        "agent:main:whatsapp_cloud:dm:15551234567",
        "STOP for session agent:main:whatsapp_cloud:dm:15551234567 — "
        "agent interrupted, session lock released",
        "Steer failed for session agent:main:whatsapp_cloud:dm:15551234567: "
        "failure",
        "Slash command /stop denied for whatsapp_cloud:15551234567 "
        "(not admin, not in user_allowed_commands)",
        "Auto voice reply skipped: mode=None adapter_auto_tts=False "
        "chat=15551234567 platform=whatsapp_cloud",
        "Watch pattern notification — injecting for whatsapp_cloud "
        "chat=15551234567 thread=15551234567",
    ],
)
def test_historical_whatsapp_gateway_identity_formats_are_redacted(record):
    redacted = _redact_log_text(record + "\n")

    assert "15551234567" not in redacted


def test_historical_whatsapp_identity_names_are_presence_only():
    record = (
        "Redelivered recovered final response to whatsapp_cloud:Private Contact "
        "(obligation o1, attempt 1)\n"
        "Watch pattern notification — injecting for whatsapp_cloud "
        "chat=Private Contact thread=Private Thread\n"
    )

    redacted = _redact_log_text(record)

    assert "Private Contact" not in redacted
    assert "Private Thread" not in redacted
    assert redacted.count("present") == 3


@pytest.mark.parametrize(
    ("record", "private_values"),
    [
        (
            "[WhatsApp] Authorization check raised for user 15551234567; "
            "treating as unknown\n",
            ("15551234567",),
        ),
        (
            "[WhatsApp] Ephemeral delete failed for 15551234567/"
            "wamid.secret: failed\n",
            ("15551234567", "wamid.secret"),
        ),
        (
            "[WhatsApp] Handler returned empty/None response for 15551234567\n",
            ("15551234567",),
        ),
        (
            "[WhatsApp] Sending response (47 chars) to 15551234567\n",
            ("15551234567",),
        ),
        (
            "[WhatsApp] response_delivery_recovered: delivering recovered "
            "original to 15551234567\n",
            ("15551234567",),
        ),
        (
            "Could not get WhatsApp chat info for 15551234567: no aiohttp\n",
            ("15551234567",),
        ),
        (
            "Profile resolution failed for Platform.WHATSAPP/15551234567, "
            "defaulting to active profile\n",
            ("15551234567",),
        ),
        (
            "[whatsapp_cloud] typing/read indicator rejected: wamid "
            "wamid.secret likely older than 30 days\n",
            ("wamid.secret",),
        ),
        (
            "[whatsapp_cloud] media metadata fetch failed (id=media.secret, "
            "status=500)\n",
            ("media.secret",),
        ),
        (
            "[whatsapp_cloud] duplicate wamid wamid.secret, skipping\n",
            ("wamid.secret",),
        ),
        (
            "[whatsapp_cloud] status delivered for wamid.secret\n",
            ("wamid.secret",),
        ),
        (
            "[whatsapp_cloud] cached inbound image media: "
            "/home/cache/media.secret.jpg\n",
            ("media.secret",),
        ),
    ],
)
def test_historical_sibling_whatsapp_logger_formats_are_redacted(
    record, private_values
):
    redacted = _redact_log_text(record)

    for private_value in private_values:
        assert private_value not in redacted


def test_cloud_identifier_logger_metadata_is_presence_and_length_only():
    identifier = "wamid.secret"

    safe = _log_whatsapp_identifier(identifier)

    assert safe == "present(len=12)"
    assert identifier not in safe


def test_historical_non_whatsapp_gateway_identity_remains_diagnostic():
    record = (
        "Auto voice reply skipped: mode=None adapter_auto_tts=False "
        "chat=123456789012345 platform=discord\n"
        "Watch pattern notification — injecting for discord "
        "chat=123456789012345 thread=987654321098765\n"
    )

    assert _redact_log_text(record) == record


def test_whatsapp_display_name_is_presence_only():
    assert _log_safe_gateway_identity(
        Platform.WHATSAPP_CLOUD, "Private Contact"
    ) == "present"


def test_non_whatsapp_display_name_remains_diagnostic():
    assert _log_safe_gateway_identity(
        Platform.DISCORD, "Private Contact"
    ) == "Private Contact"
