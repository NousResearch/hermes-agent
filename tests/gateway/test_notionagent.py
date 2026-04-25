import hashlib
import hmac
import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageType, SendResult
from gateway.platforms.notionagent import (
    NotionAgentAdapter,
    SIGNATURE_HEADER,
    _signed_json_payload,
    _validate_notionagent_signature,
)
from gateway.session import SessionSource


def _config(**extra):
    merged = {
        "secret": "test-secret",
        "callback_url": "https://notionagent.example/callback",
        "host": "127.0.0.1",
        "port": 0,
    }
    merged.update(extra)
    return PlatformConfig(enabled=True, extra=merged)


def _signature(body: bytes, secret: str = "test-secret") -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def _request(body: bytes, headers: dict):
    request = MagicMock()
    request.headers = headers

    async def read():
        return body

    request.read = read
    return request


def test_platform_enum_value():
    assert Platform.NOTIONAGENT.value == "notionagent"


def test_env_override_loading_and_connected_flag(monkeypatch):
    monkeypatch.setenv("NOTIONAGENT_SECRET", "env-secret")
    monkeypatch.setenv("NOTIONAGENT_CALLBACK_URL", "https://app.example/cb")
    monkeypatch.setenv("NOTIONAGENT_PORT", "9876")

    config = GatewayConfig()
    _apply_env_overrides(config)

    pconfig = config.platforms[Platform.NOTIONAGENT]
    assert pconfig.enabled is True
    assert pconfig.extra["secret"] == "env-secret"
    assert pconfig.extra["callback_url"] == "https://app.example/cb"
    assert pconfig.extra["port"] == 9876
    assert Platform.NOTIONAGENT in config.get_connected_platforms()


def test_connected_flag_requires_secret_and_callback_url():
    config = GatewayConfig(
        platforms={
            Platform.NOTIONAGENT: PlatformConfig(
                enabled=True,
                extra={"secret": "s"},
            )
        }
    )
    assert Platform.NOTIONAGENT not in config.get_connected_platforms()


def test_hmac_validation_accept_and_reject():
    body = b'{"session_id":"memo-1","text":"hello"}'
    assert _validate_notionagent_signature(_signature(body), body, "test-secret")
    assert not _validate_notionagent_signature("sha256=deadbeef", body, "test-secret")
    assert not _validate_notionagent_signature("", body, "test-secret")


def test_adapter_init_config_parsing():
    adapter = NotionAgentAdapter(
        _config(path="notionagent/in", port="8765", host="127.0.0.1")
    )
    assert adapter._path == "/notionagent/in"
    assert adapter._port == 8765
    assert adapter._host == "127.0.0.1"
    assert adapter._callback_url == "https://notionagent.example/callback"


@pytest.mark.asyncio
async def test_inbound_message_dispatch():
    adapter = NotionAgentAdapter(_config())
    captured = []

    async def handle(event):
        captured.append(event)

    adapter.handle_message = handle
    payload = {"session_id": "memo-123", "text": "Summarize this"}
    body = json.dumps(payload).encode("utf-8")

    resp = await adapter._handle_inbound(
        _request(body, {SIGNATURE_HEADER: _signature(body)})
    )
    assert resp.status == 202
    await asyncio.sleep(0)

    assert len(captured) == 1
    event = captured[0]
    assert event.text == "Summarize this"
    assert event.message_type == MessageType.TEXT
    assert event.source.platform == Platform.NOTIONAGENT
    assert event.source.chat_id == "memo-123"
    assert event.source.chat_type == "session"


@pytest.mark.asyncio
async def test_inbound_rejects_bad_signature():
    adapter = NotionAgentAdapter(_config())
    adapter.handle_message = AsyncMock()
    body = b'{"session_id":"memo-123","text":"hello"}'

    resp = await adapter._handle_inbound(
        _request(body, {SIGNATURE_HEADER: "sha256=bad"})
    )
    assert resp.status == 401

    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_outbound_post_shape(monkeypatch):
    posted = {}

    class Response:
        status_code = 200
        text = "ok"

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, content, headers):
            posted["url"] = url
            posted["content"] = content
            posted["headers"] = headers
            return Response()

    import gateway.platforms.notionagent as notionagent

    monkeypatch.setattr(notionagent.httpx, "AsyncClient", Client)

    adapter = NotionAgentAdapter(_config(secret="shared"))
    result = await adapter.send("memo-1", "Done")

    assert result.success is True
    assert posted["url"] == "https://notionagent.example/callback"
    payload = json.loads(posted["content"].decode("utf-8"))
    assert payload["session_id"] == "memo-1"
    assert payload["text"] == "Done"
    assert payload["message_id"]
    assert posted["headers"][SIGNATURE_HEADER] == _signature(posted["content"], "shared")


@pytest.mark.asyncio
async def test_standalone_send_notionagent_matches_adapter(monkeypatch):
    called = {}

    async def fake_post(**kwargs):
        called.update(kwargs)
        return SendResult(success=True, message_id="msg-1")

    import tools.send_message_tool as send_message_tool

    monkeypatch.setattr(
        "gateway.platforms.notionagent._post_notionagent_callback",
        fake_post,
    )
    result = await send_message_tool._send_notionagent(
        _config(secret="standalone"),
        "memo-2",
        "Hello",
    )

    assert result == {
        "success": True,
        "platform": "notionagent",
        "chat_id": "memo-2",
        "message_id": "msg-1",
    }
    assert called["secret"] == "standalone"
    assert called["session_id"] == "memo-2"
    assert called["text"] == "Hello"


def test_authorization_maps_present():
    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner._is_user_authorized)
    assert "NOTIONAGENT_ALLOWED_USERS" in source
    assert "NOTIONAGENT_ALLOW_ALL_USERS" in source


def test_authorization_defaults_to_all_signed_requests(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.delenv("NOTIONAGENT_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("NOTIONAGENT_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    source = SessionSource(
        platform=Platform.NOTIONAGENT,
        chat_id="memo-3",
        chat_type="session",
        user_id="memo-3",
    )
    assert runner._is_user_authorized(source) is True


def test_send_message_tool_routing_mentions_notionagent():
    import tools.send_message_tool as send_message_tool

    source = inspect.getsource(send_message_tool._handle_send)
    assert '"notionagent": Platform.NOTIONAGENT' in source
    source = inspect.getsource(send_message_tool._send_to_platform)
    assert "Platform.NOTIONAGENT" in source


def test_session_source_roundtrip():
    source = SessionSource(
        platform=Platform.NOTIONAGENT,
        chat_id="memo-4",
        chat_name="memo-4",
        chat_type="session",
        user_id="memo-4",
    )
    restored = SessionSource.from_dict(source.to_dict())
    assert restored.platform == Platform.NOTIONAGENT
    assert restored.chat_id == "memo-4"
    assert restored.chat_type == "session"


def test_signed_json_payload_uses_raw_body_signature():
    body, signature = _signed_json_payload(
        {"session_id": "memo-5", "text": "hi", "message_id": "msg"},
        "secret",
    )
    assert signature == _signature(body, "secret")
