"""Tests for the Sendblue platform-plugin adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_sendblue = load_plugin_adapter("sendblue")

SendblueAdapter = _sendblue.SendblueAdapter
check_requirements = _sendblue.check_requirements
validate_config = _sendblue.validate_config
is_connected = _sendblue.is_connected
register = _sendblue.register
_env_enablement = _sendblue._env_enablement
_send_result_from_response = _sendblue._send_result_from_response
_standalone_send = _sendblue._standalone_send
DEFAULT_API_BASE_URL = _sendblue.DEFAULT_API_BASE_URL


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _config(tmp_path: Path, **extra):
    base = {
        "api_key_id": "key-id",
        "api_secret_key": "secret-key",
        "from_number": "+15550000001",
        "webhook_secret": "webhook-secret",
        "sticky_state_path": str(tmp_path / "sticky.json"),
    }
    base.update(extra)
    return PlatformConfig(enabled=True, extra=base)


class _Headers(dict):
    def get(self, key, default=None):  # noqa: D401 - dict compatibility helper
        return super().get(key, super().get(str(key).lower(), default))


class _Request:
    def __init__(self, payload, headers=None):
        self._raw = json.dumps(payload).encode("utf-8")
        self.headers = _Headers(headers or {})

    async def read(self):
        return self._raw


def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform

    assert Platform("sendblue").value == "sendblue"


class TestRequirements:
    def test_check_requirements_needs_creds_and_from_number(self, monkeypatch):
        monkeypatch.setattr(_sendblue, "AIOHTTP_AVAILABLE", True)
        monkeypatch.delenv("SENDBLUE_API_KEY_ID", raising=False)
        monkeypatch.delenv("SENDBLUE_API_SECRET_KEY", raising=False)
        monkeypatch.delenv("SENDBLUE_FROM_NUMBER", raising=False)
        assert check_requirements() is False

        monkeypatch.setenv("SENDBLUE_API_KEY_ID", "key")
        monkeypatch.setenv("SENDBLUE_API_SECRET_KEY", "secret")
        monkeypatch.setenv("SENDBLUE_FROM_NUMBER", "+15550000001")
        assert check_requirements() is True

    def test_validate_config_accepts_extra(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SENDBLUE_API_KEY_ID", raising=False)
        assert validate_config(PlatformConfig(enabled=True, extra={})) is False
        assert validate_config(_config(tmp_path)) is True
        assert is_connected(_config(tmp_path)) is True


class TestAdapterInit:
    def test_reads_from_number_pool_and_default(self, tmp_path):
        adapter = SendblueAdapter(
            _config(
                tmp_path,
                from_number="+15550000001",
                from_numbers=["+15550000002", "+15550000003"],
            )
        )
        assert adapter._from_numbers == [
            "+15550000001",
            "+15550000002",
            "+15550000003",
        ]
        assert adapter._default_from_number == "+15550000001"

    def test_sticky_state_loaded(self, tmp_path):
        state = tmp_path / "sticky.json"
        state.write_text('{" +15551112222 ": "+15550000002"}', encoding="utf-8")
        adapter = SendblueAdapter(_config(tmp_path, sticky_state_path=str(state)))
        assert adapter._sticky_senders[" +15551112222 "] == "+15550000002"

    def test_env_enablement_seeds_config(self, monkeypatch):
        monkeypatch.setenv("SENDBLUE_API_KEY_ID", "key")
        monkeypatch.setenv("SENDBLUE_API_SECRET_KEY", "secret")
        monkeypatch.setenv("SENDBLUE_FROM_NUMBER", "+15550000001")
        monkeypatch.setenv("SENDBLUE_FROM_NUMBERS", "+15550000002,+15550000003")
        monkeypatch.setenv("SENDBLUE_WEBHOOK_SECRET", "whsec")
        monkeypatch.setenv("SENDBLUE_HOME_CHANNEL", "+15551112222")
        seed = _env_enablement()
        assert seed["api_base_url"] == DEFAULT_API_BASE_URL
        assert seed["from_number"] == "+15550000001"
        assert seed["from_numbers"] == ["+15550000002", "+15550000003"]
        assert seed["webhook_secret"] == "whsec"
        assert seed["home_channel"]["chat_id"] == "+15551112222"


class TestPayloads:
    def test_dm_payload_uses_send_message(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        payload, endpoint = adapter._build_send_payload(
            "+15551112222", "hello", metadata={}
        )
        assert endpoint == "/api/send-message"
        assert payload == {
            "content": "hello",
            "from_number": "+15550000001",
            "number": "+15551112222",
        }

    def test_group_payload_uses_group_endpoint(self, tmp_path):
        adapter = SendblueAdapter(
            _config(tmp_path, status_callback="https://example.com/status")
        )
        payload, endpoint = adapter._build_send_payload(
            "group:g123", "hello", metadata={}
        )
        assert endpoint == "/api/send-group-message"
        assert payload["group_id"] == "g123"
        assert payload["status_callback"] == "https://example.com/status"

    def test_sticky_sender_persists(self, tmp_path):
        adapter = SendblueAdapter(
            _config(tmp_path, from_numbers=["+15550000001", "+15550000002"])
        )
        adapter._remember_sender("+15551112222", "+15550000002")
        assert adapter._select_from_number("+15551112222") == "+15550000002"
        data = json.loads((tmp_path / "sticky.json").read_text(encoding="utf-8"))
        assert data["+15551112222"] == "+15550000002"


class TestWebhookSecurity:
    def test_accepts_configured_secret_header(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        assert (
            adapter._validate_webhook_secret({"sb-signing-secret": "webhook-secret"})
            is True
        )
        assert adapter._validate_webhook_secret({"secret": "webhook-secret"}) is True
        assert (
            adapter._validate_webhook_secret({"x-sendblue-secret": "webhook-secret"})
            is True
        )
        assert adapter._validate_webhook_secret({"sb-signing-secret": "wrong"}) is False

    def test_accepts_bearer_secret(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        assert (
            adapter._validate_webhook_secret({"authorization": "Bearer webhook-secret"})
            is True
        )


class TestWebhookHandling:
    def test_inbound_dispatches_message_and_records_sticky_sender(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        adapter.handle_message = AsyncMock()
        payload = {
            "content": "Hello",
            "is_outbound": False,
            "status": "RECEIVED",
            "message_handle": "mh-1",
            "from_number": "+15551112222",
            "to_number": "+15550000001",
            "sendblue_number": "+15550000001",
            "service": "iMessage",
        }
        response = _run(
            adapter._handle_webhook(
                _Request(payload, headers={"sb-signing-secret": "webhook-secret"})
            )
        )
        _run(asyncio.sleep(0))
        assert response.status == 200
        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "Hello"
        assert event.source.chat_id == "+15551112222"
        assert event.source.user_id == "+15551112222"
        assert adapter._sticky_senders["+15551112222"] == "+15550000001"

    def test_duplicate_message_handle_is_acked_without_dispatch(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        adapter.handle_message = AsyncMock()
        payload = {
            "content": "Hello",
            "is_outbound": False,
            "status": "RECEIVED",
            "message_handle": "mh-1",
            "from_number": "+15551112222",
            "to_number": "+15550000001",
        }
        request = _Request(payload, headers={"sb-signing-secret": "webhook-secret"})
        assert _run(adapter._handle_webhook(request)).status == 200
        assert _run(adapter._handle_webhook(request)).status == 200
        _run(asyncio.sleep(0))
        adapter.handle_message.assert_awaited_once()

    def test_outbound_status_callback_is_acked_without_dispatch(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        adapter.handle_message = AsyncMock()
        payload = {
            "content": "Hello",
            "is_outbound": True,
            "status": "SENT",
            "message_handle": "mh-out",
            "from_number": "+15550000001",
            "to_number": "+15551112222",
        }
        response = _run(
            adapter._handle_webhook(
                _Request(payload, headers={"sb-signing-secret": "webhook-secret"})
            )
        )
        assert response.status == 200
        adapter.handle_message.assert_not_called()

    def test_group_inbound_uses_group_chat_id(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        adapter.handle_message = AsyncMock()
        payload = {
            "content": "Group hello",
            "is_outbound": False,
            "status": "RECEIVED",
            "message_handle": "mh-group",
            "from_number": "+15551112222",
            "to_number": "+15550000001",
            "sendblue_number": "+15550000001",
            "group_id": "group-123",
            "group_display_name": "Friends",
        }
        response = _run(
            adapter._handle_webhook(
                _Request(payload, headers={"sb-signing-secret": "webhook-secret"})
            )
        )
        _run(asyncio.sleep(0))
        assert response.status == 200
        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_id == "group:group-123"
        assert event.source.chat_type == "group"
        assert event.source.user_id == "+15551112222"

    def test_attachment_only_inbound_uses_placeholder_text(self, tmp_path):
        adapter = SendblueAdapter(_config(tmp_path))
        adapter.handle_message = AsyncMock()
        payload = {
            "content": "",
            "is_outbound": False,
            "status": "RECEIVED",
            "message_handle": "mh-media",
            "from_number": "+15551112222",
            "to_number": "+15550000001",
            "media_url": "https://cdn.example.test/photo.jpg",
            "message_type": "image/jpeg",
        }
        response = _run(
            adapter._handle_webhook(
                _Request(payload, headers={"sb-signing-secret": "webhook-secret"})
            )
        )
        _run(asyncio.sleep(0))
        assert response.status == 200
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "(attachment)"
        assert event.message_type == _sendblue.MessageType.PHOTO
        assert event.media_urls == ["https://cdn.example.test/photo.jpg"]
        assert event.media_types == ["image/jpeg"]


class TestSendResults:
    def test_success_extracts_message_handle(self):
        result = _send_result_from_response(
            200,
            {},
            {"message_handle": "msg-123", "status": "QUEUED"},
        )
        assert result.success is True
        assert result.message_id == "msg-123"

    def test_429_is_retryable_rate_limit(self):
        result = _send_result_from_response(
            429,
            {"Retry-After": "2.5"},
            {"error_message": "Too many requests"},
        )
        assert result.success is False
        assert result.retryable is True
        assert result.retry_after == 2.5
        assert result.error_kind == "rate_limited"


class TestStandaloneSend:
    def test_uses_standalone_sender_contract(self, tmp_path, monkeypatch):
        async def fake_post(session, **kwargs):
            assert kwargs["endpoint"] == "/api/send-message"
            assert kwargs["payload"]["from_number"] == "+15550000001"
            assert kwargs["payload"]["number"] == "+15551112222"
            return 200, {}, {"message_handle": "msg-standalone"}

        monkeypatch.setattr(_sendblue, "_post_sendblue", fake_post)
        pconfig = _config(tmp_path)
        result = _run(_standalone_send(pconfig, "+15551112222", "hello"))
        assert result == {
            "success": True,
            "platform": "sendblue",
            "chat_id": "+15551112222",
            "message_id": "msg-standalone",
        }


def test_register_calls_register_platform():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    ctx.register_cli_command.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "sendblue"
    assert kwargs["cron_deliver_env_var"] == "SENDBLUE_HOME_CHANNEL"
    assert kwargs["allowed_users_env"] == "SENDBLUE_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "SENDBLUE_ALLOW_ALL_USERS"
    assert kwargs["pii_safe"] is True
    assert callable(kwargs["standalone_sender_fn"])
    assert callable(kwargs["env_enablement_fn"])
    cli_kwargs = ctx.register_cli_command.call_args.kwargs
    assert cli_kwargs["name"] == "sendblue"
    assert callable(cli_kwargs["setup_fn"])
    assert callable(cli_kwargs["handler_fn"])
