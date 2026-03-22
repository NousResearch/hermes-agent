"""Tests for Blooio gateway integration."""

import hashlib
import hmac
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import MessageType
from gateway.platforms.blooio import BlooioAdapter, check_blooio_requirements


class _FakeRequest:
    def __init__(self, body: bytes, headers: dict[str, str]):
        self._body = body
        self.headers = headers

    async def read(self) -> bytes:
        return self._body


def _signature(secret: str, body: bytes, timestamp: int | None = None) -> str:
    ts = int(timestamp or time.time())
    signed = f"{ts}.{body.decode('utf-8')}"
    digest = hmac.new(secret.encode("utf-8"), signed.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"t={ts},v1={digest}"


def _make_adapter(tmp_path: Path, monkeypatch, *, extra: dict | None = None) -> BlooioAdapter:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = PlatformConfig(
        enabled=True,
        api_key="blooio-key",
        extra={
            "public_base_url": "https://example.test",
            "bind_host": "127.0.0.1",
            "webhook_port": 8899,
            **(extra or {}),
        },
    )
    adapter = BlooioAdapter(config)
    adapter._webhook_secret = "whsec_test"
    adapter._media_secret = "media_test"
    return adapter


class TestBlooioConfig:
    def test_platform_enum_exists(self):
        assert Platform.BLOOIO.value == "blooio"

    def test_env_overrides_create_platform_config(self, monkeypatch):
        env = {
            "BLOOIO_API_KEY": "sk-blooio",
            "BLOOIO_PUBLIC_BASE_URL": "https://example.test",
            "BLOOIO_BIND_HOST": "127.0.0.1",
            "BLOOIO_WEBHOOK_PORT": "9911",
            "BLOOIO_FROM_NUMBER": "+15551234567",
            "BLOOIO_INSTANCE_ID": "instance-a",
            "BLOOIO_HOME_CHANNEL": "+15557654321",
        }
        with patch.dict("os.environ", env, clear=False):
            config = load_gateway_config()
        assert Platform.BLOOIO in config.platforms
        pconfig = config.platforms[Platform.BLOOIO]
        assert pconfig.enabled is True
        assert pconfig.api_key == "sk-blooio"
        assert pconfig.extra["public_base_url"] == "https://example.test"
        assert pconfig.extra["bind_host"] == "127.0.0.1"
        assert pconfig.extra["webhook_port"] == 9911
        assert pconfig.extra["from_number"] == "+15551234567"
        assert pconfig.extra["instance_id"] == "instance-a"
        assert pconfig.home_channel.chat_id == "+15557654321"

    def test_connected_platforms_include_blooio_when_api_key_present(self):
        config = GatewayConfig(platforms={Platform.BLOOIO: PlatformConfig(enabled=True, api_key="sk-test")})
        assert Platform.BLOOIO in config.get_connected_platforms()


class TestBlooioRequirements:
    def test_requirements_need_api_key_and_public_url(self, monkeypatch):
        monkeypatch.delenv("BLOOIO_API_KEY", raising=False)
        monkeypatch.delenv("BLOOIO_PUBLIC_BASE_URL", raising=False)
        assert check_blooio_requirements() is False


class TestBlooioPathsAndIdentity:
    def test_instance_id_derives_from_hermes_home(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "custom-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(tmp_path / "ignored-home"))
        adapter = BlooioAdapter(
            PlatformConfig(enabled=True, api_key="key", extra={"public_base_url": "https://example.test"})
        )
        expected = hashlib.sha256(str(hermes_home.resolve()).encode("utf-8")).hexdigest()[:12]
        assert adapter._instance_id == expected
        assert adapter._state_root == hermes_home / "blooio"
        assert adapter._media_root == hermes_home / "blooio" / "media"

    def test_explicit_instance_id_wins(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch, extra={"instance_id": "my-instance"})
        assert adapter._instance_id == "my-instance"
        assert adapter._webhook_path.endswith("/my-instance")


class TestBlooioWebhooks:
    def test_verify_signature_accepts_valid_payload(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        body = b'{"event":"message.received","sender":"+1555"}'
        adapter._verify_signature(body, _signature(adapter._webhook_secret, body))

    def test_verify_signature_rejects_stale_timestamp(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        body = b'{"event":"message.received","sender":"+1555"}'
        stale = int(time.time()) - 301
        with pytest.raises(ValueError):
            adapter._verify_signature(body, _signature(adapter._webhook_secret, body, timestamp=stale))

    @pytest.mark.asyncio
    async def test_control_events_do_not_dispatch_to_agent_loop(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        adapter.handle_message = AsyncMock()
        body = json.dumps(
            {
                "event": "message.delivered",
                "message_id": "msg_123",
                "external_id": "+15551234567",
            }
        ).encode("utf-8")
        request = _FakeRequest(body, {"X-Blooio-Signature": _signature(adapter._webhook_secret, body)})

        response = await adapter._handle_webhook(request)

        assert response.status == 200
        adapter.handle_message.assert_not_called()
        assert adapter._status_by_message_id["msg_123"]["event"] == "message.delivered"

    @pytest.mark.asyncio
    async def test_message_received_dispatches_dm_event(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        adapter.handle_message = AsyncMock()
        adapter._cache_attachment = AsyncMock(
            return_value={"path": "/tmp/image.png", "media_type": "image/png", "kind": "image"}
        )
        payload = {
            "event": "message.received",
            "message_id": "msg_1",
            "sender": "+15551234567",
            "text": "hello",
            "attachments": [{"url": "https://cdn.example.test/image.png", "name": "image.png"}],
            "is_group": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = _FakeRequest(body, {"X-Blooio-Signature": _signature(adapter._webhook_secret, body)})

        response = await adapter._handle_webhook(request)

        assert response.status == 200
        adapter.handle_message.assert_called_once()
        event = adapter.handle_message.call_args.args[0]
        assert event.source.platform == Platform.BLOOIO
        assert event.source.chat_id == "+15551234567"
        assert event.source.user_id == "+15551234567"
        assert event.message_type == MessageType.PHOTO
        assert event.media_urls == ["/tmp/image.png"]

    @pytest.mark.asyncio
    async def test_build_message_event_uses_group_identity(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        adapter._cache_attachment = AsyncMock(return_value=None)
        payload = {
            "event": "message.received",
            "message_id": "msg_group",
            "sender": "+15550001111",
            "is_group": True,
            "group_id": "grp_123",
            "group_name": "Team Chat",
            "text": "hi group",
            "attachments": [],
        }
        event = await adapter._build_message_event(payload)
        assert event is not None
        assert event.source.chat_id == "grp_123"
        assert event.source.chat_name == "Team Chat"
        assert event.source.user_id == "+15550001111"
        assert event.source.chat_type == "group"


class TestBlooioMediaStaging:
    def test_stage_file_uses_resolved_hermes_home(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.4")

        staged = adapter._stage_file(str(source))

        assert staged["url"].startswith("https://example.test/blooio/media/")
        staged_dir = adapter._media_root / adapter._instance_id
        assert staged_dir.exists()
        assert any(path.is_file() for path in staged_dir.iterdir())

    def test_parse_media_token_round_trip(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        token = adapter._build_media_token("deadbeef", int(time.time()) + 60)
        stored_key, _expires = adapter._parse_media_token(token)
        assert stored_key == "deadbeef"


class TestBlooioEndpoints:
    @pytest.mark.asyncio
    async def test_typing_and_read_calls_expected_routes(self, tmp_path, monkeypatch):
        adapter = _make_adapter(tmp_path, monkeypatch)
        adapter._api_request = AsyncMock(return_value={})

        await adapter.send_typing("+15551234567")
        await adapter.stop_typing("+15551234567")
        await adapter.mark_read("+15551234567")

        calls = [call.args[:2] for call in adapter._api_request.await_args_list]
        assert ("POST", "/chats/%2B15551234567/typing") in calls
        assert ("DELETE", "/chats/%2B15551234567/typing") in calls
        assert ("POST", "/chats/%2B15551234567/read") in calls
