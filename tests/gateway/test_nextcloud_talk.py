"""Tests for the Nextcloud Talk gateway adapter."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, SendResult
from gateway.platforms.nextcloud_talk import (
    NextcloudTalkAdapter,
    _decode_talk_message,
    _sanitize_chat_id,
    _sign_payload,
    _validate_backend_url,
    check_nextcloud_talk_requirements,
)


def _make_config(**extra_overrides):
    extra = {
        "base_url": "https://cloud.example.com",
        "host": "127.0.0.1",
        "port": 0,
        "path": "/nextcloud-talk",
        "chat_type": "group",
    }
    extra.update(extra_overrides)
    return PlatformConfig(enabled=True, token="super-secret-value", extra=extra)


def _make_adapter(**extra_overrides):
    return NextcloudTalkAdapter(_make_config(**extra_overrides))


def _create_app(adapter: NextcloudTalkAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/nextcloud-talk", adapter._handle_webhook)
    return app


def _talk_headers(secret_value: str, body: bytes, *, backend: str = "https://cloud.example.com") -> dict[str, str]:
    random_header = "a" * 64
    signature = _sign_payload(secret_value, random_header, body)
    return {
        "X-Nextcloud-Talk-Signature": signature,
        "X-Nextcloud-Talk-Random": random_header,
        "X-Nextcloud-Talk-Backend": backend,
        "Content-Type": "application/json",
    }


class TestNextcloudTalkConfig:
    def test_enum_exists(self):
        assert Platform.NEXTCLOUD_TALK.value == "nextcloud_talk"

    def test_apply_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NEXTCLOUD_TALK_SECRET", "shared-secret")
        monkeypatch.setenv("NEXTCLOUD_TALK_BASE_URL", "https://cloud.example.com/")
        monkeypatch.setenv("NEXTCLOUD_TALK_WEBHOOK_PORT", "9765")
        monkeypatch.setenv("NEXTCLOUD_TALK_WEBHOOK_PATH", "/hooks/talk")
        monkeypatch.setenv("NEXTCLOUD_TALK_HOME_CHANNEL", "room-token")

        from gateway.config import _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.NEXTCLOUD_TALK in config.platforms
        pconfig = config.platforms[Platform.NEXTCLOUD_TALK]
        assert pconfig.enabled is True
        assert pconfig.token == "shared-secret"
        assert pconfig.extra["base_url"] == "https://cloud.example.com"
        assert pconfig.extra["port"] == 9765
        assert pconfig.extra["path"] == "/hooks/talk"
        assert config.get_home_channel(Platform.NEXTCLOUD_TALK).chat_id == "room-token"

    def test_connected_platforms_includes_nextcloud_talk(self):
        config = GatewayConfig(
            platforms={
                Platform.NEXTCLOUD_TALK: PlatformConfig(enabled=True, token="secret")
            }
        )
        assert Platform.NEXTCLOUD_TALK in config.get_connected_platforms()


class TestNextcloudTalkHelpers:
    def test_decode_talk_message_extracts_inner_message(self):
        raw = json.dumps({"message": "hello there", "parameters": {}})
        assert _decode_talk_message(raw) == "hello there"

    def test_decode_talk_message_none(self):
        assert _decode_talk_message(None) == ""

    def test_decode_talk_message_plain_string(self):
        assert _decode_talk_message("just text") == "just text"

    def test_decode_talk_message_non_string(self):
        assert _decode_talk_message(42) == "42"

    def test_decode_talk_message_dict_without_message_key(self):
        raw = json.dumps({"parameters": {}, "other": "data"})
        assert _decode_talk_message(raw) == raw

    def test_signature_matches_reference_hmac(self):
        body = b'{"message":"hello"}'
        random_header = "b" * 64
        expected = hmac.new(b"secret", digestmod=hashlib.sha256)
        expected.update(random_header.encode("utf-8"))
        expected.update(body)
        assert _sign_payload("secret", random_header, body) == expected.hexdigest()


class TestChatIdSanitization:
    def test_valid_token_lowercase(self):
        assert _sanitize_chat_id("abc123def") == "abc123def"

    def test_valid_token_with_hyphen(self):
        assert _sanitize_chat_id("room-token") == "room-token"

    def test_valid_token_with_underscore(self):
        assert _sanitize_chat_id("room_token") == "room_token"

    def test_valid_token_mixed_case(self):
        assert _sanitize_chat_id("AbCdEf") == "AbCdEf"

    def test_rejects_path_traversal(self):
        assert _sanitize_chat_id("../../../etc/passwd") is None

    def test_rejects_slashes(self):
        assert _sanitize_chat_id("room/token") is None

    def test_rejects_special_characters(self):
        assert _sanitize_chat_id("room;DROP TABLE") is None

    def test_rejects_url_query_chars(self):
        assert _sanitize_chat_id("room?param=1") is None

    def test_rejects_empty(self):
        assert _sanitize_chat_id("") is None

    def test_rejects_too_long(self):
        assert _sanitize_chat_id("a" * 65) is None


class TestBackendUrlValidation:
    def test_accepts_https(self):
        assert _validate_backend_url("https://cloud.example.com") is True

    def test_accepts_http(self):
        assert _validate_backend_url("http://nextcloud.local") is True

    def test_accepts_path_under_host(self):
        assert _validate_backend_url("https://cloud.example.com/nextcloud") is True

    def test_rejects_ftp(self):
        assert _validate_backend_url("ftp://evil.com") is False

    def test_rejects_javascript(self):
        assert _validate_backend_url("javascript:alert(1)") is False

    def test_rejects_empty(self):
        assert _validate_backend_url("") is False

    def test_rejects_missing_host(self):
        assert _validate_backend_url("https:///missing-host") is False


class TestNextcloudTalkWebhook:
    @pytest.mark.asyncio
    async def test_accepts_valid_signed_message(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        app = _create_app(adapter)

        payload = {
            "type": "Create",
            "actor": {"type": "Person", "id": "users/alice", "name": "Alice"},
            "object": {
                "type": "Note",
                "id": "1567",
                "name": "message",
                "content": json.dumps({"message": "hello hermes", "parameters": {}}),
                "mediaType": "text/markdown",
                "inReplyTo": {
                    "actor": {"type": "Person", "id": "users/bob", "name": "Bob"},
                    "object": {
                        "type": "Note",
                        "id": "1444",
                        "content": json.dumps({"message": "parent text", "parameters": {}}),
                    },
                },
            },
            "target": {"type": "Collection", "id": "room-token", "name": "General"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = _talk_headers("super-secret-value", body)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=body, headers=headers)
            assert resp.status == 202

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert isinstance(event, MessageEvent)
        assert event.text == "hello hermes"
        assert event.source.platform == Platform.NEXTCLOUD_TALK
        assert event.source.chat_id == "room-token"
        assert event.source.user_id == "users/alice"
        assert event.reply_to_message_id == "1444"
        assert event.reply_to_text == "Bob: parent text"
        assert adapter._chat_backends["room-token"] == "https://cloud.example.com"

    @pytest.mark.asyncio
    async def test_rejects_invalid_signature(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        body = json.dumps({"type": "Create", "target": {"id": "room-token"}}).encode("utf-8")
        headers = {
            "X-Nextcloud-Talk-Signature": "deadbeef",
            "X-Nextcloud-Talk-Random": "a" * 64,
            "X-Nextcloud-Talk-Backend": "https://cloud.example.com",
            "Content-Type": "application/json",
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=body, headers=headers)
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_ignores_bot_messages(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        app = _create_app(adapter)
        payload = {
            "type": "Create",
            "actor": {"type": "Application", "id": "bots/bot-123", "name": "Hermes"},
            "object": {"id": "1", "content": json.dumps({"message": "self", "parameters": {}})},
            "target": {"id": "room-token", "name": "General"},
        }
        body = json.dumps(payload).encode("utf-8")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=body, headers=_talk_headers("super-secret-value", body))
            assert resp.status == 202

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_wrong_content_type(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        body = b"not json"
        headers = {
            "Content-Type": "text/plain",
            "X-Nextcloud-Talk-Signature": "abc",
            "X-Nextcloud-Talk-Random": "a" * 64,
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=body, headers=headers)
            assert resp.status == 415

    @pytest.mark.asyncio
    async def test_rejects_oversized_body(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        # Send a body larger than the 1MB limit via Content-Length header
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(2 * 1024 * 1024),
            "X-Nextcloud-Talk-Signature": "abc",
            "X-Nextcloud-Talk-Random": "a" * 64,
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=b"{}", headers=headers)
            assert resp.status == 413

    @pytest.mark.asyncio
    async def test_rejects_invalid_chat_id(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        app = _create_app(adapter)
        payload = {
            "type": "Create",
            "actor": {"type": "Person", "id": "users/alice", "name": "Alice"},
            "object": {"id": "1", "content": json.dumps({"message": "test", "parameters": {}})},
            "target": {"id": "../../../admin", "name": "Evil"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = _talk_headers("super-secret-value", body)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=body, headers=headers)
            assert resp.status == 400

        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_non_https_backend(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        app = _create_app(adapter)
        payload = {
            "type": "Create",
            "actor": {"type": "Person", "id": "users/alice", "name": "Alice"},
            "object": {"id": "1", "content": json.dumps({"message": "test", "parameters": {}})},
            "target": {"id": "abcd1234", "name": "Room"},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = _talk_headers("super-secret-value", body, backend="ftp://evil.com")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/nextcloud-talk", data=body, headers=headers)
            assert resp.status == 202

        # Message should be processed but the invalid backend should NOT be cached
        assert "abcd1234" not in adapter._chat_backends


class TestNextcloudTalkSend:
    @pytest.mark.asyncio
    async def test_send_uses_backend_mapping_and_signs_payload(self):
        adapter = _make_adapter(base_url="")
        adapter._chat_backends["roomtoken"] = "https://cloud.example.com"

        response = MagicMock()
        response.status = 201
        response.text = AsyncMock(return_value='{"ocs":{"meta":{"status":"ok"}}}')
        context = AsyncMock()
        context.__aenter__.return_value = response
        context.__aexit__.return_value = False

        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.close = AsyncMock()
        session.post.return_value = context

        with patch("gateway.platforms.nextcloud_talk.ClientSession", return_value=session):
            result = await adapter.send("roomtoken", "hello back", reply_to="1567")

        assert isinstance(result, SendResult)
        assert result.success is True
        session.post.assert_called_once()
        args, kwargs = session.post.call_args
        assert args[0] == "https://cloud.example.com/ocs/v2.php/apps/spreed/api/v1/bot/roomtoken/message"
        assert kwargs["headers"]["OCS-APIRequest"] == "true"
        assert kwargs["headers"]["X-Nextcloud-Talk-Bot-Random"]
        body = kwargs["data"]
        payload = json.loads(body.decode("utf-8"))
        assert payload["message"] == "hello back"
        assert payload["replyTo"] == 1567

    @pytest.mark.asyncio
    async def test_send_requires_backend(self):
        adapter = _make_adapter(base_url="")
        result = await adapter.send("roomtoken", "hello")
        assert result.success is False
        assert "backend URL is unknown" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_rejects_invalid_chat_id(self):
        adapter = _make_adapter()
        result = await adapter.send("../../../etc/passwd", "hello")
        assert result.success is False
        assert "Invalid Talk conversation token" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_rejects_invalid_backend_scheme(self):
        adapter = _make_adapter(base_url="ftp://evil.com")
        result = await adapter.send("roomtoken", "hello")
        assert result.success is False
        assert "Invalid backend URL scheme" in (result.error or "")

    @pytest.mark.asyncio
    async def test_send_empty_content_is_noop(self):
        adapter = _make_adapter()
        result = await adapter.send("roomtoken", "")
        assert result.success is True


class TestNextcloudTalkRequirements:
    def test_requirements_flag_matches_import(self):
        assert check_nextcloud_talk_requirements() is True
