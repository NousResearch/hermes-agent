from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_zalo = load_plugin_adapter("zalo")
EVENT_IMAGE = _zalo.EVENT_IMAGE
EVENT_TEXT = _zalo.EVENT_TEXT
EVENT_UNSUPPORTED = _zalo.EVENT_UNSUPPORTED
EVENT_VOICE = _zalo.EVENT_VOICE
ZALO_API_BASE = _zalo.ZALO_API_BASE
ZaloAdapter = _zalo.ZaloAdapter
ZaloApiError = _zalo.ZaloApiError
_apply_yaml_config = _zalo._apply_yaml_config
_env_enablement = _zalo._env_enablement
register = _zalo.register


class TestRegister:
    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    def test_register_exposes_standard_plugin_hooks(self):
        ctx = self._FakeCtx()

        register(ctx)

        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "zalo"
        assert ctx.kwargs["required_env"] == ["ZALO_BOT_TOKEN"]
        assert ctx.kwargs["allowed_users_env"] == "ZALO_ALLOWED_USERS"
        assert ctx.kwargs["allow_all_env"] == "ZALO_ALLOW_ALL_USERS"
        assert ctx.kwargs["standalone_sender_fn"] is not None
        assert "Zalo Bot Platform" in ctx.kwargs["platform_hint"]


class TestConfig:
    def test_env_enablement_seeds_credentials_and_home_channel(self, monkeypatch):
        monkeypatch.setenv("ZALO_BOT_TOKEN", "token")
        monkeypatch.setenv("ZALO_HOME_CHANNEL", "chat-1")
        monkeypatch.setenv("ZALO_HOME_CHANNEL_NAME", "Operations")

        assert _env_enablement() == {
            "bot_token": "token",
            "home_channel": {"chat_id": "chat-1", "name": "Operations"},
        }

    def test_yaml_bridge_uses_central_auth_and_keeps_secrets_out(self, monkeypatch):
        monkeypatch.delenv("ZALO_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("ZALO_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("ZALO_HOME_CHANNEL", raising=False)

        seeded = _apply_yaml_config(
            {},
            {
                "allow_from": ["u2", "u1"],
                "group_allow_from": ["u3"],
                "allow_all_users": False,
                "connection_mode": "webhook",
                "parse_mode": "html",
                "poll_timeout_seconds": 20,
                "webhook_url": "https://bot.example.test/zalo",
                "webhook_path": "/zalo",
                "webhook_host": "0.0.0.0",
                "webhook_port": 18080,
                "home_channel": {"chat_id": "chat-1", "name": "Home"},
                "bot_token": "must-not-be-copied",
                "webhook_secret": "must-not-be-copied",
            },
        )

        assert seeded == {
            "connection_mode": "webhook",
            "parse_mode": "html",
            "poll_timeout_seconds": 20,
            "webhook_url": "https://bot.example.test/zalo",
            "webhook_path": "/zalo",
            "webhook_host": "0.0.0.0",
            "webhook_port": 18080,
        }
        assert "bot_token" not in seeded
        assert "webhook_secret" not in seeded
        assert _zalo.os.environ["ZALO_ALLOWED_USERS"] == "u1,u2,u3"
        assert _zalo.os.environ["ZALO_ALLOW_ALL_USERS"] == "false"
        assert _zalo.os.environ["ZALO_HOME_CHANNEL"] == "chat-1"
        assert _zalo.os.environ["ZALO_HOME_CHANNEL_NAME"] == "Home"

    def test_yaml_bridge_preserves_explicit_env_allowlist(self, monkeypatch):
        monkeypatch.setenv("ZALO_ALLOWED_USERS", "env-user")

        _apply_yaml_config({}, {"allow_from": ["yaml-user"]})

        assert _zalo.os.environ["ZALO_ALLOWED_USERS"] == "env-user"

    def test_adapter_defaults_match_official_api(self, monkeypatch):
        monkeypatch.delenv("ZALO_WEBHOOK_SECRET", raising=False)

        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))

        assert adapter._bot_base_url == f"{ZALO_API_BASE}/bottoken"
        assert adapter.poll_timeout_seconds == 30
        assert adapter.connection_mode == "auto"
        assert adapter._uses_webhook is False
        assert adapter.splits_long_messages is True

    def test_webhook_requires_https_url_and_valid_secret(self, monkeypatch):
        monkeypatch.setenv("ZALO_WEBHOOK_SECRET", "12345678")
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "connection_mode": "webhook",
                    "webhook_url": "http://bot.example.test/zalo",
                },
            )
        )

        assert adapter._validate_transport_config() == (
            "platforms.zalo.webhook_url must be a public HTTPS URL"
        )


class TestHttp:
    class _FakeClient:
        def __init__(self, response):
            self.response = response

        async def post(self, url, json):
            return self.response

    @pytest.mark.asyncio
    async def test_api_converts_http_status_to_retryable_zalo_error(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        adapter._client = self._FakeClient(httpx.Response(429, text="rate limited"))

        with pytest.raises(ZaloApiError) as exc_info:
            await adapter._api("sendMessage", {"chat_id": "u1", "text": "hello"})

        assert exc_info.value.error_code == 429
        assert exc_info.value.retryable is True
        assert "rate limited" in exc_info.value.description

    def test_poll_backoff_has_capped_jitter(self, monkeypatch):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        monkeypatch.setattr(_zalo.random, "random", lambda: 1.0)

        assert adapter._poll_backoff_sleep(0) == pytest.approx(1.25)
        assert adapter._poll_backoff_sleep(1) == pytest.approx(2.5)
        assert adapter._poll_backoff_sleep(99) == pytest.approx(37.5)

    @pytest.mark.asyncio
    async def test_send_image_uses_documented_send_photo(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        calls = []

        async def fake_api(method, payload):
            calls.append((method, payload))
            return {"ok": True, "result": {"message_id": "photo-1"}}

        adapter._api = fake_api

        result = await adapter.send_image("chat-1", "https://cdn.example/image.jpg", "caption")

        assert result.success is True
        assert result.message_id == "photo-1"
        assert calls == [
            (
                "sendPhoto",
                {
                    "chat_id": "chat-1",
                    "photo": "https://cdn.example/image.jpg",
                    "caption": "caption",
                },
            )
        ]

    @pytest.mark.asyncio
    async def test_send_splits_at_documented_2000_character_limit(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        calls = []

        async def fake_api(method, payload):
            calls.append((method, payload))
            return {"ok": True, "result": {"message_id": f"message-{len(calls)}"}}

        adapter._api = fake_api

        result = await adapter.send("chat-1", "x" * 2001)

        assert len(calls) == 2
        assert all(len(payload["text"]) <= 2000 for _method, payload in calls)
        assert result.message_id == "message-2"
        assert result.continuation_message_ids == ("message-1",)


class TestPollingLifecycle:
    class _FakeClient:
        def __init__(self):
            self.calls = []

        async def post(self, url, json):
            method = url.rsplit("/", 1)[-1]
            self.calls.append((method, json))
            if method == "getMe":
                return httpx.Response(200, json={"ok": True, "result": {"id": "bot-1"}})
            if method == "getUpdates":
                return httpx.Response(200, json={"ok": True, "result": []})
            return httpx.Response(200, json={"ok": True, "result": True})

        async def aclose(self):
            pass

    @pytest.mark.asyncio
    async def test_polling_clears_webhook_before_get_updates(self, monkeypatch):
        fake_client = self._FakeClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={"bot_token": "token", "connection_mode": "polling"},
            )
        )

        try:
            assert await adapter.connect() is True
            await asyncio.sleep(0)
        finally:
            await adapter.disconnect()

        methods = [method for method, _payload in fake_client.calls]
        assert methods[:3] == ["getMe", "deleteWebhook", "getUpdates"]

    @pytest.mark.asyncio
    async def test_invalid_webhook_config_fails_before_network(self, monkeypatch):
        monkeypatch.delenv("ZALO_WEBHOOK_SECRET", raising=False)
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "connection_mode": "webhook",
                    "webhook_url": "https://bot.example.test/zalo",
                },
            )
        )

        assert await adapter.connect() is False
        assert adapter._client is None


class TestWebhookLifecycle:
    @pytest.mark.asyncio
    async def test_webhook_mode_registers_documented_url_and_secret(self, monkeypatch):
        fake_client = TestPollingLifecycle._FakeClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
        monkeypatch.setenv("ZALO_WEBHOOK_SECRET", "secret-123")
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "connection_mode": "webhook",
                    "webhook_url": "https://bot.example.test/zalo",
                },
            )
        )

        async def fake_start_webhook_server():
            return True

        adapter._start_webhook_server = fake_start_webhook_server

        try:
            assert await adapter.connect() is True
        finally:
            await adapter.disconnect()

        assert fake_client.calls[:2] == [
            ("getMe", {}),
            (
                "setWebhook",
                {
                    "url": "https://bot.example.test/zalo",
                    "secret_token": "secret-123",
                },
            ),
        ]


class TestInboundContract:
    @pytest.mark.asyncio
    async def test_documented_text_event_reaches_gateway(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        events = []

        async def capture(event):
            events.append(event)

        adapter.handle_message = capture
        await adapter._handle_update(
            {
                "event_name": EVENT_TEXT,
                "message": {
                    "from": {"id": "user-1", "display_name": "Ted", "is_bot": False},
                    "chat": {"id": "chat-1", "chat_type": "PRIVATE"},
                    "text": "Xin chào",
                    "message_id": "message-1",
                },
            }
        )

        assert len(events) == 1
        assert events[0].text == "Xin chào"
        assert events[0].message_type is MessageType.TEXT
        assert events[0].source.user_id == "user-1"
        assert events[0].source.chat_id == "chat-1"

    @pytest.mark.asyncio
    async def test_documented_media_fields_are_normalized(self, monkeypatch):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))

        async def fake_cache(url, *, kind):
            return f"/cache/{kind}"

        monkeypatch.setattr(adapter, "_cache_media_url", fake_cache)

        image = await adapter._event_content(
            EVENT_IMAGE,
            {"photo": "https://cdn.example/image.jpg", "caption": "A photo"},
        )
        voice = await adapter._event_content(
            EVENT_VOICE,
            {"voice_url": "https://cdn.example/voice.m4a"},
        )

        assert image == ("A photo", MessageType.PHOTO, ["/cache/image"], ["image/jpeg"])
        assert voice == (
            "[Zalo voice message]",
            MessageType.VOICE,
            ["/cache/audio"],
            ["audio/mpeg"],
        )

    @pytest.mark.asyncio
    async def test_unsupported_event_does_not_guess_omitted_content(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))

        text, message_type, media_urls, media_types = await adapter._event_content(
            EVENT_UNSUPPORTED,
            {"attachments": [{"url": "https://undocumented.example/value"}]},
        )

        assert "intentionally did not provide" in text
        assert "undocumented.example" not in text
        assert message_type is MessageType.TEXT
        assert media_urls == []
        assert media_types == []

    @pytest.mark.asyncio
    async def test_unknown_event_is_ignored(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        called = False

        async def capture(event):
            nonlocal called
            called = True

        adapter.handle_message = capture
        await adapter._handle_update(
            {
                "event_name": "message.future.received",
                "message": {
                    "from": {"id": "user-1"},
                    "chat": {"id": "chat-1", "chat_type": "PRIVATE"},
                    "text": "future payload",
                },
            }
        )

        assert called is False


class TestWebhookSecurity:
    class _Request:
        headers = {"X-Bot-Api-Secret-Token": "không-hợp-lệ"}

        async def json(self):
            return {}

    @pytest.mark.asyncio
    async def test_non_ascii_secret_header_is_rejected_without_crashing(self, monkeypatch):
        monkeypatch.setenv("ZALO_WEBHOOK_SECRET", "12345678")
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={"bot_token": "token", "webhook_url": "https://example.test/zalo"},
            )
        )

        response = await adapter._handle_webhook_request(self._Request())

        assert response.status == 403
