from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_zalo = load_plugin_adapter("zalo")
ZALO_API_BASE = _zalo.ZALO_API_BASE
ZaloAdapter = _zalo.ZaloAdapter
ZaloApiError = _zalo.ZaloApiError
_env_enablement = _zalo._env_enablement
_apply_yaml_config = _zalo._apply_yaml_config
register = _zalo.register


class TestRegister:
    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    def test_register_exposes_generic_zalo_platform(self):
        ctx = self._FakeCtx()

        register(ctx)

        assert ctx.kwargs is not None
        assert ctx.kwargs["name"] == "zalo"
        assert ctx.kwargs["label"] == "Zalo"
        assert ctx.kwargs["required_env"] == ["ZALO_BOT_TOKEN"]
        assert ctx.kwargs["allowed_users_env"] == "ZALO_ALLOWED_USERS"
        assert ctx.kwargs["allow_all_env"] == "ZALO_ALLOW_ALL_USERS"
        assert ctx.kwargs["standalone_sender_fn"] is not None
        assert "Zalo Bot Platform" in ctx.kwargs["platform_hint"]


class TestConfig:
    def test_env_enablement_maps_platform_env_without_profile_logic(self, monkeypatch):
        monkeypatch.setenv("ZALO_BOT_TOKEN", "token")
        monkeypatch.setenv("ZALO_API_BASE", "https://example.test")
        monkeypatch.setenv("ZALO_ALLOWED_USERS", "u1, u2")
        monkeypatch.setenv("ZALO_ALLOW_ALL_USERS", "false")
        monkeypatch.setenv("ZALO_DM_ONLY", "true")
        monkeypatch.setenv("ZALO_SUPPRESS_NOISY_STATUS", "false")
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "webhook")
        monkeypatch.setenv("ZALO_WEBHOOK_PUBLIC_URL", "https://bot.example.test/zalo")
        monkeypatch.setenv("ZALO_WEBHOOK_SECRET", "secret")
        monkeypatch.setenv("ZALO_WEBHOOK_PATH", "zalo-hook")
        monkeypatch.setenv("ZALO_WEBHOOK_HOST", "0.0.0.0")
        monkeypatch.setenv("ZALO_WEBHOOK_PORT", "18080")
        monkeypatch.setenv("ZALO_WEBHOOK_AUTO_REGISTER", "yes")
        monkeypatch.setenv("ZALO_DELETE_WEBHOOK_ON_POLLING_START", "true")
        monkeypatch.setenv("ZALO_DELETE_WEBHOOK_ON_DISCONNECT", "true")

        seeded = _env_enablement()

        assert seeded == {
            "bot_token": "token",
            "api_base": "https://example.test",
            "allowed_users": ["u1", "u2"],
            "allow_all_users": False,
            "dm_only": True,
            "suppress_noisy_status": False,
            "connection_mode": "webhook",
            "webhook_url": "https://bot.example.test/zalo",
            "webhook_secret": "secret",
            "webhook_path": "zalo-hook",
            "webhook_host": "0.0.0.0",
            "webhook_port": 18080,
            "webhook_auto_register": True,
            "delete_webhook_on_polling_start": True,
            "delete_webhook_on_disconnect": True,
        }

    def test_apply_yaml_config_maps_non_secret_operator_settings(self):
        seeded = _apply_yaml_config(
            {},
            {
                "api_base": "https://zalo.example.test",
                "allow_from": ["u1", "u2"],
                "allowed_users": "u3, u4",
                "allowed_chats": ["chat-1"],
                "allow_all_users": False,
                "dm_only": True,
                "private_only": True,
                "poll_timeout_seconds": 12,
                "poll_interval_seconds": 0.5,
                "connection_mode": "polling",
                "parse_mode": "",
                "suppress_noisy_status": False,
                "webhook_url": "https://bot.example.test/zalo",
                "webhook_path": "zalo-hook",
                "webhook_host": "0.0.0.0",
                "webhook_port": 18080,
                "webhook_auto_register": True,
                "delete_webhook_on_polling_start": True,
                "delete_webhook_on_disconnect": False,
                "url_intake_public_base": "https://intake.example.test",
                "url_intake_pending_file": "/tmp/zalo-intake.json",
                "bot_token": "ignored-token",
                "webhook_secret": "ignored-secret",
            },
        )

        assert seeded == {
            "api_base": "https://zalo.example.test",
            "allowed_users": ["chat-1", "u1", "u2", "u3", "u4"],
            "allow_all_users": False,
            "dm_only": True,
            "private_only": True,
            "poll_timeout_seconds": 12,
            "poll_interval_seconds": 0.5,
            "connection_mode": "polling",
            "parse_mode": "",
            "suppress_noisy_status": False,
            "webhook_url": "https://bot.example.test/zalo",
            "webhook_path": "zalo-hook",
            "webhook_host": "0.0.0.0",
            "webhook_port": 18080,
            "webhook_auto_register": True,
            "delete_webhook_on_polling_start": True,
            "delete_webhook_on_disconnect": False,
            "url_intake_public_base": "https://intake.example.test",
            "url_intake_pending_file": "/tmp/zalo-intake.json",
        }
        assert "bot_token" not in seeded
        assert "webhook_secret" not in seeded

    def test_adapter_defaults_to_official_api_base(self, monkeypatch):
        monkeypatch.delenv("ZALO_API_BASE", raising=False)

        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))

        assert adapter.api_base == ZALO_API_BASE

    def test_connection_mode_polling_overrides_webhook_alias(self, monkeypatch):
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "polling")
        monkeypatch.setenv("ZALO_WEBHOOK_PUBLIC_URL", "https://bot.example.test/zalo")
        monkeypatch.setenv("ZALO_WEBHOOK_SECRET", "12345678")

        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))

        assert adapter.connection_mode == "polling"
        assert adapter.webhook_url == "https://bot.example.test/zalo"
        assert adapter._webhook_enabled is False
        assert adapter._webhook_config_incomplete is False


class TestStatusFiltering:
    def test_noisy_compression_status_is_suppressed_by_default(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))

        assert (
            adapter.prepare_gateway_status_message(
                "lifecycle",
                "📦 Preflight compression: ~147,901 tokens >= 136,000 threshold. "
                "This may take a moment.",
            )
            is None
        )
        assert (
            adapter.prepare_gateway_status_message(
                "lifecycle",
                "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
            )
            is None
        )

    def test_noisy_status_filter_can_be_disabled(self):
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={"bot_token": "token", "suppress_noisy_status": False},
            )
        )
        message = "🗜️ Compacting context — summarizing earlier conversation so I can continue..."

        assert adapter.prepare_gateway_status_message("lifecycle", message) == message

    def test_normal_status_is_preserved(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        message = "Working on the website now."

        assert adapter.prepare_gateway_status_message("progress", message) == message


class TestHttp:
    class _FakeClient:
        def __init__(self, response):
            self.response = response

        async def post(self, url, json):
            return self.response

    @pytest.mark.asyncio
    async def test_api_converts_http_status_to_zalo_api_error(self):
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
    async def test_send_image_uses_native_send_photo(self):
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
    async def test_send_animation_uses_native_send_sticker(self):
        adapter = ZaloAdapter(PlatformConfig(enabled=True, extra={"bot_token": "token"}))
        calls = []

        async def fake_api(method, payload):
            calls.append((method, payload))
            return {"ok": True, "result": {"message_id": "sticker-1"}}

        adapter._api = fake_api

        result = await adapter.send_animation("chat-1", "zalo-sticker-id")

        assert result.success is True
        assert result.message_id == "sticker-1"
        assert calls == [
            ("sendSticker", {"chat_id": "chat-1", "sticker": "zalo-sticker-id"})
        ]


class TestWebhookPollingMode:
    class _FakeClient:
        def __init__(self):
            self.calls = []

        async def post(self, url, json):
            self.calls.append((url.rsplit("/", 1)[-1], json))
            if url.endswith("/getMe"):
                return httpx.Response(200, json={"ok": True, "result": {"id": "bot-1"}})
            if url.endswith("/deleteWebhook"):
                return httpx.Response(200, json={"ok": True, "result": True})
            if url.endswith("/getUpdates"):
                return httpx.Response(200, json={"ok": True, "result": []})
            return httpx.Response(200, json={"ok": True, "result": {}})

        async def aclose(self):
            pass

    @pytest.mark.asyncio
    async def test_polling_can_delete_stale_webhook_before_get_updates(self, monkeypatch):
        fake_client = self._FakeClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "delete_webhook_on_polling_start": True,
                    "poll_interval_seconds": 0,
                    "poll_timeout_seconds": 1,
                },
            )
        )

        try:
            assert await adapter.connect() is True
            await asyncio.sleep(0)
        finally:
            await adapter.disconnect()

        methods = [method for method, _payload in fake_client.calls]
        assert methods[:2] == ["getMe", "deleteWebhook"]

    @pytest.mark.asyncio
    async def test_disconnect_can_delete_webhook_when_opted_in(self):
        fake_client = self._FakeClient()
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "delete_webhook_on_disconnect": True,
                },
            )
        )
        adapter._client = fake_client

        await adapter.disconnect()

        methods = [method for method, _payload in fake_client.calls]
        assert methods == ["deleteWebhook"]

    @pytest.mark.asyncio
    async def test_partial_webhook_config_fails_closed(self, monkeypatch):
        fake_client = self._FakeClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={"bot_token": "token", "webhook_url": "https://bot.example/zalo"},
            )
        )

        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_short_webhook_secret_fails_closed(self, monkeypatch):
        fake_client = self._FakeClient()
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: fake_client)
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "webhook_url": "https://bot.example/zalo",
                    "webhook_secret": "short",
                },
            )
        )

        assert await adapter.connect() is False


class TestUrlIntake:
    def test_url_intake_pop_is_generic_and_consumes_entries(self, tmp_path):
        pending_path = tmp_path / "pending.json"
        pending_path.write_text(
            json.dumps(
                {
                    "chat-1": [
                        {
                            "url": "https://docs.google.com/spreadsheets/d/sheet/edit",
                            "note": "rooms",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "bot_token": "token",
                    "url_intake_pending_file": str(pending_path),
                },
            )
        )

        text = adapter._pop_url_intake_text("chat-1")

        assert "[Zalo URL intake submissions for this chat]" in text
        assert "https://docs.google.com/spreadsheets/d/sheet/edit" in text
        assert json.loads(pending_path.read_text(encoding="utf-8")) == {}


class TestSourcePolicy:
    def test_dm_only_rejects_group_chat_type(self):
        adapter = ZaloAdapter(
            PlatformConfig(enabled=True, extra={"bot_token": "token", "dm_only": True})
        )
        update = {
            "message": {
                "chat": {"id": "group-1", "type": "GROUP"},
                "sender": {"id": "user-1"},
                "text": "hello",
            }
        }
        message = update["message"]

        assert adapter._chat_type(update, message) == "group"
        assert adapter.dm_only is True

    def test_allowed_source_accepts_user_or_chat_id(self):
        adapter = ZaloAdapter(
            PlatformConfig(
                enabled=True,
                extra={"bot_token": "token", "allowed_users": ["user-1", "chat-2"]},
            )
        )

        assert adapter._allowed_source("user-1", "chat-1")
        assert adapter._allowed_source("user-2", "chat-2")
        assert not adapter._allowed_source("user-2", "chat-3")
