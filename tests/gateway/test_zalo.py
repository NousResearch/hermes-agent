"""Focused tests for the bundled Zalo Bot Platform plugin."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.zalo import adapter as zalo


class TestZaloConfiguration:
    def test_combined_requirement_probe_needs_env_token(self, monkeypatch):
        monkeypatch.setattr(zalo, "HTTPX_AVAILABLE", True)
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        assert zalo.check_requirements() is True
        assert zalo.check_zalo_requirements() is False

        monkeypatch.setenv("ZALO_BOT_TOKEN", "tok")
        assert zalo.check_zalo_requirements() is True

    def test_validate_config_accepts_env_or_yaml_token(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        assert zalo.validate_config(PlatformConfig(enabled=True)) is False
        assert zalo.validate_config(PlatformConfig(enabled=True, token="yaml")) is True
        assert (
            zalo.validate_config(
                PlatformConfig(enabled=True, extra={"bot_token": "extra"})
            )
            is True
        )

    def test_webhook_config_requires_https_url_and_bounded_secret(self, monkeypatch):
        for name in (
            "ZALO_BOT_TOKEN",
            "ZALO_CONNECTION_MODE",
            "ZALO_WEBHOOK_PUBLIC_URL",
            "ZALO_WEBHOOK_SECRET",
        ):
            monkeypatch.delenv(name, raising=False)
        base = {"connection_mode": "webhook"}
        assert not zalo.validate_config(
            PlatformConfig(enabled=True, token="tok", extra=base)
        )
        assert not zalo.validate_config(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={
                    **base,
                    "webhook_public_url": "http://example.com/hook",
                    "webhook_secret": "12345678",
                },
            )
        )
        assert zalo.validate_config(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={
                    **base,
                    "webhook_public_url": "https://example.com/hook",
                    "webhook_secret": "12345678",
                },
            )
        )

    def test_webhook_environment_overrides_yaml_transport(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "webhook")
        monkeypatch.setenv("ZALO_WEBHOOK_PUBLIC_URL", "https://env.example/hook")
        monkeypatch.setenv("ZALO_WEBHOOK_SECRET", "env-secret")
        monkeypatch.setenv("ZALO_WEBHOOK_HOST", "127.0.0.1")
        monkeypatch.setenv("ZALO_WEBHOOK_PORT", "9911")
        monkeypatch.setenv("ZALO_WEBHOOK_PATH", "/env-hook")
        adapter = zalo.ZaloBotAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={
                    "connection_mode": "polling",
                    "webhook_public_url": "https://yaml.example/yaml",
                    "webhook_secret": "yaml-secret",
                    "webhook_host": "0.0.0.0",
                    "webhook_port": 8811,
                    "webhook_path": "/yaml-hook",
                },
            )
        )
        assert adapter._connection_mode == "webhook"
        assert adapter._webhook_public_url == "https://env.example/hook"
        assert adapter._webhook_secret == "env-secret"
        assert adapter._webhook_host == "127.0.0.1"
        assert adapter._webhook_port == 9911
        assert adapter._webhook_path == "/env-hook"

    def test_profile_secret_scope_is_authoritative_for_mode_and_token(
        self, monkeypatch
    ):
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from gateway.config import platform_binds_port

        monkeypatch.setenv("ZALO_BOT_TOKEN", "wrong-process-token")
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "polling")
        scope_token = set_secret_scope({
            "ZALO_BOT_TOKEN": "scoped-token",
            "ZALO_CONNECTION_MODE": "webhook",
            "ZALO_WEBHOOK_PUBLIC_URL": "https://scope.example/hook",
            "ZALO_WEBHOOK_SECRET": "scope-secret",
        })
        try:
            adapter = zalo.ZaloBotAdapter(
                PlatformConfig(
                    enabled=True,
                    token="yaml-token",
                    extra={"connection_mode": "polling"},
                )
            )
            assert adapter._token == "scoped-token"
            assert adapter._connection_mode == "webhook"
            seed = zalo._env_enablement()
            assert seed is not None
            effective_extra = {"connection_mode": "polling"}
            effective_extra.update(seed)
            assert platform_binds_port("zalo", effective_extra) is True
        finally:
            reset_secret_scope(scope_token)

    def test_env_enablement_seeds_mode_without_env_token(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "webhook")
        assert zalo._env_enablement() == {"connection_mode": "webhook"}

    def test_scoped_mode_is_materialized_before_port_guard(self, tmp_path, monkeypatch):
        from agent.secret_scope import (
            is_multiplex_active,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from gateway.config import load_gateway_config, platform_binds_port

        (tmp_path / "config.yaml").write_text(
            "platforms:\n"
            "  zalo:\n"
            "    enabled: true\n"
            "    token: yaml-token\n"
            "    extra:\n"
            "      connection_mode: polling\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "polling")
        previous_multiplex = is_multiplex_active()
        set_multiplex_active(True)
        scope_token = set_secret_scope({"ZALO_CONNECTION_MODE": "webhook"})
        try:
            loaded = load_gateway_config()
        finally:
            reset_secret_scope(scope_token)
            set_multiplex_active(previous_multiplex)

        config = loaded.platforms[Platform("zalo")]
        assert config.extra["connection_mode"] == "webhook"
        # This call intentionally runs after the profile scope has exited,
        # exactly like GatewayRunner._start_one_profile_adapters.
        assert platform_binds_port("zalo", config.extra) is True

    def test_env_enablement_seeds_transport_and_home(self, monkeypatch):
        monkeypatch.setenv("ZALO_BOT_TOKEN", "tok")
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "webhook")
        monkeypatch.setenv("ZALO_WEBHOOK_PUBLIC_URL", "https://example.com/zalo")
        monkeypatch.setenv("ZALO_WEBHOOK_PORT", "9001")
        monkeypatch.setenv("ZALO_POLL_TIMEOUT", "25")
        monkeypatch.setenv("ZALO_HOME_CHANNEL", "chat-1")
        monkeypatch.setenv("ZALO_HOME_CHANNEL_NAME", "Primary")

        seed = zalo._env_enablement()

        assert seed is not None
        assert seed["bot_token"] == "tok"
        assert seed["connection_mode"] == "webhook"
        assert seed["webhook_public_url"] == "https://example.com/zalo"
        assert seed["webhook_port"] == 9001
        assert seed["poll_timeout"] == "25"
        assert seed["home_channel"] == {
            "chat_id": "chat-1",
            "name": "Primary",
        }

    def test_adapter_uses_env_precedence_and_parses_webhook(self, monkeypatch):
        monkeypatch.setenv("ZALO_BOT_TOKEN", "env-token")
        monkeypatch.setenv("ZALO_POLL_TIMEOUT", "22")
        cfg = PlatformConfig(
            enabled=True,
            token="yaml-token",
            extra={
                "connection_mode": "webhook",
                "webhook_public_url": "https://example.com/hook",
                "webhook_secret": "12345678",
                "webhook_port": 9001,
            },
        )

        adapter = zalo.ZaloBotAdapter(cfg)

        assert adapter.platform == Platform("zalo")
        assert adapter._token == "env-token"
        assert adapter._poll_timeout == 22
        assert adapter._connection_mode == "webhook"
        assert adapter._webhook_path == "/hook"

    @pytest.mark.parametrize("port", ["invalid", 0, 65536])
    def test_invalid_webhook_port_falls_back_safely(self, monkeypatch, port):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(
            PlatformConfig(enabled=True, token="tok", extra={"webhook_port": port})
        )
        assert adapter._webhook_port == 8790

    @pytest.mark.parametrize(
        "api_base",
        [
            "http://example.com/bot",
            "https://user:password@example.com/bot",
            "not-a-url",
        ],
    )
    def test_unsafe_api_base_cannot_redirect_bot_token(self, monkeypatch, api_base):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"api_base": api_base},
            )
        )
        assert adapter._api_base == zalo.ZALO_API_BASE


class TestWebhookPath:
    @pytest.mark.parametrize(
        ("public_url", "override", "expected"),
        [
            ("https://example.com/api/zalo", None, "/api/zalo"),
            ("https://example.com", None, "/zalo/webhook"),
            ("https://example.com", "custom", "/custom"),
        ],
    )
    def test_parse_webhook_path(self, public_url, override, expected):
        assert zalo._parse_webhook_path(public_url, override) == expected

    def test_root_public_url_is_registered_at_listener_path(self):
        assert (
            zalo._webhook_url_for_path("https://example.com", "/zalo/webhook")
            == "https://example.com/zalo/webhook"
        )
        assert (
            zalo._webhook_url_for_path(
                "https://example.com/old?token=unsafe#fragment", "/custom"
            )
            == "https://example.com/custom"
        )

    @pytest.mark.asyncio
    async def test_webhook_secret_is_required_and_unicode_safe(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)

        class FakeResponse:
            def __init__(self, *, status=200, text="ok"):
                self.status = status
                self.text = text

        class FakeRequest:
            def __init__(self, token):
                self.headers = {"X-Bot-Api-Secret-Token": token}
                self.json = AsyncMock(
                    return_value={
                        "result": {
                            "event_name": "message.text.received",
                            "message": {"message_id": "m1"},
                        }
                    }
                )

        monkeypatch.setattr(zalo, "web", SimpleNamespace(Response=FakeResponse))
        adapter = zalo.ZaloBotAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"webhook_secret": "비밀-secret-123"},
            )
        )
        adapter._dispatch_update = AsyncMock()  # type: ignore[method-assign]

        rejected = await adapter._handle_webhook_post(FakeRequest("wrong"))
        assert rejected.status == 403
        adapter._dispatch_update.assert_not_awaited()

        accepted = await adapter._handle_webhook_post(FakeRequest("비밀-secret-123"))
        assert accepted.status == 200
        adapter._dispatch_update.assert_awaited_once()


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_connect_probes_get_me_before_starting_polling(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        posts = []

        class FakeResponse:
            text = ""

            def json(self):
                return {"ok": True, "result": {"id": "bot-1"}}

        class FakeClient:
            async def post(self, url, json=None):
                posts.append((url, json))
                return FakeResponse()

            async def aclose(self):
                return None

        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        monkeypatch.setattr(zalo.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
        monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
        connect_polling = AsyncMock(return_value=True)
        monkeypatch.setattr(adapter, "_connect_polling", connect_polling)

        assert await adapter.connect() is True
        assert posts == [(zalo._api_url("tok", "getMe"), None)]
        connect_polling.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_accepts_runner_reconnect_keyword(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        monkeypatch.setattr(
            zalo.httpx,
            "AsyncClient",
            lambda **_kwargs: SimpleNamespace(aclose=AsyncMock()),
        )
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
        monkeypatch.setattr(adapter, "_probe_bot", AsyncMock(return_value=True))
        connect_polling = AsyncMock(return_value=True)
        monkeypatch.setattr(adapter, "_connect_polling", connect_polling)

        assert await adapter.connect(is_reconnect=True) is True
        connect_polling.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_webhook_application_enforces_body_limit(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={
                    "connection_mode": "webhook",
                    "webhook_public_url": "https://example.com/hook",
                    "webhook_secret": "secret-123",
                },
            )
        )
        response = SimpleNamespace(
            content=b"{}",
            status_code=200,
            json=lambda: {"ok": True},
        )
        adapter._http_client = SimpleNamespace(  # type: ignore[assignment]
            post=AsyncMock(return_value=response),
            aclose=AsyncMock(),
        )
        app = SimpleNamespace(router=SimpleNamespace(add_post=MagicMock()))
        application = MagicMock(return_value=app)
        runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())
        site = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        fake_web = SimpleNamespace(
            Application=application,
            AppRunner=MagicMock(return_value=runner),
            TCPSite=MagicMock(return_value=site),
        )
        monkeypatch.setattr(zalo, "web", fake_web)
        monkeypatch.setattr(zalo, "_ensure_webhook_dependency", lambda: True)

        assert await adapter._connect_webhook() is True
        application.assert_called_once_with(client_max_size=zalo.WEBHOOK_MAX_BYTES)
        app.router.add_post.assert_called_once_with(
            "/hook", adapter._handle_webhook_post
        )

    @pytest.mark.asyncio
    async def test_connect_releases_lock_when_get_me_rejects_token(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)

        class FakeResponse:
            text = "bad token"
            status_code = 401

            def json(self):
                return {"ok": False, "description": "Unauthorized"}

        class FakeClient:
            async def post(self, _url, json=None):
                return FakeResponse()

            async def aclose(self):
                return None

        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        release = MagicMock()
        monkeypatch.setattr(zalo.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
        monkeypatch.setattr(adapter, "_release_platform_lock", release)
        monkeypatch.setattr(adapter, "_write_runtime_status_safe", MagicMock())

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "zalo_auth_failed"
        release.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_cleans_up_when_transport_startup_raises(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)

        fake_client = SimpleNamespace(
            aclose=AsyncMock(side_effect=RuntimeError("close failed"))
        )

        adapter = zalo.ZaloBotAdapter(
            PlatformConfig(
                enabled=True,
                token="tok",
                extra={"connection_mode": "webhook"},
            )
        )
        monkeypatch.setattr(zalo.httpx, "AsyncClient", lambda **_kwargs: fake_client)
        release = MagicMock()
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
        monkeypatch.setattr(adapter, "_release_platform_lock", release)
        monkeypatch.setattr(adapter, "_probe_bot", AsyncMock(return_value=True))
        monkeypatch.setattr(
            adapter,
            "_connect_webhook",
            AsyncMock(side_effect=RuntimeError("startup failed")),
        )
        cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        monkeypatch.setattr(adapter, "_disconnect_webhook_server", cleanup)

        assert await adapter.connect() is False
        cleanup.assert_awaited_once()
        fake_client.aclose.assert_awaited_once()
        release.assert_called_once()
        assert adapter._http_client is None

    @pytest.mark.asyncio
    async def test_disconnect_releases_lock_when_every_cleanup_step_fails(
        self, monkeypatch
    ):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        adapter._webhook_site = SimpleNamespace(
            stop=AsyncMock(side_effect=RuntimeError("site stop failed"))
        )
        adapter._webhook_runner = SimpleNamespace(
            cleanup=AsyncMock(side_effect=RuntimeError("runner cleanup failed"))
        )
        fake_client = SimpleNamespace(
            aclose=AsyncMock(side_effect=RuntimeError("client close failed"))
        )
        adapter._http_client = fake_client  # type: ignore[assignment]
        release = MagicMock()
        monkeypatch.setattr(adapter, "_release_platform_lock", release)

        await adapter.disconnect()

        release.assert_called_once()
        fake_client.aclose.assert_awaited_once()
        assert adapter._webhook_site is None
        assert adapter._webhook_runner is None
        assert adapter._http_client is None

    def test_poll_backoff_stays_in_jitter_bound(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        for index in range(len(zalo._POLL_BACKOFF_SEC) + 2):
            delay = adapter._poll_backoff_sleep(index)
            cap = zalo._POLL_BACKOFF_SEC[min(index, len(zalo._POLL_BACKOFF_SEC) - 1)]
            assert cap <= delay <= cap * 1.25


class TestInboundDispatch:
    @staticmethod
    def _text_update(**message_overrides):
        message = {
            "message_id": "mid-1",
            "text": "hello",
            "from": {"id": "user-1", "display_name": "User", "is_bot": False},
            "chat": {"id": "chat-1", "chat_type": "PRIVATE"},
            "date": 1_700_000_000_000,
        }
        message.update(message_overrides)
        return {"event_name": "message.text.received", "message": message}

    @pytest.mark.asyncio
    async def test_dispatches_text_and_skips_bot_messages(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        seen = []

        async def capture(event):
            seen.append(event)

        adapter.set_message_handler(capture)
        await adapter._dispatch_update(self._text_update())
        await adapter._dispatch_update(
            self._text_update(
                message_id="bot-mid",
                **{"from": {"id": "bot-1", "is_bot": True}},
            )
        )
        await asyncio.sleep(0.05)

        assert len(seen) == 1
        assert seen[0].text == "hello"
        assert seen[0].source.user_id == "user-1"
        assert seen[0].source.chat_id == "chat-1"

    @pytest.mark.asyncio
    async def test_dispatches_image_and_sticker(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)

        async def fake_cache(url):
            assert url == "https://cdn.example/photo.jpg"
            return "/cached/photo.jpg"

        monkeypatch.setattr(zalo, "cache_image_from_url", fake_cache)
        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        seen = []

        async def capture(event):
            seen.append(event)

        adapter.set_message_handler(capture)
        common = {
            "from": {"id": "user-1", "display_name": "User", "is_bot": False},
            "chat": {"id": "chat-1", "chat_type": "PRIVATE"},
            "date": 1_700_000_000_000,
        }
        await adapter._dispatch_update({
            "event_name": "message.image.received",
            "message": {
                **common,
                "message_id": "image-1",
                "photo_url": "https://cdn.example/photo.jpg",
                "caption": "caption",
            },
        })
        await adapter._dispatch_update({
            "event_name": "message.sticker.received",
            "message": {
                **common,
                "message_id": "sticker-1",
                "sticker": "sticker-id",
            },
        })
        await asyncio.sleep(0.05)

        assert [event.message_type for event in seen] == [
            MessageType.PHOTO,
            MessageType.STICKER,
        ]
        assert seen[0].media_urls == ["/cached/photo.jpg"]
        assert "sticker-id" in seen[1].text

    @pytest.mark.asyncio
    async def test_unauthorized_sender_cannot_trigger_image_fetch(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        cache = AsyncMock(return_value="/cached/photo.jpg")
        monkeypatch.setattr(zalo, "cache_image_from_url", cache)
        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        adapter.set_authorization_check(lambda *_args: False)
        seen = []

        async def capture(event):
            seen.append(event)

        adapter.set_message_handler(capture)
        await adapter._dispatch_update({
            "event_name": "message.image.received",
            "message": {
                "message_id": "image-unauthorized",
                "photo_url": "https://cdn.example/photo.jpg",
                "from": {"id": "user-1", "is_bot": False},
                "chat": {"id": "chat-1", "chat_type": "PRIVATE"},
            },
        })
        await asyncio.sleep(0.05)

        cache.assert_not_awaited()
        assert len(seen) == 1
        assert seen[0].media_urls == []


class TestOutboundDelivery:
    @pytest.mark.asyncio
    async def test_live_send_does_not_expose_token_from_transport_error(
        self, monkeypatch, caplog
    ):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        token = "top-secret-token"

        class LeakyClient:
            async def post(self, _url, json=None):
                raise RuntimeError(f"request failed for {_url}")

        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token=token))
        adapter._http_client = LeakyClient()  # type: ignore[assignment]

        result = await adapter.send("chat-1", "hello")

        assert result.success is False
        assert result.error == "Zalo send failed"
        assert token not in caplog.text

    @pytest.mark.asyncio
    async def test_api_error_body_cannot_reflect_token(self, monkeypatch, caplog):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        token = "top-secret-token"

        class EchoResponse:
            status_code = 400
            text = f"https://evil.invalid/bot{token}/sendMessage"

            def json(self):
                return {"ok": False, "description": self.text}

        class EchoClient:
            async def post(self, _url, json=None):
                return EchoResponse()

        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token=token))
        adapter._http_client = EchoClient()  # type: ignore[assignment]

        result = await adapter.send("chat-1", "hello")

        assert result.success is False
        assert token not in (result.error or "")
        assert "[REDACTED]" in (result.error or "")
        assert token not in caplog.text

    @pytest.mark.asyncio
    async def test_live_adapter_send_photo(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        posts = []

        class FakeResponse:
            text = ""
            status_code = 200

            def json(self):
                return {"ok": True, "result": {"message_id": "photo-1"}}

        class FakeClient:
            async def post(self, url, json=None):
                posts.append((url, json))
                return FakeResponse()

        adapter = zalo.ZaloBotAdapter(PlatformConfig(enabled=True, token="tok"))
        adapter._http_client = FakeClient()  # type: ignore[assignment]

        result = await adapter.send_image(
            "chat-1", "https://img.example/photo.png", caption="caption"
        )

        assert result.success is True
        assert result.message_id == "photo-1"
        assert posts[0][0].endswith("/sendPhoto")
        assert posts[0][1] == {
            "chat_id": "chat-1",
            "photo": "https://img.example/photo.png",
            "caption": "caption",
        }

    @pytest.mark.asyncio
    async def test_standalone_sender_supports_cron_and_send_message(self, monkeypatch):
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        posts = []

        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"ok": True, "result": {"message_id": "message-1"}}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, url, json=None):
                posts.append((url, json))
                return FakeResponse()

        monkeypatch.setattr(zalo.httpx, "AsyncClient", lambda **_kwargs: FakeClient())
        result = await zalo._standalone_send(
            PlatformConfig(enabled=True, token="tok"), "chat-1", "hello"
        )

        assert result == {
            "success": True,
            "platform": "zalo",
            "chat_id": "chat-1",
            "message_id": "message-1",
        }
        assert posts[0][0].endswith("/sendMessage")
        assert posts[0][1] == {"chat_id": "chat-1", "text": "hello"}


class TestPluginIntegration:
    def test_interactive_setup_saves_polling_configuration(self, monkeypatch):
        import hermes_cli.config as config_module
        import hermes_cli.secret_prompt as secret_prompt_module

        saved = {}
        answers = iter(["user-1,user-2", "chat-1", "polling"])
        monkeypatch.setattr(config_module, "get_env_value", lambda _name: None)
        monkeypatch.setattr(
            config_module,
            "save_env_value",
            lambda name, value: saved.__setitem__(name, value),
        )
        monkeypatch.setattr(
            secret_prompt_module,
            "masked_secret_prompt",
            lambda _prompt: "token-1",
        )
        monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

        zalo.interactive_setup()

        assert saved == {
            "ZALO_BOT_TOKEN": "token-1",
            "ZALO_ALLOWED_USERS": "user-1,user-2",
            "ZALO_HOME_CHANNEL": "chat-1",
            "ZALO_CONNECTION_MODE": "polling",
        }

    def test_env_only_config_auto_enables_zalo(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("ZALO_BOT_TOKEN", "tok")
        monkeypatch.setenv("ZALO_HOME_CHANNEL", "chat-1")
        from gateway.config import load_gateway_config

        config = load_gateway_config()
        platform = Platform("zalo")

        assert config.platforms[platform].enabled is True
        assert config.platforms[platform].extra["bot_token"] == "tok"
        home_channel = config.platforms[platform].home_channel
        assert home_channel is not None
        assert home_channel.chat_id == "chat-1"
        assert platform in config.get_connected_platforms()

    def test_yaml_only_config_constructs_registered_adapter(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("ZALO_BOT_TOKEN", raising=False)
        (tmp_path / "config.yaml").write_text(
            "platforms:\n  zalo:\n    enabled: true\n    token: yaml-only-token\n",
            encoding="utf-8",
        )
        from gateway.config import load_gateway_config
        from gateway.platform_registry import platform_registry
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        config = load_gateway_config()
        platform_config = config.platforms[Platform("zalo")]

        adapter = platform_registry.create_adapter("zalo", platform_config)

        assert adapter is not None
        assert adapter.__class__.__name__ == "ZaloBotAdapter"
        assert adapter.platform == Platform("zalo")
        assert adapter._token == "yaml-only-token"

    def test_register_exposes_setup_status_send_and_cron_hooks(self):
        ctx = SimpleNamespace(register_platform=MagicMock())

        zalo.register(ctx)

        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "zalo"
        assert kwargs["required_env"] == ["ZALO_BOT_TOKEN"]
        assert callable(kwargs["setup_fn"])
        assert callable(kwargs["is_connected"])
        assert callable(kwargs["env_enablement_fn"])
        assert callable(kwargs["standalone_sender_fn"])
        assert kwargs["cron_deliver_env_var"] == "ZALO_HOME_CHANNEL"
        assert kwargs["allowed_users_env"] == "ZALO_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "ZALO_ALLOW_ALL_USERS"
        assert kwargs["max_message_length"] == 2000
        assert zalo.ZaloBotAdapter.splits_long_messages is True

    def test_webhook_mode_participates_in_port_binding_guards(self, monkeypatch):
        from gateway.config import platform_binds_port
        from hermes_cli.web_server import _PORT_BINDING_PLATFORM_PORTS

        monkeypatch.delenv("ZALO_CONNECTION_MODE", raising=False)
        assert platform_binds_port("zalo", {"connection_mode": "polling"}) is False
        assert platform_binds_port("zalo", {"connection_mode": "webhook"}) is True
        assert _PORT_BINDING_PLATFORM_PORTS["zalo"] == ("webhook_port", 8790)

        monkeypatch.setenv("ZALO_CONNECTION_MODE", "webhook")
        webhook_seed = zalo._env_enablement()
        assert webhook_seed is not None
        webhook_extra = {"connection_mode": "polling"}
        webhook_extra.update(webhook_seed)
        assert platform_binds_port("zalo", webhook_extra) is True
        monkeypatch.setenv("ZALO_CONNECTION_MODE", "polling")
        polling_seed = zalo._env_enablement()
        assert polling_seed is not None
        polling_extra = {"connection_mode": "webhook"}
        polling_extra.update(polling_seed)
        assert platform_binds_port("zalo", polling_extra) is False

    def test_bundled_discovery_registers_zalo_for_cron(self, monkeypatch):
        monkeypatch.setenv("ZALO_BOT_TOKEN", "tok")
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get("zalo")
        assert entry is not None
        assert entry.cron_deliver_env_var == "ZALO_HOME_CHANNEL"
        assert callable(entry.standalone_sender_fn)
        assert entry.standalone_sender_fn.__name__ == "_standalone_send"

        from cron.scheduler import _resolve_home_env_var

        assert _resolve_home_env_var("zalo") == "ZALO_HOME_CHANNEL"
