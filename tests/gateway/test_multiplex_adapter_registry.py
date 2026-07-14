"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway.run import GatewayRunner


class _FakeAdapter:
    def __init__(self, token=None, config=None):
        self.token = token
        self.config = config


class TestCredentialFingerprint:
    def test_none_without_token(self):
        assert GatewayRunner._adapter_credential_fingerprint(_FakeAdapter()) is None

    def test_stable_and_log_safe(self):
        a = _FakeAdapter(token="secret-bot-token")
        fp1 = GatewayRunner._adapter_credential_fingerprint(a)
        fp2 = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="secret-bot-token"))
        assert fp1 == fp2  # stable
        assert "secret-bot-token" not in (fp1 or "")  # never the raw token
        assert len(fp1) == 16

    def test_distinct_tokens_distinct_fp(self):
        a = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-A"))
        b = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="tok-B"))
        assert a != b

    def test_reads_alt_attrs(self):
        class _AltAdapter:
            def __init__(self):
                self.bot_token = "alt-token"
        assert GatewayRunner._adapter_credential_fingerprint(_AltAdapter()) is not None

    def test_reads_platform_config_token(self):
        class _Config:
            token = "config-token"

        fp = GatewayRunner._adapter_credential_fingerprint(
            _FakeAdapter(token=None, config=_Config())
        )

        assert fp is not None
        assert "config-token" not in fp


class TestProfileMessageHandler:
    @pytest.mark.asyncio
    async def test_stamps_profile_on_unstamped_source(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()

        result = await handler(_Evt())
        assert result == "ok"
        assert seen["profile"] == "coder"

    @pytest.mark.asyncio
    async def test_does_not_override_existing_profile(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = "writer"  # already stamped (e.g. by URL prefix)

        class _Evt:
            source = _Src()

        await handler(_Evt())
        assert seen["profile"] == "writer"


class TestPortBindingHardError:
    """A secondary profile enabling a port-binding platform aborts startup."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_raises(self, monkeypatch):
        from gateway.run import MultiplexConfigError
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        # reviewer profile config enables webhook (a port-binding platform)
        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.WEBHOOK: PlatformConfig(enabled=True, extra={"port": 8644}),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        with pytest.raises(MultiplexConfigError) as ei:
            await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert "webhook" in str(ei.value)
        assert "reviewer" in str(ei.value)

    @pytest.mark.asyncio
    async def test_secondary_non_binding_platform_ok(self, monkeypatch):
        """A non-port-binding platform (e.g. telegram) is NOT rejected."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        # _create_adapter returns None here (no real telegram token wiring), so
        # the loop simply connects nothing — the key assertion is NO raise.
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: None)

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0  # nothing connected, but no MultiplexConfigError

    @pytest.mark.asyncio
    async def test_secondary_telegram_profile_reaches_active_topic_lookup(self, monkeypatch):
        """Production startup stamps the profile used by Telegram's pre-dispatch gate."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.session import build_session_key
        from plugins.platforms.telegram.adapter import TelegramAdapter

        profile_name = "reviewer"
        platform_config = PlatformConfig(enabled=True, token="secondary-token")
        adapter = TelegramAdapter(platform_config)

        class _SessionStore:
            config = SimpleNamespace(
                group_sessions_per_user=True,
                thread_sessions_per_user=False,
                multiplex_profiles=True,
            )

            def __init__(self):
                self._entries = set()

            def _ensure_loaded(self):
                return None

            def _generate_session_key(self, source):
                return build_session_key(
                    source,
                    group_sessions_per_user=self.config.group_sessions_per_user,
                    thread_sessions_per_user=self.config.thread_sessions_per_user,
                    profile=source.profile,
                )

        store = _SessionStore()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}
        runner.session_store = cast(Any, store)
        runner._busy_text_mode = "interrupt"
        runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
        runner._is_user_authorized = lambda source: True

        profile_cfg = GatewayConfig(multiplex_profiles=True)
        profile_cfg.platforms = {Platform.TELEGRAM: platform_config}
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)
        monkeypatch.setattr(runner, "_create_adapter", lambda _platform, _config: adapter)

        connected = await runner._start_one_profile_adapters(profile_name, Path("/tmp/x"), {})

        assert connected == 1
        assert adapter._session_profile == profile_name
        source = adapter.build_source(
            chat_id="-1001234567890",
            chat_type="group",
            user_id="111",
            thread_id="42",
        )
        assert source.profile == profile_name
        store._entries.add(store._generate_session_key(source))

        message = SimpleNamespace(
            chat=SimpleNamespace(
                id=-1001234567890,
                type="supergroup",
                title="Example Forum",
                is_forum=True,
            ),
            from_user=SimpleNamespace(id=111),
            message_thread_id=42,
            is_topic_message=True,
        )
        assert adapter._forum_topic_has_active_session(message) is True

    @pytest.mark.asyncio
    async def test_secondary_same_config_token_is_refused(self, monkeypatch):
        """Adapters that keep their token on config still trip the mux guard."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _ConfigTokenAdapter:
            def __init__(self, token):
                self.config = PlatformConfig(enabled=True, token=token)
                self.disconnected = False

            async def connect(self):
                raise AssertionError("duplicate adapter must not connect")

            async def disconnect(self):
                self.disconnected = True

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="same-token"),
        }
        duplicate = _ConfigTokenAdapter("same-token")
        claimed = {
            (
                Platform.TELEGRAM,
                GatewayRunner._adapter_credential_fingerprint(
                    _ConfigTokenAdapter("same-token")
                ),
            ): "default"
        }

        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: duplicate)
        monkeypatch.setattr(runner, "_adapter_disconnect_timeout_secs", lambda: 0)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", claimed
        )

        assert connected == 0
        assert duplicate.disconnected is True
        assert runner._profile_adapters["reviewer"] == {}

    def test_port_binding_set_covers_known_listeners(self):
        from gateway.run import _PORT_BINDING_PLATFORM_VALUES
        # Every adapter that binds a TCP port must be in the guard set.
        for p in ("webhook", "api_server", "msgraph_webhook", "feishu",
                  "wecom_callback", "bluebubbles", "sms"):
            assert p in _PORT_BINDING_PLATFORM_VALUES
