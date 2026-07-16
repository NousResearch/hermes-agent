"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
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


    def test_reads_config_token(self):
        """Adapters like Discord store token on `config`, not on self.

        Without the config-token fallback, every Discord adapter in a
        multiplexed gateway returns None here and the same-token conflict
        check is silently skipped — N adapters start polling the same bot
        token and race on every inbound message.
        """
        class _Config:
            token = "discord-bot-token"
        class _ConfigBackedAdapter:
            config = _Config()
        fp = GatewayRunner._adapter_credential_fingerprint(_ConfigBackedAdapter())
        assert fp is not None
        assert "discord-bot-token" not in fp
        assert len(fp) == 16

    def test_distinct_config_tokens_distinct_fp(self):
        class _CfgA:
            token = "tok-A"
        class _CfgB:
            token = "tok-B"
        class _A:
            config = _CfgA()
        class _B:
            config = _CfgB()
        a = GatewayRunner._adapter_credential_fingerprint(_A())
        b = GatewayRunner._adapter_credential_fingerprint(_B())
        assert a is not None and b is not None
        assert a != b

    def test_direct_token_takes_precedence_over_config(self):
        """If both `adapter.token` and `adapter.config.token` exist, direct wins."""
        class _Cfg:
            token = "from-config"
        class _Both:
            token = "from-direct"
            config = _Cfg()
        fp = GatewayRunner._adapter_credential_fingerprint(_Both())
        import hashlib
        expected = hashlib.sha256(b"hermes-mux:from-direct").hexdigest()[:16]
        assert fp == expected

    def test_config_without_token_returns_none(self):
        """config present but no token attribute → None (no false positive)."""
        class _Cfg:
            pass
        class _Adapter:
            config = _Cfg()
        assert GatewayRunner._adapter_credential_fingerprint(_Adapter()) is None


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
    async def test_multiplex_secondary_skips_relay_but_starts_direct_adapter(
        self, monkeypatch
    ):
        """Relay is process-shared; direct adapters remain per-profile."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _DirectAdapter:
            platform = Platform.TELEGRAM

            def set_message_handler(self, handler):
                self.message_handler = handler

            def set_fatal_error_handler(self, handler):
                self.fatal_error_handler = handler

            def set_session_store(self, store):
                self.session_store = store

            def set_busy_session_handler(self, handler):
                self.busy_session_handler = handler

            def set_topic_recovery_fn(self, handler):
                self.topic_recovery_fn = handler

            def set_authorization_check(self, handler):
                self.authorization_check = handler

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}
        runner.session_store = object()
        runner._handle_adapter_fatal_error = object()
        runner._handle_active_session_busy_message = object()
        runner._recover_telegram_topic_thread_id = object()
        runner._busy_text_mode = "queue"
        runner._make_adapter_auth_check = lambda platform: object()

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.RELAY: PlatformConfig(enabled=True),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="reviewer-token"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        direct = _DirectAdapter()
        factory_calls = []

        def _create_adapter(platform, config):
            factory_calls.append(platform)
            if platform is Platform.RELAY:
                raise AssertionError("secondary Relay factory must not be invoked")
            return direct

        connect_calls = []

        async def _connect(adapter, platform):
            connect_calls.append((adapter, platform))
            return True

        monkeypatch.setattr(runner, "_create_adapter", _create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", {}
        )

        assert connected == 1
        assert factory_calls == [Platform.TELEGRAM]
        assert connect_calls == [(direct, Platform.TELEGRAM)]
        assert runner._profile_adapters["reviewer"] == {
            Platform.TELEGRAM: direct,
        }

    @pytest.mark.asyncio
    async def test_non_multiplex_profile_adapter_start_keeps_relay(self, monkeypatch):
        """The Relay skip is gated to multiplex mode."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _RelayAdapter:
            platform = Platform.RELAY

            def set_message_handler(self, handler):
                pass

            def set_fatal_error_handler(self, handler):
                pass

            def set_session_store(self, store):
                pass

            def set_busy_session_handler(self, handler):
                pass

            def set_topic_recovery_fn(self, handler):
                pass

            def set_authorization_check(self, handler):
                pass

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=False)
        runner._profile_adapters = {}
        runner.session_store = object()
        runner._handle_adapter_fatal_error = object()
        runner._handle_active_session_busy_message = object()
        runner._recover_telegram_topic_thread_id = object()
        runner._busy_text_mode = "queue"
        runner._make_adapter_auth_check = lambda platform: object()

        profile_cfg = GatewayConfig(multiplex_profiles=False)
        profile_cfg.platforms = {
            Platform.RELAY: PlatformConfig(enabled=True),
        }
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)

        relay = _RelayAdapter()
        factory_calls = []
        connect_calls = []

        def _create_adapter(platform, config):
            factory_calls.append(platform)
            return relay

        async def _connect(adapter, platform):
            connect_calls.append((adapter, platform))
            return True

        monkeypatch.setattr(runner, "_create_adapter", _create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", {}
        )

        assert connected == 1
        assert factory_calls == [Platform.RELAY]
        assert connect_calls == [(relay, Platform.RELAY)]

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
