"""Tests for Kasia gateway integration."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


class TestKasiaPlatformEnum:
    def test_kasia_enum_exists(self):
        assert Platform.KASIA.value == "kasia"

    def test_kasia_in_platform_list(self):
        assert "kasia" in [platform.value for platform in Platform]


class TestKasiaConfigLoading:
    def test_apply_env_overrides_kasia(self, monkeypatch):
        monkeypatch.setenv("KASIA_ENABLED", "true")
        monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
        monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
        monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "ws://127.0.0.1:17110")
        monkeypatch.setenv("KASIA_NETWORK", "mainnet")
        monkeypatch.setenv("KASIA_BRIDGE_PORT", "3011")
        monkeypatch.setenv("KASIA_SEND_WAIT_MS", "7000")
        monkeypatch.setenv("KASIA_MAX_MULTIPARTS", "6")
        monkeypatch.setenv("KASIA_HOME_CHANNEL", "kaspa:qhomeaddress")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.KASIA in config.platforms
        kasia_config = config.platforms[Platform.KASIA]
        assert kasia_config.enabled is True
        assert kasia_config.extra["seed_phrase"] == "seed words go here"
        assert kasia_config.extra["indexer_url"] == "https://indexer.example.com"
        assert kasia_config.extra["node_wborsh_url"] == "ws://127.0.0.1:17110"
        assert kasia_config.extra["network"] == "mainnet"
        assert kasia_config.extra["bridge_port"] == 3011
        assert kasia_config.extra["send_wait_ms"] == 7000
        assert kasia_config.extra["max_multipart_parts"] == 6
        assert kasia_config.home_channel.chat_id == "kaspa:qhomeaddress"

    def test_connected_platforms_includes_kasia(self, monkeypatch):
        monkeypatch.setenv("KASIA_ENABLED", "true")
        monkeypatch.setenv("KASIA_SEED_PHRASE", "seed words go here")
        monkeypatch.setenv("KASIA_INDEXER_URL", "https://indexer.example.com")
        monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "ws://127.0.0.1:17110")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.KASIA in config.get_connected_platforms()


class TestKasiaRequirements:
    def test_check_requirements(self):
        from gateway.platforms.kasia import check_kasia_requirements

        config = PlatformConfig(
            enabled=True,
            extra={
                "seed_phrase": "seed words",
                "indexer_url": "https://indexer.example.com",
                "node_wborsh_url": "ws://127.0.0.1:17110",
            },
        )

        with patch("gateway.platforms.kasia.subprocess.run") as run_mock:
            run_mock.return_value = SimpleNamespace(returncode=0)
            assert check_kasia_requirements(config) is True

    def test_check_requirements_missing_values(self):
        from gateway.platforms.kasia import check_kasia_requirements

        config = PlatformConfig(enabled=True, extra={})
        assert check_kasia_requirements(config) is False


class TestKasiaAuthorizationMaps:
    def test_kasia_in_adapter_factory(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._create_adapter)
        assert "Platform.KASIA" in source

    def test_kasia_in_allowed_users_map(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        assert "KASIA_ALLOWED_USERS" in source

    def test_kasia_in_allow_all_map(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        assert "KASIA_ALLOW_ALL_USERS" in source


class TestKasiaGatewaySetup:
    def test_setup_kasia_hides_existing_seed_default(self):
        import hermes_cli.gateway as gateway_cli

        env_values = {
            "KASIA_SEED_PHRASE": "existing twelve words",
            "KASIA_INDEXER_URL": "https://indexer.example.com",
            "KASIA_NODE_WBORSH_URL": "ws://127.0.0.1:17110",
            "KASIA_NETWORK": "mainnet",
        }
        prompt_calls = []
        saved = {}

        def fake_get_env(name):
            return env_values.get(name, "")

        def fake_prompt(question, default=None, password=False):
            prompt_calls.append((question, default, password))
            if password:
                return ""
            return default or ""

        def fake_save_env(name, value):
            saved[name] = value

        with patch.object(gateway_cli, "get_env_value", side_effect=fake_get_env), patch.object(
            gateway_cli, "prompt", side_effect=fake_prompt
        ), patch.object(gateway_cli, "save_env_value", side_effect=fake_save_env), patch(
            "builtins.input", side_effect=["kaspa:qpeeraddress", "kaspa:qhomeaddress"]
        ):
            gateway_cli._setup_kasia()

        assert prompt_calls[0] == (
            "  Seed phrase (leave blank to keep current value)",
            None,
            True,
        )
        assert saved["KASIA_SEED_PHRASE"] == "existing twelve words"

    def test_setup_kasia_clears_open_access_when_allowlist_is_saved(self):
        import hermes_cli.gateway as gateway_cli

        saved = {}

        def fake_save_env(name, value):
            saved[name] = value

        prompt_values = iter(
            [
                "seed words go here",
                "https://indexer.example.com",
                "ws://127.0.0.1:17110",
                "mainnet",
            ]
        )

        with patch.object(gateway_cli, "get_env_value", return_value=""), patch.object(
            gateway_cli, "prompt", side_effect=lambda *args, **kwargs: next(prompt_values)
        ), patch.object(gateway_cli, "save_env_value", side_effect=fake_save_env), patch(
            "builtins.input", side_effect=["kaspa:qpeeraddress", "kaspa:qhomeaddress"]
        ):
            gateway_cli._setup_kasia()

        assert saved["KASIA_ALLOWED_USERS"] == "kaspa:qpeeraddress"
        assert saved["KASIA_ALLOW_ALL_USERS"] == ""


class TestKasiaAdapter:
    def _make_config(self, **extra):
        return PlatformConfig(
            enabled=True,
            extra={
                "seed_phrase": "seed words",
                "indexer_url": "https://indexer.example.com",
                "node_wborsh_url": "ws://127.0.0.1:17110",
                "network": "mainnet",
                **extra,
            },
        )

    def test_allow_all_authorizes_any_address(self, monkeypatch):
        monkeypatch.setenv("KASIA_ALLOW_ALL_USERS", "true")
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        assert adapter._is_address_authorized("kaspa:qpeeraddress") is True

    def test_allowlist_matches_full_address(self, monkeypatch):
        monkeypatch.setenv(
            "KASIA_ALLOWED_USERS",
            "kaspa:qpeeraddress,kaspa:qotheraddress",
        )
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        assert adapter._is_address_authorized("kaspa:qpeeraddress") is True
        assert adapter._is_address_authorized("kaspa:qunknownaddress") is False

    def test_build_message_event(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        event = adapter._build_message_event(
            {
                "messageId": "tx-1",
                "chatId": "kaspa:qpeeraddress",
                "senderId": "kaspa:qpeeraddress",
                "senderName": "kaspa:qpeeraddress",
                "body": "hello from kasia",
                "timestampMs": 1710000000000,
            }
        )

        assert event is not None
        assert event.text == "hello from kasia"
        assert event.source.platform == Platform.KASIA
        assert event.source.chat_id == "kaspa:qpeeraddress"
        assert event.source.user_id == "kaspa:qpeeraddress"

    @pytest.mark.asyncio
    async def test_handshake_request_auto_responds_for_authorized_user(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter.handle_message = AsyncMock()
        adapter._request_json = AsyncMock(return_value={"status": "sent", "txId": "tx-2"})

        with patch.object(adapter, "_is_address_authorized", return_value=True):
            await adapter._handle_bridge_event(
                {
                    "eventType": "handshake_request",
                    "chatId": "kaspa:qpeeraddress",
                    "senderId": "kaspa:qpeeraddress",
                }
            )

        adapter._request_json.assert_awaited_once_with(
            "POST",
            "/handshakes/respond",
            payload={"chatId": "kaspa:qpeeraddress"},
            total=30,
        )
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_message_event_routes_into_handle_message(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter.handle_message = AsyncMock()

        await adapter._handle_bridge_event(
            {
                "eventType": "message",
                "messageId": "tx-3",
                "chatId": "kaspa:qpeeraddress",
                "senderId": "kaspa:qpeeraddress",
                "senderName": "kaspa:qpeeraddress",
                "body": "hello",
                "timestampMs": 1710000000000,
            }
        )

        adapter.handle_message.assert_awaited_once()
        forwarded_event = adapter.handle_message.await_args.args[0]
        assert forwarded_event.text == "hello"

    @pytest.mark.asyncio
    async def test_send_uses_bridge_txid(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter._mark_connected()
        adapter._request_json = AsyncMock(
            return_value={
                "status": "sent",
                "txId": "abc123",
                "wallet": {"usedPendingInput": True, "inputCount": 1},
            }
        )

        result = await adapter.send("kaspa:qpeeraddress", "hello")

        assert result.success is True
        assert result.message_id == "abc123"
        assert result.raw_response["wallet"]["usedPendingInput"] is True

    @pytest.mark.asyncio
    async def test_send_accepts_queued_bridge_job(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter._mark_connected()
        adapter._request_json = AsyncMock(
            return_value={
                "status": "queued",
                "jobId": "job-123",
                "partCount": 3,
                "completedParts": 0,
            }
        )

        result = await adapter.send("kaspa:qpeeraddress", "hello")

        assert result.success is True
        assert result.message_id == "job-123"
        assert result.raw_response["status"] == "queued"

    @pytest.mark.asyncio
    async def test_send_surfaces_rejected_bridge_job(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter._mark_connected()
        adapter._request_json = AsyncMock(
            return_value={
                "status": "rejected",
                "jobId": "job-456",
                "error": "Message is too long for Kasia delivery.",
            }
        )

        result = await adapter.send("kaspa:qpeeraddress", "A" * 450)

        assert result.success is False
        assert "too long" in result.error
        adapter._request_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_refuses_busy_port_without_killing_other_processes(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        adapter = KasiaAdapter(self._make_config())
        adapter._bridge_script = bridge_script

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=True
        ), patch("gateway.platforms.kasia.subprocess.Popen") as popen_mock:
            result = await adapter.connect()

        assert result is False
        popen_mock.assert_not_called()
