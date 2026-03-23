"""Tests for Kasia gateway integration."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource


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
        monkeypatch.setenv("KASIA_INDEXER_URLS", "https://indexer-a.example.com,https://indexer-b.example.com")
        monkeypatch.setenv("KASIA_NODE_WBORSH_URL", "ws://127.0.0.1:17110")
        monkeypatch.setenv("KASIA_NODE_WBORSH_URLS", "ws://127.0.0.1:17110,ws://127.0.0.1:17111")
        monkeypatch.setenv("KASIA_NETWORK", "mainnet")
        monkeypatch.setenv("KASIA_KNS_URL", "https://kns.example.com/api/v1")
        monkeypatch.setenv("KASIA_FEE_POLICY", "auto")
        monkeypatch.setenv("KASIA_BRIDGE_PORT", "3011")
        monkeypatch.setenv("KASIA_SEND_WAIT_MS", "7000")
        monkeypatch.setenv("KASIA_MAX_MULTIPARTS", "6")
        monkeypatch.setenv("KASIA_BROADCAST_SUBSCRIPTIONS", "news=kaspa:qpub1|kaspa:qpub2")
        monkeypatch.setenv("KASIA_ALLOWED_BROADCAST_CHANNELS", "news,alerts")
        monkeypatch.setenv("KASIA_HOME_CHANNEL", "kaspa:qhomeaddress")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.KASIA in config.platforms
        kasia_config = config.platforms[Platform.KASIA]
        assert kasia_config.enabled is True
        assert kasia_config.extra["seed_phrase"] == "seed words go here"
        assert kasia_config.extra["indexer_url"] == "https://indexer.example.com"
        assert kasia_config.extra["indexer_urls"] == [
            "https://indexer-a.example.com",
            "https://indexer-b.example.com",
        ]
        assert kasia_config.extra["node_wborsh_url"] == "ws://127.0.0.1:17110"
        assert kasia_config.extra["node_wborsh_urls"] == [
            "ws://127.0.0.1:17110",
            "ws://127.0.0.1:17111",
        ]
        assert kasia_config.extra["network"] == "mainnet"
        assert kasia_config.extra["kns_url"] == "https://kns.example.com/api/v1"
        assert kasia_config.extra["fee_policy"] == "auto"
        assert kasia_config.extra["bridge_port"] == 3011
        assert kasia_config.extra["send_wait_ms"] == 7000
        assert kasia_config.extra["max_multipart_parts"] == 6
        assert kasia_config.extra["broadcast_subscriptions"] == "news=kaspa:qpub1|kaspa:qpub2"
        assert kasia_config.extra["allowed_broadcast_channels"] == ["news", "alerts"]
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

    def test_load_kasia_settings_prefers_explicit_config_endpoints(self, monkeypatch):
        monkeypatch.setenv("KASIA_INDEXER_URLS", "https://env-a.example.com,https://env-b.example.com")
        monkeypatch.setenv("KASIA_NODE_WBORSH_URLS", "ws://env-a.example.com,ws://env-b.example.com")

        from gateway.kasia_config import load_kasia_settings

        settings = load_kasia_settings(
            extra={
                "indexer_url": "https://config.example.com",
                "node_wborsh_url": "ws://config.example.com:17110",
            }
        )

        assert settings.indexer_url == "https://config.example.com"
        assert settings.indexer_urls == ("https://config.example.com",)
        assert settings.node_wborsh_url == "ws://config.example.com:17110"
        assert settings.node_wborsh_urls == ("ws://config.example.com:17110",)

    def test_load_kasia_settings_exports_platform_and_bridge_settings(self):
        from gateway.kasia_config import load_kasia_settings

        settings = load_kasia_settings(
            extra={
                "seed_phrase": "seed words",
                "indexer_url": "https://indexer.example.com",
                "indexer_urls": ["https://indexer.example.com", "https://indexer-backup.example.com"],
                "node_wborsh_url": "ws://node.example.com:17110",
                "node_wborsh_urls": ["ws://node.example.com:17110", "ws://node-backup.example.com:17110"],
                "kns_url": "https://kns.example.com/api/v1",
                "fee_policy": "priority",
                "broadcast_subscriptions": "news=kaspa:qpub1;alerts=kaspa:qpub2",
                "allowed_broadcast_channels": ["alerts", "ops"],
                "allow_all_broadcast_channels": True,
                "max_multipart_parts": 6,
            }
        )

        assert settings.platform_extra()["allowed_broadcast_channels"] == ["alerts", "ops"]
        assert settings.bridge_env()["KASIA_ALLOWED_BROADCAST_CHANNELS"] == "alerts,ops"
        assert settings.bridge_env()["KASIA_BROADCAST_SUBSCRIPTIONS"] == "news=kaspa:qpub1;alerts=kaspa:qpub2"
        assert settings.bridge_env()["KASIA_ALLOW_ALL_BROADCAST_CHANNELS"] == "true"
        assert settings.bridge_env()["KASIA_MAX_MULTIPARTS"] == "6"

    def test_load_kasia_settings_warns_and_falls_back_for_invalid_ints(self, caplog):
        from gateway.kasia_config import (
            DEFAULT_KASIA_BRIDGE_PORT,
            DEFAULT_KASIA_SEND_WAIT_MS,
            load_kasia_settings,
        )
        import logging

        settings = load_kasia_settings(
            env={
                "KASIA_BRIDGE_PORT": "not-a-number",
                "KASIA_SEND_WAIT_MS": "still-not-a-number",
            },
            logger=logging.getLogger("tests.kasia"),
        )

        assert settings.bridge_port == DEFAULT_KASIA_BRIDGE_PORT
        assert settings.send_wait_ms == DEFAULT_KASIA_SEND_WAIT_MS
        assert "Invalid KASIA_BRIDGE_PORT" in caplog.text
        assert "Invalid KASIA_SEND_WAIT_MS" in caplog.text

    def test_configured_broadcast_channels_merge_subscriptions_and_allowlist(self):
        from gateway.kasia_config import load_kasia_settings

        settings = load_kasia_settings(
            extra={
                "broadcast_subscriptions": "news=kaspa:qpub1;alerts=kaspa:qpub2",
                "allowed_broadcast_channels": ["alerts", "ops"],
            }
        )

        assert settings.configured_broadcast_channels() == ["news", "alerts", "ops"]

    def test_kasia_authorization_uses_platform_and_global_allowlists(self):
        from gateway.kasia_config import is_kasia_address_authorized

        env = {
            "KASIA_ALLOWED_USERS": "kaspa:qpeeraddress",
            "GATEWAY_ALLOWED_USERS": "qglobalpeer",
        }

        assert is_kasia_address_authorized("qpeeraddress", env=env) is True
        assert is_kasia_address_authorized("kaspa:qglobalpeer", env=env) is True
        assert is_kasia_address_authorized("kaspa:qunknown", env=env) is False


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

    def test_check_requirements_missing_values(self, monkeypatch):
        from gateway.platforms.kasia import check_kasia_requirements

        for name in (
            "KASIA_SEED_PHRASE",
            "KASIA_INDEXER_URL",
            "KASIA_INDEXER_URLS",
            "KASIA_NODE_WBORSH_URL",
            "KASIA_NODE_WBORSH_URLS",
        ):
            monkeypatch.delenv(name, raising=False)

        config = PlatformConfig(enabled=True, extra={})
        assert check_kasia_requirements(config) is False


class TestKasiaGatewayRunnerBehavior:
    def test_create_adapter_builds_kasia_adapter(self):
        import gateway.platforms.kasia as kasia_platform
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(group_sessions_per_user=True)
        config = PlatformConfig(enabled=True, extra={})
        adapter_instance = object()

        with patch.object(kasia_platform, "check_kasia_requirements", return_value=True), patch.object(
            kasia_platform,
            "KasiaAdapter",
            return_value=adapter_instance,
        ) as adapter_cls:
            result = GatewayRunner._create_adapter(runner, Platform.KASIA, config)

        assert result is adapter_instance
        assert config.extra["group_sessions_per_user"] is True
        adapter_cls.assert_called_once_with(config)

    def test_create_adapter_bridges_unauthorized_dm_behavior_for_kasia(self):
        import gateway.platforms.kasia as kasia_platform
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(
            group_sessions_per_user=False,
            get_unauthorized_dm_behavior=lambda platform: "ignore",
        )
        config = PlatformConfig(enabled=True, extra={})
        adapter_instance = object()

        with patch.object(kasia_platform, "check_kasia_requirements", return_value=True), patch.object(
            kasia_platform,
            "KasiaAdapter",
            return_value=adapter_instance,
        ):
            result = GatewayRunner._create_adapter(runner, Platform.KASIA, config)

        assert result is adapter_instance
        assert config.extra["unauthorized_dm_behavior"] == "ignore"

    def test_create_adapter_normalizes_null_extra_for_kasia(self):
        import gateway.platforms.kasia as kasia_platform
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(
            group_sessions_per_user=False,
            get_unauthorized_dm_behavior=lambda platform: "ignore",
        )
        config = PlatformConfig(enabled=True, extra=None)
        adapter_instance = object()

        with patch.object(kasia_platform, "check_kasia_requirements", return_value=True), patch.object(
            kasia_platform,
            "KasiaAdapter",
            return_value=adapter_instance,
        ):
            result = GatewayRunner._create_adapter(runner, Platform.KASIA, config)

        assert result is adapter_instance
        assert config.extra == {
            "group_sessions_per_user": False,
            "unauthorized_dm_behavior": "ignore",
        }

    def test_create_adapter_returns_none_when_requirements_fail(self):
        import gateway.platforms.kasia as kasia_platform
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = SimpleNamespace(group_sessions_per_user=False)
        config = PlatformConfig(enabled=True, extra={})

        with patch.object(kasia_platform, "check_kasia_requirements", return_value=False):
            result = GatewayRunner._create_adapter(runner, Platform.KASIA, config)

        assert result is None

    def test_is_user_authorized_accepts_platform_allow_all_for_kasia(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *args: False)
        source = SessionSource(
            platform=Platform.KASIA,
            chat_id="kaspa:qpeeraddress",
            user_id="kaspa:qpeeraddress",
        )

        monkeypatch.setenv("KASIA_ALLOW_ALL_USERS", "true")

        assert GatewayRunner._is_user_authorized(runner, source) is True

    def test_is_user_authorized_accepts_kasia_allowlist(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *args: False)
        source = SessionSource(
            platform=Platform.KASIA,
            chat_id="kaspa:qpeeraddress",
            user_id="kaspa:qpeeraddress",
        )

        monkeypatch.setenv("KASIA_ALLOWED_USERS", "kaspa:qpeeraddress")
        monkeypatch.delenv("KASIA_ALLOW_ALL_USERS", raising=False)

        assert GatewayRunner._is_user_authorized(runner, source) is True

    def test_is_user_authorized_accepts_kasia_kns_allowlist(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *args: False)
        source = SessionSource(
            platform=Platform.KASIA,
            chat_id="kaspa:qpeeraddress",
            user_id="kaspa:qpeeraddress",
            user_name="peer.kas",
        )

        monkeypatch.setenv("KASIA_ALLOWED_USERS", "peer.kas")
        monkeypatch.delenv("KASIA_ALLOW_ALL_USERS", raising=False)

        with patch("gateway.kasia_identity.kasia_target_matches", return_value=True):
            assert GatewayRunner._is_user_authorized(runner, source) is True

    def test_is_user_authorized_accepts_matching_kasia_kns_without_lookup(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *args: False)
        source = SessionSource(
            platform=Platform.KASIA,
            chat_id="kaspa:qpeeraddress",
            user_id="kaspa:qpeeraddress",
            user_name="peer.kas",
        )

        monkeypatch.setenv("KASIA_ALLOWED_USERS", "peer.kas")
        monkeypatch.delenv("KASIA_ALLOW_ALL_USERS", raising=False)

        with patch(
            "gateway.kasia_identity.resolve_kasia_kns_name",
            side_effect=AssertionError("should not resolve"),
        ):
            assert GatewayRunner._is_user_authorized(runner, source) is True


class TestKasiaGatewaySetup:
    def test_setup_kasia_delegates_to_dedicated_command(self):
        import hermes_cli.gateway as gateway_cli

        with patch("hermes_cli.main.cmd_kasia") as kasia_cmd:
            gateway_cli._setup_kasia()

        kasia_cmd.assert_called_once()
        assert kasia_cmd.call_args.args[0].kasia_command == "setup"


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

    def test_allowlist_matches_bare_address_variant(self, monkeypatch):
        monkeypatch.setenv("KASIA_ALLOWED_USERS", "qpeeraddress")
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        assert adapter._is_address_authorized("kaspa:qpeeraddress") is True

    def test_allowlist_matches_kns_target(self, monkeypatch):
        monkeypatch.setenv("KASIA_ALLOWED_USERS", "peer.kas")
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        with patch("gateway.kasia_identity.resolve_kasia_kns_name", return_value="kaspa:qpeeraddress"):
            assert adapter._is_address_authorized("kaspa:qpeeraddress") is True

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
    async def test_handshake_request_triggers_pairing_flow_for_unauthorized_user(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter.handle_message = AsyncMock()
        adapter._request_json = AsyncMock(return_value={"status": "sent", "txId": "tx-2"})

        with patch.object(adapter, "_is_address_authorized", return_value=False):
            await adapter._handle_bridge_event(
                {
                    "eventType": "handshake_request",
                    "chatId": "kaspa:qpeeraddress",
                    "senderId": "kaspa:qpeeraddress",
                    "senderName": "peer.kas",
                    "messageId": "tx-handshake",
                }
            )

        adapter._request_json.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()
        forwarded_event = adapter.handle_message.await_args.args[0]
        assert forwarded_event.text == ""
        assert forwarded_event.message_id == "tx-handshake"
        assert forwarded_event.source.chat_id == "kaspa:qpeeraddress"
        assert forwarded_event.source.user_id == "kaspa:qpeeraddress"
        assert forwarded_event.source.user_name == "peer.kas"

    def test_authorization_handler_controls_kasia_address_authorization(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter.set_authorization_handler(lambda source: source.user_id == "kaspa:qpeeraddress")

        assert adapter._is_address_authorized("kaspa:qpeeraddress", user_name="peer.kas") is True
        assert adapter._is_address_authorized("kaspa:qunknownaddress", user_name="other.kas") is False

    @pytest.mark.asyncio
    async def test_handshake_request_still_ignores_unauthorized_user_when_behavior_is_ignore(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config(unauthorized_dm_behavior="ignore"))
        adapter.handle_message = AsyncMock()
        adapter._request_json = AsyncMock(return_value={"status": "sent", "txId": "tx-2"})

        with patch.object(adapter, "_is_address_authorized", return_value=False):
            await adapter._handle_bridge_event(
                {
                    "eventType": "handshake_request",
                    "chatId": "kaspa:qpeeraddress",
                    "senderId": "kaspa:qpeeraddress",
                }
            )

        adapter._request_json.assert_not_awaited()
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adapter_can_initiate_handshake_explicitly(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter._mark_connected()
        adapter._request_json = AsyncMock(
            return_value={"status": "sent", "txId": "tx-init", "chatId": "kaspa:qpeeraddress"}
        )

        with patch.object(adapter, "_is_address_authorized", return_value=True):
            result = await adapter.initiate_handshake("kaspa:qpeeraddress", display_name="Peer")

        assert result["txId"] == "tx-init"
        adapter._request_json.assert_awaited_once_with(
            "POST",
            "/handshakes/initiate",
            payload={"chatId": "kaspa:qpeeraddress", "displayName": "Peer", "retry": False},
            total=30,
        )

    @pytest.mark.asyncio
    async def test_broadcast_event_routes_into_channel_message(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter.handle_message = AsyncMock()

        await adapter._handle_bridge_event(
            {
                "eventType": "broadcast",
                "messageId": "tx-bcast-1",
                "chatId": "broadcast:news",
                "channelName": "news",
                "senderId": "kaspa:qpublisher",
                "senderName": "kaspa:qpublisher",
                "body": "announcement",
                "timestampMs": 1710000000000,
            }
        )

        adapter.handle_message.assert_awaited_once()
        forwarded_event = adapter.handle_message.await_args.args[0]
        assert forwarded_event.source.chat_type == "channel"
        assert forwarded_event.source.chat_name == "#news"
        assert forwarded_event.text == "announcement"

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
    async def test_send_prefers_job_id_for_submitted_bridge_job(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter._mark_connected()
        adapter._request_json = AsyncMock(
            return_value={
                "status": "submitted",
                "jobId": "job-789",
                "txId": "tx-789",
                "statusMessage": "Submitted to the Kaspa node. Waiting for indexer visibility.",
                "partCount": 1,
                "completedParts": 1,
                "indexedParts": 0,
                "submittedMs": 1710000000100,
            }
        )

        result = await adapter.send("kaspa:qpeeraddress", "hello")

        assert result.success is True
        assert result.message_id == "job-789"
        assert result.raw_response["status"] == "submitted"
        assert "Waiting for indexer visibility" in result.raw_response["statusMessage"]

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
    async def test_connect_refuses_busy_port_after_cleanup_attempt(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        adapter = KasiaAdapter(self._make_config())
        adapter._bridge_script = bridge_script

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._kill_port_process"
        ) as kill_port_process, patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=True
        ), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ), patch("gateway.platforms.kasia.subprocess.Popen") as popen_mock:
            result = await adapter.connect()

        assert result is False
        kill_port_process.assert_called_once_with(adapter._bridge_port)
        popen_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_kills_orphaned_bridge_before_starting(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        adapter = KasiaAdapter(self._make_config())
        adapter._bridge_script = bridge_script
        adapter._request_json = AsyncMock(return_value={"status": "connected"})

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        def fake_create_task(coro):
            coro.close()
            return SimpleNamespace(cancel=lambda: None)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._kill_port_process"
        ) as kill_port_process, patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch(
            "gateway.platforms.kasia.subprocess.Popen", side_effect=fake_popen
        ) as popen_mock, patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ), patch(
            "gateway.platforms.kasia.asyncio.create_task", side_effect=fake_create_task
        ):
            result = await adapter.connect()

        assert result is True
        kill_port_process.assert_called_once_with(adapter._bridge_port)
        popen_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_kills_orphaned_bridge_process_on_port(self):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter._bridge_process = MagicMock()
        adapter._bridge_process.pid = 1234
        adapter._bridge_process.poll.return_value = 0

        with patch("gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()), patch(
            "gateway.platforms.kasia._kill_port_process"
        ) as kill_port_process, patch.object(
            adapter, "_close_bridge_log"
        ) as close_bridge_log, patch.object(
            adapter, "_mark_disconnected"
        ) as mark_disconnected:
            await adapter.disconnect()

        kill_port_process.assert_called_once_with(adapter._bridge_port)
        close_bridge_log.assert_called_once()
        mark_disconnected.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_passes_fee_policy_to_bridge_environment(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        adapter = KasiaAdapter(self._make_config(fee_policy="normal"))
        adapter._bridge_script = bridge_script
        adapter._request_json = AsyncMock(return_value={"status": "connected"})

        popen_calls = {}

        def fake_popen(*args, **kwargs):
            popen_calls["env"] = kwargs.get("env", {}).copy()
            return SimpleNamespace(poll=lambda: None, returncode=None)

        def fake_create_task(coro):
            coro.close()
            return SimpleNamespace(cancel=lambda: None)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch("gateway.platforms.kasia.subprocess.Popen", side_effect=fake_popen), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ), patch("gateway.platforms.kasia.asyncio.create_task", side_effect=fake_create_task):
            result = await adapter.connect()

        assert result is True
        assert popen_calls["env"]["KASIA_FEE_POLICY"] == "normal"
        assert "KASIA_KNS_URL" not in popen_calls["env"]

    @pytest.mark.asyncio
    async def test_connect_warns_when_wallet_funding_is_low(self, tmp_path, caplog):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        adapter = KasiaAdapter(self._make_config())
        adapter._bridge_script = bridge_script
        adapter._request_json = AsyncMock(
            return_value={
                "status": "connected",
                "walletAddress": "kaspa:qwallet",
                "walletFundingState": "low",
                "walletBalanceSompi": "27881431",
                "availableMatureBalanceSompi": "27881431",
                "recommendedMinBalanceSompi": "40000000",
            }
        )

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        def fake_create_task(coro):
            coro.close()
            return SimpleNamespace(cancel=lambda: None)

        with caplog.at_level(logging.WARNING), patch(
            "gateway.platforms.kasia.check_kasia_requirements",
            return_value=True,
        ), patch(
            "gateway.platforms.kasia._is_local_port_in_use",
            return_value=False,
        ), patch(
            "gateway.platforms.kasia.subprocess.Popen",
            side_effect=fake_popen,
        ), patch(
            "gateway.platforms.kasia.asyncio.sleep",
            new=AsyncMock(),
        ), patch(
            "gateway.platforms.kasia.asyncio.create_task",
            side_effect=fake_create_task,
        ):
            result = await adapter.connect()

        assert result is True
        assert "Kasia wallet kaspa:qwallet is low on funds" in caplog.text
        assert "recommended >= 0.40000000 KAS" in caplog.text

    @pytest.mark.asyncio
    async def test_connect_replays_pending_handshakes_when_wallet_is_ready(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(
            json.dumps(
                {
                    "conversations": {
                        "kaspa:qpeeraddress": {
                            "peer_address": "kaspa:qpeeraddress",
                            "display_name": "peer.kas",
                            "pending_handshake": {
                                "tx_id": "tx-handshake",
                                "block_time": 1710000000000,
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        adapter = KasiaAdapter(self._make_config(state_dir=str(state_dir)))
        adapter._bridge_script = bridge_script
        adapter._poll_messages = AsyncMock(return_value=None)
        adapter._request_json = AsyncMock(
            return_value={
                "status": "connected",
                "walletFundingState": "ready",
            }
        )
        adapter._handle_bridge_event = AsyncMock()

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch("gateway.platforms.kasia.subprocess.Popen", side_effect=fake_popen), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ):
            result = await adapter.connect()

        assert result is True
        await asyncio.sleep(0)
        adapter._handle_bridge_event.assert_awaited_once()
        replayed_event = adapter._handle_bridge_event.await_args.args[0]
        assert replayed_event["eventType"] == "handshake_request"
        assert replayed_event["messageId"] == "tx-handshake"
        assert replayed_event["chatId"] == "kaspa:qpeeraddress"
        assert replayed_event["senderName"] == "peer.kas"

    @pytest.mark.asyncio
    async def test_connect_defers_pending_handshakes_when_wallet_funding_is_low(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(
            json.dumps(
                {
                    "conversations": {
                        "kaspa:qpeeraddress": {
                            "peer_address": "kaspa:qpeeraddress",
                            "pending_handshake": {
                                "tx_id": "tx-handshake",
                                "block_time": 1710000000000,
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        adapter = KasiaAdapter(self._make_config(state_dir=str(state_dir)))
        adapter._bridge_script = bridge_script
        adapter._poll_messages = AsyncMock(return_value=None)
        adapter._request_json = AsyncMock(
            return_value={
                "status": "connected",
                "walletFundingState": "low",
            }
        )
        adapter._handle_bridge_event = AsyncMock()

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch("gateway.platforms.kasia.subprocess.Popen", side_effect=fake_popen), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ):
            result = await adapter.connect()

        assert result is True
        await asyncio.sleep(0)
        adapter._handle_bridge_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_bootstraps_configured_handshakes_for_ready_wallet(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        monkeypatch.setenv("KASIA_ALLOWED_USERS", "qpeeraddress,kaspa:qsecondpeer")
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "kaspa:qglobalpeer,123456")

        config = self._make_config()
        config.home_channel = SimpleNamespace(
            platform=Platform.KASIA,
            chat_id="friend.kas",
            name="Home",
        )
        adapter = KasiaAdapter(config)
        adapter._bridge_script = bridge_script
        adapter._poll_messages = AsyncMock(return_value=None)
        initiated_chat_ids: list[str] = []
        real_sleep = asyncio.sleep

        async def fake_request_json(method, path, payload=None, total=None):
            if method == "GET" and path == "/health":
                return {
                    "status": "connected",
                    "walletFundingState": "ready",
                }
            if method == "POST" and path == "/handshakes/initiate":
                initiated_chat_ids.append(payload["chatId"])
                await real_sleep(0)
                return {
                    "status": "sent",
                    "chatId": payload["chatId"],
                    "txId": f"tx-{len(initiated_chat_ids)}",
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

        adapter._request_json = AsyncMock(side_effect=fake_request_json)

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch("gateway.platforms.kasia.subprocess.Popen", side_effect=fake_popen), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ):
            result = await adapter.connect()

        assert result is True
        for _ in range(5):
            await asyncio.sleep(0)

        assert initiated_chat_ids == [
            "friend.kas",
            "kaspa:qpeeraddress",
            "kaspa:qsecondpeer",
            "kaspa:qglobalpeer",
        ]

    @pytest.mark.asyncio
    async def test_connect_skips_configured_handshake_bootstrap_when_wallet_funding_is_low(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        monkeypatch.setenv("KASIA_ALLOWED_USERS", "kaspa:qpeeraddress")

        config = self._make_config()
        config.home_channel = SimpleNamespace(
            platform=Platform.KASIA,
            chat_id="kaspa:qhomeaddress",
            name="Home",
        )
        adapter = KasiaAdapter(config)
        adapter._bridge_script = bridge_script
        adapter._poll_messages = AsyncMock(return_value=None)
        initiated_chat_ids: list[str] = []
        real_sleep = asyncio.sleep

        async def fake_request_json(method, path, payload=None, total=None):
            if method == "GET" and path == "/health":
                return {
                    "status": "connected",
                    "walletFundingState": "low",
                }
            if method == "POST" and path == "/handshakes/initiate":
                initiated_chat_ids.append(payload["chatId"])
                await real_sleep(0)
                return {
                    "status": "sent",
                    "chatId": payload["chatId"],
                    "txId": "tx-init",
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

        adapter._request_json = AsyncMock(side_effect=fake_request_json)

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch("gateway.platforms.kasia.subprocess.Popen", side_effect=fake_popen), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ):
            result = await adapter.connect()

        assert result is True
        for _ in range(5):
            await asyncio.sleep(0)

        assert initiated_chat_ids == []

    @pytest.mark.asyncio
    async def test_connect_bootstraps_configured_handshakes_for_kns_allowlist(
        self, tmp_path, monkeypatch
    ):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        monkeypatch.setenv("KASIA_ALLOWED_USERS", "peer.kas")

        config = self._make_config()
        adapter = KasiaAdapter(config)
        adapter._bridge_script = bridge_script
        adapter._poll_messages = AsyncMock(return_value=None)
        initiated_chat_ids: list[str] = []
        real_sleep = asyncio.sleep

        async def fake_request_json(method, path, payload=None, total=None):
            if method == "GET" and path == "/health":
                return {
                    "status": "connected",
                    "walletFundingState": "ready",
                }
            if method == "POST" and path == "/handshakes/initiate":
                initiated_chat_ids.append(payload["chatId"])
                await real_sleep(0)
                return {
                    "status": "sent",
                    "chatId": payload["chatId"],
                    "txId": "tx-kns",
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

        adapter._request_json = AsyncMock(side_effect=fake_request_json)

        with patch("gateway.platforms.kasia.check_kasia_requirements", return_value=True), patch(
            "gateway.platforms.kasia._is_local_port_in_use", return_value=False
        ), patch("gateway.platforms.kasia.subprocess.Popen", return_value=SimpleNamespace(poll=lambda: None, returncode=None)), patch(
            "gateway.platforms.kasia.asyncio.sleep", new=AsyncMock()
        ), patch(
            "gateway.kasia_identity.resolve_kasia_kns_name",
            side_effect=AssertionError("should not resolve"),
        ):
            result = await adapter.connect()

        assert result is True
        for _ in range(5):
            await asyncio.sleep(0)

        assert initiated_chat_ids == ["peer.kas"]

    @pytest.mark.asyncio
    async def test_connect_replays_pending_handshakes_after_runner_registers_adapter(self, tmp_path):
        from gateway.platforms.kasia import KasiaAdapter

        bridge_dir = tmp_path / "kasia-bridge"
        bridge_dir.mkdir()
        bridge_script = bridge_dir / "bridge.js"
        bridge_script.write_text("// test bridge\n", encoding="utf-8")
        (bridge_dir / "node_modules").mkdir()

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        (state_dir / "state.json").write_text(
            json.dumps(
                {
                    "conversations": {
                        "kaspa:qpeeraddress1": {
                            "peer_address": "kaspa:qpeeraddress1",
                            "display_name": "peer-1.kas",
                            "pending_handshake": {
                                "tx_id": "tx-handshake-1",
                                "block_time": 1710000000000,
                            },
                        },
                        "kaspa:qpeeraddress2": {
                            "peer_address": "kaspa:qpeeraddress2",
                            "display_name": "peer-2.kas",
                            "pending_handshake": {
                                "tx_id": "tx-handshake-2",
                                "block_time": 1710000001000,
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        adapter = KasiaAdapter(
            self._make_config(
                state_dir=str(state_dir),
                unauthorized_dm_behavior="pair",
            )
        )
        adapter._bridge_script = bridge_script
        adapter._poll_messages = AsyncMock(return_value=None)
        real_sleep = asyncio.sleep

        async def fake_request_json(method, path, payload=None, total=None):
            if method == "GET" and path == "/health":
                return {
                    "status": "connected",
                    "walletFundingState": "ready",
                }
            if method == "POST" and path == "/handshakes/respond":
                await real_sleep(0)
                return {
                    "status": "sent",
                    "txId": f"reply-{payload['chatId']}",
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

        adapter._request_json = AsyncMock(side_effect=fake_request_json)

        registered_during_handler: list[bool] = []
        runner = SimpleNamespace(adapters={})

        async def fake_message_handler(event):
            registered_during_handler.append(
                runner.adapters.get(event.source.platform) is adapter
            )
            return None

        adapter.set_message_handler(fake_message_handler)

        def fake_popen(*args, **kwargs):
            return SimpleNamespace(poll=lambda: None, returncode=None)

        with patch.object(adapter, "_is_address_authorized", return_value=False), patch(
            "gateway.platforms.kasia.check_kasia_requirements",
            return_value=True,
        ), patch(
            "gateway.platforms.kasia._is_local_port_in_use",
            return_value=False,
        ), patch(
            "gateway.platforms.kasia.subprocess.Popen",
            side_effect=fake_popen,
        ), patch(
            "gateway.platforms.kasia.asyncio.sleep",
            new=AsyncMock(),
        ):
            result = await adapter.connect()

        assert result is True
        runner.adapters[Platform.KASIA] = adapter
        for _ in range(5):
            await asyncio.sleep(0)

        assert registered_during_handler == [True, True]

    @pytest.mark.asyncio
    async def test_unauthorized_handshake_no_longer_spends_or_logs_wallet_warning(self, caplog):
        from gateway.platforms.kasia import KasiaAdapter

        adapter = KasiaAdapter(self._make_config())
        adapter.handle_message = AsyncMock()

        async def fake_request_json(method, path, payload=None, total=None):
            if method == "POST" and path == "/handshakes/respond":
                raise RuntimeError(
                    'Kasia bridge error (500) on /handshakes/respond: {"error":"Insufficient funds after fee"}'
                )
            if method == "GET" and path == "/health":
                return {
                    "walletAddress": "kaspa:qwallet",
                    "walletFundingState": "low",
                    "walletBalanceSompi": "27881431",
                    "availableMatureBalanceSompi": "27881431",
                    "recommendedMinBalanceSompi": "40000000",
                }
            raise AssertionError(f"Unexpected request: {method} {path}")

        adapter._request_json = AsyncMock(side_effect=fake_request_json)

        with caplog.at_level(logging.WARNING), patch.object(
            adapter,
            "_is_address_authorized",
            return_value=False,
        ):
            await adapter._handle_bridge_event(
                {
                    "eventType": "handshake_request",
                    "chatId": "kaspa:qpeeraddress",
                    "senderId": "kaspa:qpeeraddress",
                }
            )

        adapter._request_json.assert_not_awaited()
        adapter.handle_message.assert_awaited_once()
        assert "Kasia wallet kaspa:qwallet is low on funds" not in caplog.text
