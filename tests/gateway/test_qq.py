"""Tests for native QQ gateway platform support."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult


class TestQQPlatformEnum:
    def test_qq_enum_exists(self):
        assert Platform.QQ.value == "qq"

    def test_qq_in_platform_list(self):
        assert "qq" in [platform.value for platform in Platform]


class TestQQConfigLoading:
    def test_apply_env_overrides_qq(self, monkeypatch):
        monkeypatch.setenv("QQ_BOT_APP_ID", "1024")
        monkeypatch.setenv("QQ_BOT_SECRET", "secret-xyz")
        monkeypatch.setenv("QQ_HOME_CHANNEL", "group:home")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.QQ in config.platforms
        qq_config = config.platforms[Platform.QQ]
        assert qq_config.enabled is True
        assert qq_config.token == "1024"
        assert qq_config.api_key == "secret-xyz"
        assert qq_config.home_channel.chat_id == "group:home"

    def test_qq_not_loaded_without_secret(self, monkeypatch):
        monkeypatch.setenv("QQ_BOT_APP_ID", "1024")
        monkeypatch.delenv("QQ_BOT_SECRET", raising=False)

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.QQ not in config.platforms

    def test_connected_platforms_includes_qq(self, monkeypatch):
        monkeypatch.setenv("QQ_BOT_APP_ID", "1024")
        monkeypatch.setenv("QQ_BOT_SECRET", "secret-xyz")

        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.QQ in config.get_connected_platforms()


class TestQQAdapterHelpers:
    def test_check_requirements_reflects_sdk_availability(self, monkeypatch):
        from gateway.platforms import qq as qq_mod

        monkeypatch.setattr(qq_mod, "QQ_AVAILABLE", True)
        assert qq_mod.check_qq_requirements() is True

        monkeypatch.setattr(qq_mod, "QQ_AVAILABLE", False)
        assert qq_mod.check_qq_requirements() is False

    @pytest.mark.asyncio
    async def test_send_routes_group_targets_via_group_api(self):
        from gateway.platforms.qq import QQAdapter

        adapter = QQAdapter(PlatformConfig(enabled=True, token="1024", api_key="secret"))
        adapter._client = SimpleNamespace(api=SimpleNamespace(
            post_group_message=AsyncMock(return_value={"id": "group-msg-1"}),
            post_c2c_message=AsyncMock(),
        ))

        result = await adapter.send("group:group-openid", "hello QQ")

        assert isinstance(result, SendResult)
        assert result.success is True
        adapter._client.api.post_group_message.assert_awaited_once_with(
            group_openid="group-openid",
            msg_type=0,
            content="hello QQ",
            msg_id=None,
        )
        adapter._client.api.post_c2c_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_routes_dm_targets_via_c2c_api(self):
        from gateway.platforms.qq import QQAdapter

        adapter = QQAdapter(PlatformConfig(enabled=True, token="1024", api_key="secret"))
        adapter._client = SimpleNamespace(api=SimpleNamespace(
            post_group_message=AsyncMock(),
            post_c2c_message=AsyncMock(return_value={"id": "dm-msg-1"}),
        ))

        result = await adapter.send("user:user-openid", "hello DM")

        assert result.success is True
        adapter._client.api.post_c2c_message.assert_awaited_once_with(
            openid="user-openid",
            msg_type=0,
            content="hello DM",
            msg_id=None,
        )
        adapter._client.api.post_group_message.assert_not_called()

    def test_normalize_group_message_event(self):
        from gateway.platforms.qq import QQAdapter

        adapter = QQAdapter(PlatformConfig(enabled=True, token="1024", api_key="secret"))
        raw_message = SimpleNamespace(
            id="m-1",
            content="hi from group",
            group_openid="group-openid",
            author=SimpleNamespace(id="member-openid", username="Alice"),
        )

        event = adapter._event_from_group_message(raw_message)

        assert event.text == "hi from group"
        assert event.message_id == "m-1"
        assert event.source.platform == Platform.QQ
        assert event.source.chat_id == "group:group-openid"
        assert event.source.chat_type == "group"
        assert event.source.user_id == "member-openid"
        assert event.source.user_name == "Alice"

    def test_normalize_c2c_message_event(self):
        from gateway.platforms.qq import QQAdapter

        adapter = QQAdapter(PlatformConfig(enabled=True, token="1024", api_key="secret"))
        raw_message = SimpleNamespace(
            id="m-2",
            content="hi from dm",
            author=SimpleNamespace(id="user-openid", username="Bob"),
        )

        event = adapter._event_from_c2c_message(raw_message)

        assert event.text == "hi from dm"
        assert event.message_id == "m-2"
        assert event.source.platform == Platform.QQ
        assert event.source.chat_id == "user:user-openid"
        assert event.source.chat_type == "dm"
        assert event.source.user_id == "user-openid"
        assert event.source.user_name == "Bob"


class TestQQIntegrationPoints:
    def test_qq_in_adapter_factory(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._create_adapter)
        assert "Platform.QQ" in source

    def test_qq_in_auth_maps(self):
        import gateway.run

        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        assert "QQ_ALLOWED_USERS" in source
        assert "QQ_ALLOW_ALL_USERS" in source

    def test_qq_in_send_message_tool(self):
        import tools.send_message_tool as send_message_tool

        handle_send_src = inspect.getsource(send_message_tool._handle_send)
        route_src = inspect.getsource(send_message_tool._send_to_platform)
        assert '"qq"' in handle_send_src
        assert "Platform.QQ" in route_src

    def test_qq_in_cron_delivery_map(self):
        import cron.scheduler

        source = inspect.getsource(cron.scheduler)
        assert '"qq"' in source

    def test_qq_in_toolsets(self):
        from toolsets import TOOLSETS

        assert "hermes-qq" in TOOLSETS
        assert "hermes-qq" in TOOLSETS["hermes-gateway"]["includes"]

    def test_qq_in_platform_hints(self):
        from agent.prompt_builder import PLATFORM_HINTS

        assert "qq" in PLATFORM_HINTS
        assert "qq" in PLATFORM_HINTS["qq"].lower()

    def test_qq_in_channel_directory(self):
        import gateway.channel_directory

        source = inspect.getsource(gateway.channel_directory.build_channel_directory)
        assert '"qq"' in source

    def test_qq_in_gateway_setup(self):
        import hermes_cli.gateway

        source = inspect.getsource(hermes_cli.gateway)
        assert '"key": "qq"' in source
        assert "QQ_BOT_APP_ID" in source
        assert "QQ_BOT_SECRET" in source
