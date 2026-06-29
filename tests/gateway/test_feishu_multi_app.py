"""Tests for multi-app Feishu list-config compatibility.

Regression guard: when platforms.feishu is a list, several code paths used to
raise AttributeError: 'list' object has no attribute 'extra'.
"""

import pytest
from unittest.mock import patch

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import BasePlatformAdapter


class DummyMixin(GatewayAuthorizationMixin):
    """Minimal mixin instance for testing list-config paths."""

    def __init__(self, config):
        self.config = config
        self.adapters = {}


class DummyAdapter(BasePlatformAdapter):
    """Small concrete adapter for runner routing tests."""

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, **kwargs):
        calls = getattr(self, "send_calls", None)
        if calls is not None:
            calls.append((chat_id, content, kwargs))
        return None

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


class TestMultiAppListConfig:
    """Cover the list-config branches introduced in PR #42499."""

    @pytest.fixture
    def multi_config(self):
        return GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        token="app1-token",
                        extra={
                            "dm_policy": "open",
                            "unauthorized_dm_behavior": "ignore",
                            "notice_delivery": "private",
                            "app_id": "cli_app1",
                        },
                    ),
                    PlatformConfig(
                        enabled=True,
                        token="app2-token",
                        extra={
                            "dm_policy": "pairing",
                            "app_id": "cli_app2",
                        },
                    ),
                ],
            }
        )

    @pytest.fixture
    def single_config(self):
        return GatewayConfig(
            platforms={
                Platform.FEISHU: PlatformConfig(
                    enabled=True,
                    token="single-token",
                    extra={
                        "dm_policy": "open",
                        "unauthorized_dm_behavior": "pair",
                        "notice_delivery": "public",
                    },
                ),
            }
        )

    # --- GatewayConfig ---

    def test_get_unauthorized_dm_behavior_list_first_wins(self, multi_config):
        """First config in the list with the key wins."""
        assert multi_config.get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"

    def test_get_unauthorized_dm_behavior_single(self, single_config):
        assert single_config.get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    def test_get_unauthorized_dm_behavior_missing_platform(self, single_config):
        assert single_config.get_unauthorized_dm_behavior(Platform.SLACK) == "pair"

    def test_get_notice_delivery_list_first_wins(self, multi_config):
        assert multi_config.get_notice_delivery(Platform.FEISHU) == "private"

    def test_get_notice_delivery_single(self, single_config):
        assert single_config.get_notice_delivery(Platform.FEISHU) == "public"

    def test_get_notice_delivery_missing_platform(self, single_config):
        assert single_config.get_notice_delivery(Platform.SLACK) == "public"

    # --- GatewayAuthorizationMixin._adapter_dm_policy ---

    def test_adapter_dm_policy_list(self, multi_config):
        mixin = DummyMixin(multi_config)
        assert mixin._adapter_dm_policy(Platform.FEISHU) == "open"

    def test_adapter_dm_policy_single(self, single_config):
        mixin = DummyMixin(single_config)
        assert mixin._adapter_dm_policy(Platform.FEISHU) == "open"

    def test_adapter_dm_policy_no_config(self):
        mixin = DummyMixin(GatewayConfig())
        assert mixin._adapter_dm_policy(Platform.FEISHU) == ""

    # --- GatewayAuthorizationMixin._get_unauthorized_dm_behavior ---

    def test_get_unauthorized_dm_behavior_list_explicit(self, multi_config):
        mixin = DummyMixin(multi_config)
        # First list item has explicit unauthorized_dm_behavior
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"

    def test_get_unauthorized_dm_behavior_single_explicit(self, single_config):
        mixin = DummyMixin(single_config)
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    def test_get_unauthorized_dm_behavior_list_dm_policy_pairing(self):
        """When no explicit unauthorized_dm_behavior but dm_policy == pairing."""
        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        extra={"dm_policy": "pairing"},
                    ),
                ],
            }
        )
        mixin = DummyMixin(config)
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    def test_get_unauthorized_dm_behavior_list_dm_policy_disabled(self):
        """When dm_policy is disabled, default to ignore."""
        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        extra={"dm_policy": "disabled"},
                    ),
                ],
            }
        )
        mixin = DummyMixin(config)
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"

    def test_get_unauthorized_dm_behavior_no_platform(self):
        mixin = DummyMixin(GatewayConfig())
        assert mixin._get_unauthorized_dm_behavior(Platform.FEISHU) == "pair"

    # --- GatewayConfig roundtrip with list ---

    def test_to_dict_from_dict_preserves_list(self, multi_config):
        d = multi_config.to_dict()
        restored = GatewayConfig.from_dict(d)
        feishu_cfgs = restored.platforms[Platform.FEISHU]
        assert isinstance(feishu_cfgs, list)
        assert len(feishu_cfgs) == 2
        assert feishu_cfgs[0].extra["app_id"] == "cli_app1"
        assert feishu_cfgs[1].extra["app_id"] == "cli_app2"

    def test_from_dict_preserves_top_level_platform_specific_keys_in_extra(self):
        restored = GatewayConfig.from_dict(
            {
                "platforms": {
                    "feishu": [
                        {"enabled": True, "app_id": "cli_app1", "app_secret": "secret1"},
                        {"enabled": True, "app_id": "cli_app2", "app_secret": "secret2"},
                    ]
                }
            }
        )

        feishu_cfgs = restored.platforms[Platform.FEISHU]

        assert isinstance(feishu_cfgs, list)
        assert feishu_cfgs[0].extra["app_id"] == "cli_app1"
        assert feishu_cfgs[0].extra["app_secret"] == "secret1"
        assert feishu_cfgs[1].extra["app_id"] == "cli_app2"
        assert feishu_cfgs[1].extra["app_secret"] == "secret2"

    def test_env_overrides_do_not_clobber_explicit_multi_app_credentials(self):
        config = GatewayConfig.from_dict(
            {
                "platforms": {
                    "feishu": [
                        {"enabled": True, "app_id": "cli_app1", "app_secret": "secret1"},
                        {"enabled": True, "app_id": "cli_app2", "app_secret": "secret2"},
                    ]
                }
            }
        )

        with patch.dict(
            "os.environ",
            {
                "FEISHU_APP_ID": "cli_env",
                "FEISHU_APP_SECRET": "secret_env",
                "FEISHU_DOMAIN": "lark",
                "FEISHU_CONNECTION_MODE": "websocket",
            },
            clear=False,
        ):
            _apply_env_overrides(config)

        feishu_cfgs = config.platforms[Platform.FEISHU]

        assert isinstance(feishu_cfgs, list)
        assert feishu_cfgs[0].extra["app_id"] == "cli_app1"
        assert feishu_cfgs[0].extra["app_secret"] == "secret1"
        assert feishu_cfgs[1].extra["app_id"] == "cli_app2"
        assert feishu_cfgs[1].extra["app_secret"] == "secret2"
        assert feishu_cfgs[0].extra["domain"] == "lark"
        assert feishu_cfgs[1].extra["domain"] == "lark"

    def test_get_home_channel_list(self, multi_config):
        """get_home_channel should iterate list and return first match."""
        # Default home_channel is None; test that it doesn't crash
        assert multi_config.get_home_channel(Platform.FEISHU) is None

    def test_connected_platforms_list(self, multi_config):
        """connected_platforms should include platform if any list item is connected."""
        # No real connection, but ensure it doesn't crash on list
        connected = multi_config.get_connected_platforms()
        assert isinstance(connected, list)

    def test_build_source_carries_adapter_id(self):
        adapter = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}),
            Platform.FEISHU,
        )
        adapter.adapter_id = "feishu:cli_app1"

        source = adapter.build_source(chat_id="chat-1", user_id="user-1")

        assert source.adapter_id == "feishu:cli_app1"

    def test_session_key_isolates_matching_chat_ids_across_feishu_apps(self):
        """Different Feishu apps must not share agent/session state."""
        first = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_same",
            chat_type="dm",
            user_id="ou_same",
            adapter_id="feishu:cli_app1",
        )
        second = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_same",
            chat_type="dm",
            user_id="ou_same",
            adapter_id="feishu:cli_app2",
        )

        first_key = build_session_key(first)
        second_key = build_session_key(second)

        assert first_key == "agent:main:feishu:adapter=feishu%3Acli_app1:dm:oc_same"
        assert second_key == "agent:main:feishu:adapter=feishu%3Acli_app2:dm:oc_same"
        assert first_key != second_key

    def test_runner_routes_source_to_matching_feishu_app_adapter(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}

        app1 = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app1"}),
            Platform.FEISHU,
        )
        app2 = DummyAdapter(
            PlatformConfig(enabled=True, extra={"app_id": "cli_app2"}),
            Platform.FEISHU,
        )

        runner._register_connected_adapter(Platform.FEISHU, app1)
        runner._register_connected_adapter(Platform.FEISHU, app2)

        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="chat-from-app2",
            adapter_id="feishu:cli_app2",
        )

        assert runner.adapters[Platform.FEISHU] is app1
        assert runner._adapter_for_source(source) is app2

    @pytest.mark.asyncio
    async def test_startup_notifications_use_each_feishu_app_home_channel(self):
        config = GatewayConfig(
            platforms={
                Platform.FEISHU: [
                    PlatformConfig(
                        enabled=True,
                        extra={"app_id": "cli_app1"},
                        home_channel=HomeChannel(
                            platform=Platform.FEISHU,
                            chat_id="chat-app1",
                            name="App 1",
                        ),
                    ),
                    PlatformConfig(
                        enabled=True,
                        extra={"app_id": "cli_app2"},
                        home_channel=HomeChannel(
                            platform=Platform.FEISHU,
                            chat_id="chat-app2",
                            name="App 2",
                        ),
                    ),
                ],
            }
        )
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = config
        runner.adapters = {}
        runner.adapters_by_id = {}
        runner._platform_adapter_ids = {}

        app1 = DummyAdapter(config.platforms[Platform.FEISHU][0], Platform.FEISHU)
        app1.send_calls = []
        app2 = DummyAdapter(config.platforms[Platform.FEISHU][1], Platform.FEISHU)
        app2.send_calls = []

        runner._register_connected_adapter(Platform.FEISHU, app1)
        runner._register_connected_adapter(Platform.FEISHU, app2)

        delivered = await runner._send_home_channel_startup_notifications()

        assert ("feishu", "chat-app1", None) in delivered
        assert ("feishu", "chat-app2", None) in delivered
        assert [call[0] for call in app1.send_calls] == ["chat-app1"]
        assert [call[0] for call in app2.send_calls] == ["chat-app2"]
