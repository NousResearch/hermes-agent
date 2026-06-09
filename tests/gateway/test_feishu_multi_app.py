"""Tests for multi-app Feishu list-config compatibility.

Regression guard: when platforms.feishu is a list, several code paths used to
raise AttributeError: 'list' object has no attribute 'extra'.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.authz_mixin import GatewayAuthorizationMixin


class DummyMixin(GatewayAuthorizationMixin):
    """Minimal mixin instance for testing list-config paths."""

    def __init__(self, config):
        self.config = config
        self.adapters = {}


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

    def test_get_home_channel_list(self, multi_config):
        """get_home_channel should iterate list and return first match."""
        # Default home_channel is None; test that it doesn't crash
        assert multi_config.get_home_channel(Platform.FEISHU) is None

    def test_connected_platforms_list(self, multi_config):
        """connected_platforms should include platform if any list item is connected."""
        # No real connection, but ensure it doesn't crash on list
        connected = multi_config.get_connected_platforms()
        assert isinstance(connected, list)
