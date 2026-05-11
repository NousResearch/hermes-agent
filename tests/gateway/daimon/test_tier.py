"""Tests for gateway.daimon.tier module."""

from gateway.daimon.config import DaimonConfig
from gateway.daimon.tier import Tier, resolve_tier


class TestTier:
    """Test Tier enum behavior."""

    def test_admin_is_admin(self):
        assert Tier.ADMIN.is_admin is True

    def test_user_is_not_admin(self):
        assert Tier.USER.is_admin is False

    def test_admin_model(self):
        cfg = DaimonConfig(admin_model="anthropic/claude-opus-4")
        assert Tier.ADMIN.model(cfg) == "anthropic/claude-opus-4"

    def test_user_model(self):
        cfg = DaimonConfig(user_model="openai/gpt-4o-mini")
        assert Tier.USER.model(cfg) == "openai/gpt-4o-mini"

    def test_default_models(self):
        cfg = DaimonConfig()
        assert Tier.ADMIN.model(cfg) == "anthropic/claude-sonnet-4.6"
        assert Tier.USER.model(cfg) == "xiaomi/mimo-v2.5-pro"


class TestResolveTier:
    """Test resolve_tier function."""

    def test_admin_user_detected(self):
        cfg = DaimonConfig(admin_users=["111", "222", "333"])
        assert resolve_tier("222", cfg) is Tier.ADMIN

    def test_non_admin_is_user(self):
        cfg = DaimonConfig(admin_users=["111", "222"])
        assert resolve_tier("999", cfg) is Tier.USER

    def test_empty_admin_list(self):
        cfg = DaimonConfig(admin_users=[])
        assert resolve_tier("111", cfg) is Tier.USER

    def test_model_routing_for_admin(self):
        cfg = DaimonConfig(
            admin_users=["admin_id"],
            admin_model="big-model",
            user_model="small-model",
        )
        tier = resolve_tier("admin_id", cfg)
        assert tier.model(cfg) == "big-model"

    def test_model_routing_for_user(self):
        cfg = DaimonConfig(
            admin_users=["admin_id"],
            admin_model="big-model",
            user_model="small-model",
        )
        tier = resolve_tier("regular_user", cfg)
        assert tier.model(cfg) == "small-model"
