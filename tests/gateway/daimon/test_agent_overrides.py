"""Tests for gateway.daimon.agent_overrides module."""
from __future__ import annotations

import pytest

from gateway.daimon.agent_overrides import AgentOverrides, compute_overrides
from gateway.daimon.tier import Tier


def _make_config(admin_users=None, user_model=None, admin_model=None, tool_limits=None, **kwargs):
    """Build a raw config dict matching gateway.discord.daimon namespace."""
    daimon = {}
    if admin_users is not None:
        daimon["admin_users"] = admin_users
    if user_model is not None:
        daimon["user_model"] = user_model
    if admin_model is not None:
        daimon["admin_model"] = admin_model
    if tool_limits is not None:
        daimon["tool_limits"] = tool_limits
    daimon.update(kwargs)
    return {"gateway": {"discord": {"daimon": daimon}}}


class TestNonDiscordReturnsNone:
    """Platform != 'discord' should return None."""

    def test_telegram_returns_none(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "admin1", "telegram")
        assert result is None

    def test_slack_returns_none(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "admin1", "slack")
        assert result is None

    def test_cli_returns_none(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "admin1", "cli")
        assert result is None


class TestNoAdminUsersReturnsNone:
    """Empty or missing admin_users means Daimon is inactive."""

    def test_empty_admin_list(self):
        cfg = _make_config(admin_users=[])
        result = compute_overrides(cfg, "user123", "discord")
        assert result is None

    def test_missing_admin_users(self):
        cfg = {"gateway": {"discord": {"daimon": {}}}}
        result = compute_overrides(cfg, "user123", "discord")
        assert result is None

    def test_no_daimon_section(self):
        cfg = {"gateway": {"discord": {}}}
        result = compute_overrides(cfg, "user123", "discord")
        assert result is None

    def test_empty_config(self):
        result = compute_overrides({}, "user123", "discord")
        assert result is None


class TestAdminGetsAdminModel:
    """Admin user should get the admin_model and no iteration cap override."""

    def test_admin_model_default(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "admin1", "discord")
        assert result is not None
        assert result.model == "anthropic/claude-sonnet-4.6"
        assert result.tier == Tier.ADMIN
        assert result.max_iterations is None
        assert result.disabled_toolsets is None
        assert result.gateway_timeout is None

    def test_admin_model_custom(self):
        cfg = _make_config(admin_users=["admin1"], admin_model="openai/gpt-4o")
        result = compute_overrides(cfg, "admin1", "discord")
        assert result.model == "openai/gpt-4o"
        assert result.tier == Tier.ADMIN


class TestUserGetsUserModelAndCaps:
    """Regular user should get user_model, max_iterations, and other caps."""

    def test_user_defaults(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result is not None
        assert result.model == "xiaomi/mimo-v2.5-pro"
        assert result.max_iterations == 30
        assert result.tier == Tier.USER
        assert result.gateway_timeout == 600

    def test_user_custom_model(self):
        cfg = _make_config(admin_users=["admin1"], user_model="openai/gpt-4o-mini")
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result.model == "openai/gpt-4o-mini"

    def test_user_custom_iterations(self):
        cfg = _make_config(admin_users=["admin1"], max_iterations=15)
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result.max_iterations == 15


class TestUserDisabledToolsets:
    """Tools with limit=0 should appear in disabled_toolsets for users."""

    def test_default_disabled_tools(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result.disabled_toolsets is not None
        # Default config disables text_to_speech, cronjob, send_message
        assert "text_to_speech" in result.disabled_toolsets
        assert "cronjob" in result.disabled_toolsets
        assert "send_message" in result.disabled_toolsets

    def test_custom_disabled_tools(self):
        cfg = _make_config(
            admin_users=["admin1"],
            tool_limits={"browser": 0, "image_generate": 0},
        )
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result.disabled_toolsets is not None
        # Defaults with limit=0 + custom overrides
        assert "browser" in result.disabled_toolsets
        assert "image_generate" in result.disabled_toolsets
        assert "text_to_speech" in result.disabled_toolsets
        assert "cronjob" in result.disabled_toolsets
        assert "send_message" in result.disabled_toolsets

    def test_tools_with_positive_limits_not_disabled(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "regular_user", "discord")
        assert "web_search" not in result.disabled_toolsets
        assert "terminal" not in result.disabled_toolsets
        assert "read_file" not in result.disabled_toolsets

    def test_admin_no_disabled_toolsets(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "admin1", "discord")
        assert result.disabled_toolsets is None


class TestUserGatewayTimeout:
    """gateway_timeout override is set for users."""

    def test_default_timeout(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result.gateway_timeout == 600

    def test_custom_timeout(self):
        cfg = _make_config(admin_users=["admin1"], gateway_timeout=300)
        result = compute_overrides(cfg, "regular_user", "discord")
        assert result.gateway_timeout == 300

    def test_admin_no_timeout_override(self):
        cfg = _make_config(admin_users=["admin1"])
        result = compute_overrides(cfg, "admin1", "discord")
        assert result.gateway_timeout is None
