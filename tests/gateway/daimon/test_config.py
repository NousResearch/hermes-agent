"""Tests for gateway.daimon.config module."""

from gateway.daimon.config import (
    DaimonConfig,
    _DEFAULT_ADMIN_ONLY_COMMANDS,
    _DEFAULT_TOOL_LIMITS,
    load_daimon_config,
)


class TestDaimonConfigDefaults:
    """Test that DaimonConfig has correct defaults."""

    def test_default_models(self):
        cfg = DaimonConfig()
        assert cfg.user_model == "xiaomi/mimo-v2.5-pro"
        assert cfg.admin_model == "anthropic/claude-sonnet-4.6"

    def test_default_limits(self):
        cfg = DaimonConfig()
        assert cfg.max_iterations == 30
        assert cfg.max_threads_per_day == 5
        assert cfg.gateway_timeout == 600
        assert cfg.max_active_sessions == 50

    def test_default_flags(self):
        cfg = DaimonConfig()
        assert cfg.queue_enabled is True
        assert cfg.per_user_concurrent is True

    def test_default_tool_limits(self):
        cfg = DaimonConfig()
        assert cfg.tool_limits == _DEFAULT_TOOL_LIMITS
        assert cfg.tool_limits["web_search"] == 15
        assert cfg.tool_limits["text_to_speech"] == 0
        assert cfg.tool_limits["execute_code"] == 10

    def test_default_admin_only_commands(self):
        cfg = DaimonConfig()
        assert cfg.admin_only_commands == _DEFAULT_ADMIN_ONLY_COMMANDS
        assert "daimon" in cfg.admin_only_commands
        assert "config" in cfg.admin_only_commands

    def test_default_responders(self):
        cfg = DaimonConfig()
        assert cfg.responders == ["creator", "admins"]

    def test_default_admin_users_empty(self):
        cfg = DaimonConfig()
        assert cfg.admin_users == []


class TestLoadDaimonConfig:
    """Test load_daimon_config with various inputs."""

    def test_empty_config(self):
        cfg = load_daimon_config({})
        assert cfg.admin_users == []
        assert cfg.user_model == "xiaomi/mimo-v2.5-pro"
        assert cfg.tool_limits == _DEFAULT_TOOL_LIMITS

    def test_override_model(self):
        raw = {
            "gateway": {
                "discord": {
                    "daimon": {
                        "user_model": "openai/gpt-4o",
                        "admin_model": "anthropic/claude-opus-4",
                    }
                }
            }
        }
        cfg = load_daimon_config(raw)
        assert cfg.user_model == "openai/gpt-4o"
        assert cfg.admin_model == "anthropic/claude-opus-4"

    def test_override_admin_users(self):
        raw = {
            "gateway": {
                "discord": {
                    "daimon": {
                        "admin_users": ["123456", "789012"],
                    }
                }
            }
        }
        cfg = load_daimon_config(raw)
        assert cfg.admin_users == ["123456", "789012"]

    def test_tool_limits_merge(self):
        """User overrides merge on top of defaults (not replace)."""
        raw = {
            "gateway": {
                "discord": {
                    "daimon": {
                        "tool_limits": {
                            "web_search": 50,
                            "custom_tool": 3,
                        }
                    }
                }
            }
        }
        cfg = load_daimon_config(raw)
        # Overridden value
        assert cfg.tool_limits["web_search"] == 50
        # Added custom tool
        assert cfg.tool_limits["custom_tool"] == 3
        # Default preserved
        assert cfg.tool_limits["web_extract"] == 10
        assert cfg.tool_limits["image_generate"] == 3

    def test_admin_only_commands_override(self):
        """admin_only_commands replaces entirely when provided."""
        raw = {
            "gateway": {
                "discord": {
                    "daimon": {
                        "admin_only_commands": ["config", "model"],
                    }
                }
            }
        }
        cfg = load_daimon_config(raw)
        assert cfg.admin_only_commands == ["config", "model"]

    def test_override_numeric_fields(self):
        raw = {
            "gateway": {
                "discord": {
                    "daimon": {
                        "max_iterations": 50,
                        "max_threads_per_day": 10,
                        "gateway_timeout": 300,
                    }
                }
            }
        }
        cfg = load_daimon_config(raw)
        assert cfg.max_iterations == 50
        assert cfg.max_threads_per_day == 10
        assert cfg.gateway_timeout == 300

    def test_override_responders(self):
        raw = {
            "gateway": {
                "discord": {
                    "daimon": {
                        "responders": ["everyone"],
                    }
                }
            }
        }
        cfg = load_daimon_config(raw)
        assert cfg.responders == ["everyone"]
