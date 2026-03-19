"""Tests for the social relay platform adapter."""

import os
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.social import (
    SocialAdapter,
    check_social_adapter_requirements,
    _load_social_relay_config,
)


@pytest.fixture
def social_config_enabled(tmp_path):
    config = """
social:
  enabled: true
  relay: "http://localhost:8787"
  poll_interval: 5
  profile:
    display_name: "Test Agent"
    bio: "Testing"
"""
    (tmp_path / "config.yaml").write_text(config)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def social_config_disabled(tmp_path):
    config = """
social:
  enabled: false
  relay: "http://localhost:8787"
"""
    (tmp_path / "config.yaml").write_text(config)
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


@pytest.fixture
def no_config(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


class TestCheckRequirements:
    def test_returns_true_when_enabled(self, social_config_enabled):
        assert check_social_adapter_requirements() is True

    def test_returns_false_when_disabled(self, social_config_disabled):
        assert check_social_adapter_requirements() is False

    def test_returns_false_when_no_config(self, no_config):
        assert check_social_adapter_requirements() is False


class TestLoadConfig:
    def test_loads_relay_url(self, social_config_enabled):
        config = _load_social_relay_config()
        assert config["relay"] == "http://localhost:8787"

    def test_loads_poll_interval(self, social_config_enabled):
        config = _load_social_relay_config()
        assert config["poll_interval"] == 5

    def test_defaults_when_no_config(self, no_config):
        config = _load_social_relay_config()
        assert config["relay"] == ""
        assert config["poll_interval"] == 30


class TestSocialAdapter:
    def test_creates_with_platform_social(self, social_config_enabled):
        from gateway.config import PlatformConfig
        pc = PlatformConfig(enabled=True)
        adapter = SocialAdapter(pc)
        assert adapter.platform == Platform.SOCIAL
        assert adapter._relay_url == "http://localhost:8787"
        assert adapter._poll_interval == 5

    def test_send_without_identity_fails(self, social_config_enabled):
        from gateway.config import PlatformConfig
        import asyncio

        pc = PlatformConfig(enabled=True)
        adapter = SocialAdapter(pc)
        adapter._identity = None

        result = asyncio.get_event_loop().run_until_complete(
            adapter.send("target_pk", "hello")
        )
        assert result.success is False
        assert "identity" in result.error.lower()
