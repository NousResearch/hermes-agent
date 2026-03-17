"""Tests for set_config_value — verifying secrets route to .env and config to config.yaml."""

import os
from pathlib import Path
from unittest.mock import patch, call

import pytest

from hermes_cli.config import set_config_value


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    env_file = tmp_path / ".env"
    env_file.touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def _read_env(tmp_path):
    return (tmp_path / ".env").read_text()


def _read_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    return config_path.read_text() if config_path.exists() else ""


# ---------------------------------------------------------------------------
# Explicit allowlist keys → .env
# ---------------------------------------------------------------------------

class TestExplicitAllowlist:
    """Keys in the hardcoded allowlist should always go to .env."""

    @pytest.mark.parametrize("key", [
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "WANDB_API_KEY",
        "TINKER_API_KEY",
        "HONCHO_API_KEY",
        "FIRECRAWL_API_KEY",
        "BROWSERBASE_API_KEY",
        "FAL_KEY",
        "SUDO_PASSWORD",
        "GITHUB_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
    ])
    def test_explicit_key_routes_to_env(self, key, _isolated_hermes_home):
        set_config_value(key, "test-value-123")
        env_content = _read_env(_isolated_hermes_home)
        assert f"{key}=test-value-123" in env_content
        # Must NOT appear in config.yaml
        assert key not in _read_config(_isolated_hermes_home)


# ---------------------------------------------------------------------------
# Catch-all patterns → .env
# ---------------------------------------------------------------------------

class TestCatchAllPatterns:
    """Any key ending in _API_KEY or _TOKEN should route to .env."""

    @pytest.mark.parametrize("key", [
        "DAYTONA_API_KEY",
        "ELEVENLABS_API_KEY",
        "SOME_FUTURE_SERVICE_API_KEY",
        "MY_CUSTOM_TOKEN",
        "WHATSAPP_BOT_TOKEN",
    ])
    def test_api_key_suffix_routes_to_env(self, key, _isolated_hermes_home):
        set_config_value(key, "secret-456")
        env_content = _read_env(_isolated_hermes_home)
        assert f"{key}=secret-456" in env_content
        assert key not in _read_config(_isolated_hermes_home)

    def test_case_insensitive(self, _isolated_hermes_home):
        """Keys should be uppercased regardless of input casing."""
        set_config_value("openai_api_key", "sk-test")
        env_content = _read_env(_isolated_hermes_home)
        assert "OPENAI_API_KEY=sk-test" in env_content

    def test_terminal_ssh_prefix_routes_to_env(self, _isolated_hermes_home):
        set_config_value("TERMINAL_SSH_PORT", "2222")
        env_content = _read_env(_isolated_hermes_home)
        assert "TERMINAL_SSH_PORT=2222" in env_content


# ---------------------------------------------------------------------------
# Non-secret keys → config.yaml
# ---------------------------------------------------------------------------

class TestConfigYamlRouting:
    """Regular config keys should go to config.yaml, NOT .env."""

    def test_simple_key(self, _isolated_hermes_home):
        set_config_value("model", "gpt-4o")
        config = _read_config(_isolated_hermes_home)
        assert "gpt-4o" in config
        assert "model" not in _read_env(_isolated_hermes_home)

    def test_nested_key(self, _isolated_hermes_home):
        set_config_value("terminal.backend", "docker")
        config = _read_config(_isolated_hermes_home)
        assert "docker" in config
        assert "terminal" not in _read_env(_isolated_hermes_home)

    def test_terminal_image_goes_to_config(self, _isolated_hermes_home):
        """TERMINAL_DOCKER_IMAGE doesn't match _API_KEY or _TOKEN, so config.yaml."""
        set_config_value("terminal.docker_image", "python:3.12")
        config = _read_config(_isolated_hermes_home)
        assert "python:3.12" in config

    def test_terminal_docker_cwd_mount_flag_goes_to_config_and_env(self, _isolated_hermes_home):
        set_config_value("terminal.docker_mount_cwd_to_workspace", "true")
        config = _read_config(_isolated_hermes_home)
        env_content = _read_env(_isolated_hermes_home)
        assert "docker_mount_cwd_to_workspace: 'true'" in config or "docker_mount_cwd_to_workspace: true" in config
        assert (
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=true" in env_content
            or "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=True" in env_content
        )
