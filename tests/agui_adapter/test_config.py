"""Integration tests for AG-UI config.yaml propagation."""

import pytest

from agui_adapter.entry import _load_listener_config
from agui_adapter.session import AgentConfig
from hermes_cli import config as config_module
from hermes_cli.config import save_config


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Use a real, isolated config.yaml and keep loader caches test-local."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    yield tmp_path
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()


def test_listener_settings_come_from_config_yaml(isolated_hermes_home, monkeypatch):
    save_config({"agui": {"host": "0.0.0.0", "port": 9123}})
    monkeypatch.setenv("HERMES_AGUI_HOST", "legacy.example.com")
    monkeypatch.setenv("HERMES_AGUI_PORT", "9999")
    monkeypatch.setenv("PORT", "7777")

    assert _load_listener_config() == ("0.0.0.0", 9123)


def test_agent_settings_come_from_config_yaml(isolated_hermes_home, monkeypatch):
    save_config(
        {
            "agui": {
                "base_url": "http://localhost:4010/v1",
                "model": "configured-model",
                "provider": "custom",
                "api_mode": "chat_completions",
                "toolsets": ["hermes-acp", "web"],
            }
        }
    )
    monkeypatch.setenv("HERMES_AGUI_BASE_URL", "https://legacy.invalid/v1")
    monkeypatch.setenv("HERMES_AGUI_MODEL", "legacy-model")
    monkeypatch.setenv("HERMES_AGUI_PROVIDER", "legacy-provider")
    monkeypatch.setenv("HERMES_AGUI_API_MODE", "legacy-mode")
    monkeypatch.setenv("HERMES_AGUI_TOOLSETS", "terminal")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-from-credential-env")

    config = AgentConfig()

    assert config.base_url == "http://localhost:4010/v1"
    assert config.model == "configured-model"
    assert config.provider == "custom"
    assert config.api_mode == "chat_completions"
    assert config.enabled_toolsets == ["hermes-acp", "web"]
    assert config.api_key == "secret-from-credential-env"


def test_agent_settings_use_adapter_defaults_without_user_config(isolated_hermes_home):
    config = AgentConfig()

    assert config.base_url is None
    assert config.model == ""
    assert config.provider is None
    assert config.api_mode is None
    assert config.enabled_toolsets == ["hermes-acp"]
