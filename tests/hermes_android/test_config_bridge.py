from hermes_android.auth_bridge import provider_env_key, read_provider_api_key, write_provider_api_key
from hermes_android.config_bridge import read_runtime_config, write_runtime_config


def test_write_runtime_config_updates_model_section(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    updated = write_runtime_config(
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        base_url="https://openrouter.ai/api/v1",
    )

    assert updated["model"]["provider"] == "openrouter"
    assert updated["model"]["default"] == "anthropic/claude-sonnet-4"
    assert updated["model"]["base_url"] == "https://openrouter.ai/api/v1"
    assert read_runtime_config()["model"]["provider"] == "openrouter"


def test_auth_bridge_reads_and_writes_provider_api_key(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = write_provider_api_key("openrouter", "sk-test")

    assert result == {"provider": "openrouter", "env_key": "OPENROUTER_API_KEY", "saved": True}
    assert provider_env_key("openrouter") == "OPENROUTER_API_KEY"
    assert read_provider_api_key("openrouter") == "sk-test"
