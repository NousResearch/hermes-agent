import yaml

from hermes_cli.auth import _update_config_for_provider, DEFAULT_CODEX_BASE_URL
from hermes_cli.main import _model_flow_named_custom


def test_update_config_for_provider_clears_stale_custom_api_fields(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "claude-sonnet-4-6",
                    "provider": "custom",
                    "base_url": "https://sub2api7.baijia.com",
                    "api_key": "secret-key",
                    "api_mode": "anthropic_messages",
                }
            },
            sort_keys=False,
        )
    )

    _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)

    updated = yaml.safe_load(config_path.read_text())
    model_cfg = updated["model"]
    assert model_cfg["provider"] == "openai-codex"
    assert model_cfg["base_url"] == DEFAULT_CODEX_BASE_URL
    assert "api_key" not in model_cfg
    assert "api_mode" not in model_cfg



def test_model_flow_named_custom_persists_api_mode_for_saved_model(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openai-codex",
                    "base_url": DEFAULT_CODEX_BASE_URL,
                }
            },
            sort_keys=False,
        )
    )

    saved = {"deactivated": False}

    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: saved.__setitem__("deactivated", True))

    _model_flow_named_custom(
        {},
        {
            "name": "gaotu",
            "base_url": "https://sub2api7.baijia.com",
            "api_key": "secret-key",
            "model": "claude-sonnet-4-6",
            "api_mode": "anthropic_messages",
        },
    )

    updated = yaml.safe_load(config_path.read_text())
    model_cfg = updated["model"]
    assert saved["deactivated"] is True
    assert model_cfg["default"] == "claude-sonnet-4-6"
    assert model_cfg["provider"] == "custom"
    assert model_cfg["base_url"] == "https://sub2api7.baijia.com"
    assert model_cfg["api_key"] == "secret-key"
    assert model_cfg["api_mode"] == "anthropic_messages"
