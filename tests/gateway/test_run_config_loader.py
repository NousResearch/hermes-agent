import yaml

import gateway.run as gateway_run


def test_load_gateway_config_uses_shared_normalized_config(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "provider": "openrouter",
                "model": {
                    "default": "openai/gpt-4.1-mini",
                },
                "max_turns": 41,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    cfg = gateway_run._load_gateway_config()

    assert cfg["model"]["provider"] == "openrouter"
    assert cfg["agent"]["max_turns"] == 41
    assert cfg["terminal"]["backend"] == "local"
