import yaml


def test_config_set_model_key_creates_model_config_backup(tmp_path, monkeypatch, capsys):
    from hermes_cli import config as config_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "old-model",
                    "provider": "openai-codex",
                },
                "custom_providers": [
                    {
                        "name": "local-ollama",
                        "base_url": "http://127.0.0.1:11434/v1",
                    }
                ],
                "providers": {
                    "local": {
                        "base_url": "http://127.0.0.1:11434/v1",
                        "model": "qwen3",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    config_mod.set_config_value("model.base_url", "https://127.0.0.1:11434/v1")

    capsys.readouterr()
    backups = sorted(cfg_path.parent.glob(f"{cfg_path.name}.model-backup.*.bak"))
    assert len(backups) == 1
    backup = yaml.safe_load(backups[0].read_text(encoding="utf-8"))
    assert backup["reason"] == "config-set:model.base_url"
    assert backup["model"] == {
        "default": "old-model",
        "provider": "openai-codex",
    }
    assert backup["custom_providers"] == [
        {
            "name": "local-ollama",
            "base_url": "http://127.0.0.1:11434/v1",
        }
    ]
    assert backup["providers"] == {
        "local": {
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3",
        }
    }


def test_config_set_canonical_provider_key_creates_backup(tmp_path, monkeypatch, capsys):
    from hermes_cli import config as config_mod

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    original = {
        "providers": {
            "local": {
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "qwen3",
            }
        }
    }
    cfg_path.write_text(yaml.safe_dump(original, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    config_mod.set_config_value(
        "providers.local.base_url", "https://localhost.example/v1"
    )

    capsys.readouterr()
    backups = sorted(cfg_path.parent.glob(f"{cfg_path.name}.model-backup.*.bak"))
    assert len(backups) == 1
    backup = yaml.safe_load(backups[0].read_text(encoding="utf-8"))
    assert backup["reason"] == "config-set:providers.local.base_url"
    assert backup["had_providers"] is True
    assert backup["providers"] == original["providers"]
