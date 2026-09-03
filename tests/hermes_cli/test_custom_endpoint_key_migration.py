"""Regression tests for the v41 custom-endpoint credential migration."""

import yaml


def test_migration_moves_plaintext_custom_key_to_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 39,
                "model": {
                    "provider": "custom",
                    "base_url": "https://text.example.com/v1",
                    "default": "model-a",
                    "api_key": "sk-legacy-secret",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.config import get_env_value
    from hermes_cli.config_migrations import run_migrations

    results = {"env_added": [], "config_added": [], "warnings": []}
    run_migrations(39, results, quiet=True)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    key_env = raw["model"]["key_env"]
    assert "api_key" not in raw["model"]
    assert get_env_value(key_env) == "sk-legacy-secret"
    assert "sk-legacy-secret" not in config_path.read_text(encoding="utf-8")
    assert any("key_env" in item for item in results["config_added"])


def test_migration_preserves_existing_env_reference(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 39,
                "model": {
                    "provider": "custom",
                    "base_url": "https://text.example.com/v1",
                    "default": "model-a",
                    "api_key": "${EXISTING_KEY}",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.config_migrations import run_migrations

    results = {"env_added": [], "config_added": [], "warnings": []}
    run_migrations(39, results, quiet=True)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["model"]["api_key"] == "${EXISTING_KEY}"
    assert "key_env" not in raw["model"]
