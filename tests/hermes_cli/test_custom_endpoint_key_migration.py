"""Regression tests for the v51 custom-endpoint credential migration."""

import hermes_yaml as yaml
import pytest


def test_migration_moves_plaintext_custom_key_to_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "_config_version": 39,
                "cron": {"model_drift_guard": True, "max_iterations": 17},
                "security": {"tirith_enabled": True, "redact_secrets": True},
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
    from hermes_cli.config_migrations import MIGRATIONS, run_migrations

    versions = [version for version, _step in MIGRATIONS]
    assert versions == sorted(versions)
    results = {"env_added": [], "config_added": [], "warnings": []}
    run_migrations(39, results, quiet=True)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "model_drift_guard" not in raw["cron"]
    assert raw["cron"]["max_iterations"] == 17
    assert "tirith_enabled" not in raw["security"]
    assert raw["security"]["redact_secrets"] is True
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


@pytest.mark.parametrize("version", [49, 50, None])
def test_current_and_unversioned_configs_migrate_plaintext(tmp_path, monkeypatch, version):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"model": {"provider": "custom", "base_url": "https://current.example/v1",
                        "api_key": "sk-current-secret", "default": "model-a"}}
    if version is not None:
        config["_config_version"] = version
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    from hermes_cli.config import get_env_value
    from hermes_cli.config_migrations import run_migrations

    results = {"env_added": [], "config_added": [], "warnings": []}
    run_migrations(version or 0, results, quiet=True, unversioned=version is None)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "api_key" not in raw["model"]
    assert get_env_value(raw["model"]["key_env"]) == "sk-current-secret"
    assert "sk-current-secret" not in path.read_text(encoding="utf-8")
    assert not results["warnings"]


@pytest.mark.parametrize("binding", ["key_env", "api_key_env"])
def test_migration_preserves_populated_declared_binding(tmp_path, monkeypatch, binding):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SHARED_PROVIDER_KEY", "sk-current-bound-secret")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "_config_version": 49,
        "model": {"provider": "custom", "default": "model-a",
                  "base_url": "https://bound.example/v1", binding: "SHARED_PROVIDER_KEY",
                  "api_key": "sk-stale-inline-secret"},
    }), encoding="utf-8")
    from hermes_cli.config import get_env_value
    from hermes_cli.config_migrations import run_migrations

    run_migrations(49, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["model"]["key_env"] == "SHARED_PROVIDER_KEY"
    assert "api_key" not in raw["model"] and "api_key_env" not in raw["model"]
    assert get_env_value("SHARED_PROVIDER_KEY") == "sk-current-bound-secret"
    assert "sk-stale-inline-secret" not in path.read_text(encoding="utf-8")
