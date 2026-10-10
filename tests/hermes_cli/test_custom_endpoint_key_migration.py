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


@pytest.mark.parametrize("version", [50, None])
@pytest.mark.parametrize("location", ["model", "providers", "custom_providers"])
@pytest.mark.parametrize("failure", ["env_error", "env_noop", "config_error"])
def test_failed_secret_move_keeps_migration_retryable(
    tmp_path, monkeypatch, version, location, failure,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import config as cfg

    entry = {"base_url": "https://retry.example/v1", "api_key": "sk-retry-secret"}
    config = {}
    if location == "model":
        config[location] = {**entry, "provider": "custom", "default": "model-a"}
    elif location == "providers":
        config[location] = {"retry": entry}
    else:
        config[location] = [{**entry, "name": "retry"}]
    if version is not None:
        config["_config_version"] = version
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def fail_write(*args, **kwargs):
        raise OSError("credential migration write failed")

    with monkeypatch.context() as failed_write:
        if failure == "env_noop":
            failed_write.setattr(cfg, "save_env_value", lambda *args: None)
            expected_error = RuntimeError
        else:
            boundary = "save_config" if failure == "config_error" else "save_env_value"
            failed_write.setattr(cfg, boundary, fail_write)
            expected_error = OSError
        with pytest.raises(expected_error):
            cfg.migrate_config(interactive=False, quiet=True)

    failed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert failed.get("_config_version") == version
    assert "sk-retry-secret" in path.read_text(encoding="utf-8")

    cfg.migrate_config(interactive=False, quiet=True)

    migrated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert migrated["_config_version"] == cfg.DEFAULT_CONFIG["_config_version"]
    if location == "model":
        migrated_entry = migrated[location]
    elif location == "providers" or version is None:
        # Unversioned legacy lists first migrate to the upstream providers map.
        migrated_entry = migrated["providers"]["retry"]
        if location == "custom_providers":
            assert "custom_providers" not in migrated
    else:
        migrated_entry = migrated[location][0]
    assert "api_key" not in migrated_entry
    assert cfg.get_env_value(migrated_entry["key_env"]) == "sk-retry-secret"
    assert "sk-retry-secret" not in path.read_text(encoding="utf-8")
