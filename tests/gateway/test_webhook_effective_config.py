"""Table-driven coverage for effective webhook configuration resolution."""

from pathlib import Path

import pytest

from gateway.webhook_config import resolve_effective_webhook_config


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("WEBHOOK_ENABLED", raising=False)
    monkeypatch.delenv("WEBHOOK_HOST", raising=False)
    monkeypatch.delenv("WEBHOOK_PORT", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    return tmp_path


def _write_yaml(home: Path, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("case", "yaml_body", "env", "expected", "sources"),
    [
        (
            "default",
            "",
            {},
            {"enabled": False, "host": None, "port": 8644, "secret_ref": None},
            {"enabled": "default", "host": "default", "port": "default", "global_secret_ref": "default"},
        ),
        (
            "yaml-only",
            "platforms:\n  webhook:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 9123\n      secret: yaml-secret\n",
            {},
            {"enabled": True, "host": "127.0.0.1", "port": 9123, "secret_ref": "WEBHOOK_SECRET"},
            {"enabled": "yaml", "host": "yaml", "port": "yaml", "global_secret_ref": "yaml"},
        ),
        (
            "env-only",
            "",
            {"WEBHOOK_ENABLED": "true", "WEBHOOK_HOST": "env.example", "WEBHOOK_PORT": "9234", "WEBHOOK_SECRET": "env-secret"},
            {"enabled": True, "host": "env.example", "port": 9234, "secret_ref": "WEBHOOK_SECRET"},
            {"enabled": "env", "host": "env", "port": "env", "global_secret_ref": "env"},
        ),
        (
            "env-over-yaml",
            "platforms:\n  webhook:\n    enabled: false\n    extra:\n      host: yaml.example\n      port: 9123\n      secret: yaml-secret\n",
            {"WEBHOOK_ENABLED": "true", "WEBHOOK_HOST": "env.example", "WEBHOOK_PORT": "9234", "WEBHOOK_SECRET": "env-secret"},
            {"enabled": True, "host": "env.example", "port": 9234, "secret_ref": "WEBHOOK_SECRET"},
            {"enabled": "env", "host": "env", "port": "env", "global_secret_ref": "env"},
        ),
    ],
)
def test_effective_webhook_config_precedence(
    isolated_profiles, case, yaml_body, env, expected, sources, monkeypatch
):
    home = isolated_profiles / "home"
    _write_yaml(home, yaml_body)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    config = resolve_effective_webhook_config()

    assert {
        "enabled": config.enabled,
        "host": config.host,
        "port": config.port,
        "secret_ref": config.global_secret_ref,
    } == expected, case
    assert {key: config.source_map[key] for key in sources} == sources
    assert config.profile == "default"
    assert config.routes_path == home / "webhook_subscriptions.json"


def test_named_profile_uses_its_own_yaml_and_profile_environment(isolated_profiles, monkeypatch):
    root = isolated_profiles
    default_home = root / "home"
    profile_home = default_home / "profiles" / "worker"
    _write_yaml(
        default_home,
        "platforms:\n  webhook:\n    enabled: true\n    extra:\n      port: 8001\n",
    )
    _write_yaml(
        profile_home,
        "platforms:\n  webhook:\n    enabled: false\n    extra:\n      host: worker.example\n      port: 8002\n",
    )
    (profile_home / ".env").write_text(
        "WEBHOOK_ENABLED=true\nWEBHOOK_HOST=profile.example\nWEBHOOK_PORT=8003\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WEBHOOK_HOST", "process.example")
    monkeypatch.setenv("WEBHOOK_PORT", "8999")

    config = resolve_effective_webhook_config("worker")

    assert config.enabled is True
    assert config.host == "profile.example"
    assert config.port == 8003
    assert config.source_map["enabled"] == "profile"
    assert config.source_map["host"] == "profile"
    assert config.source_map["port"] == "profile"
    assert config.routes_path == profile_home / "webhook_subscriptions.json"
