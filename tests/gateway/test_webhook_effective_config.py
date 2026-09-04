"""Behavior contracts for the canonical webhook config projection."""

from pathlib import Path

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.run import _profile_runtime_scope
from gateway.webhook_config import (
    resolve_effective_webhook_config,
    resolve_effective_webhook_secret,
)


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in (
        "WEBHOOK_ENABLED",
        "WEBHOOK_HOST",
        "WEBHOOK_PORT",
        "WEBHOOK_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    set_multiplex_active(False)
    yield root
    set_multiplex_active(False)


def _write_config(home: Path, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_defaults_are_profile_qualified(profile_root):
    effective = resolve_effective_webhook_config()

    assert effective.enabled is False
    assert effective.host is None
    assert effective.port == 8644
    assert effective.profile == "default"
    assert effective.global_secret_ref is None
    assert effective.routes_path == profile_root / "webhook_subscriptions.json"
    assert effective.source_map == {
        "enabled": "default",
        "host": "default",
        "port": "default",
        "global_secret_ref": "default",
        "routes_path": "profile",
    }


def test_yaml_values_are_projected_without_secret_egress(profile_root):
    _write_config(
        profile_root,
        "platforms:\n"
        "  webhook:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      host: 127.0.0.1\n"
        "      port: 9123\n"
        "      secret: yaml-secret-sentinel\n",
    )

    effective = resolve_effective_webhook_config()

    assert effective.enabled is True
    assert effective.host == "127.0.0.1"
    assert effective.port == 9123
    assert effective.global_secret_ref is None
    assert effective.source_map["enabled"] == "yaml"
    assert effective.source_map["host"] == "yaml"
    assert effective.source_map["port"] == "yaml"
    assert effective.source_map["global_secret_ref"] == "yaml"
    assert "yaml-secret-sentinel" not in repr(effective)


def test_explicit_named_profile_reads_own_yaml_and_restores_scope(profile_root):
    worker = profile_root / "profiles" / "worker"
    _write_config(
        profile_root,
        "platforms:\n"
        "  webhook:\n"
        "    enabled: false\n"
        "    extra:\n"
        "      host: default.example\n"
        "      port: 8101\n",
    )
    _write_config(
        worker,
        "platforms:\n"
        "  webhook:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      host: worker.example\n"
        "      port: 8102\n",
    )

    worker_effective = resolve_effective_webhook_config("worker")
    default_effective = resolve_effective_webhook_config()

    assert worker_effective.profile == "worker"
    assert worker_effective.enabled is True
    assert worker_effective.host == "worker.example"
    assert worker_effective.port == 8102
    assert worker_effective.routes_path == worker / "webhook_subscriptions.json"

    assert default_effective.profile == "default"
    assert default_effective.enabled is False
    assert default_effective.host == "default.example"
    assert default_effective.port == 8101
    assert default_effective.routes_path == profile_root / "webhook_subscriptions.json"


def test_process_environment_is_the_runtime_and_cli_override(profile_root, monkeypatch):
    monkeypatch.setenv("WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("WEBHOOK_HOST", "env.example")
    monkeypatch.setenv("WEBHOOK_PORT", "9234")
    monkeypatch.setenv("WEBHOOK_SECRET", "env-secret-sentinel")

    effective = resolve_effective_webhook_config()

    assert effective.enabled is True
    assert effective.host == "env.example"
    assert effective.port == 9234
    assert effective.global_secret_ref == "WEBHOOK_SECRET"
    assert effective.source_map["enabled"] == "env"
    assert effective.source_map["host"] == "env"
    assert effective.source_map["port"] == "env"
    assert effective.source_map["global_secret_ref"] == "env"
    assert "env-secret-sentinel" not in repr(effective)
    assert resolve_effective_webhook_secret() == "env-secret-sentinel"


def test_explicit_false_and_sibling_env_values_override_yaml(profile_root, monkeypatch):
    _write_config(
        profile_root,
        "platforms:\n"
        "  webhook:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      host: yaml.example\n"
        "      port: 8123\n",
    )
    monkeypatch.setenv("WEBHOOK_ENABLED", "false")
    monkeypatch.setenv("WEBHOOK_PORT", "8234")
    monkeypatch.setenv("WEBHOOK_SECRET", "env-secret-sentinel")

    effective = resolve_effective_webhook_config()

    assert effective.enabled is False
    assert effective.host == "yaml.example"
    assert effective.port == 8234
    assert effective.global_secret_ref == "WEBHOOK_SECRET"
    assert effective.source_map["enabled"] == "env"
    assert effective.source_map["host"] == "yaml"
    assert effective.source_map["port"] == "env"


def test_profile_scope_cannot_borrow_default_process_webhook_values(
    profile_root, monkeypatch
):
    worker = profile_root / "profiles" / "worker"
    _write_config(
        worker,
        "platforms:\n"
        "  webhook:\n"
        "    enabled: true\n"
        "    extra:\n"
        "      host: worker.example\n"
        "      port: 8002\n",
    )
    (worker / ".env").write_text(
        "WEBHOOK_PORT=8003\nWEBHOOK_SECRET=worker-secret-sentinel\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WEBHOOK_ENABLED", "false")
    monkeypatch.setenv("WEBHOOK_HOST", "default.example")
    monkeypatch.setenv("WEBHOOK_PORT", "8999")
    monkeypatch.setenv("WEBHOOK_SECRET", "default-secret-sentinel")
    set_multiplex_active(True)

    with _profile_runtime_scope(worker):
        effective = resolve_effective_webhook_config()
        runtime_secret = resolve_effective_webhook_secret()

    assert effective.profile == "worker"
    assert effective.enabled is True
    assert effective.host == "worker.example"
    assert effective.port == 8003
    assert effective.global_secret_ref == "WEBHOOK_SECRET"
    assert effective.routes_path == worker / "webhook_subscriptions.json"
    assert effective.source_map["enabled"] == "yaml"
    assert effective.source_map["host"] == "yaml"
    assert effective.source_map["port"] == "profile"
    assert effective.source_map["global_secret_ref"] == "profile"
    assert runtime_secret == "worker-secret-sentinel"
    assert "worker-secret-sentinel" not in repr(effective)
    assert "default-secret-sentinel" not in repr(effective)
