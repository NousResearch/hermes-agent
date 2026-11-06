"""Regression edges: unresolved lazy plugins and blank dotenv provenance are not deletions/overrides."""
import os

import pytest

from gateway.config_loader import load_yaml_layer
from gateway.platforms._shared import bridged_env_marker, yaml_env_setter


@pytest.fixture(autouse=True)
def isolated_bridge(monkeypatch):
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", set())
    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    for key in ("TELEGRAM_REACTIONS", bridged_env_marker("TELEGRAM_REACTIONS")):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


def test_lazy_plugin_failure_preserves_configured_value_on_first_and_later_loads(tmp_path, monkeypatch):
    from gateway import platform_registry as registry_module
    from hermes_cli import plugins

    yaml_env_setter()("TELEGRAM_REACTIONS", "false")
    (tmp_path / "config.yaml").write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    registry = registry_module.PlatformRegistry()
    calls = []

    def missing_dependency():
        calls.append("load")
        raise ImportError("test missing platform dependency")

    registry.register_deferred("telegram", missing_dependency)
    monkeypatch.setattr(registry_module, "platform_registry", registry)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    for _ in range(2):
        load_yaml_layer(tmp_path, {})
        assert os.environ.get("TELEGRAM_REACTIONS") == "false"
        assert os.environ.get(bridged_env_marker("TELEGRAM_REACTIONS")) == "false"
    assert calls == ["load"]


@pytest.mark.parametrize("replacement,expected", [("telegram:\n  reactions: true\n", "true"), ("{}\n", None)])
def test_blank_real_dotenv_does_not_claim_a_later_yaml_bridge_value(tmp_path, replacement, expected):
    from hermes_cli.env_loader import load_hermes_dotenv

    (tmp_path / ".env").write_text("TELEGRAM_REACTIONS=\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
    (tmp_path / "config.yaml").write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    load_yaml_layer(tmp_path, {})
    assert os.environ.get("TELEGRAM_REACTIONS") == "false"
    (tmp_path / "config.yaml").write_text(replacement, encoding="utf-8")

    load_yaml_layer(tmp_path, {})

    assert os.environ.get("TELEGRAM_REACTIONS") == expected
    assert os.environ.get(bridged_env_marker("TELEGRAM_REACTIONS")) == expected
