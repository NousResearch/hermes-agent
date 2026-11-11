"""Regression edges: unresolved lazy plugins and blank dotenv provenance are not deletions/overrides."""
import os

import pytest

from gateway.config_loader import load_yaml_layer
from gateway.platforms._shared import bridged_env_digest, bridged_env_marker, yaml_env_setter


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
        assert os.environ.get(bridged_env_marker("TELEGRAM_REACTIONS")) == bridged_env_digest("false")
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
    assert os.environ.get(bridged_env_marker("TELEGRAM_REACTIONS")) == bridged_env_digest(expected)


def test_platform_plugin_whose_import_fails_keeps_its_bridged_value(tmp_path, monkeypatch):
    """The real loader disposes a failed plugin's deferred lease; that is still not a YAML key deletion."""
    from gateway import platform_registry as registry_module
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager, PluginManifest

    plugin_dir = tmp_path / "plugins" / "bridgeprobe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        "import bridgeprobe_missing_sdk  # noqa: F401\n\ndef register(ctx):\n    pass\n", encoding="utf-8")
    manifest = PluginManifest(name="bridgeprobe-platform", kind="platform", source="bundled",
                              path=str(plugin_dir), key="bridgeprobe-platform")
    registry = registry_module.PlatformRegistry()
    monkeypatch.setattr(registry_module, "platform_registry", registry)
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    # a previous gateway generation's bridge wrote it
    monkeypatch.setenv("BRIDGEPROBE_FLAG", "false")
    monkeypatch.setenv(bridged_env_marker("BRIDGEPROBE_FLAG"), bridged_env_digest("false"))
    (tmp_path / "config.yaml").write_text("bridgeprobe:\n  flag: false\n", encoding="utf-8")
    PluginManager()._register_deferred_platform(manifest)

    for _ in range(2):
        load_yaml_layer(tmp_path, {})
        assert os.environ.get("BRIDGEPROBE_FLAG") == "false"
        assert os.environ.get(bridged_env_marker("BRIDGEPROBE_FLAG")) == bridged_env_digest("false")
    assert registry.get("bridgeprobe") is None


@pytest.mark.platforms("windows")
def test_lowercase_dotenv_key_is_the_operators_value_on_windows(tmp_path, monkeypatch):
    """Windows env names are case-insensitive: ``telegram_reactions`` in .env owns TELEGRAM_REACTIONS."""
    from hermes_cli.env_loader import load_hermes_dotenv

    # inherited from the previous generation's bridge
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))
    (tmp_path / ".env").write_text("telegram_reactions=false\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    load_yaml_layer(tmp_path, {})

    assert os.environ.get("TELEGRAM_REACTIONS") == "false"
