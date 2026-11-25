"""Regression edges: unresolved lazy plugins and blank dotenv provenance are not deletions/overrides."""
import os

import pytest

from gateway.config_loader import load_yaml_layer
from gateway.platforms._shared import bridged_env_digest, bridged_env_marker, yaml_env_setter
from hermes_constants import hermes_home_key


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


def test_a_force_reload_drops_the_failed_load_record(tmp_path, monkeypatch):
    """A plugin whose import failed, then disabled: a force reload must let YAML cleanup resume."""
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
    (tmp_path / "config.yaml").write_text("bridgeprobe:\n  flag: false\n", encoding="utf-8")
    manager = PluginManager()
    manager._register_deferred_platform(manifest)
    load_yaml_layer(tmp_path, {})  # the import fails here and is recorded
    # An unrelated setting the bridge wrote earlier, now removed from YAML.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))

    manager.unload()  # force rediscovery's unload-all; the disabled plugin never registers again
    load_yaml_layer(tmp_path, {})

    assert "TELEGRAM_REACTIONS" not in os.environ


def test_another_profiles_reload_keeps_this_profiles_failed_load_record(tmp_path, monkeypatch):
    """A force reload of a sibling profile's plugins re-decides only that profile's plugins."""
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
    (tmp_path / "config.yaml").write_text("bridgeprobe:\n  flag: false\n", encoding="utf-8")
    PluginManager()._register_deferred_platform(manifest)
    load_yaml_layer(tmp_path, {})  # this profile's import fails and is recorded
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_REACTIONS"), bridged_env_digest("false"))

    PluginManager(scope_key=str(tmp_path / "sibling")).unload()
    load_yaml_layer(tmp_path, {})

    assert os.environ.get("TELEGRAM_REACTIONS") == "false"  # the failed plugin may still own it


@pytest.mark.parametrize("yaml_text", ["{}\n", "telegram:\n  proxy_url: 'http://other:3128'\n"])
def test_a_value_an_external_secret_source_supplies_is_the_operators(tmp_path, monkeypatch, yaml_text):
    """The proxy moved from config.yaml into the vault: neither a YAML removal nor a YAML edit touches it."""
    from hermes_cli import env_loader

    proxy = "http://user:pw@proxy:3128"
    monkeypatch.setenv("TELEGRAM_PROXY", proxy)
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_PROXY"), bridged_env_digest(proxy))  # inherited
    monkeypatch.setitem(env_loader._SECRET_SOURCE_VALUES_BY_HOME, _launch_home_key(), {"TELEGRAM_PROXY": proxy})
    (tmp_path / "config.yaml").write_text(yaml_text, encoding="utf-8")

    load_yaml_layer(tmp_path, {})

    assert os.environ.get("TELEGRAM_PROXY") == proxy


def _launch_home_key() -> str:
    from hermes_constants import get_routing_process_hermes_home

    return hermes_home_key(get_routing_process_hermes_home())


@pytest.mark.parametrize("history", ["removed_mapping", "other_profile"])
def test_source_history_the_launch_home_no_longer_owns_does_not_block_a_yaml_edit(tmp_path, monkeypatch, history):
    """Only the launch home's CURRENT source snapshot owns a value: a vault mapping removed since, or a
    sibling profile's vault supplying the same name, must not freeze the bridge's own leftover."""
    from hermes_cli import env_loader

    proxy = "http://user:pw@proxy:3128"
    monkeypatch.setenv("TELEGRAM_PROXY", proxy)
    monkeypatch.setenv(bridged_env_marker("TELEGRAM_PROXY"), bridged_env_digest(proxy))  # inherited
    monkeypatch.setattr(env_loader, "_SOURCE_SUPPLIED_NAMES", {"TELEGRAM_PROXY"})
    if history == "other_profile":
        monkeypatch.setitem(env_loader._SECRET_SOURCE_VALUES_BY_HOME, hermes_home_key(tmp_path / "other"),
                            {"TELEGRAM_PROXY": proxy})
    (tmp_path / "config.yaml").write_text("telegram:\n  proxy_url: 'http://other:3128'\n", encoding="utf-8")

    load_yaml_layer(tmp_path, {})

    assert os.environ.get("TELEGRAM_PROXY") == "http://other:3128"


def test_a_dotenv_value_equal_to_an_earlier_bridged_one_reaches_child_programs(tmp_path):
    """Moving TELEGRAM_REACTIONS=false from config.yaml into .env: the operator's value must survive
    the child-env scrub, which drops any value still matching a bridge marker's digest."""
    from gateway.bridged_env import drop_bridge_markers
    from hermes_cli.env_loader import load_hermes_dotenv

    (tmp_path / "config.yaml").write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    load_yaml_layer(tmp_path, {})
    assert os.environ.get(bridged_env_marker("TELEGRAM_REACTIONS")) == bridged_env_digest("false")
    (tmp_path / ".env").write_text("TELEGRAM_REACTIONS=false\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=tmp_path, load_external_secrets=False)
    load_yaml_layer(tmp_path, {})

    assert drop_bridge_markers(dict(os.environ)).get("TELEGRAM_REACTIONS") == "false"


def test_a_vault_value_equal_to_an_earlier_bridged_one_reaches_child_programs(tmp_path, monkeypatch):
    """The vault twin of the .env move: a secret source skips a name the env already holds, so it
    publishes nothing, yet the value is the operator's from then on and must survive the child-env scrub."""
    from pathlib import Path

    import agent.secret_sources.bitwarden as bw_module
    from agent.secret_sources import registry as reg_module
    from gateway.bridged_env import drop_bridge_markers
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_SECRET_SOURCE_VALUES_BY_HOME", {})
    monkeypatch.setattr(env_loader, "_SECRET_SOURCES", {})
    env_loader.reset_secret_source_cache()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", lambda **_kw: ({"TELEGRAM_REACTIONS": "false"}, []))
    reg_module._reset_registry_for_tests()
    (tmp_path / "config.yaml").write_text("telegram:\n  reactions: false\n", encoding="utf-8")
    load_yaml_layer(tmp_path, {})
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  bitwarden:\n    enabled: true\n    project_id: p\n    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8")
    env_loader._apply_external_secret_sources(tmp_path)
    load_yaml_layer(tmp_path, {})

    assert drop_bridge_markers(dict(os.environ)).get("TELEGRAM_REACTIONS") == "false"
