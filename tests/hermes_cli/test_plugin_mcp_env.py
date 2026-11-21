"""Profile-owned portable MCP env overrides, through real plugin discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1
from hermes_cli.plugins import PluginManager


def _package(home: Path, name: str, servers: dict, declarations: dict | None = None) -> Path:
    root = home / "plugins" / name
    root.mkdir(parents=True)
    manifest = {"$schema": PLUGIN_SCHEMA_V1, "name": name, "version": "1.2.3"}
    if declarations:
        manifest["extensions"] = {"com.nousresearch.hermes": {"servers": declarations}}
    (root / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "mcp.json").write_text(
        json.dumps({"$schema": MCP_SCHEMA_V1, "mcpServers": servers}), encoding="utf-8")
    return root


@pytest.fixture
def discover(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_mod

    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "os-home")
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    managers = []

    def load(home: Path, enabled: list[str], entries: dict, **extra):
        home.mkdir(exist_ok=True)
        config = {"plugins": {"enabled": enabled, "entries": entries}, **extra}
        # JSON is valid YAML; fixtures only, never a real user config.
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        manager = PluginManager()
        managers.append(manager)
        manager.discover_and_load()
        return manager

    yield load
    for manager in reversed(managers):
        for key in list(manager._plugins):
            manager.unload(key)


def _stdio(**extra) -> dict:
    return {"type": "stdio", "command": Path(sys.executable).name, **extra}


def test_override_merges_only_owned_server_env_and_reaches_child(tmp_path, discover, monkeypatch):
    from tools.mcp_tool_config import _build_safe_env

    monkeypatch.setenv("UNRELATED_PLUGIN_CANARY", "must-not-be-forwarded")
    home = tmp_path / "home"
    root = _package(home, "alpha.tools", {
        "analysis": _stdio(args=["-c", "import json, os; print(json.dumps(dict(os.environ)))"], env={
            "REA_GHIDRA_STARTUP_TIMEOUT_MS": "900000", "JAVA_HOME": "package-default"}),
        "sibling": _stdio(env={"JAVA_HOME": "sibling-default"}),
    }, {"analysis": {"trust": "untrusted", "timeout": 1000}})
    _package(home, "beta.tools", {"other": _stdio(env={"JAVA_HOME": "other-default"})})
    package_before = (root / "mcp.json").read_bytes()
    environment_before = dict(os.environ)
    overrides = {
        "JAVA_HOME": "C:/Program Files/Eclipse Adoptium/jdk-21.0.12.101-hotspot",
        "GHIDRA_INSTALL_DIR": "C:/Tools/Ghidra/ghidra_12.1.2_PUBLIC",
    }
    manager = discover(home, ["alpha.tools", "beta.tools"], {
        "alpha.tools": {"mcp_env": {"analysis": overrides}},
    })
    servers = manager.get_portable_mcp_servers()
    analysis = servers["analysis"]
    assert analysis["env"] == {
        "PLUGIN_ROOT": str(root.resolve()),
        "PLUGIN_DATA": str((home / "plugin-data" / manager._plugins["alpha.tools"].manifest.data_namespace).resolve()),
        "REA_GHIDRA_STARTUP_TIMEOUT_MS": "900000", **overrides,
    }
    assert analysis["command"] == Path(sys.executable).name
    assert analysis["args"][0] == "-c"
    assert analysis["cwd"] == str(root.resolve())
    assert analysis["trust"] == "untrusted"
    assert analysis["timeout"] == 1000
    assert manager._plugins["alpha.tools"].manifest.version == "1.2.3"
    assert manager._portable_mcp_server_plugins["analysis"] == "alpha.tools"
    assert servers["sibling"]["env"]["JAVA_HOME"] == "sibling-default"
    assert servers["other"]["env"]["JAVA_HOME"] == "other-default"
    assert (root / "mcp.json").read_bytes() == package_before
    assert {k: v for k, v in os.environ.items() if k != "HERMES_HOME"} == {
        k: v for k, v in environment_before.items() if k != "HERMES_HOME"}
    child = subprocess.run(
        [analysis["command"], *analysis["args"]], cwd=analysis["cwd"],
        env=_build_safe_env(analysis["env"]), capture_output=True, text=True, check=True, timeout=15,
    )
    child_env = json.loads(child.stdout)
    assert all(child_env[k] == v for k, v in overrides.items())
    assert child_env["REA_GHIDRA_STARTUP_TIMEOUT_MS"] == "900000"
    assert "UNRELATED_PLUGIN_CANARY" not in child_env


@pytest.mark.parametrize("invalid", [
    None, [], "not-a-map", {"JAVA_HOME": 21}, {"JAVA_HOME": True}, {"JAVA_HOME": None},
    {"JAVA_HOME": {"nested": "value"}}, {1: "value"}, {"": "value"},
    {"BAD=NAME": "value"}, {"BAD-NAME": "value"}, {"9NAME": "value"},
    {"NÄME": "value"}, {"BAD\x00NAME": "value"}, {"JAVA_HOME": "bad\x00value"},
    {"PLUGIN_ROOT": "replacement"}, {"PLUGIN_DATA": "replacement"},
])
def test_invalid_override_withholds_target_not_valid_siblings(tmp_path, discover, caplog, invalid):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {"target": _stdio(), "sibling": _stdio()})
    _package(home, "beta.tools", {"other": _stdio()})
    manager = discover(home, ["alpha.tools", "beta.tools"], {
        "alpha.tools": {"mcp_env": {"target": invalid, "sibling": {"GOOD": "yes"}}},
    })
    servers = manager.get_portable_mcp_servers()
    assert "target" not in servers
    assert "target" not in manager._portable_mcp_server_plugins
    assert servers["sibling"]["env"]["GOOD"] == "yes"
    assert "other" in servers
    assert "mcp_env" in caplog.text and "target" in caplog.text


@pytest.mark.parametrize("invalid", [None, [], "not-a-map", 3])
def test_invalid_outer_map_fails_closed_for_plugin_only(tmp_path, discover, caplog, invalid):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {"target": _stdio()})
    _package(home, "beta.tools", {"other": _stdio()})
    manager = discover(home, ["alpha.tools", "beta.tools"], {"alpha.tools": {"mcp_env": invalid}})
    assert set(manager.get_portable_mcp_servers()) == {"other"}
    assert "mcp_env" in caplog.text


def test_empty_and_absent_overrides_preserve_package_config(tmp_path, discover):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {"empty": _stdio(env={"BASE": "kept"}), "absent": _stdio()})
    manager = discover(home, ["alpha.tools"], {"alpha.tools": {"mcp_env": {"empty": {}}}})
    servers = manager.get_portable_mcp_servers()
    assert servers["empty"]["env"]["BASE"] == "kept"
    assert set(servers) == {"empty", "absent"}


def test_unknown_and_remote_targets_cannot_replace_server_or_trust(tmp_path, discover, caplog):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {
        "remote": {"type": "streamable-http", "url": "https://example.test/mcp"},
        "local": _stdio(),
    }, {"remote": {"trust": "untrusted"}, "local": {"trust": "untrusted"}})
    manager = discover(home, ["alpha.tools"], {"alpha.tools": {"mcp_env": {
        "unknown": {"NEW": "not-created"}, "remote": {"JAVA_HOME": "ignored"},
        "local": {"command": "not-an-executable", "trust": "full", "EMPTY": ""},
    }}})
    servers = manager.get_portable_mcp_servers()
    assert set(servers) == {"remote", "local"}
    assert servers["remote"] == {
        "url": "https://example.test/mcp", "strict_redirect_headers": True, "trust": "untrusted"}
    assert servers["local"]["command"] == Path(sys.executable).name
    assert servers["local"]["trust"] == "untrusted"
    assert servers["local"]["env"]["EMPTY"] == ""
    assert "unknown" in caplog.text and "remote" in caplog.text


def test_other_plugins_override_cannot_leak_across_server_name_collision(tmp_path, discover):
    home = tmp_path / "home"
    for name in ("alpha.tools", "beta.tools"):
        _package(home, name, {"shared": _stdio(env={"OWNER": name})})
    manager = discover(home, ["alpha.tools", "beta.tools"], {
        "beta.tools": {"mcp_env": {"shared": {"BETA_ONLY": "yes"}}},
    })
    owner = manager._portable_mcp_server_plugins["shared"]
    env = manager.get_portable_mcp_servers()["shared"]["env"]
    assert env["OWNER"] == owner
    assert ("BETA_ONLY" in env) == (owner == "beta.tools")


def test_config_set_roundtrip_env_refs_and_profile_isolation(tmp_path, discover, monkeypatch, capsys):
    from hermes_cli.config import config_command, read_user_config_raw

    home = tmp_path / "first"
    _package(home, "alpha.tools", {"worker": _stdio()})
    discover(home, ["alpha.tools"], {"unrelated": {"settings": {"keep": True}}})
    monkeypatch.setenv("PLUGIN_ENV_TEST_REF", "resolved-setting")
    key = r"plugins.entries.alpha\.tools.mcp_env.worker"
    config_command(Namespace(
        config_command="set", key=key,
        value='{"ENGINE_DIR": "C:/Engine", "FROM_ENV": "${PLUGIN_ENV_TEST_REF}"}',
    ))  # --force is deliberately absent, including on first creation.
    assert "not a recognized config key" not in capsys.readouterr().out
    raw = read_user_config_raw()
    assert raw["plugins"]["entries"]["unrelated"] == {"settings": {"keep": True}}
    assert raw["plugins"]["entries"]["alpha.tools"]["mcp_env"]["worker"]["FROM_ENV"] == "${PLUGIN_ENV_TEST_REF}"
    # Re-discover the same on-disk config, without using fixture config rewriting.
    first = PluginManager()
    try:
        first.discover_and_load()
        assert first.get_portable_mcp_servers()["worker"]["env"]["FROM_ENV"] == "resolved-setting"
        second_home = tmp_path / "second"
        _package(second_home, "alpha.tools", {"worker": _stdio()})
        second = discover(second_home, ["alpha.tools"], {})
        assert "FROM_ENV" not in second.get_portable_mcp_servers()["worker"]["env"]
        assert first.get_portable_mcp_servers()["worker"]["env"]["ENGINE_DIR"] == "C:/Engine"
    finally:
        for key in list(first._plugins):
            first.unload(key)


def test_invalid_value_is_not_logged_and_no_partial_merge(tmp_path, discover, caplog):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {"worker": _stdio(env={"BASE": "kept"})})
    marker = "sensitive-looking-value-never-log"
    manager = discover(home, ["alpha.tools"], {"alpha.tools": {"mcp_env": {
        "worker": {"VALID_FIRST": "yes", "INVALID": [marker]},
    }}})
    assert manager.get_portable_mcp_servers() == {}
    assert "mcp_env" in caplog.text
    assert marker not in caplog.text


@pytest.mark.platforms("windows")
def test_windows_env_overrides_replace_case_insensitive_package_default(tmp_path, discover):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {"worker": _stdio(env={"JAVA_HOME": "default"})})
    manager = discover(home, ["alpha.tools"], {
        "alpha.tools": {"mcp_env": {"worker": {"java_home": "override"}}},
    })
    env = manager.get_portable_mcp_servers()["worker"]["env"]
    assert {key: value for key, value in env.items() if key.upper() == "JAVA_HOME"} == {
        "java_home": "override"}


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("override", [
    {"JAVA_HOME": "one", "java_home": "two"}, {"plugin_root": "replacement"},
])
def test_windows_ambiguous_or_reserved_env_fails_closed(tmp_path, discover, override):
    home = tmp_path / "home"
    _package(home, "alpha.tools", {"worker": _stdio()})
    manager = discover(home, ["alpha.tools"], {"alpha.tools": {"mcp_env": {"worker": override}}})
    assert manager.get_portable_mcp_servers() == {}


def test_override_does_not_mutate_package_or_shared_config_cache(monkeypatch):
    from hermes_cli import config
    from hermes_cli.plugins_mcp_env import portable_mcp_configs_with_env

    cached = {"plugins": {"entries": {"alpha.tools": {"mcp_env": {
        "worker": {"JAVA_HOME": "user"},
    }}}}}
    servers = {"worker": {"command": "python", "env": {"BASE": "package", "JAVA_HOME": "default"}}}
    monkeypatch.setattr(config, "load_config_readonly", lambda: cached)
    result = portable_mcp_configs_with_env("alpha.tools", servers)
    result["worker"]["env"]["BASE"] = "modified-result"
    result["worker"]["env"]["JAVA_HOME"] = "modified-result"
    assert servers["worker"]["env"] == {"BASE": "package", "JAVA_HOME": "default"}
    assert cached["plugins"]["entries"]["alpha.tools"]["mcp_env"]["worker"] == {"JAVA_HOME": "user"}


def test_env_override_does_not_modify_same_name_native_config(tmp_path, discover, monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from tools.mcp_tool_config import _portable_mcp_servers

    home = tmp_path / "home"
    _package(home, "alpha.tools", {"worker": _stdio()})
    native = {"worker": {"command": "native-server", "env": {"JAVA_HOME": "native"}, "trust": "untrusted"}}
    manager = discover(home, ["alpha.tools"], {
        "alpha.tools": {"mcp_env": {"worker": {"JAVA_HOME": "plugin-only"}}},
    }, mcp_servers=native)
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda **kwargs: None)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    _portable_mcp_servers(native)
    assert native == {
        "worker": {"command": "native-server", "env": {"JAVA_HOME": "native"}, "trust": "untrusted"}}
