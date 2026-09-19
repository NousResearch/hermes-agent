"""pm.plugins_state: complete, order-preserving cross-profile discovery."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pm.plugins_state as pstate


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Default home + one profile, each with a config.yaml."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "work"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "get_default_hermes_root", lambda: default_home
    )
    monkeypatch.setattr(pstate, "_profiles_root", lambda: tmp_path / "profiles")
    return default_home, profile_home


def _write_config(home: Path, enabled: list) -> None:
    import hermes_yaml as yaml

    config = {"plugins": {"enabled": enabled}} if enabled else {"plugins": {}}
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def test_enabled_plugins_ordered_reads_all_homes(homes):
    default_home, profile_home = homes
    _write_config(default_home, ["a-plug", "b-plug"])
    _write_config(profile_home, ["c-plug"])

    by_root = pstate.enabled_plugins_ordered()
    assert by_root.get(default_home / "plugins") == ["a-plug", "b-plug"]
    assert by_root.get(profile_home / "plugins") == ["c-plug"]


@pytest.mark.parametrize("boundary", ["profile-listing", "profile-stat", "config-read", "plugin-stat", "manifest-read", "manifest-stat", "provider-stat"])
def test_unreadable_profile_state_is_not_an_empty_selection(homes, monkeypatch, boundary):
    from pm.workspace import enabled_member_dirs

    default_home, profile_home = homes
    _write_config(profile_home, ["keep-plug"])
    plugin = profile_home / "plugins" / "keep-plug"
    plugin.mkdir(parents=True)
    manifest = plugin / "plugin.yaml"
    manifest.write_text("python_dependencies: [fixture-dep]\n", encoding="utf-8")
    if boundary == "provider-stat":
        (profile_home / "config.yaml").write_text("memory:\n  provider: keep-plug\n", encoding="utf-8")
    method, target = {
        "profile-listing": ("iterdir", profile_home.parent),
        "profile-stat": ("stat", profile_home),
        "config-read": ("read_text", profile_home / "config.yaml"),
        "plugin-stat": ("stat", plugin),
        "manifest-read": ("read_text", manifest),
        "manifest-stat": ("stat", manifest),
        "provider-stat": ("stat", plugin),
    }[boundary]
    original = getattr(Path, method)

    def unreadable(path, *args, **kwargs):
        if path == target:
            raise PermissionError("access denied by fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, unreadable)
    with pytest.raises(ValueError, match=re.escape(str(target))):
        enabled_member_dirs()


def test_missing_config_parser_is_not_an_empty_plugin_selection(homes, monkeypatch):
    import sys

    default_home, _ = homes
    _write_config(default_home, ["keep-plug"])
    before = (default_home / "config.yaml").read_bytes()
    monkeypatch.setitem(sys.modules, "utils", None)
    with pytest.raises(ImportError):
        pstate.enabled_plugins_ordered()
    assert (default_home / "config.yaml").read_bytes() == before



@pytest.mark.parametrize("content", ["{ not yaml", "[]", "plugins: wrong", "plugins:\n  enabled: wrong", "memory: wrong"])
def test_enabled_read_refuses_invalid_existing_config(homes, content):
    default_home, _ = homes
    config = default_home / "config.yaml"
    config.write_text(content, encoding="utf-8")
    before = config.read_bytes()
    with pytest.raises(ValueError, match=re.escape(str(config))):
        pstate.enabled_plugins_ordered()
    assert config.read_bytes() == before


@pytest.mark.parametrize("content", ["", "# empty config\n", "null", "{}", "plugins: {}", "plugins:\n  enabled: []"])
def test_empty_config_is_an_explicit_empty_selection(homes, content):
    default_home, _ = homes
    (default_home / "config.yaml").write_text(content, encoding="utf-8")
    assert pstate.enabled_plugins_ordered() == {}


@pytest.mark.parametrize("enabled, provider, exists, expected", [
    (["z-first", "a-second"], "provider", True, ["z-first", "a-second", "provider"]),
    (["provider"], "provider", True, ["provider"]),
    ([], "ghost", False, []),
])
def test_memory_provider_joins_ordered_selection(homes, enabled, provider, exists, expected):
    import hermes_yaml as yaml
    home, sibling = homes
    if exists:
        (home / "plugins" / provider).mkdir(parents=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": enabled}, "memory": {"provider": provider}}))
    _write_config(sibling, ["sibling"])
    by_root = pstate.enabled_plugins_ordered()
    assert by_root.get(home / "plugins", []) == expected
    assert by_root[sibling / "plugins"] == ["sibling"]


def test_member_discovery_uses_real_enabled_config(homes):
    from pm.workspace import enabled_member_dirs
    home, _ = homes
    manifests = {
        "modern/pyproject.toml": "[project]\n",
        "legacy/plugin.yaml": 'name: legacy\npip_dependencies: ["requests>=2"]\n',
        "plain/plugin.yaml": "name: plain\n",
        "orphan/pyproject.toml": "[project]\n",
    }
    for relative, body in manifests.items():
        path = home / "plugins" / relative
        path.parent.mkdir(parents=True)
        path.write_text(body)
    _write_config(home, ["legacy", "modern", "plain"])
    assert enabled_member_dirs() == [home / "plugins/legacy", home / "plugins/modern"]
    _write_config(home, [])
    assert enabled_member_dirs() == []




def test_read_parses_config_once_per_home(homes, monkeypatch):
    """enabled_plugins_ordered must parse each home's config.yaml once,
    not once for plugins.enabled and again for memory.provider."""
    default_home, profile_home = homes
    _write_config(default_home, ["a-plug"])
    _write_config(profile_home, ["c-plug"])

    import utils

    calls: list = []
    real = utils.fast_safe_load

    def counting(stream):
        calls.append(stream)
        return real(stream)

    monkeypatch.setattr(utils, "fast_safe_load", counting)
    pstate.enabled_plugins_ordered()
    assert len(calls) == 2  # one per home, not one per query
