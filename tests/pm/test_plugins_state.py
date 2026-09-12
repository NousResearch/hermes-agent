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


@pytest.mark.parametrize("boundary", ["profile-listing", "profile-stat", "config-read", "plugin-stat", "manifest-read", "provider-stat"])
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


def test_enabled_list_preserves_config_order(homes):
    default_home, _ = homes
    # NOT alphabetical: recency order must survive the read
    _write_config(default_home, ["z-first-enabled", "a-second"])
    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["z-first-enabled", "a-second"]


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


def test_active_memory_provider_joins_union(homes, tmp_path):
    """The mnemosyne path: a provider installed via memory.provider (not
    plugins.enabled) must join the union — deps ride the lock either way."""
    default_home, _ = homes
    provider_dir = default_home / "plugins" / "mnemosyne-like"
    provider_dir.mkdir(parents=True)
    (provider_dir / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    import hermes_yaml as yaml

    with (default_home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"plugins": {"enabled": ["regular-plug"]},
             "memory": {"provider": "mnemosyne-like"}},
            f,
        )

    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["regular-plug", "mnemosyne-like"]


def test_memory_provider_without_dir_is_skipped(homes):
    """memory.provider set but no plugin dir on disk — not a member."""
    default_home, _ = homes
    import hermes_yaml as yaml

    with (default_home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"memory": {"provider": "ghost-provider"}}, f)

    assert pstate.enabled_plugins_ordered() == {}


def test_memory_provider_already_enabled_not_duplicated(homes):
    default_home, _ = homes
    (default_home / "plugins" / "dual").mkdir(parents=True)
    import hermes_yaml as yaml

    with (default_home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"plugins": {"enabled": ["dual"]}, "memory": {"provider": "dual"}}, f
        )

    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["dual"]  # once, not twice


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
