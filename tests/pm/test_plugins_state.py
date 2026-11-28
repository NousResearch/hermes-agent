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


def test_only_live_profiles_join_the_dependency_union(homes):
    from hermes_constants import mark_named_profile_deleted

    default_home, profile_home = homes
    profiles = profile_home.parent
    _write_config(default_home, ["a-plug"])
    _write_config(profile_home, ["live-plug"])
    for name in (".work.staging-123", "Bad Name", "retired", "ghost"):
        (profiles / name).mkdir()
    for name in (".work.staging-123", "Bad Name", "retired"):
        _write_config(profiles / name, [f"{name}-plug"])
    mark_named_profile_deleted(profiles / "retired")
    (profiles / "ghost" / "plugins").mkdir()  # runtime side-effect dir, no identity marker

    assert pstate.dependency_homes() == [default_home, profile_home]
    by_root = pstate.enabled_plugins_ordered()
    assert set(by_root) == {default_home / "plugins", profile_home / "plugins"}


@pytest.mark.parametrize("boundary", ["profile-listing", "profile-stat", "plugin-stat", "manifest-read", "manifest-stat", "provider-stat"])
def test_unreadable_profile_state_is_not_an_empty_selection(homes, monkeypatch, boundary):
    from pm.workspace import enabled_member_dirs

    _default_home, profile_home = homes
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


@pytest.mark.platforms("posix")  # a self-referencing symlink needs no privilege only on POSIX
def test_unreadable_config_link_is_not_an_empty_selection(homes):
    # Only a missing file means "no config": a symlink loop must refuse, not drop the home's plugins.
    default_home, _ = homes
    config = default_home / "config.yaml"
    config.unlink(missing_ok=True)
    config.symlink_to(config)
    with pytest.raises(ValueError, match=re.escape(str(config))):
        pstate.enabled_plugins_ordered()


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

def test_verify_skips_bad_secondary_but_admission_refuses_it(homes, caplog):
    from pm.workspace import enabled_member_dirs, enabled_plugin_dirs

    default_home, profile_home = homes
    plugin = default_home / "plugins" / "working"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: working\npython_dependencies: [fixturedep]\n", encoding="utf-8")
    _write_config(default_home, ["working"])
    bad = profile_home / "config.yaml"
    bad.write_text("plugins: [invalid]\n", encoding="utf-8")

    assert enabled_member_dirs() == [plugin]
    assert str(bad) in caplog.text
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        enabled_plugin_dirs(proposed_home=default_home, enabled=["working"])
    bad.write_text("plugins: {enabled: []}\n", encoding="utf-8")
    assert enabled_member_dirs() == [plugin]


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
        "modern/pyproject.toml": '[project]\ndependencies = ["fixturedep>=1"]\n',
        "legacy/plugin.yaml": 'name: legacy\npip_dependencies: ["fixturedep>=1"]\n',
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


def test_a_bundled_key_that_left_core_keeps_its_catalog_plugin_a_member(homes):
    """``plugins.enabled: [google_meet]`` (core's key, kept as an alias) still selects ``plugins/google-meet``,
    so a sync keeps Playwright for the plugin the loader loads under that key."""
    from pm.workspace import enabled_member_dirs
    home, _ = homes
    plugin = home / "plugins" / "google-meet"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text('[project]\ndependencies = ["playwright==1.64.0"]\n')
    _write_config(home, ["google_meet"])
    assert enabled_member_dirs() == [plugin]


def test_an_install_in_flight_under_the_catalog_name_counts_for_the_old_key(homes):
    """First install with only ``google_meet`` saved: the pending ``plugins/google-meet`` is the member."""
    from pm.workspace import selected_plugin_dir
    home, _ = homes
    pending = home / "plugins" / "google-meet"
    assert selected_plugin_dir(home / "plugins", "google_meet", installing=pending) == pending


@pytest.mark.parametrize(("enabled", "disabled"), [
    (["google_meet"], ["google-meet"]),   # core's key enabled, the catalog name disabled
    (["google-meet"], ["google_meet"]),   # and the other way round: the loader honours both spellings
])
def test_a_left_core_plugin_disabled_under_either_name_adds_no_dependencies(homes, enabled, disabled):
    """The loader keeps the plugin off, so its pins (Playwright) must not join the union where they could
    block an unrelated install."""
    import hermes_yaml as yaml
    from pm.workspace import enabled_member_dirs
    home, _ = homes
    plugin = home / "plugins" / "google-meet"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text('[project]\ndependencies = ["playwright==1.64.0"]\n')
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"plugins": {"enabled": enabled, "disabled": disabled}}, f)
    assert enabled_member_dirs() == []


def test_a_local_plugin_under_an_old_bundled_key_keeps_its_own_dependencies(homes):
    """A home's own ``plugins/google_meet`` is its own plugin to the loader: a disable of the catalog name
    ``google-meet`` does not turn it off, so PM keeps its dependencies too."""
    import hermes_yaml as yaml
    from pm.workspace import enabled_member_dirs
    home, _ = homes
    local = home / "plugins" / "google_meet"
    local.mkdir(parents=True)
    (local / "pyproject.toml").write_text('[project]\ndependencies = ["requests==2.32.3"]\n')
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"plugins": {"enabled": ["google_meet"], "disabled": ["google-meet"]}}, f)
    assert enabled_member_dirs() == [local]


def test_a_staged_catalog_update_never_replaces_a_local_plugin_under_the_old_key(homes):
    """A staged ``plugins/google-meet`` replaces the tree at its own path. While the home's own
    ``plugins/google_meet`` is what ``google_meet`` selects, the union keeps the local plugin, not the
    staged catalog copy, whether or not ``google-meet`` is disabled."""
    import hermes_yaml as yaml
    from pm.workspace import enabled_plugin_entries, selected_plugin_dir
    home, _ = homes
    local = home / "plugins" / "google_meet"
    local.mkdir(parents=True)
    staged = home / "plugins" / "google-meet"
    assert selected_plugin_dir(home / "plugins", "google_meet", installing=staged) == local
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"plugins": {"enabled": ["google_meet"], "disabled": ["google-meet"]}}, f)
    assert [tree for _d, _n, tree in enabled_plugin_entries(installing=staged)] == [local]
