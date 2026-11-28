"""A non-main profile discovers plugins installed in the main home (``<root>/plugins``).

Enablement stays per profile (``plugins.enabled``); the shared copy only becomes discoverable. A
profile's own copy wins a same-name conflict, and a shared copy never replaces a bundled plugin.
Driven through ``PluginManager._collect_directory_manifests`` with a real temp root/profile layout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.plugins import PluginManager, resolve_manifest_winners


def _plugin(base: Path, name: str, *, dirname: str | None = None) -> Path:
    plugin_dir = base / (dirname or name)
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(f"name: {name}\nversion: 0.1.0\n")
    return plugin_dir


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile = root / "profiles" / "p"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return type("Layout", (), {"root": root, "profile": profile, "bundled": bundled})


def _found(layout=None):
    return {(m.key, m.source) for m in PluginManager()._collect_directory_manifests()}


def test_non_main_profile_discovers_a_main_home_plugin(layout):
    _plugin(layout.root / "plugins", "foo")

    assert ("foo", "shared") in _found()


def test_main_profile_does_not_scan_its_own_plugins_twice(layout, monkeypatch):
    _plugin(layout.root / "plugins", "foo")
    monkeypatch.setenv("HERMES_HOME", str(layout.root))

    assert _found() == {("foo", "user")}


def test_profile_local_copy_beats_the_shared_copy(layout):
    _plugin(layout.root / "plugins", "foo")
    local = _plugin(layout.profile / "plugins", "foo")

    winner = resolve_manifest_winners(PluginManager()._collect_directory_manifests())["foo"]

    assert (winner.source, Path(winner.path)) == ("user", local)


def test_shared_copy_never_replaces_a_bundled_plugin_of_the_same_name(layout):
    """A shared plugin cannot take over a bundled key: enabling the bundled name must keep running
    the bundled code, whatever sits in the main home."""
    bundled_copy = _plugin(layout.bundled, "foo")
    _plugin(layout.root / "plugins", "foo")

    winner = resolve_manifest_winners(PluginManager()._collect_directory_manifests())["foo"]

    assert (winner.source, Path(winner.path)) == ("bundled", bundled_copy)


def test_shared_impostor_directory_cannot_claim_a_bundled_key(layout):
    _plugin(layout.bundled, "foo")
    _plugin(layout.root / "plugins", "foo", dirname="other-dir")

    winner = resolve_manifest_winners(PluginManager()._collect_directory_manifests())["foo"]

    assert winner.source == "bundled"


def test_shared_discovery_can_be_turned_off(layout):
    _plugin(layout.root / "plugins", "foo")
    assert _found() == {("foo", "shared")}

    (layout.profile / "config.yaml").write_text("plugins:\n  shared_discovery: false\n")

    assert _found() == set()


def test_shared_scan_does_not_follow_a_symlink_out_of_the_main_home(layout, tmp_path):
    outside = _plugin(tmp_path / "outside", "evil")
    _plugin(layout.root / "plugins", "foo")
    (layout.root / "plugins" / "evil").symlink_to(outside, target_is_directory=True)

    assert _found() == {("foo", "shared")}


def test_discovery_does_not_enable_the_shared_plugin(layout):
    from hermes_cli.plugins_discovery import gate_manifest

    _plugin(layout.root / "plugins", "foo")
    (manifest,) = PluginManager()._collect_directory_manifests()

    assert gate_manifest(manifest, set(), set()).action == "placeholder"
    assert gate_manifest(manifest, set(), {"foo"}).action == "load"
