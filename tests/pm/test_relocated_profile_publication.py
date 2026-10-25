"""A live profile that is a symlink onto another volume is still Hermes state.

The dependency root is unresolved, so ``target.resolve()`` leaves it. Publication
may use that profile's real ``plugins/<name>`` (and the profile dir / its
config.yaml) and must not treat the rest of the volume as a root. Staging
overlap and a symlinked plugin directory stay rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm.environments import contained_in_dependency_state


def _relocated(tmp_path: Path, name: str = "claude") -> tuple[Path, Path]:
    """Return ``(profile_link, volume_dir)`` under the PM-isolated Hermes home."""
    home = tmp_path / ".hermes"
    volume = tmp_path / "hc-volume" / name
    volume.mkdir(parents=True)
    (volume / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    profiles = home / "profiles"
    profiles.mkdir(exist_ok=True)
    link = profiles / name
    link.symlink_to(volume, target_is_directory=True)
    return link, volume


def _staged(tmp_path: Path, name: str) -> Path:
    staged = tmp_path / f"staged-{name}"
    staged.mkdir()
    (staged / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    return staged


def test_symlinked_live_profile_plugin_is_contained(tmp_path):
    link, volume = _relocated(tmp_path)
    target = link / "plugins" / "homeassistant"
    assert not target.resolve().is_relative_to((tmp_path / ".hermes").resolve())
    assert contained_in_dependency_state(target, role="plugin")
    assert contained_in_dependency_state(volume, role="home")
    assert contained_in_dependency_state(volume / "config.yaml", role="config")


def test_volume_sibling_and_unregistered_symlink_are_not_contained(tmp_path):
    link, volume = _relocated(tmp_path)
    sibling = tmp_path / "hc-volume" / "other" / "plugins" / "homeassistant"
    assert not contained_in_dependency_state(sibling, role="plugin")
    assert not contained_in_dependency_state(volume.parent, role="home")
    assert not contained_in_dependency_state(volume / "not-plugins" / "homeassistant", role="plugin")

    stray = (tmp_path / ".hermes" / "profiles" / "stray")
    stray_volume = tmp_path / "hc-volume" / "stray"
    stray_volume.mkdir()
    stray.symlink_to(stray_volume, target_is_directory=True)
    assert not contained_in_dependency_state(stray_volume / "plugins" / "homeassistant", role="plugin")
    assert link.name == "claude"


def test_tombstoned_profile_is_not_a_publication_root(tmp_path):
    _link, volume = _relocated(tmp_path, "claude")
    deleted = tmp_path / ".hermes" / "profiles" / ".deleted"
    deleted.mkdir()
    (deleted / "claude").write_text("deleted\n", encoding="utf-8")
    assert not contained_in_dependency_state(volume / "plugins" / "homeassistant", role="plugin")
    assert not contained_in_dependency_state(volume, role="home")


def test_staged_plugin_accepts_relocated_profile_and_keeps_the_other_checks(tmp_path):
    from pm.publication import StagedPlugin

    link, volume = _relocated(tmp_path)
    target = link / "plugins" / "homeassistant"
    staged = _staged(tmp_path, "homeassistant")
    record = {"revision": "new", "source": "https://example.test/homeassistant.git", "pinned": False}
    plugin = StagedPlugin({
        "staged": str(staged), "target": str(target), "target_digest": None,
        "old_metadata": {}, "new_metadata": {"homeassistant": record},
    })
    assert plugin.target.resolve() == (volume / "plugins" / "homeassistant").resolve()

    overlap = volume / "plugins" / "homeassistant" / "inner"
    overlap.mkdir(parents=True)
    (overlap / "plugin.yaml").write_text("name: homeassistant\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escape or overlap"):
        StagedPlugin({
            "staged": str(overlap), "target": str(target), "target_digest": None,
            "old_metadata": {}, "new_metadata": {"homeassistant": record},
        })

    real = volume / "plugins" / "real-plugin"
    real.mkdir(parents=True)
    linked_name = "linkedplugin"
    linked = volume / "plugins" / linked_name
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="escape or overlap"):
        StagedPlugin({
            "staged": str(_staged(tmp_path, linked_name)),
            "target": str(link / "plugins" / linked_name),
            "target_digest": None, "old_metadata": {},
            "new_metadata": {linked_name: record},
        })

    sibling = tmp_path / "hc-volume" / "other" / "plugins" / "homeassistant"
    with pytest.raises(ValueError, match="escape or overlap"):
        StagedPlugin({
            "staged": str(staged), "target": str(sibling), "target_digest": None,
            "old_metadata": {}, "new_metadata": {"homeassistant": record},
        })


def test_plugin_selection_accepts_the_relocated_profile_home_only(tmp_path):
    from pm.publication import PluginSelection

    _link, volume = _relocated(tmp_path)
    selection = PluginSelection({"home": str(volume), "enabled": [], "disabled": []})
    assert selection.home == volume.resolve()

    sibling = tmp_path / "hc-volume" / "other"
    sibling.mkdir()
    with pytest.raises(ValueError, match="outside Hermes state"):
        PluginSelection({"home": str(sibling), "enabled": [], "disabled": []})


def test_recovery_accepts_a_relocated_profile_plugin_target(tmp_path):
    from hermes_cli.runtime_state import _recover_plugin_publication

    link, _volume = _relocated(tmp_path)
    target = link / "plugins" / "homeassistant"
    target.parent.mkdir(parents=True)
    journal = tmp_path / "publication.json"
    journal.write_text("{}", encoding="utf-8")
    row = {
        "target": str(target),
        "backup": str(target.parent / ".previous-abc"),
        "metadata": str(target.parent / ".install-metadata.json"),
        "committed": True,
        "facts_before": "unused",
    }
    _recover_plugin_publication(tmp_path / "project", row, journal)
    assert not journal.exists()

    outside = tmp_path / "hc-volume" / "other" / "plugins" / "homeassistant"
    journal.write_text(json.dumps({"kept": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="escape their home"):
        _recover_plugin_publication(tmp_path / "project", {
            "target": str(outside),
            "backup": str(outside.parent / ".previous-abc"),
            "metadata": str(outside.parent / ".install-metadata.json"),
            "committed": True,
        }, journal)
    assert journal.exists()
