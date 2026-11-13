"""Catalog ``default: true`` plugins: every home gets each one installed and enabled once.

Real config.yaml files under temp homes and the real catalog parser; the network install is a stand-in
that places the plugin directory (the live install is exercised end to end outside the unit suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm
from hermes_cli import default_plugins
from hermes_cli.plugin_catalog import entry_from_mapping


def _entry(name: str, tier: str = "official"):
    return entry_from_mapping({"name": name, "repo": "https://github.com/NousResearch/hermes-official-plugins",
                               "sha": "a" * 40, "description": "d", "maintainer": "NousResearch", "tier": tier,
                               "default": True}, name)


@pytest.fixture
def installs(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)  # the suite turns fetching off
    calls: list[tuple[Path, str]] = []

    def install_into(home, consent=None):
        def install(name):
            calls.append((home, name))
            (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    return calls


def test_default_is_installed_once_and_a_removal_sticks(tmp_path, installs):
    home = tmp_path / "home"
    home.mkdir()
    entries = [_entry("accent-picker")]

    assert default_plugins.install_defaults(home, entries=entries)["installed"] == ["accent-picker"]
    assert installs == [(home, "accent-picker")]

    (home / "plugins" / "accent-picker").rmdir()  # the user removes it
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": [], "owed": []}
    assert installs == [(home, "accent-picker")], "a removed default came back"
    assert not (home / "plugins" / "accent-picker").exists()

    # A default added to the catalog later still reaches the home; a tree a refused enable left on disk
    # (in neither plugins.enabled nor plugins.disabled) is installed again, not recorded as done.
    entries.append(_entry("second"))
    (home / "plugins" / "second").mkdir()
    assert default_plugins.install_defaults(home, entries=entries)["installed"] == ["second"]


def test_only_official_and_verified_entries_can_be_default():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_plugin_catalog.py"
    spec = importlib.util.spec_from_file_location("validate_plugin_catalog", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    assert _entry("o").default and _entry("v", tier="verified").default
    assert _entry("c", tier="community").default is False
    errors, _ = validator.validate_entry({"name": "c", "repo": "https://github.com/a/b", "sha": "a" * 40,
                                          "description": "d", "maintainer": "m", "tier": "community",
                                          "default": True})
    assert any("default" in error for error in errors)
