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

    # A default added to the catalog later still reaches the home.
    entries.append(_entry("second"))
    assert default_plugins.install_defaults(home, entries=entries)["installed"] == ["second"]


def test_a_refused_enable_is_retried_until_the_user_decides(tmp_path, monkeypatch):
    """Co-authored with @BearHuddleston's review patch on #154858."""
    import json
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    home = tmp_path / "home"
    for name in ("kept-off", "by-hand"):  # disabled by the user; installed by hand with the enable declined
        (home / "plugins" / name).mkdir(parents=True)
    config = home / "config.yaml"
    config.write_text("plugins:\n  disabled: [kept-off]\n", encoding="utf-8")
    calls: list[str] = []

    def install_into(h, consent=None):
        def install(name):
            calls.append(name)
            installed = name.removeprefix("hermes-")  # hermes-user-off installs as user-off, like hermes-memory-wiki
            (h / "plugins" / installed).mkdir(parents=True, exist_ok=True)  # the clone lands before the enable
            sidecar = h / "plugins" / ".install-metadata.json"
            rows = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            sidecar.write_text(json.dumps({**rows, installed: {"catalog": {"name": name}}}))
            if calls.count(name) == 1:
                return {"ok": False, "error": "enable refused: dependency resolution failed"}
            return {"ok": True}
        return install
    monkeypatch.setattr(lcm, "_install_into", install_into)
    entries = [_entry(name) for name in ("kept-off", "by-hand", "accent-picker", "user-on", "hermes-user-off")]

    first = default_plugins.install_defaults(home, entries=entries)
    assert first["owed"] == ["accent-picker", "user-on", "hermes-user-off"]
    # The user settles two of the leftover trees by hand before the next attempt (by their installed names).
    config.write_text("plugins:\n  enabled: [user-on]\n  disabled: [kept-off, user-off]\n", encoding="utf-8")
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": ["accent-picker"], "owed": []}
    assert calls == ["accent-picker", "user-on", "hermes-user-off", "accent-picker"], "retried the wrong trees"
    assert default_plugins.install_defaults(home, entries=entries) == {"installed": [], "owed": []}


def test_a_home_that_turned_lazy_installs_off_fetches_nothing(tmp_path, installs):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")
    assert default_plugins.install_defaults(home, entries=[_entry("accent-picker")])["skipped"] == "lazy-installs-disabled"
    assert installs == []


def test_an_owed_boot_step_stays_owed_until_it_succeeds(tmp_path, monkeypatch):
    """An owed step that raises on a retry is still owed; only a success clears it."""
    from hermes_cli import boot_bootstrap, post_update
    monkeypatch.setattr(boot_bootstrap, "current_install_identity", lambda root: "a" * 40)
    script = iter([{"ok": True, "owed": True}, OSError("profiles dir unreadable"), {"ok": True}])

    def step():
        outcome = next(script)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("install_default_plugins", step),))
    root = tmp_path / "install"

    def boot():
        return boot_bootstrap.run_boot_bootstrap(root)["home"]

    assert boot()["install_default_plugins"]["owed"]
    assert boot()["install_default_plugins"]["owed"], "a retry that raised cleared the debt"
    assert boot()["install_default_plugins"] == {"ok": True}
    assert boot() == "skipped"


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
