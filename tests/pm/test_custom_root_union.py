"""E2E: custom HERMES_HOME roots (Docker, non-default local) join the
plugin-deps union the same as standard ~/.hermes ones.

get_default_hermes_root() is the ONE authority: HERMES_HOME outside the
default (e.g. /opt/data in Docker) → that root directly; profile-mode
HERMES_HOME (<root>/profiles/<name>) → <root>. The union's profile scan
and the bisect disable write-back must both flow through it — the bug
this guards: hardcoded Path.home()/.hermes/profiles silently omitted
custom-root profiles (enabled dep plugins never joined the union; disable
decisions never wrote back).
"""

from __future__ import annotations

from pathlib import Path

import yaml

import pm.plugins_state as pstate
import pm.workspace as ws


def _write_enabled(home: Path, enabled: list) -> None:
    home.mkdir(parents=True, exist_ok=True)
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"plugins": {"enabled": enabled}}, f)


def _make_dep_plugin(plugins_dir: Path, name: str) -> Path:
    plug = plugins_dir / name
    plug.mkdir(parents=True)
    (plug / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (plug / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["pyfiglet==1.0.2"]\n',
        encoding="utf-8",
    )
    return plug


def test_custom_hermes_home_profile_joins_union(tmp_path, monkeypatch):
    """Docker shape: HERMES_HOME=/opt/data-like root, a profile under it
    with an enabled dep plugin. Discovery must find the member through
    the REAL get_default_hermes_root (no path mocks)."""
    custom_root = tmp_path / "opt-data"  # the /opt/data shape
    profile_home = custom_root / "profiles" / "worker"
    _write_enabled(profile_home, ["dep-plug"])
    _make_dep_plugin(profile_home / "plugins", "dep-plug")

    monkeypatch.setenv("HERMES_HOME", str(custom_root))
    import hermes_constants

    # the real authority must resolve the custom root (the contract this
    # test exists to pin: env → get_default_hermes_root → union)
    assert hermes_constants.get_default_hermes_root() == custom_root

    # union discovery: the profile's enabled dep plugin IS a member
    members = ws.enabled_member_dirs()
    assert [p.name for p in members] == ["dep-plug"], (
        "custom-root profile's enabled dep plugin missing from the union — "
        "profile scan must derive from get_default_hermes_root"
    )
    # ordered enabled reads see it too
    ordered = pstate.enabled_plugins_ordered()
    assert ordered.get(profile_home / "plugins") == ["dep-plug"]


def test_custom_hermes_home_disable_writes_back(tmp_path, monkeypatch):
    """Bisect disable decisions must write back to the CUSTOM-root
    profile's config — the resolver disabling a plugin there updates
    that profile's enabled list."""
    custom_root = tmp_path / "opt-data"
    profile_home = custom_root / "profiles" / "worker"
    _write_enabled(profile_home, ["bad-plug", "keep-plug"])
    _make_dep_plugin(profile_home / "plugins", "bad-plug")
    _make_dep_plugin(profile_home / "plugins", "keep-plug")

    monkeypatch.setenv("HERMES_HOME", str(custom_root))

    removed = pstate.disable_plugins(["bad-plug"])
    assert removed[str(profile_home)] == ["bad-plug"]

    # the profile's config reflects the removal
    with (profile_home / "config.yaml").open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    assert cfg["plugins"]["enabled"] == ["keep-plug"]

    # and the union no longer sees the disabled member
    members = ws.enabled_member_dirs()
    assert [p.name for p in members] == ["keep-plug"]


def test_standard_layout_still_works(tmp_path, monkeypatch):
    """The default-home path: profile under the (monkeypatched) default
    root, standard layout. Guards the derivation change didn't break the
    common case."""
    default_root = tmp_path / "home"  # stands in for ~/.hermes
    profile_home = default_root / "profiles" / "coder"
    _write_enabled(profile_home, ["dep-plug"])
    _make_dep_plugin(profile_home / "plugins", "dep-plug")

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "get_default_hermes_root", lambda: default_root
    )
    # NOTE: pm.plugins_state imports the function lazily inside
    # _profiles_root, so the attribute patch reaches it.

    members = ws.enabled_member_dirs()
    assert [p.name for p in members] == ["dep-plug"]


def test_profile_mode_hermes_home_resolves_parent(tmp_path, monkeypatch):
    """Profile-mode shape: HERMES_HOME=<root>/profiles/<name> — the
    authority returns <root> so sibling profiles join the union."""
    root = tmp_path / "data-root"
    active_home = root / "profiles" / "active"
    sibling_home = root / "profiles" / "sibling"
    _write_enabled(sibling_home, ["dep-plug"])
    _make_dep_plugin(sibling_home / "plugins", "dep-plug")
    active_home.mkdir(parents=True)  # active profile has no dep plugins

    monkeypatch.setenv("HERMES_HOME", str(active_home))
    import hermes_constants

    assert hermes_constants.get_default_hermes_root() == root

    members = ws.enabled_member_dirs()
    assert [p.name for p in members] == ["dep-plug"], (
        "profile-mode HERMES_HOME must not hide sibling profiles' plugins "
        "from the union"
    )
