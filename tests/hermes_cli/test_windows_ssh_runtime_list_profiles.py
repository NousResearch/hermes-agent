"""list-profiles operation of the Windows SSH runtime helper.

Desktop SSH roster inventory asks a native Windows host which Hermes profiles
it would serve via `python -m hermes_cli.windows_ssh_runtime list-profiles`.
The operation must stay spawn-free and answer from the canonical profile
registry (`default` plus every live, valid named profile directory).
"""

import json

import pytest

from hermes_cli import windows_ssh_runtime


def _make_profile(profiles_dir, name):
    """A named profile is a dir PLUS an identity marker (see hermes_constants)."""
    profile_dir = profiles_dir / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("model: default\n", encoding="utf-8")
    return profile_dir


def _make_install(root, profiles=()):
    """Lay out a Hermes install under *root*: profiles dir + junk entries."""
    profiles_dir = root / "profiles"
    profiles_dir.mkdir(parents=True)
    for name in profiles:
        _make_profile(profiles_dir, name)
    return root


def test_list_profiles_returns_canonical_default_when_no_profiles_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    (tmp_path / "hermes-home").mkdir()

    assert windows_ssh_runtime.dispatch(["list-profiles"]) == {"profiles": ["default"]}


def test_list_profiles_invents_default_plus_valid_named_dirs(tmp_path, monkeypatch):
    home = _make_install(
        tmp_path / "hermes-home",
        ["work", "coder-2", ".hidden", "Bad Name", "bob.rollback-old", "UPPER"],
    )
    # A plain file must not be mistaken for a profile.
    (home / "profiles" / "notes.txt").write_text("not a profile")
    # A dir with no identity marker is not a profile either (cron/log ghosts).
    (home / "profiles" / "ghost").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    assert windows_ssh_runtime.dispatch(["list-profiles"]) == {"profiles": ["default", "coder-2", "work"]}


def test_list_profiles_skips_tombstoned_profiles(tmp_path, monkeypatch):
    home = _make_install(tmp_path / "hermes-home", ["live", "gone"])
    tombstones = home / "profiles" / ".deleted"
    tombstones.mkdir()
    (tombstones / "gone").write_text("")

    monkeypatch.setenv("HERMES_HOME", str(home))

    assert windows_ssh_runtime.dispatch(["list-profiles"]) == {"profiles": ["default", "live"]}


def test_list_profiles_anchors_profile_mode_hermes_home_to_the_root(tmp_path, monkeypatch):
    # HERMES_HOME pointing INTO a profile (<root>/profiles/coder) must still
    # inventory the whole installation — same anchoring get_default_hermes_root
    # applies everywhere else.
    root = tmp_path / "docker-root"
    _make_install(root, ["coder", "ops"])
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))

    assert windows_ssh_runtime.dispatch(["list-profiles"]) == {"profiles": ["default", "coder", "ops"]}


def test_list_profiles_rejects_extra_arguments():
    with pytest.raises(ValueError):
        windows_ssh_runtime.dispatch(["list-profiles", "extra"])


def test_list_profiles_main_emits_single_line_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    (tmp_path / "hermes-home").mkdir()

    import sys

    monkeypatch.setattr(sys, "argv", ["windows_ssh_runtime.py", "list-profiles"])
    windows_ssh_runtime.main()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"profiles": ["default"]}
