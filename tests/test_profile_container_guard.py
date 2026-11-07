import os

import pytest

from hermes_cli import config
from hermes_cli.backup import _sibling_profile_homes


def _symlink_or_skip(link, target):
    """Skip only when this platform/runtime cannot create symlinks."""
    if not hasattr(os, "symlink"):
        pytest.skip("platform has no os.symlink support")
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"platform/runtime cannot create symlinks: {exc}")


def test_profiles_container_home_fails_before_initialization(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profiles = home / "profiles"
    monkeypatch.setenv("HERMES_HOME", str(profiles))

    with pytest.raises(ValueError, match="profiles container"):
        config.ensure_hermes_home()

    assert not profiles.exists()
    assert not (profiles / "SOUL.md").exists()


def test_custom_root_profiles_container_fails_before_initialization(tmp_path, monkeypatch):
    profiles = tmp_path / "custom-hermes" / "profiles"
    monkeypatch.setenv("HERMES_HOME", str(profiles))

    with pytest.raises(ValueError, match="profiles container"):
        config.ensure_hermes_home()

    assert not profiles.exists()
    assert not (profiles / "SOUL.md").exists()


def test_symlink_alias_to_profiles_container_fails_before_initialization(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profiles = root / "profiles"
    profiles.mkdir(parents=True)
    alias = tmp_path / "profile-storage"
    _symlink_or_skip(alias, profiles)
    monkeypatch.setenv("HERMES_HOME", str(alias))

    with pytest.raises(ValueError, match="profiles container"):
        config.ensure_hermes_home()

    assert not (profiles / "SOUL.md").exists()


def test_symlink_named_profiles_to_named_profile_is_accepted(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "orchestrator"
    profile.mkdir(parents=True)
    alias_root = tmp_path / "alias-root"
    alias_root.mkdir()
    _symlink_or_skip(alias_root / "profiles", profile)
    monkeypatch.setenv("HERMES_HOME", str(alias_root / "profiles"))

    config.ensure_hermes_home()

    assert (profile / "SOUL.md").is_file()


def test_concrete_named_profile_home_is_accepted(tmp_path, monkeypatch):
    profile = tmp_path / ".hermes" / "profiles" / "orchestrator"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    config.ensure_hermes_home()

    assert (profile / "SOUL.md").is_file()


def test_siblings_exclude_markerless_infrastructure(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    invoking = root / "profiles" / "orchestrator"
    (root / "profiles" / "audio_cache").mkdir(parents=True)
    real = root / "profiles" / "writer"
    real.mkdir(parents=True)
    (real / "config.yaml").write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(invoking))

    siblings = _sibling_profile_homes(invoking)

    assert ("writer", real) in siblings
    assert all(name != "audio_cache" for name, _ in siblings)
