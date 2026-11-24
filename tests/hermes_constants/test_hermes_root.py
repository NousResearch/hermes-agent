"""``get_hermes_root`` resolves the install root without the memoized helper."""

from __future__ import annotations

from pathlib import Path

import hermes_constants


def test_profile_home_resolves_to_install_root(tmp_path: Path, monkeypatch) -> None:
    native = tmp_path / "native"
    root = tmp_path / "install"
    profile = root / "profiles" / "grid-lead"
    profile.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert hermes_constants.get_hermes_root() == root


def test_default_home_with_hermes_home_unset_is_its_own_root(tmp_path: Path, monkeypatch) -> None:
    native = tmp_path / "hermes"
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert hermes_constants.get_hermes_root() == native


def test_does_not_use_the_memoized_default_root(tmp_path: Path, monkeypatch) -> None:
    """A test that changes HERMES_HOME must not see ``get_default_hermes_root``'s memo."""
    native = tmp_path / "native"
    root = tmp_path / "install"
    profile = root / "profiles" / "grid-lead"
    profile.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    hermes_constants.get_default_hermes_root()
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(
        hermes_constants,
        "get_default_hermes_root",
        lambda **_kwargs: tmp_path / "stale-memo",
    )
    assert hermes_constants.get_hermes_root() == root
