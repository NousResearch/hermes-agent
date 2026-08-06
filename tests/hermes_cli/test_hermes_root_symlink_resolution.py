"""Regression tests for Hermes roots whose profiles directory is symlinked."""

from pathlib import Path

import pytest

from hermes_constants import get_default_hermes_root


@pytest.fixture
def symlinked_profiles_layout(tmp_path, monkeypatch):
    deployment = tmp_path / "deployment"
    hermes_root = deployment / "home"
    real_profiles = deployment / "profiles-real"
    profile = real_profiles / "main"
    profile.mkdir(parents=True)
    hermes_root.mkdir()
    (hermes_root / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (hermes_root / "profiles").symlink_to(
        Path("..") / "profiles-real", target_is_directory=True
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "native-home")
    return hermes_root, profile, hermes_root / "profiles" / "main"


def test_real_profile_path_resolves_to_symlink_owning_hermes_root(
    symlinked_profiles_layout, monkeypatch
):
    hermes_root, real_profile, _linked_profile = symlinked_profiles_layout
    monkeypatch.setenv("HERMES_HOME", str(real_profile))

    assert get_default_hermes_root() == hermes_root


def test_symlinked_profile_path_keeps_literal_hermes_root(
    symlinked_profiles_layout, monkeypatch
):
    hermes_root, _real_profile, linked_profile = symlinked_profiles_layout
    monkeypatch.setenv("HERMES_HOME", str(linked_profile))

    assert get_default_hermes_root() == hermes_root
