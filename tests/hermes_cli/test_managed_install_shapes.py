"""Managed-mode detection across the Nix install shapes.

The NixOS module and the Home Manager module both mark the install as
managed, and the CLI then refuses a configuration change that it cannot
keep. The two modules write different values, and an install from an
earlier version writes an empty marker, so detection must handle all three.
"""

import os

import pytest

from hermes_cli import config as config_mod


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    return home


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("nixos", "nixos"),
        ("home-manager", "home-manager"),
        ("HOME-MANAGER", "home-manager"),
        # An install from an earlier version sets a bare "true". Only the
        # NixOS module did that.
        ("true", "nixos"),
        ("1", "nixos"),
        # Homebrew is not a distribution method, so these must not block a
        # configuration change.
        ("brew", None),
        ("homebrew", None),
        (None, None),
    ],
)
def test_env_var_names_the_managing_system(hermes_home, monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
    else:
        monkeypatch.setenv("HERMES_MANAGED", env_value)

    assert config_mod.get_managed_system() == expected
    assert config_mod.is_managed() is (expected is not None)


@pytest.mark.parametrize(
    ("marker_text", "expected"),
    [
        ("home-manager", "home-manager"),
        ("nixos", "nixos"),
        # An install from an earlier version has an empty marker. Only the
        # NixOS module wrote one.
        ("", "nixos"),
    ],
)
def test_marker_file_names_the_managing_system(
    hermes_home, monkeypatch, marker_text, expected
):
    """An interactive shell reads .managed, not the HERMES_MANAGED of the service."""
    (hermes_home / ".managed").write_text(marker_text, encoding="utf-8")
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    assert config_mod.get_managed_system() == expected
    assert config_mod.is_managed() is True


def test_env_var_wins_over_the_marker(hermes_home, monkeypatch):
    (hermes_home / ".managed").write_text("nixos", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED", "home-manager")

    assert config_mod.get_managed_system() == "home-manager"


@pytest.mark.parametrize("managed_value", ["nixos", "home-manager"])
def test_managed_install_names_its_system_and_offers_an_update(
    hermes_home, monkeypatch, tmp_path, managed_value
):
    """The message names the system, so the user knows what owns the install."""
    monkeypatch.setenv("HERMES_MANAGED", managed_value)

    # This test uses an install tree of its own. The real checkout can carry
    # a stamp from the install shape of the contributor. A stamp answers
    # first, and detection never reaches the managed state under test.
    install_tree = tmp_path / "install"
    install_tree.mkdir()

    assert managed_value in config_mod.format_managed_message("set model")
    assert "set model" in config_mod.format_managed_message("set model")
    assert config_mod.get_managed_update_command()
    assert config_mod.detect_install_method(install_tree) == managed_value
    # `hermes update` cannot run on a managed install, so the advice must not
    # name it.
    assert config_mod.recommended_update_command() != "hermes update"


@pytest.mark.parametrize("managed_value", ["nixos", "home-manager"])
def test_a_stamp_can_name_every_managed_system(
    hermes_home, monkeypatch, tmp_path, managed_value
):
    """A stamp must give back every value that detection can return.

    Detection reads the stamp against an allowlist. A managed system that is
    absent from that allowlist gives "unknown". The update guidance then
    names a command that the managed guard refuses.
    """
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    install_tree = tmp_path / "install"
    install_tree.mkdir()

    (install_tree / ".install_method").write_text(managed_value + "\n", encoding="utf-8")

    assert config_mod.detect_install_method(install_tree) == managed_value


def test_unmanaged_install_offers_no_update_command(hermes_home, monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    assert config_mod.get_managed_update_command() is None


def test_unreadable_marker_still_reports_managed(hermes_home, monkeypatch):
    """A marker we cannot read is still a marker. Fail closed, not open."""
    marker = hermes_home / ".managed"
    marker.write_text("home-manager", encoding="utf-8")
    marker.chmod(0o000)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    try:
        if os.access(marker, os.R_OK):
            pytest.skip("running as a user that ignores file modes (for example root)")
        assert config_mod.is_managed() is True
    finally:
        marker.chmod(0o600)

# ── Home-aware (profile-scoped) detection ────────────────────────────────────
# The dashboard scopes config reads/writes to a non-default profile by setting
# a home override (hermes_constants.set_hermes_home_override). A package
# manager manages the *main* install's config.yaml, not per-profile configs,
# so a profile with no `.managed` marker must stay writable even when the
# global install is managed via HERMES_MANAGED.


@pytest.fixture
def profile_home(tmp_path):
    profile = tmp_path / "profiles" / "maintainer"
    profile.mkdir(parents=True)
    return profile


def test_profile_override_without_marker_is_unmanaged(hermes_home, profile_home, monkeypatch):
    """A profile with no `.managed` marker must stay writable even when the
    global install is managed via HERMES_MANAGED."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    token = set_hermes_home_override(str(profile_home))
    try:
        assert config_mod.get_managed_system() is None
        assert config_mod.is_managed() is False
    finally:
        reset_hermes_home_override(token)

    # Outside the profile scope the global managed state still applies.
    assert config_mod.is_managed() is True


def test_profile_override_with_marker_is_managed(hermes_home, profile_home, monkeypatch):
    """A profile with its own `.managed` marker is managed, independent of the
    process-global HERMES_MANAGED value."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    (profile_home / ".managed").write_text("nixos", encoding="utf-8")
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    token = set_hermes_home_override(str(profile_home))
    try:
        assert config_mod.get_managed_system() == "nixos"
        assert config_mod.is_managed() is True
    finally:
        reset_hermes_home_override(token)


def test_profile_override_ignores_global_env_managed(hermes_home, profile_home, monkeypatch):
    """A profile without a marker stays unmanaged even when the global env var
    names a managing system — the env var must not leak into the profile scope."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    monkeypatch.setenv("HERMES_MANAGED", "home-manager")
    token = set_hermes_home_override(str(profile_home))
    try:
        assert config_mod.get_managed_system() is None
        assert config_mod.is_managed() is False
    finally:
        reset_hermes_home_override(token)
