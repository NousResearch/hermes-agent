"""Regression coverage for hermes doctor command-link install layouts."""

from pathlib import Path
import sys

import pytest

from hermes_cli import doctor_platform
import hermes_constants


@pytest.mark.skipif(sys.platform != "linux", reason="FHS install layout is Linux-only")
def test_fhs_checkout_uses_system_link_even_for_non_root(monkeypatch):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    # The active checkout path, not effective UID or a stale legacy checkout,
    # identifies a system-wide FHS install for all its users.
    location, label = doctor_platform._command_link_location(Path("/usr/local/lib/hermes-agent"))
    assert (location, label) == (Path("/usr/local/bin"), "/usr/local/bin")


def test_termux_precedes_fhs_check(monkeypatch):
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setenv("TERMUX_VERSION", "1")
    location, label = doctor_platform._command_link_location(Path("/usr/local/lib/hermes-agent"))
    assert (location, label) == (Path("/data/data/com.termux/files/usr/bin"), "$PREFIX/bin")


def test_profile_home_resolves_preserved_real_home(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    profile_home = tmp_path / "profile" / "home"
    real_home = tmp_path / "custom-installer-home"
    monkeypatch.setattr(Path, "home", lambda: profile_home)
    monkeypatch.setattr(hermes_constants, "get_subprocess_home", lambda: str(profile_home))
    monkeypatch.setattr(hermes_constants, "get_real_home", lambda: str(real_home))
    location, label = doctor_platform._command_link_location(tmp_path / "checkout")
    assert (location, label) == (real_home / ".local/bin", "~/.local/bin")


def test_non_isolated_custom_home_is_respected(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    custom_home = tmp_path / "user-chosen-home"
    monkeypatch.setattr(Path, "home", lambda: custom_home)
    monkeypatch.setattr(hermes_constants, "get_subprocess_home", lambda: None)
    location, label = doctor_platform._command_link_location(tmp_path / "checkout")
    assert (location, label) == (custom_home / ".local/bin", "~/.local/bin")


def test_existing_profile_helper_does_not_override_real_home(monkeypatch, tmp_path):
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    custom_home = tmp_path / "user-chosen-home"
    profile_home = tmp_path / "different-profile-home"
    monkeypatch.setattr(Path, "home", lambda: custom_home)
    monkeypatch.setattr(hermes_constants, "get_subprocess_home", lambda: str(profile_home))
    location, label = doctor_platform._command_link_location(tmp_path / "checkout")
    assert (location, label) == (custom_home / ".local/bin", "~/.local/bin")
