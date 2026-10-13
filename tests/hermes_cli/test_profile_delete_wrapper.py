"""`hermes profile delete` removes the profile's wrapper (#131753).

Windows wrappers are ``<alias>.bat``; delete used to look only for the bare name, so the
``.bat`` survived as an orphan alias. A Windows wrapper written before ``.bat`` (#34796) is
bare-named and must still go.
"""

from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.profiles import create_profile, create_wrapper_script, delete_profile


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    for stub in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service", "_stop_profile_backends"):
        monkeypatch.setattr(profiles, stub, lambda *a, **k: None)
    return tmp_path


def _delete_and_check(profile_dir, wrapper):
    delete_profile("imported", yes=True)
    assert not profile_dir.exists()
    assert not wrapper.exists()


@pytest.mark.platforms("any")  # the wrapper name differs per host (.bat on Windows)
def test_delete_removes_the_profile_wrapper(home):
    profile_dir = create_profile("imported", no_alias=True)
    wrapper = create_wrapper_script("imported")
    assert wrapper is not None
    _delete_and_check(profile_dir, wrapper)


@pytest.mark.platforms("windows")
def test_delete_removes_a_pre_bat_windows_wrapper(home):
    profile_dir = create_profile("imported", no_alias=True)
    wrapper = home / ".local" / "bin" / "imported"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("@echo off\r\nhermes -p imported %*\r\n", encoding="utf-8")
    _delete_and_check(profile_dir, wrapper)
