"""POSIX mode invariants for a relocated provider credential residence."""

import os
import stat

import pytest

from hermes_constants import (
    reset_hermes_home_override,
    secure_parent_dir,
    set_hermes_home_override,
)
from hermes_cli.auth import (
    _auth_store_lock,
    _load_auth_store,
    _save_auth_store,
    _store_provider_state,
)


def _persist_named_profile(runtime_root, profile):
    token = set_hermes_home_override(runtime_root / "profiles" / profile)
    try:
        with _auth_store_lock():
            store = _load_auth_store()
            _store_provider_state(
                store,
                "nous",
                {"access_token": "secret"},
                set_active=True,
            )
            return _save_auth_store(store)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_named_profile_first_auth_residence_modes(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(runtime_root))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    previous_umask = os.umask(0o022)
    try:
        auth_file = _persist_named_profile(runtime_root, "work")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(residence.stat().st_mode) == 0o700
    assert stat.S_IMODE((residence / "profiles").stat().st_mode) == 0o700
    assert stat.S_IMODE((residence / "profiles" / "work").stat().st_mode) == 0o700
    assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_auth_residence_modes_do_not_change_external_parent(monkeypatch, tmp_path):
    external_parent = tmp_path / "operator-owned"
    external_parent.mkdir(mode=0o751)
    external_parent.chmod(0o751)
    runtime_root = tmp_path / "runtime"
    residence = external_parent / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(runtime_root))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    previous_umask = os.umask(0o022)
    try:
        _persist_named_profile(runtime_root, "work")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(external_parent.stat().st_mode) == 0o751


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced")
def test_auth_residence_modes_do_not_follow_internal_symlink(
    monkeypatch, tmp_path
):
    runtime_root = tmp_path / "runtime"
    residence = tmp_path / "auth-residence"
    external_parent = tmp_path / "operator-owned"
    external_profile = external_parent / "work"
    residence.mkdir()
    external_profile.mkdir(parents=True)
    external_profile.chmod(0o751)
    (residence / "profiles").symlink_to(external_parent, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(runtime_root))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    secure_parent_dir(residence / "profiles" / "work" / "auth.json")

    assert stat.S_IMODE(residence.stat().st_mode) == 0o700
    assert stat.S_IMODE(external_profile.stat().st_mode) == 0o751
