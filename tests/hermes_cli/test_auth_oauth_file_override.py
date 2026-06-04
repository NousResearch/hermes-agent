import json
from pathlib import Path

import pytest


def test_auth_file_path_defaults_to_hermes_home(tmp_path, monkeypatch):
    from hermes_cli.auth import _auth_file_path

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_OAUTH_FILE", raising=False)

    assert _auth_file_path() == hermes_home / "auth.json"


def test_auth_file_path_uses_hermes_oauth_file_override(tmp_path, monkeypatch):
    from hermes_cli.auth import _auth_file_path

    hermes_home = tmp_path / "hermes"
    shared_auth = tmp_path / "shared" / "auth.json"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_OAUTH_FILE", str(shared_auth))

    assert _auth_file_path() == shared_auth


def test_auth_file_path_ignores_blank_hermes_oauth_file(tmp_path, monkeypatch):
    from hermes_cli.auth import _auth_file_path

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_OAUTH_FILE", "  ")

    assert _auth_file_path() == hermes_home / "auth.json"


def test_auth_file_path_expands_user_override(tmp_path, monkeypatch):
    from hermes_cli.auth import _auth_file_path

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_OAUTH_FILE", "~/custom-auth.json")

    assert _auth_file_path() == fake_home / "custom-auth.json"


def test_auth_lock_path_follows_hermes_oauth_file(tmp_path, monkeypatch):
    from hermes_cli.auth import _auth_lock_path

    shared_auth = tmp_path / "shared" / "auth.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_OAUTH_FILE", str(shared_auth))

    assert _auth_lock_path() == shared_auth.with_suffix(".lock")


def test_save_and_load_auth_store_use_hermes_oauth_file(tmp_path, monkeypatch):
    from hermes_cli.auth import _load_auth_store, _save_auth_store

    hermes_home = tmp_path / "hermes"
    shared_auth = tmp_path / "shared" / "auth.json"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_OAUTH_FILE", str(shared_auth))

    saved_to = _save_auth_store({
        "providers": {
            "nous": {"access_token": "shared-token"},
        },
        "active_provider": "nous",
    })

    assert saved_to == shared_auth
    assert shared_auth.exists()
    assert not (hermes_home / "auth.json").exists()
    loaded = _load_auth_store()
    assert loaded["providers"]["nous"]["access_token"] == "shared-token"


def test_pytest_seat_belt_checks_hermes_oauth_file_final_path(monkeypatch):
    from hermes_cli.auth import _auth_file_path

    real_home_auth = Path.home() / ".hermes" / "auth.json"
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-test-home")
    monkeypatch.setenv("HERMES_OAUTH_FILE", str(real_home_auth))

    with pytest.raises(RuntimeError, match="Refusing to touch real user auth store"):
        _auth_file_path()


def test_hermes_oauth_file_disables_profile_global_fallback(tmp_path, monkeypatch):
    from hermes_cli.auth import _global_auth_file_path

    fake_home = tmp_path / "home"
    global_home = fake_home / ".hermes"
    profile_home = global_home / "profiles" / "work"
    shared_auth = tmp_path / "shared" / "auth.json"
    profile_home.mkdir(parents=True)
    shared_auth.parent.mkdir(parents=True)
    shared_auth.write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_OAUTH_FILE", str(shared_auth))

    assert _global_auth_file_path() is None
