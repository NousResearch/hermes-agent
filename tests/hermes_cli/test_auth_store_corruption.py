import json
import os
import stat

import pytest

from hermes_cli.auth import (
    _AUTH_STORE_CORRUPT_COPY_KEY,
    _AUTH_STORE_LOAD_FAILED_KEY,
    _load_auth_store,
    _save_auth_store,
)


def test_load_auth_store_corrupt_backup_is_unique_and_does_not_clobber(tmp_path):
    auth_file = tmp_path / "auth.json"
    legacy_corrupt = tmp_path / "auth.json.corrupt"
    legacy_corrupt.write_text('{"valid": true}\n', encoding="utf-8")
    auth_file.write_text("{not json", encoding="utf-8")

    store = _load_auth_store(auth_file)

    assert store["providers"] == {}
    assert store[_AUTH_STORE_LOAD_FAILED_KEY]
    assert legacy_corrupt.read_text(encoding="utf-8") == '{"valid": true}\n'

    backups = list(tmp_path.glob("auth.json.corrupt.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not json"
    assert store[_AUTH_STORE_CORRUPT_COPY_KEY] == str(backups[0])
    if os.name == "posix":
        assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600


def test_save_auth_store_refuses_store_loaded_after_parse_failure(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    auth_file = hermes_home / "auth.json"
    original = "{not json"
    auth_file.write_text(original, encoding="utf-8")

    store = _load_auth_store()
    store.setdefault("credential_pool", {})["openrouter"] = [
        {"id": "new", "access_token": "sk-new"}
    ]

    with pytest.raises(RuntimeError, match="Refusing to overwrite auth.json"):
        _save_auth_store(store)

    assert auth_file.read_text(encoding="utf-8") == original


def test_save_auth_store_refuses_explicit_target_loaded_after_parse_failure(tmp_path):
    global_auth = tmp_path / "global-auth.json"
    original = "{not json"
    global_auth.write_text(original, encoding="utf-8")

    store = _load_auth_store(global_auth)
    store.setdefault("credential_pool", {})["openrouter"] = [
        {"id": "new", "access_token": "sk-new"}
    ]

    with pytest.raises(RuntimeError, match="Refusing to overwrite auth.json"):
        _save_auth_store(store, target_path=global_auth)

    assert global_auth.read_text(encoding="utf-8") == original


def test_load_auth_store_valid_json_not_marked_failed(tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {"nous": {"access_token": "tok"}},
                "credential_pool": {"openrouter": []},
            }
        ),
        encoding="utf-8",
    )

    store = _load_auth_store(auth_file)

    assert _AUTH_STORE_LOAD_FAILED_KEY not in store
    assert store["providers"]["nous"]["access_token"] == "tok"
    assert "credential_pool" in store


def test_load_auth_store_read_errors_are_not_treated_as_json_corruption(monkeypatch, tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"version": 1, "providers": {}}', encoding="utf-8")

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("no read")

    monkeypatch.setattr(type(auth_file), "read_text", raise_permission_error)

    with pytest.raises(PermissionError, match="no read"):
        _load_auth_store(auth_file)

    assert not list(tmp_path.glob("auth.json.corrupt.*"))
