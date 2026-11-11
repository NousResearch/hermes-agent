"""An unparseable auth.json is copied to a unique file and is never written over."""

import json
import os
import stat

import pytest

from hermes_cli import auth


def test_corrupt_copy_is_unique_and_keeps_an_earlier_copy(tmp_path):
    auth_file = tmp_path / "auth.json"
    earlier = tmp_path / "auth.json.corrupt"
    earlier.write_text('{"providers": {"nous": {"api_key": "older"}}}', encoding="utf-8")

    for payload in ("{not json", "{still not json"):
        auth_file.write_text(payload, encoding="utf-8")
        assert auth._load_auth_store(auth_file)["providers"] == {}

    assert earlier.read_text(encoding="utf-8") == '{"providers": {"nous": {"api_key": "older"}}}'
    copies = sorted(tmp_path.glob("auth.json.corrupt.*"))
    assert sorted(c.read_text(encoding="utf-8") for c in copies) == ["{not json", "{still not json"]
    if os.name == "posix":
        assert all(stat.S_IMODE(c.stat().st_mode) == 0o600 for c in copies)


def test_save_refuses_to_overwrite_the_corrupt_active_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{not json", encoding="utf-8")

    store = auth._load_auth_store()
    store.setdefault("credential_pool", {})["openrouter"] = [{"id": "new", "access_token": "sk-new"}]
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        auth._save_auth_store(store)

    assert auth_file.read_text(encoding="utf-8") == "{not json"


def test_save_refuses_an_explicit_target_loaded_from_a_corrupt_file(tmp_path):
    target = tmp_path / "global-auth.json"
    target.write_text("{not json", encoding="utf-8")

    store = auth._load_auth_store(target)
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        auth._save_auth_store(store, target_path=target)

    assert target.read_text(encoding="utf-8") == "{not json"


def test_store_from_a_healthy_file_still_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"version": 1, "providers": {"nous": {"api_key": "tok"}}}), encoding="utf-8")

    store = auth._load_auth_store()
    store["providers"]["nous"]["api_key"] = "tok2"
    auth._save_auth_store(store)

    assert json.loads(auth_file.read_text(encoding="utf-8"))["providers"]["nous"]["api_key"] == "tok2"
    assert not list(tmp_path.glob("auth.json.corrupt*"))


def test_store_from_a_missing_file_still_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = auth._load_auth_store()
    store["providers"]["nous"] = {"api_key": "tok"}
    auth._save_auth_store(store)
    assert json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))["providers"]["nous"]["api_key"] == "tok"


def test_repeated_loads_of_the_same_corrupt_bytes_keep_one_copy_and_warn_once(tmp_path, caplog):
    # _load_auth_store runs on every credential lookup; a corrupt store must not fan out into
    # one secret-bearing copy (and one warning) per call.
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{not json", encoding="utf-8")

    with caplog.at_level("DEBUG", logger="hermes_cli.auth_store_corruption"):
        stores = [auth._load_auth_store(auth_file) for _ in range(5)]

    copies = list(tmp_path.glob("auth.json.corrupt*"))
    assert len(copies) == 1
    assert {s["_auth_store_corrupt_copy"] for s in stores} == {str(copies[0])}
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "failed to parse" in r.getMessage()]
    assert len(warnings) == 1

    auth_file.write_text("{different corruption", encoding="utf-8")
    auth._load_auth_store(auth_file)
    assert len(list(tmp_path.glob("auth.json.corrupt*"))) == 2


def test_an_identical_legacy_copy_is_reused(tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{not json", encoding="utf-8")
    legacy = tmp_path / "auth.json.corrupt"
    legacy.write_text("{not json", encoding="utf-8")

    store = auth._load_auth_store(auth_file)

    assert store["_auth_store_corrupt_copy"] == str(legacy)
    assert list(tmp_path.glob("auth.json.corrupt*")) == [legacy]
