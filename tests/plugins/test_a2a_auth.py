from __future__ import annotations

import json
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from plugins.platforms.a2a import auth


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_inbound_token_is_salted_hashed_and_owner_only(hermes_home):
    token = auth.create_inbound_credential("inbound:laptop")

    path = hermes_home / "a2a" / "credentials.json"
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    credential_id, _secret = token.removeprefix("a2a_").split(".", 1)
    record = data["inbound"][credential_id]

    assert token not in raw
    assert "inbound:laptop" not in token
    assert record["credential_ref"] == "inbound:laptop"
    assert record["algorithm"] == "scrypt"
    assert record["salt"]
    assert record["digest"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert auth.verify_inbound_token("inbound:laptop", token)
    assert auth.resolve_inbound_token(token) == "inbound:laptop"
    assert not auth.verify_inbound_token("inbound:laptop", token + "wrong")
    assert not auth.verify_inbound_token("missing", token)


def test_rotation_invalidates_old_token_and_uses_new_salt(hermes_home):
    old = auth.create_inbound_credential("inbound:laptop")
    old_id = old.removeprefix("a2a_").split(".", 1)[0]
    old_salt = auth._load_credentials()["inbound"][old_id]["salt"]

    new = auth.rotate_inbound_credential("inbound:laptop")
    new_id = new.removeprefix("a2a_").split(".", 1)[0]
    new_salt = auth._load_credentials()["inbound"][new_id]["salt"]

    assert old != new
    assert old_salt != new_salt
    assert not auth.verify_inbound_token("inbound:laptop", old)
    assert auth.verify_inbound_token("inbound:laptop", new)
    assert new not in repr(new)


def test_inbound_resolution_performs_at_most_one_scrypt(hermes_home, monkeypatch):
    token = auth.create_inbound_credential("inbound:laptop")
    original = auth._derive
    calls = 0

    def counted_derive(candidate, salt):
        nonlocal calls
        calls += 1
        return original(candidate, salt)

    monkeypatch.setattr(auth, "_derive", counted_derive)
    assert auth.resolve_inbound_token(token) == "inbound:laptop"
    assert calls == 1

    calls = 0
    assert auth.resolve_inbound_token("a2a_unknown.public-secret-that-is-long-enough") is None
    assert calls == 0


def test_outbound_token_is_returned_only_by_explicit_secret_lookup(hermes_home):
    token = "outbound-token-with-at-least-thirty-two-characters"
    auth.store_outbound_credential("outbound:norbert", token)

    loaded = auth.load_outbound_token("outbound:norbert")
    assert loaded == token
    assert token not in repr(loaded)
    summary = auth.credential_summary()
    assert token not in repr(summary)
    assert summary == {
        "inbound": [],
        "outbound": ["outbound:norbert"],
    }


def test_outbound_token_validation_never_echoes_secret(hermes_home):
    token = "short-secret"
    with pytest.raises(ValueError) as exc:
        auth.store_outbound_credential("outbound:norbert", token)
    assert token not in str(exc.value)


def test_credentials_are_profile_aware(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(first))
    token = auth.create_inbound_credential("inbound:peer")
    assert auth.verify_inbound_token("inbound:peer", token)

    monkeypatch.setenv("HERMES_HOME", str(second))
    assert not auth.verify_inbound_token("inbound:peer", token)
    assert not (second / "a2a" / "credentials.json").exists()


def test_corrupt_hash_record_fails_closed(hermes_home):
    path = auth.credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"version": 1, "inbound": {"inbound:x": {"salt": "!"}}, "outbound": {}}),
        encoding="utf-8",
    )

    assert not auth.verify_inbound_token("inbound:x", "x" * 40)


def test_store_symlink_is_rejected_without_exposing_target_secret(hermes_home, tmp_path):
    secret = "stored-secret-that-must-not-appear"
    target = tmp_path / "target.json"
    target.write_text(secret, encoding="utf-8")
    path = auth.credentials_path()
    path.parent.mkdir(parents=True)
    path.symlink_to(target)

    with pytest.raises(auth.CredentialStoreError) as exc:
        auth.credential_summary()

    assert secret not in str(exc.value)


def test_symlink_store_parent_is_rejected_on_write(hermes_home, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (hermes_home / "a2a").symlink_to(outside, target_is_directory=True)

    with pytest.raises(auth.CredentialStoreError):
        auth.create_inbound_credential("inbound:laptop")

    assert not (outside / "credentials.json").exists()


def test_non_regular_store_is_rejected(hermes_home):
    path = auth.credentials_path()
    path.parent.mkdir(parents=True)
    path.mkdir()

    with pytest.raises(auth.CredentialStoreError):
        auth.credential_summary()


def test_wrong_owner_store_is_rejected(hermes_home, monkeypatch):
    path = auth.credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(auth._empty_store()), encoding="utf-8")
    real_uid = auth.os.getuid()
    monkeypatch.setattr(auth, "_validate_owned_directory", lambda info, label: None)
    monkeypatch.setattr(auth.os, "getuid", lambda: real_uid + 1)

    with pytest.raises(auth.CredentialStoreError):
        auth.credential_summary()


def test_permissive_store_mode_is_normalized_before_read(hermes_home):
    path = auth.credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(auth._empty_store()), encoding="utf-8")
    path.chmod(0o644)

    assert auth.credential_summary() == {"inbound": [], "outbound": []}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_mutations_reload_under_one_cross_process_lock(hermes_home, monkeypatch):
    original_load = auth._load_credentials
    first_loaded = threading.Event()
    release_first = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def interleaved_load(directory_fd=None):
        nonlocal calls
        data = original_load(directory_fd)
        with calls_lock:
            calls += 1
            position = calls
        if position == 1:
            first_loaded.set()
            assert release_first.wait(timeout=2)
        return data

    monkeypatch.setattr(auth, "_load_credentials", interleaved_load)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            auth.store_outbound_credential,
            "outbound:first",
            "first-token-with-more-than-thirty-two-characters",
        )
        assert first_loaded.wait(timeout=2)
        second = executor.submit(
            auth.store_outbound_credential,
            "outbound:second",
            "second-token-with-more-than-thirty-two-characters",
        )
        time.sleep(0.05)
        assert calls == 1
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert auth.credential_summary()["outbound"] == ["outbound:first", "outbound:second"]


def test_credential_read_rejects_file_swap_between_stat_and_open(hermes_home, monkeypatch):
    auth.store_outbound_credential(
        "outbound:peer", "original-token-with-more-than-thirty-two-characters"
    )
    path = auth.credentials_path()
    original_open = auth.os.open
    swapped = False

    def swapping_open(name, flags, *args, **kwargs):
        nonlocal swapped
        if name == path.name and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            path.replace(path.with_suffix(".old"))
            path.write_text(json.dumps(auth._empty_store()), encoding="utf-8")
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(auth.os, "open", swapping_open)
    with pytest.raises(auth.CredentialStoreError, match="changed"):
        auth.credential_summary()


def test_pinned_credential_directory_survives_parent_swap_for_read_and_write(
    hermes_home,
):
    original_token = "original-token-with-more-than-thirty-two-characters"
    auth.store_outbound_credential("outbound:original", original_token)
    visible = auth.credentials_path().parent
    pinned = visible.with_name("a2a-pinned")

    with auth._locked_credential_mutation() as directory_fd:
        visible.rename(pinned)
        visible.mkdir(mode=0o700)
        data = auth._load_credentials(directory_fd)
        assert data["outbound"]["outbound:original"]["token"] == original_token
        data["outbound"]["outbound:new"] = {
            "token": "new-token-with-more-than-thirty-two-characters",
            "created_at": auth._now(),
        }
        auth._save_credentials(data, directory_fd)

    assert not (visible / "credentials.json").exists()
    stored = json.loads((pinned / "credentials.json").read_text(encoding="utf-8"))
    assert set(stored["outbound"]) == {"outbound:new", "outbound:original"}
