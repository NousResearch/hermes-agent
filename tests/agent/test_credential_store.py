"""Encrypted credential store: round-trip, corruption safety, profile scoping.

The corruption tests pin the contract the original /creds PR got wrong: an
unreadable store must raise, never read as ``{}`` — otherwise the next
``set()`` rewrites the file and silently deletes every other credential.
"""

import json
import os
import stat

import pytest

from agent.credential_store import (
    CredentialStore,
    CredentialStoreError,
    get_store,
    reset_store,
    validate_name,
)


def _store(tmp_path) -> CredentialStore:
    return CredentialStore(tmp_path / "credentials")


def test_round_trip_set_get_list_delete(tmp_path):
    store = _store(tmp_path)
    result = store.set("supabase_service_key", "sk-live-abc123")
    assert result == {"success": True, "name": "supabase_service_key", "created": True,
                      "updated": False}
    assert store.get("supabase_service_key") == "sk-live-abc123"
    assert store.get("not_stored") is None

    entries = store.list()
    assert [entry["name"] for entry in entries] == ["supabase_service_key"]
    assert "value" not in entries[0]  # metadata only

    assert store.delete("supabase_service_key")["success"] is True
    assert store.get("supabase_service_key") is None
    assert store.delete("supabase_service_key")["success"] is False


def test_store_file_is_ciphertext_and_key_is_0600(tmp_path):
    store = _store(tmp_path)
    store.set("stripe_secret_key", "sk_live_51SecretValue")
    raw = store._store_path.read_bytes()
    assert b"sk_live_51SecretValue" not in raw
    assert b"stripe_secret_key" not in raw  # names are inside the ciphertext too
    mode = stat.S_IMODE(os.stat(store._key_path).st_mode)
    assert mode == 0o600, oct(mode)


def test_overwrite_is_opt_in_and_keeps_created_at(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.set("stripe_secret_key", "first")
    denied = store.set("stripe_secret_key", "second")
    assert denied["success"] is False
    assert "already exists" in denied["error"]
    assert store.get("stripe_secret_key") == "first"

    created_before = store.list()[0]["created_at"]
    monkeypatch.setattr("agent.credential_store.time.time", lambda: created_before + 100)
    accepted = store.set("stripe_secret_key", "second", overwrite=True)
    assert accepted["success"] is True and accepted["updated"] is True
    entry = store.list()[0]
    assert entry["created_at"] == created_before          # not reset by the overwrite
    assert entry["updated_at"] == created_before + 100
    assert store.get("stripe_secret_key") == "second"


def test_clear_removes_everything(tmp_path):
    store = _store(tmp_path)
    store.set("alpha_token_one", "v1")
    store.set("beta_token_two", "v2")
    assert store.clear() == {"success": True, "removed": 2}
    assert store.list() == []


@pytest.mark.parametrize("bad", ["", "ab", "key", "token", "password", "path", "env_x", "os_x",
                                "has space", "do/t", "x" * 129])
def test_invalid_names_rejected(bad):
    with pytest.raises(ValueError):
        validate_name(bad)


def test_valid_names_accepted():
    for name in ("supabase_service_key", "stripe-secret-key", "db_password2"):
        validate_name(name)  # no raise


def test_corrupt_ciphertext_raises_and_is_never_rewritten(tmp_path):
    """A decryptable-but-garbage or truncated store must fail loudly and stay intact."""
    store = _store(tmp_path)
    store.set("stripe_secret_key", "value")
    assert store._store_path.read_bytes()  # sanity: ciphertext exists

    store._store_path.write_bytes(b"\x00truncated-ciphertext")
    with pytest.raises(CredentialStoreError):
        store.list()
    with pytest.raises(CredentialStoreError):
        store.set("another_credential", "value")
    assert store._store_path.read_bytes() == b"\x00truncated-ciphertext"  # not wiped

    # Same contract for a well-formed envelope whose payload is not JSON.
    store._store_path.write_bytes(store._fernet().encrypt(b"{not json"))
    broken = store._store_path.read_bytes()
    with pytest.raises(CredentialStoreError):
        store.set("another_credential", "value")
    assert store._store_path.read_bytes() == broken  # still not wiped


def test_zero_byte_store_reads_as_empty(tmp_path):
    store = _store(tmp_path)
    store.base_dir.mkdir(parents=True, exist_ok=True)
    store._store_path.write_bytes(b"")
    assert store.list() == []
    assert store.set("alpha_token_one", "value")["success"] is True


def test_get_on_corrupt_store_raises_rather_than_reporting_missing(tmp_path):
    store = _store(tmp_path)
    store.base_dir.mkdir(parents=True, exist_ok=True)
    store._store_path.write_bytes(store._fernet().encrypt(json.dumps({"entries": 7}).encode()))
    with pytest.raises(CredentialStoreError):
        store.get("stripe_secret_key")


def test_singleton_is_keyed_by_home(tmp_path, monkeypatch):
    """A multiplexed gateway resolving a different home must get a different store."""
    import agent.credential_store as cs

    monkeypatch.setattr(cs, "get_hermes_home", lambda: tmp_path / "profile_a")
    reset_store()
    first = get_store()
    assert get_store() is first  # same home → same instance

    monkeypatch.setattr(cs, "get_hermes_home", lambda: tmp_path / "profile_b")
    second = get_store()
    assert second is not first
    second.set("profile_a_secret_key", "from-b")
    monkeypatch.setattr(cs, "get_hermes_home", lambda: tmp_path / "profile_a")
    assert get_store().get("profile_a_secret_key") is None  # A must not see B's store
    reset_store()