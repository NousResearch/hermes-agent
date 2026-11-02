"""The vault key against the host's REAL OS keychain (macOS Keychain, Windows Credential Manager).

Everything else in the vault suite uses an in-memory keyring, which proves Hermes' plumbing but not
that the platform store accepts, returns and deletes an entry the way ``agent.vault_keychain``
expects. These run only on the macOS and Windows CI lanes (never on Linux, so a developer's Secret
Service is never touched), use uniquely named dummy entries, and delete them in teardown.
"""

from __future__ import annotations

import uuid

import pytest
from cryptography.fernet import Fernet

from agent import vault_keychain as kc
from agent.vault_store import VaultStore

_LOGIN = {"identifier_type": "email", "identifier": "ci@example.com", "password": "dummy-pw-not-real-42"}


@pytest.fixture
def os_keychain():
    """The real backend keyring resolves to; skip where the host has none (not a failure)."""
    try:
        kc.os_keyring()
    except kc.KeychainUnavailable as exc:
        pytest.skip(f"no OS keychain on this host: {exc}")
    made: list[dict] = []
    yield made
    for ref in made:
        kc.delete_key(ref)


@pytest.mark.platforms("macos", "windows")
def test_real_keychain_stores_reads_and_deletes_a_dummy_entry(os_keychain):
    ref = {"storage": "keychain", "service": kc.SERVICE, "account": f"hermes-ci-smoke-{uuid.uuid4().hex}"}
    os_keychain.append(ref)
    key = Fernet.generate_key()
    kc.store_key(ref, key)  # raises unless the store hands the same bytes back
    assert kc.load_key(ref) == key
    assert kc.delete_key(ref)
    assert kc.load_key(ref) is None


@pytest.mark.platforms("macos", "windows")
def test_keychain_mode_vault_round_trips_through_the_real_keychain(os_keychain, tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text("vault:\n  key_storage: keychain\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    vault = home / "vault"

    item = VaultStore().add_item("login", "Example", dict(_LOGIN), origin="https://example.com").id
    ref = kc.read_ref(vault / "vault.keyref")
    assert ref is not None, "keychain mode must keep only a pointer beside the ciphertext"
    os_keychain.append(ref)
    assert not (vault / "vault.key").exists()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]

    assert VaultStore().migrate_key("file") == "file"
    assert (vault / "vault.key").exists() and not (vault / "vault.keyref").exists()
    assert kc.load_key(ref) is None, "moving to the file deletes the vault's own keychain entry"

    assert VaultStore().migrate_key("keychain") == "keychain"
    back = kc.read_ref(vault / "vault.keyref")
    os_keychain.append(back)
    assert not (vault / "vault.key").exists()
    assert VaultStore().resolve_secret(item)["password"] == _LOGIN["password"]
