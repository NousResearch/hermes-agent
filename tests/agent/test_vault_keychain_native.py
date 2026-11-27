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
from hermes_platform.host.facts import os_family

_LOGIN = {"identifier_type": "email", "identifier": "ci@example.com", "password": "dummy-pw-not-real-42"}
_NO_SUCH_LOGON_SESSION = 1312


def _skip_without_logon_session(backend) -> None:
    """Skip when Credential Manager refuses a bare write with ERROR_NO_SUCH_LOGON_SESSION; any other write
    error fails setup, and read-back is left to the tests' own checks. A probe that was written is removed
    again, and one that cannot be removed fails setup."""
    account = f"hermes-ci-probe-{uuid.uuid4().hex}"
    try:
        backend.set_password(kc.SERVICE, account, "probe")
    except Exception as exc:  # health: allow BLE001 -- keyring backends raise their own types
        if getattr(exc, "winerror", None) == _NO_SUCH_LOGON_SESSION:
            pytest.skip(f"this runner's session has no logon session for Credential Manager: {exc}")
        raise
    assert kc.delete_key({"storage": "keychain", "service": kc.SERVICE, "account": account}), \
        f"the keychain probe entry {account} could not be removed"


@pytest.fixture
def os_keychain():
    """The real backend keyring resolves to; skip where the host has none (not a failure).

    A Windows runner whose job session has no logon session (``ERROR_NO_SUCH_LOGON_SESSION``, 1312) has a
    Credential Manager backend that refuses every write: a property of that runner's session, not of
    Hermes, so it skips too. Any other refusal still fails."""
    try:
        backend = kc.os_keyring()
    except kc.KeychainUnavailable as exc:
        pytest.skip(f"no OS keychain on this host: {exc}")
    if os_family() == "win32":
        _skip_without_logon_session(backend)
    made: list[dict] = []
    yield made
    for ref in made:
        kc.delete_key(ref)


class _RefusingBackend:
    def __init__(self, error: Exception):
        self.error = error

    def set_password(self, service: str, account: str, password: str) -> None:
        raise self.error


class _CredWriteError(OSError):
    def __init__(self, winerror: int):
        super().__init__("CredWrite failed")
        self.winerror = winerror


@pytest.mark.parametrize("error", [_CredWriteError(5), OSError("CredWrite failed")],
                         ids=["access-denied", "plain-os-error"])
def test_only_no_logon_session_skips_a_refused_probe_write(error):
    """The skip is for WinError 1312 alone: any other write refusal still fails the native tests."""
    with pytest.raises(pytest.skip.Exception):
        _skip_without_logon_session(_RefusingBackend(_CredWriteError(1312)))  # the literal, not the constant
    # Catch the skip outcome too, so a widened skip fails here instead of skipping this test.
    with pytest.raises((OSError, pytest.skip.Exception)) as raised:
        _skip_without_logon_session(_RefusingBackend(error))
    assert raised.value is error


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
