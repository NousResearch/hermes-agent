"""Fixtures for the hermes_crypto test suite.

Builds on the hermetic global conftest (isolated HERMES_HOME, scrubbed
credential env vars). Adds two encryption-specific guarantees:

* An **in-memory keyring backend** so keyring-mode tests never read from or
  write to the developer's real OS keyring (Credential Manager / Keychain /
  Secret Service).
* The process-wide DEK cache is cleared around every test so one test's
  unlocked keystore cannot leak into the next.
"""

from __future__ import annotations

import pytest

# Skip the whole package when the optional encryption extra is absent.
pytest.importorskip("cryptography")
pytest.importorskip("keyring")


import keyring.backend  # noqa: E402


class _InMemoryKeyring(keyring.backend.KeyringBackend):
    """Minimal keyring backend that keeps secrets in a process-local dict.

    keyring inspects a backend's ``__module__`` to classify it as secure;
    this test module is not in the insecure-prefix list, so
    ``keystore.keyring_is_secure()`` treats it as a real backend — which is
    what keyring-mode tests need.
    """

    priority = 99

    def __init__(self) -> None:
        super().__init__()
        self._store: dict = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        self._store.pop((service, username), None)


@pytest.fixture(autouse=True)
def _in_memory_keyring():
    """Swap the OS keyring for an in-memory fake for the duration of a test."""
    import keyring

    previous = keyring.get_keyring()
    keyring.set_keyring(_InMemoryKeyring())
    try:
        yield
    finally:
        keyring.set_keyring(previous)


@pytest.fixture(autouse=True)
def _clear_dek_cache():
    """Drop any cached data-encryption key before and after each test."""
    import hermes_crypto
    from hermes_crypto import keystore

    keystore.lock()
    hermes_crypto._stashed_passphrase = None
    try:
        yield
    finally:
        keystore.lock()
        hermes_crypto._stashed_passphrase = None
