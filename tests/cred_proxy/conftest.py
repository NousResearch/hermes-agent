"""Shared fixtures for cred_proxy tests."""

import keyring
import keyring.backend
import keyring.errors
import pytest


class _MemoryKeyring(keyring.backend.KeyringBackend):
    """In-memory keyring backend for isolated unit tests."""

    priority = 1

    def __init__(self):
        self._data: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._data[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._data.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self._data:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._data[key]


@pytest.fixture(autouse=True)
def memory_keyring():
    """Replace the active keyring with a fresh in-memory store for each test."""
    kr = _MemoryKeyring()
    keyring.set_keyring(kr)
    yield kr
