"""Shared fakes for OAuth broker tests.

Everything here is synthetic and in-memory. No fixture may reach the real
Keychain, the real Codex OAuth endpoint, or any non-loopback network.
"""

from __future__ import annotations

import pytest

from agent.keychain_secret import KeychainNotFound


class FakeKeychainBackend:
    """In-memory KeychainBackend recording (service, account) call tuples."""

    def __init__(self, items=None):
        self.items = dict(items or {})
        self.read_calls = []
        self.write_calls = []
        self.delete_calls = []

    def read(self, ref):
        self.read_calls.append((ref.service, ref.account))
        try:
            return self.items[(ref.service, ref.account)]
        except KeyError:
            raise KeychainNotFound(
                service=ref.service, account=ref.account
            ) from None

    def write(self, ref, secret):
        self.write_calls.append((ref.service, ref.account))
        self.items[(ref.service, ref.account)] = secret

    def delete(self, ref):
        self.delete_calls.append((ref.service, ref.account))
        if (ref.service, ref.account) not in self.items:
            raise KeychainNotFound(service=ref.service, account=ref.account)
        del self.items[(ref.service, ref.account)]


@pytest.fixture
def fake_keychain():
    return FakeKeychainBackend()
