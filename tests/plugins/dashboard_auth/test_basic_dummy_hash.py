"""BasicAuthProvider: the unknown-user dummy hash costs nothing at import but still runs scrypt per login.

Plugin discovery imports this module on every CLI/gateway start, so an import-time scrypt is a startup
tax; the constant-time verify for an unknown username must survive the move to first use.
"""

from __future__ import annotations

import hashlib
import importlib
import secrets
import sys

import pytest

from hermes_cli.dashboard_auth import InvalidCredentialsError

_MODULE = "plugins.dashboard_auth.basic"


@pytest.fixture
def scrypt_calls(monkeypatch):
    calls: list[bytes] = []
    real = hashlib.scrypt

    def spy(password, **kwargs):
        calls.append(password)
        return real(password, **kwargs)

    monkeypatch.setattr(hashlib, "scrypt", spy)
    return calls


@pytest.fixture
def fresh_basic(monkeypatch, scrypt_calls):
    monkeypatch.delitem(sys.modules, _MODULE, raising=False)
    module = importlib.import_module(_MODULE)
    yield module
    sys.modules.pop(_MODULE, None)


def test_import_does_not_run_scrypt(fresh_basic, scrypt_calls):
    assert scrypt_calls == []


def test_unknown_user_login_still_verifies_with_scrypt(fresh_basic, scrypt_calls):
    provider = fresh_basic.BasicAuthProvider(
        username="admin", password_hash=fresh_basic.hash_password("hunter2"), secret=secrets.token_bytes(32))
    scrypt_calls.clear()

    for _ in range(2):
        with pytest.raises(InvalidCredentialsError):
            provider.complete_password_login(username="nobody", password="hunter2")

    # Every unknown-user attempt verifies the submitted password against a scrypt hash.
    assert scrypt_calls.count(b"hunter2") == 2
