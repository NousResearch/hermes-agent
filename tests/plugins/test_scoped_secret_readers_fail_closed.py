"""Credential shims outside the memory plugins honour the secret-scope contract.

``langfuse._secret`` and ``azure_identity_adapter._scoped_env`` used to catch ``UnscopedSecretError``
and fall back to ``os.environ`` / ``""``. Under multiplex ``os.environ`` is the DEFAULT profile's
``.env``, so that fallback either shipped another profile's credentials or hid the spawn-site bug the
exception exists to surface. Contract: scope wins over environ; no scope while multiplexing raises.
"""
from __future__ import annotations

import pytest

from agent import secret_scope
from agent.azure_identity_adapter import _scoped_env as azure_scoped_env
from plugins.observability.langfuse import _secret as langfuse_secret

_READERS = {"langfuse": (langfuse_secret, "LANGFUSE_SECRET_KEY"),
            "azure": (azure_scoped_env, "AZURE_CLIENT_SECRET")}


@pytest.fixture
def multiplex(monkeypatch):
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("name", sorted(_READERS))
def test_scoped_read_prefers_profile_scope_over_default_environ(name, monkeypatch, multiplex):
    reader, var = _READERS[name]
    monkeypatch.setenv(var, "default-profile-value")
    token = secret_scope.set_secret_scope({var: " profile-b-value "})
    try:
        assert reader(var) == "profile-b-value"
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("name", sorted(_READERS))
def test_scopeless_multiplex_read_fails_loud(name, monkeypatch, multiplex):
    reader, var = _READERS[name]
    monkeypatch.setenv(var, "default-profile-value")
    token = secret_scope.set_secret_scope(None)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            reader(var)
    finally:
        secret_scope.reset_secret_scope(token)
