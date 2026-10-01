
import auth.providers.xai as _auth_auth_providers_xai
"""Regression tests for xAI OAuth auth resolution in profile/cron contexts."""
import auth.store as auth_storage

import pytest

from hermes_cli import auth
from auth.errors import AuthError


def test_read_xai_oauth_tokens_uses_credential_pool_when_provider_tokens_empty(monkeypatch):
    """Profile auth can have fresh pool tokens while singleton provider state is empty.

    This mirrors profiled cron after re-auth/credential-pool sync: the xAI
    OAuth credential is usable, but `providers.xai-oauth.tokens` may be empty
    or stale. Treating that as missing auth makes cron keep failing after the
    user has successfully re-authenticated.
    """
    store = {
        "providers": {"xai-oauth": {"tokens": {}, "last_auth_error": {}}},
        "credential_pool": {
            "xai-oauth": [
                {
                    "access_token": "pool-access",
                    "refresh_token": "pool-refresh",
                    "token_type": "Bearer",
                    "last_refresh": "2026-06-03T19:00:00Z",
                }
            ]
        },
    }
    monkeypatch.setattr(auth_storage, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth_storage, "_load_global_auth_store", lambda: {})

    resolved = _auth_auth_providers_xai._read_xai_oauth_tokens(_lock=False)

    assert resolved["tokens"]["access_token"] == "pool-access"
    assert resolved["tokens"]["refresh_token"] == "pool-refresh"
    assert resolved["tokens"]["token_type"] == "Bearer"
    assert resolved["last_refresh"] == "2026-06-03T19:00:00Z"


