"""Regression tests for xAI OAuth auth resolution in profile/cron contexts."""

import pytest

from hermes_cli import auth
from hermes_cli.auth import AuthError


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
    monkeypatch.setattr(auth, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    resolved = auth._read_xai_oauth_tokens(_lock=False)

    assert resolved["tokens"]["access_token"] == "pool-access"
    assert resolved["tokens"]["refresh_token"] == "pool-refresh"
    assert resolved["tokens"]["token_type"] == "Bearer"
    assert resolved["last_refresh"] == "2026-06-03T19:00:00Z"


def test_read_xai_oauth_tokens_uses_global_store_when_profile_state_empty(monkeypatch):
    """A profile/cron process should see root xAI auth after user re-auths there."""
    profile_store = {"providers": {"xai-oauth": {"tokens": {}}}}
    global_store = {
        "providers": {
            "xai-oauth": {
                "tokens": {
                    "access_token": "global-access",
                    "refresh_token": "global-refresh",
                    "token_type": "Bearer",
                },
                "last_refresh": "2026-06-03T19:05:00Z",
            }
        }
    }
    monkeypatch.setattr(auth, "_load_auth_store", lambda: profile_store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: global_store)

    resolved = auth._read_xai_oauth_tokens(_lock=False)

    assert resolved["tokens"]["access_token"] == "global-access"
    assert resolved["tokens"]["refresh_token"] == "global-refresh"
    assert resolved["last_refresh"] == "2026-06-03T19:05:00Z"


def test_read_xai_oauth_tokens_still_requires_usable_tokens(monkeypatch):
    """Fallback should not hide genuinely broken xAI auth state."""
    store = {
        "providers": {"xai-oauth": {"tokens": {}}},
        "credential_pool": {"xai-oauth": [{"access_token": "", "refresh_token": ""}]},
    }
    monkeypatch.setattr(auth, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    with pytest.raises(AuthError) as exc:
        auth._read_xai_oauth_tokens(_lock=False)

    assert exc.value.code == "xai_auth_missing_access_token"
    assert exc.value.relogin_required is True
