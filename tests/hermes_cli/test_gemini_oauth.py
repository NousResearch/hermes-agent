"""Regression tests for Google Gemini OAuth auth resolution and model listing.

Mirrors the xai-oauth test pattern (test_xai_oauth_profile_auth.py) but
adapted for gemini-oauth's user-provided client credentials and Google's
fixed OAuth endpoints (no OIDC discovery).
"""

import pytest

from hermes_cli import auth
from hermes_cli.auth import AuthError
from hermes_cli import models


# ---------------------------------------------------------------------------
# Auth state resolution
# ---------------------------------------------------------------------------

def test_read_gemini_oauth_tokens_uses_provider_state(monkeypatch):
    """Provider state tokens should be returned when present."""
    store = {
        "providers": {
            "gemini-oauth": {
                "tokens": {
                    "access_token": "test-access",
                    "refresh_token": "test-refresh",
                    "token_type": "Bearer",
                },
                "client_credentials": {
                    "client_id": "test-client-id",
                    "client_secret": "test-client-secret",
                },
                "last_refresh": "2026-07-25T12:00:00Z",
            }
        },
    }
    monkeypatch.setattr(auth, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    resolved = auth._read_gemini_oauth_tokens(_lock=False)

    assert resolved["tokens"]["access_token"] == "test-access"
    assert resolved["tokens"]["refresh_token"] == "test-refresh"
    assert resolved["tokens"]["token_type"] == "Bearer"


def test_read_gemini_oauth_tokens_uses_credential_pool_when_provider_empty(monkeypatch):
    """Pool tokens should be used when provider state tokens are empty.

    Mirrors the xai-oauth profile/cron fallback: a profile may have fresh
    pool tokens while the singleton provider state is empty.
    """
    store = {
        "providers": {"gemini-oauth": {"tokens": {}, "last_auth_error": {}}},
        "credential_pool": {
            "gemini-oauth": [
                {
                    "access_token": "pool-access",
                    "refresh_token": "pool-refresh",
                    "token_type": "Bearer",
                    "last_refresh": "2026-07-25T19:00:00Z",
                }
            ]
        },
    }
    monkeypatch.setattr(auth, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    resolved = auth._read_gemini_oauth_tokens(_lock=False)

    assert resolved["tokens"]["access_token"] == "pool-access"
    assert resolved["tokens"]["refresh_token"] == "pool-refresh"
    assert resolved["tokens"]["token_type"] == "Bearer"


def test_read_gemini_oauth_tokens_uses_global_store_when_profile_empty(monkeypatch):
    """A profile/cron process should see root gemini-oauth auth."""
    profile_store = {"providers": {"gemini-oauth": {"tokens": {}}}}
    global_store = {
        "providers": {
            "gemini-oauth": {
                "tokens": {
                    "access_token": "global-access",
                    "refresh_token": "global-refresh",
                    "token_type": "Bearer",
                },
                "last_refresh": "2026-07-25T19:05:00Z",
            }
        }
    }
    monkeypatch.setattr(auth, "_load_auth_store", lambda: profile_store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: global_store)

    resolved = auth._read_gemini_oauth_tokens(_lock=False)

    assert resolved["tokens"]["access_token"] == "global-access"
    assert resolved["tokens"]["refresh_token"] == "global-refresh"


def test_read_gemini_oauth_tokens_raises_when_no_credentials(monkeypatch):
    """Should raise AuthError when no usable credentials exist anywhere."""
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {"providers": {}})
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    with pytest.raises(AuthError) as exc_info:
        auth._read_gemini_oauth_tokens(_lock=False)
    assert exc_info.value.provider == "gemini-oauth"


def test_get_gemini_oauth_auth_status_not_logged_in(monkeypatch):
    """Status should report not logged in when no credentials exist."""
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {"providers": {}})
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    status = auth.get_gemini_oauth_auth_status()
    assert status.get("logged_in") is False


def test_get_gemini_oauth_auth_status_logged_in(monkeypatch):
    """Status should report logged in when valid tokens exist."""
    store = {
        "providers": {
            "gemini-oauth": {
                "tokens": {
                    "access_token": "valid-access",
                    "refresh_token": "valid-refresh",
                    "token_type": "Bearer",
                }
            }
        }
    }
    monkeypatch.setattr(auth, "_load_auth_store", lambda: store)
    monkeypatch.setattr(auth, "_load_global_auth_store", lambda: {})

    status = auth.get_gemini_oauth_auth_status()
    assert status.get("logged_in") is True


# ---------------------------------------------------------------------------
# Constants and provider registry
# ---------------------------------------------------------------------------

def test_gemini_oauth_constants():
    """OAuth endpoints should be Google's well-known URLs."""
    assert auth.DEFAULT_GEMINI_OAUTH_BASE_URL == "https://generativelanguage.googleapis.com/v1beta"
    assert auth.GEMINI_OAUTH_DEVICE_CODE_URL == "https://oauth2.googleapis.com/device/code"
    assert auth.GEMINI_OAUTH_TOKEN_URL == "https://oauth2.googleapis.com/token"


def test_gemini_oauth_provider_config():
    """ProviderConfig should be registered with oauth_external auth type."""
    assert "gemini-oauth" in auth.PROVIDER_REGISTRY
    pc = auth.PROVIDER_REGISTRY["gemini-oauth"]
    assert pc.auth_type == "oauth_external"
    assert pc.inference_base_url == auth.DEFAULT_GEMINI_OAUTH_BASE_URL


# ---------------------------------------------------------------------------
# Model listing and provider aliases
# ---------------------------------------------------------------------------

def test_gemini_oauth_model_list():
    """gemini-oauth should have its own model list mirroring gemini."""
    oauth_models = models._PROVIDER_MODELS.get("gemini-oauth", [])
    gemini_models = models._PROVIDER_MODELS.get("gemini", [])
    assert len(oauth_models) > 0
    # The OAuth provider should offer the same Gemini models as the API-key provider
    assert set(oauth_models) == set(gemini_models)


def test_gemini_oauth_fetch_models():
    """provider_model_ids should return gemini-oauth models."""
    result = models.provider_model_ids("gemini-oauth")
    assert len(result) > 0
    assert "gemini-3.1-pro-preview" in result


def test_gemini_oauth_provider_aliases():
    """Provider aliases should resolve to gemini-oauth."""
    from hermes_cli.models import _PROVIDER_ALIASES
    assert _PROVIDER_ALIASES.get("gemini-oauth") == "gemini-oauth"
    assert _PROVIDER_ALIASES.get("google-oauth") == "gemini-oauth"
    assert _PROVIDER_ALIASES.get("google-gemini-oauth") == "gemini-oauth"


def test_gemini_oauth_in_canonical_providers():
    """gemini-oauth should be in CANONICAL_PROVIDERS."""
    slugs = [p.slug for p in models.CANONICAL_PROVIDERS]
    assert "gemini-oauth" in slugs


def test_gemini_oauth_in_provider_groups():
    """gemini-oauth should be in the google provider group alongside gemini."""
    google_group = models.PROVIDER_GROUPS.get("google")
    assert google_group is not None
    assert "gemini-oauth" in google_group[2]
    assert "gemini" in google_group[2]