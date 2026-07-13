"""Regression test for #63737: sk-ant-oat tokens must be OAuth, not api_key."""
from agent.credential_pool import (
    PooledCredential,
    AUTH_TYPE_OAUTH,
    AUTH_TYPE_API_KEY,
)


def test_manual_anthropic_oat_normalized_to_oauth():
    # A setup-token added manually (dashboard / hermes auth add) defaults to
    # api_key; it must be normalized to OAuth so it is sent with Bearer auth.
    entry = PooledCredential.from_dict(
        "anthropic",
        {"label": "MainKey", "source": "manual",
         "auth_type": "api_key", "access_token": "sk-ant-oat01-EXAMPLE"},
    )
    assert entry.auth_type == AUTH_TYPE_OAUTH


def test_anthropic_real_api_key_unchanged():
    entry = PooledCredential.from_dict(
        "anthropic",
        {"auth_type": "api_key", "access_token": "sk-ant-api03-EXAMPLE"},
    )
    assert entry.auth_type == AUTH_TYPE_API_KEY


def test_non_anthropic_provider_unchanged():
    entry = PooledCredential.from_dict(
        "openrouter",
        {"auth_type": "api_key", "access_token": "sk-ant-oat01-WHATEVER"},
    )
    assert entry.auth_type == AUTH_TYPE_API_KEY
