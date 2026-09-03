"""Credential pools must support policy-filtered selection without fallback."""

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)


def _entry(entry_id, auth_type, priority):
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label=entry_id,
        auth_type=auth_type,
        priority=priority,
        source="manual",
        access_token=f"token-{entry_id}",
    )


def test_select_auth_type_skips_higher_priority_api_key():
    pool = CredentialPool(
        "openai-codex",
        [
            _entry("api", AUTH_TYPE_API_KEY, 0),
            _entry("oauth", AUTH_TYPE_OAUTH, 1),
        ],
    )

    selected = pool.select(auth_type=AUTH_TYPE_OAUTH)

    assert selected is not None
    assert selected.id == "oauth"
    assert selected.auth_type == AUTH_TYPE_OAUTH
    assert pool.current().id == "oauth"


def test_select_auth_type_returns_none_instead_of_falling_back():
    pool = CredentialPool(
        "openai-codex",
        [_entry("api", AUTH_TYPE_API_KEY, 0)],
    )

    selected = pool.select(auth_type=AUTH_TYPE_OAUTH)

    assert selected is None
    assert pool.current() is None


def test_unfiltered_select_preserves_existing_behavior():
    pool = CredentialPool(
        "openai-codex",
        [
            _entry("api", AUTH_TYPE_API_KEY, 0),
            _entry("oauth", AUTH_TYPE_OAUTH, 1),
        ],
    )

    selected = pool.select()

    assert selected is not None
    assert selected.id == "api"
