"""Regression coverage for pooled Anthropic OAuth recovery (#138337)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from run_agent import AIAgent


def _entry(entry_id, token):
    return PooledCredential(
        provider="anthropic", id=entry_id, label=entry_id, auth_type=AUTH_TYPE_OAUTH,
        priority=0, source="manual:hermes_pkce", access_token=token,
        refresh_token=f"{entry_id}-refresh",
    )


def test_stale_failed_token_adopts_sibling_refresh_without_another_post():
    """A shared pool must return its newer token instead of revoking it again."""
    entry = _entry("leased", "new-access-token")
    pool = CredentialPool("anthropic", [entry])
    pool._try_refresh_current_unlocked = MagicMock(return_value=None)

    assert pool.try_refresh_matching(api_key_hint="old-access-token", credential_id="leased") is entry
    pool._try_refresh_current_unlocked.assert_not_called()


def test_refresh_cap_rotates_before_another_refresh_post():
    """A repeatedly rejected leased entry is benched before a third refresh."""
    old, replacement = _entry("leased", "old-access-token"), _entry("other", "other-access-token")
    pool = MagicMock(provider="anthropic")
    pool.entries.return_value = [old, replacement]
    pool.try_refresh_matching.side_effect = AssertionError("must not refresh past the cap")
    pool.mark_exhausted_and_rotate.return_value = replacement
    agent = SimpleNamespace(
        _credential_pool=pool, _credential_pool_entry_id="leased",
        api_key="old-access-token", provider="anthropic", base_url=None, model=None,
        _auth_pool_refresh_counts={("anthropic", "leased"): 2},
        _credential_pool_revert_id=None,
        _is_entitlement_failure=lambda _context, _status: False,
        _swap_credential=MagicMock(return_value=True),
    )

    assert recover_with_credential_pool(agent, status_code=401, has_retried_429=False) == (True, False)
    pool.mark_exhausted_and_rotate.assert_called_once()


def test_rate_limit_proves_auth_and_resets_refresh_cap():
    """A 429 after refresh authenticates, so a later 401 gets a fresh budget."""
    pool = MagicMock(provider="anthropic")
    agent = SimpleNamespace(
        _credential_pool=pool, _credential_pool_entry_id="leased",
        api_key="access-token", provider="anthropic", base_url=None, model=None,
        _auth_pool_refresh_counts={("anthropic", "leased"): 2},
        _is_entitlement_failure=lambda _context, _status: False,
    )

    assert recover_with_credential_pool(agent, status_code=429, has_retried_429=False) == (False, True)
    assert agent._auth_pool_refresh_counts == {}


def test_pooled_lease_does_not_adopt_a_different_disk_token():
    """The per-request resolver must leave a leased pooled client on its lease."""
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_api_key = "leased-token"
    agent._credential_pool = object()
    agent._credential_pool_entry_id = "leased"

    with patch("agent.anthropic_credentials.resolve_anthropic_token", return_value="disk-token") as resolve:
        assert agent._try_refresh_anthropic_client_credentials() is False

    resolve.assert_not_called()
