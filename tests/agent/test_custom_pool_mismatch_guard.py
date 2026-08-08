"""Regression tests for the credential-pool provider-mismatch guard with
custom providers (Bernard's Fireworks report, June 2026).

Custom endpoints carry two naming conventions for the same provider: the
agent's ``provider`` attribute is the generic ``"custom"`` label while the
pool is keyed ``custom:<normalized-name>`` (``CUSTOM_POOL_PREFIX``).  The
defensive guard in ``recover_with_credential_pool`` compared the two
literally, logged "Credential pool provider mismatch: pool=custom:<name>,
agent=custom", and skipped recovery — so 401/429 recovery (refresh,
rotation) never ran for ANY custom-provider user.

The fix accepts the pair only when the agent's current base_url resolves to
the same pool key, preserving the guard's original purpose (#33088/#33163:
never mutate the primary's pool while a fallback provider is active).
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason


FIREWORKS_URL = "https://api.fireworks.ai/inference/v1"


def _agent(provider, base_url, pool_provider):
    agent = MagicMock()
    agent.provider = provider
    agent.base_url = base_url
    agent.requested_provider = ""
    pool = MagicMock()
    pool.provider = pool_provider
    agent._credential_pool = pool
    return agent, pool


class TestCustomPoolMismatchGuard:

    def test_unrelated_custom_pool_still_guarded(self):
        """agent=custom pointed at a DIFFERENT endpoint than the pool's
        custom provider must still skip pool mutation."""
        agent, pool = _agent(
            "custom", "https://other-endpoint.example/v1", "custom:fireworks"
        )
        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            return_value="custom:other",
        ):
            recovered, _ = recover_with_credential_pool(
                agent,
                status_code=401,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
            )
        assert recovered is False
        assert not pool.method_calls

    def test_fallback_provider_still_guarded(self):
        """Original #33088/#33163 contract: when a fallback provider is
        active (agent.provider != pool.provider, non-custom), the pool is
        never mutated."""
        agent, pool = _agent("openai-codex", "https://chatgpt.com/backend-api", "custom:fireworks")
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=401,
            has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        assert recovered is False
        assert not pool.method_calls

    def test_second_provider_sharing_base_url_guard_passes_with_name(self):
        """Two custom providers sharing one base_url with different api_keys
        (issue #81789): the pool of the SECOND entry must pass the guard when
        the agent carries its requested identity, even though a bare base_url
        lookup resolves the first URL owner."""
        agent, pool = _agent(
            "custom", "https://relay.example/v1", "custom:slomerex-alt"
        )
        agent.requested_provider = "custom:slomerex-alt"
        # Drive the auth-recovery path past the entitlement check to the
        # rotation branch (this test is about the guard, not refresh logic).
        agent._is_entitlement_failure = lambda *a, **k: False
        agent.api_key = "sk-keyB"
        pool.try_refresh_matching = lambda **kw: None
        pool.mark_exhausted_and_rotate = lambda **kw: MagicMock(id="next-entry")
        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            side_effect=lambda url, provider_name=None: (
                "custom:slomerex-alt" if provider_name else "custom:slomerex-grok"
            ),
        ) as lookup:
            recovered, _ = recover_with_credential_pool(
                agent,
                status_code=401,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
            )
        assert recovered is True
        # Recovery actually ran: the failed credential was rotated and the
        # agent swapped onto the new entry (not blocked by the guard).
        agent._swap_credential.assert_called_once()
        # The guard must have consulted the requested identity, not just URL.
        kwargs = lookup.call_args
        assert kwargs.kwargs.get("provider_name") == "custom:slomerex-alt"

    def test_second_provider_sharing_base_url_without_name_still_guarded(self):
        """Without the requested identity the base_url-only lookup resolves
        the first URL owner, so the second pool stays guarded (no mutation)."""
        agent, pool = _agent(
            "custom", "https://relay.example/v1", "custom:slomerex-alt"
        )
        agent.requested_provider = ""
        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            return_value="custom:slomerex-grok",
        ):
            recovered, _ = recover_with_credential_pool(
                agent,
                status_code=401,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
            )
        assert recovered is False
        assert not pool.method_calls

