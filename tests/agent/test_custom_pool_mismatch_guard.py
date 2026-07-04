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


def _agent(provider, base_url, pool_provider, requested_provider=""):
    agent = MagicMock()
    agent.provider = provider
    agent.base_url = base_url
    # Default to bare "custom" so older test setups (and the bug repro in
    # #45715) don't need to set it explicitly. The guard only consults
    # requested_provider inside the bare-custom branch.
    agent.requested_provider = requested_provider or ""
    pool = MagicMock()
    pool.provider = pool_provider
    agent._credential_pool = pool
    return agent, pool


class TestCustomPoolMismatchGuard:
    def test_matching_custom_pool_reaches_recovery(self):
        """agent=custom + pool=custom:<name> whose base_url matches must NOT
        be treated as a cross-provider mismatch."""
        agent, pool = _agent("custom", FIREWORKS_URL, "custom:fireworks")
        # Rate-limit path deterministically calls pool.current() once past
        # the guard (the auth path consults agent._is_entitlement_failure,
        # which a MagicMock would answer truthily).
        pool.current.return_value = None
        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            return_value="custom:fireworks",
        ):
            recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=False,
                classified_reason=FailoverReason.rate_limit,
            )
        assert pool.current.called, (
            "guard short-circuited: pool never touched despite matching custom base_url"
        )

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
        agent, pool = _agent(
            "openai-codex", "https://chatgpt.com/backend-api", "custom:fireworks"
        )
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=401,
            has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        assert recovered is False
        assert not pool.method_calls

    def test_plain_provider_mismatch_still_guarded(self):
        agent, pool = _agent("openrouter", "https://openrouter.ai/api/v1", "anthropic")
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
        )
        assert recovered is False
        assert not pool.method_calls

    def test_relayer_routed_custom_matches_via_requested_provider(self):
        """#45715: agent=custom + pool=custom:<name> + base_url is a relayer
        that resolves to NO custom_providers entry — must still match when
        agent.requested_provider carries the named form the user originally
        requested (propagated from resolve_runtime_provider)."""
        RELAYER_URL = "https://relayer.internal.example/v1"
        agent, pool = _agent(
            "custom", RELAYER_URL, "custom:claude", requested_provider="custom:claude"
        )
        pool.current.return_value = None
        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            # Relayer URL is NOT in custom_providers → key resolution returns None
            return_value=None,
        ):
            recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=False,
                classified_reason=FailoverReason.rate_limit,
            )
        assert pool.current.called, (
            "guard short-circuited: pool never touched despite "
            "agent.requested_provider matching the named pool"
        )

    def test_relayer_with_unrelated_requested_provider_still_guarded(self):
        """#45715 defensive case: relayer + agent.requested_provider pointing
        at a DIFFERENT custom pool than the one loaded must still be guarded
        — prevents mutating the wrong pool during a fallback chain."""
        RELAYER_URL = "https://relayer.internal.example/v1"
        agent, pool = _agent(
            "custom", RELAYER_URL, "custom:claude", requested_provider="custom:minimax"
        )
        with patch(
            "agent.credential_pool.get_custom_provider_pool_key",
            return_value=None,
        ):
            recovered, _ = recover_with_credential_pool(
                agent,
                status_code=401,
                has_retried_429=False,
                classified_reason=FailoverReason.auth,
            )
        assert recovered is False
        assert not pool.method_calls

    def test_requested_provider_ignored_for_non_custom(self):
        """#45715 narrow scope: agent.requested_provider is only consulted
        inside the bare-custom branch. A non-custom agent with a matching
        requested_provider string must NOT short-circuit the original
        fallback-guard contract (#33088/#33163)."""
        agent, pool = _agent(
            "openai-codex",
            "https://chatgpt.com/backend-api",
            "custom:claude",
            requested_provider="custom:claude",
        )
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=401,
            has_retried_429=False,
            classified_reason=FailoverReason.auth,
        )
        assert recovered is False
        assert not pool.method_calls
