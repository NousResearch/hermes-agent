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
    pool = MagicMock()
    pool.provider = pool_provider
    agent._credential_pool = pool
    return agent, pool


class TestCustomPoolMismatchGuard:

    def test_named_custom_runtime_identity_can_recover_matching_pool(self):
        """A new-style ``providers.<name>`` runtime keeps the bare config key,
        while its pool uses the endpoint-scoped ``custom:<name>`` identity."""
        agent, pool = _agent(
            "sensenova", "https://token.sensenova.cn/v1", "custom:sensenova"
        )
        pool.current.return_value = None

        with patch(
            "hermes_cli.runtime_provider.load_config",
            return_value={
                "providers": {
                    "sensenova": {
                        "name": "SenseNova",
                        "api": "https://token.sensenova.cn/v1",
                    }
                }
            },
        ):
            recovered, retried = recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=False,
                classified_reason=FailoverReason.rate_limit,
            )

        assert recovered is False
        assert retried is True
        pool.current.assert_called_once_with()

    def test_named_custom_runtime_identity_with_other_endpoint_is_guarded(self):
        agent, pool = _agent(
            "sensenova", "https://other-endpoint.example/v1", "custom:sensenova"
        )

        with patch(
            "hermes_cli.runtime_provider.load_config",
            return_value={
                "providers": {
                    "sensenova": {
                        "name": "SenseNova",
                        "api": "https://token.sensenova.cn/v1",
                    }
                }
            },
        ):
            recovered, retried = recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=False,
                classified_reason=FailoverReason.rate_limit,
            )

        assert recovered is False
        assert retried is False
        assert not pool.method_calls

    def test_named_custom_recovery_disambiguates_shared_endpoint(self):
        shared_url = "https://shared.example/v1"
        agent, pool = _agent("second", shared_url, "custom:second")
        pool.current.return_value = None

        with patch(
            "hermes_cli.runtime_provider.load_config",
            return_value={
                "providers": {
                    "first": {"name": "First", "api": shared_url},
                    "second": {"name": "Second", "api": shared_url},
                }
            },
        ):
            recovered, retried = recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=False,
                classified_reason=FailoverReason.rate_limit,
            )

        assert recovered is False
        assert retried is True
        pool.current.assert_called_once_with()

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

