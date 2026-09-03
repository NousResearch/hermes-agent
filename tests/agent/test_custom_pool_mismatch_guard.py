"""Regression tests for the credential-pool provider-mismatch guard with
custom providers (Bernard's Fireworks report, June 2026).

Custom endpoints can carry a generic ``"custom"`` label or retain their
configured name/provider key while the pool is keyed
``custom:<normalized-name>`` (``CUSTOM_POOL_PREFIX``). The defensive guard in
``recover_with_credential_pool`` must recognize each identity without letting
a different endpoint or fallback provider mutate the pool.

The fix accepts the pair only when the agent's current base_url resolves to
the same pool key, preserving the guard's original purpose (#33088/#33163:
never mutate the primary's pool while a fallback provider is active).
"""
from types import SimpleNamespace
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

    @staticmethod
    def _gemini_config():
        return [
            (
                "gemini-no-filter",
                {
                    "name": "Gemini No Filter",
                    "provider_key": "gemini-no-filter",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta",
                },
            )
        ]

    def test_named_custom_provider_rotates_its_matching_pool(self):
        agent, pool = _agent(
            "gemini-no-filter",
            "https://generativelanguage.googleapis.com/v1beta",
            "custom:gemini-no-filter",
        )
        agent.api_key = "key-a"
        agent._credential_pool_entry_id = None
        agent._swap_credential = MagicMock()
        pool.entries.return_value = []
        pool.current.return_value = None
        next_entry = SimpleNamespace(id="key-b", runtime_api_key="key-b")
        pool.mark_exhausted_and_rotate.return_value = next_entry
        with patch(
            "agent.credential_pool._iter_custom_providers",
            return_value=self._gemini_config(),
        ):
            recovered, retried = recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=True,
                classified_reason=FailoverReason.rate_limit,
            )

        assert recovered is True
        assert retried is False
        pool.mark_exhausted_and_rotate.assert_called_once()
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_exact_custom_identity_requires_matching_endpoint(self):
        agent, pool = _agent(
            "custom:gemini-no-filter",
            "https://fallback.example/v1",
            "custom:gemini-no-filter",
        )

        with patch(
            "agent.credential_pool._iter_custom_providers",
            return_value=self._gemini_config(),
        ):
            recovered, retried = recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=True,
                classified_reason=FailoverReason.rate_limit,
            )

        assert recovered is False
        assert retried is True
        assert not pool.method_calls

    def test_exact_custom_identity_rotates_at_matching_endpoint(self):
        agent, pool = _agent(
            "custom:gemini-no-filter",
            "https://generativelanguage.googleapis.com/v1beta",
            "custom:gemini-no-filter",
        )
        agent.api_key = "key-a"
        agent._credential_pool_entry_id = None
        agent._swap_credential = MagicMock()
        pool.entries.return_value = []
        pool.current.return_value = None
        next_entry = SimpleNamespace(id="key-b", runtime_api_key="key-b")
        pool.mark_exhausted_and_rotate.return_value = next_entry

        with patch(
            "agent.credential_pool._iter_custom_providers",
            return_value=self._gemini_config(),
        ):
            recovered, retried = recover_with_credential_pool(
                agent,
                status_code=429,
                has_retried_429=True,
                classified_reason=FailoverReason.rate_limit,
            )

        assert recovered is True
        assert retried is False
        pool.mark_exhausted_and_rotate.assert_called_once()
        agent._swap_credential.assert_called_once_with(next_entry)

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

