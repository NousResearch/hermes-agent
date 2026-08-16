"""Regression tests for the credential-pool provider-mismatch guard with
ALIASES-canonicalized providers (#87526).

Two code paths name the same provider differently. The pool is seeded from
the auth-registry slug (e.g. ``opencode-zen``), while a ``/model`` switch runs
the provider through ``normalize_provider`` and stores the ALIASES-canonical
id (``opencode``) on the session override. The defensive guard in
``recover_with_credential_pool`` compared the two literally, logged
"Credential pool provider mismatch: pool=opencode-zen, agent=opencode", and
skipped recovery — so pool rotation / 429 retry / billing recovery never ran
for ANY provider whose ALIASES canonical id differs from its slug.

The fix canonicalizes both sides through the shared alias map before
comparing, so the aliased pair matches while genuinely different providers
still trip the guard (#33088/#33163: never mutate the primary's pool while a
fallback provider is active).
"""
from unittest.mock import MagicMock

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.error_classifier import FailoverReason


def _agent(provider, pool_provider):
    agent = MagicMock()
    agent.provider = provider
    pool = MagicMock()
    pool.provider = pool_provider
    agent._credential_pool = pool
    return agent, pool


class TestAliasPoolMismatchGuard:

    def test_aliased_provider_pair_is_not_guarded(self):
        """agent carries the ALIASES-canonical id (``opencode``) while the
        pool is seeded from the slug (``opencode-zen``) — the same provider,
        so recovery must proceed and rotate rather than skip."""
        agent, pool = _agent("opencode", "opencode-zen")
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=402,
            has_retried_429=False,
            classified_reason=FailoverReason.billing,
        )
        assert recovered is True
        pool.mark_exhausted_and_rotate.assert_called_once()

    def test_genuinely_different_provider_still_guarded(self):
        """A different provider (not an alias of the pool's) must still skip
        pool mutation — the #33088/#33163 fallback-isolation contract."""
        agent, pool = _agent("anthropic", "opencode-zen")
        recovered, _ = recover_with_credential_pool(
            agent,
            status_code=402,
            has_retried_429=False,
            classified_reason=FailoverReason.billing,
        )
        assert recovered is False
        assert not pool.method_calls
