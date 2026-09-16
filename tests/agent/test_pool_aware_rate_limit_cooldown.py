"""Rate-limit cooldown sizing from provider-known quota resets.

Cluster: ``_arm_rate_limit_cooldown`` sizes the primary's fallback cooldown with the
exponential backoff table (60s → 2m → …) even when the credential pool carries a
provider-declared quota reset far in the future (Codex ``resets_at``). The notice then
claims "Primary retry eligible in ~60 s" for a proven-empty weekly window, and the
long-lived gateway agent re-probes the primary every turn. When the pool reports a
future reset, the cooldown must honor it (capped) so restore stays gated and the
notice states the real horizon.
"""

import time
from types import SimpleNamespace


def _agent_with_pool(next_available_at):
    pool = SimpleNamespace(next_available_at=next_available_at)
    return SimpleNamespace(
        provider="openai-codex", _primary_runtime={"provider": "openai-codex"},
        _fallback_activated=False, _rate_limit_backoff_count=0,
        _rate_limited_until=0, _credential_pool=pool,
    )


class TestPoolAwareCooldown:
    def test_provider_reset_overrides_backoff(self):
        """A pool-known reset 58h out arms the cooldown at the reset, not the 60s step."""
        from agent.fallback_cooldown import _arm_rate_limit_cooldown
        from agent.error_classifier import FailoverReason

        agent = _agent_with_pool(lambda: time.time() + 58 * 3600)
        seconds = _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit)
        assert seconds is not None and seconds > 3600
        remaining = agent._rate_limited_until - time.monotonic()
        assert remaining > 3600

    def test_short_reset_keeps_backoff_table(self):
        """A transient 30s cooldown from the pool stays under the 60s backoff step, so
        the backoff table governs as before (no regression for genuine throttles)."""
        from agent.fallback_cooldown import _arm_rate_limit_cooldown
        from agent.error_classifier import FailoverReason

        agent = _agent_with_pool(lambda: time.time() + 30)
        seconds = _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit)
        assert seconds == 60

    def test_no_pool_information_uses_backoff(self):
        """``None``/absent pool info means 'no wait information': backoff table rules."""
        from agent.fallback_cooldown import _arm_rate_limit_cooldown
        from agent.error_classifier import FailoverReason

        agent = _agent_with_pool(lambda: None)
        assert _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit) == 60

    def test_reset_cap(self):
        """A pool reset absurdly far out is capped so one stale timestamp can't pin the
        agent to fallback for months (7 days)."""
        from agent.fallback_cooldown import _arm_rate_limit_cooldown
        from agent.error_classifier import FailoverReason

        agent = _agent_with_pool(lambda: time.time() + 400 * 24 * 3600)
        seconds = _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit)
        assert seconds is not None and seconds <= 7 * 24 * 3600
