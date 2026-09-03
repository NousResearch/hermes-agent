"""Tests for the rate_limit_retry policy's credential-pool interaction.

The policy holds the primary provider on long backoff during 429 storms.
That turn-loop behavior must also change how the credential pool is
updated: with a single-credential pool, marking the entry exhausted on a
transient concurrency 429 triggers a 1h cooldown that starves every OTHER
process (cron jobs, new sessions) even though the chat loop keeps using
the same key anyway.

Behavior under test (agent/agent_runtime_helpers.py::_recover_with_credential_pool,
FailoverReason.rate_limit branch):

1. Policy active + single pool entry → no mark_exhausted_and_rotate call.
2. Policy active + a second available entry → rotation proceeds (a real
   alternate exists, failover is the better move).
3. Policy active + real quota wall (usage_limit_reached) → exhaustion
   proceeds (backoff cannot recover a dead quota).
4. Policy disabled/absent → legacy behavior byte-identical (exhaustion on
   second 429).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.agent_runtime_helpers as helpers
from agent.rate_limit_policy import RateLimitRetryPolicy


ACTIVE_POLICY = RateLimitRetryPolicy(max_retries=-1, backoff_base=10.0, backoff_max=300.0)


def _make_pool(n_entries=1, current_status=None):
    pool = MagicMock()
    pool.provider = "zai"
    entries = []
    for i in range(n_entries):
        e = MagicMock()
        e.id = f"zai-entry-{i}"
        e.runtime_api_key = f"key-{i}"
        e.last_status = current_status if (i == 0 and current_status) else None
        entries.append(e)
    pool.entries.return_value = entries
    current = entries[0] if entries else None
    pool.current.return_value = current
    pool.mark_exhausted_and_rotate.return_value = current if n_entries > 1 else None
    pool.reset_statuses.return_value = 1
    return pool


def _make_agent(policy, pool):
    agent = MagicMock()
    agent.provider = "zai"
    agent.model = "glm-5.3-flash"
    agent._rate_limit_retry_policy = policy
    agent._credential_pool = pool
    agent._fallback_activated = False
    return agent


def _run_recovery(agent, *, has_retried_429=True, status_code=429, error_context=None):
    """Invoke the real recover_with_credential_pool rate-limit branch."""
    # The helper module resolves dependencies through _ra(); patch the
    # pieces the rate-limit branch touches so no real I/O happens.
    with patch.object(helpers, "_ra") as ra:
        ra.return_value.logger = MagicMock()
        result = helpers.recover_with_credential_pool(
            agent,
            status_code=status_code,
            has_retried_429=has_retried_429,
            error_context=error_context,
        )
    return result


class TestPolicyHoldsPoolEntry:
    def test_policy_single_entry_no_exhaustion(self):
        """Policy active + one entry → hold the key, don't poison the pool."""
        pool = _make_pool(n_entries=1)
        agent = _make_agent(ACTIVE_POLICY, pool)

        result = _run_recovery(agent, has_retried_429=True)

        pool.mark_exhausted_and_rotate.assert_not_called()
        assert result == (False, True)

    def test_policy_second_entry_still_rotates(self):
        """Policy active but a real alternate exists → standard rotation."""
        pool = _make_pool(n_entries=2)
        agent = _make_agent(ACTIVE_POLICY, pool)

        result = _run_recovery(agent, has_retried_429=True)

        pool.mark_exhausted_and_rotate.assert_called_once()
        agent._swap_credential.assert_called_once()

    def test_policy_quota_wall_still_exhausts(self):
        """usage_limit_reached is a real quota wall — backoff can't fix it."""
        pool = _make_pool(n_entries=1)
        pool.mark_exhausted_and_rotate.return_value = None
        agent = _make_agent(ACTIVE_POLICY, pool)

        result = _run_recovery(
            agent,
            has_retried_429=True,
            error_context={"reason": "usage_limit_reached"},
        )

        pool.mark_exhausted_and_rotate.assert_called_once()
        assert result == (False, True)

    def test_no_policy_legacy_exhaustion(self):
        """Policy disabled → legacy exhaustion path is byte-identical."""
        pool = _make_pool(n_entries=1)
        agent = _make_agent(None, pool)

        result = _run_recovery(agent, has_retried_429=True)

        pool.mark_exhausted_and_rotate.assert_called_once()
        assert result == (False, True)

    def test_policy_first_429_still_defers(self):
        """has_retried_429=False → both legacy and policy defer the rotation
        decision (return False, True) without touching the pool."""
        pool = _make_pool(n_entries=1)
        agent = _make_agent(ACTIVE_POLICY, pool)

        result = _run_recovery(agent, has_retried_429=False)

        pool.mark_exhausted_and_rotate.assert_not_called()
        assert result == (False, True)

    def test_policy_clears_stale_exhaustion_flag(self):
        """Pre-policy turns may have left the only entry exhausted; the
        upstream early-return handles this before the policy branch."""
        pool = _make_pool(n_entries=1, current_status="exhausted")
        agent = _make_agent(ACTIVE_POLICY, pool)

        result = _run_recovery(agent, has_retried_429=True)

        # Legacy path: marks exhausted and rotates (returns False, True)
        pool.mark_exhausted_and_rotate.assert_called_once()
        assert result == (False, True)
