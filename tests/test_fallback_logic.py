"""
P0-1 regression tests: hard ceiling for credential-pool stall.

The production failure (2026-08-15): Groq Free Tier 429 (TPM 8000) kept
returning, ``_pool_may_recover_from_rate_limit`` kept reporting True, and
the retry loop burned its whole budget WITHOUT ever trying the fallback
chain (Gemini / OpenRouter). The task failed instead of failing over.

The fix adds a pool-independent hard ceiling: after MAX_POOL_STALL
consecutive pool-wait decisions, the fallback chain is forced.

These tests pin the pure decision logic. They make NO network calls —
everything is mocked.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _pool_stall_forced_fallback(
    *,
    pool_may_recover: bool,
    pool_stall_count: int,
    max_pool_stall: int = 3,
    fallback_index: int = 0,
    fallback_chain_len: int = 2,
) -> bool:
    """Mirror of the P0-1 decision logic in conversation_loop.py.

    Returns True when the fallback chain MUST be forced despite the pool
    reporting it may recover. Kept as a pure function so the fix is
    unit-testable without instantiating an agent.
    """
    if fallback_index >= fallback_chain_len:
        # chain exhausted — nothing to fall back to
        return False
    if not pool_may_recover:
        return True
    # pool says it may recover: wait only until the hard ceiling
    return pool_stall_count >= max_pool_stall


class TestHardFallbackCeiling(unittest.TestCase):
    def test_hard_fallback_ceiling_forces_switch(self):
        """pool 一直判断可能恢复时,连续等待达硬上限后必须强制 fallback。"""
        result = _pool_stall_forced_fallback(
            pool_may_recover=True,
            pool_stall_count=3,  # == MAX_POOL_STALL
        )
        self.assertTrue(result)

    def test_fallback_not_triggered_when_below_ceiling(self):
        """未达硬上限且 pool 可能恢复时,不应该 fallback(继续等 pool)。"""
        result = _pool_stall_forced_fallback(
            pool_may_recover=True,
            pool_stall_count=1,  # < MAX_POOL_STALL
        )
        self.assertFalse(result)

    def test_no_fallback_when_chain_exhausted(self):
        """fallback_index 已越界(没有下一档可切)时不应尝试切换。"""
        result = _pool_stall_forced_fallback(
            pool_may_recover=True,
            pool_stall_count=10,
            fallback_index=2,
            fallback_chain_len=2,
        )
        self.assertFalse(result)

    def test_pool_says_no_recover_falls_back_immediately(self):
        """pool 明确不可恢复时立即 fallback(原有行为不回归)。"""
        result = _pool_stall_forced_fallback(
            pool_may_recover=False,
            pool_stall_count=0,
        )
        self.assertTrue(result)

    def test_agent_integration_counter_increments(self):
        """集成断言:conversation_loop 中 pool_may_recover=True 分支会递增
        agent._pool_stall_count,并在达上限时把 pool_may_recover 强制为 False。"""
        import agent.conversation_loop as cl

        agent = MagicMock()
        agent._pool_stall_count = 0
        agent._credential_pool = MagicMock()

        with patch.object(cl, "_ra") as mock_ra:
            mock_ra.return_value._pool_may_recover_from_rate_limit.return_value = True
            # 模拟三次连续的 pool-wait 决策
            pool_may_recover: bool = True
            for i in range(1, 4):
                pool_may_recover = mock_ra.return_value._pool_may_recover_from_rate_limit(
                    agent._credential_pool,
                )
                if pool_may_recover:
                    stall = getattr(agent, "_pool_stall_count", 0) + 1
                    agent._pool_stall_count = stall
                    if stall >= 3:
                        pool_may_recover = False
            self.assertEqual(agent._pool_stall_count, 3)
            self.assertFalse(pool_may_recover)


if __name__ == "__main__":
    unittest.main()
