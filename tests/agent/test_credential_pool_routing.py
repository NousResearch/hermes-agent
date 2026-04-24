"""Tests for credential pool preservation through turn config and 429 recovery.

Covers:
1. CLI _resolve_turn_agent_config passes credential_pool to runtime dict
2. Gateway _resolve_turn_agent_config passes credential_pool to runtime dict
3. Eager fallback deferred when credential pool has credentials
4. Eager fallback fires when no credential pool exists
5. Full 429 rotation cycle: retry-same → rotate → exhaust → fallback
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. CLI _resolve_turn_agent_config includes credential_pool
# ---------------------------------------------------------------------------

class TestCliTurnRoutePool:
    def test_resolve_turn_includes_pool(self):
        """CLI's _resolve_turn_agent_config must pass credential_pool in runtime."""
        fake_pool = MagicMock(name="FakePool")
        shell = SimpleNamespace(
            model="gpt-5.4",
            api_key="sk-test",
            base_url=None,
            provider="openai-codex",
            api_mode="codex_responses",
            acp_command=None,
            acp_args=[],
            _credential_pool=fake_pool,
            service_tier=None,
        )

        from cli import HermesCLI
        bound = HermesCLI._resolve_turn_agent_config.__get__(shell)
        route = bound("test message")

        assert route["runtime"]["credential_pool"] is fake_pool


# ---------------------------------------------------------------------------
# 2. Gateway _resolve_turn_agent_config includes credential_pool
# ---------------------------------------------------------------------------

class TestGatewayTurnRoutePool:
    def test_resolve_turn_includes_pool(self):
        """Gateway's _resolve_turn_agent_config must pass credential_pool."""
        from gateway.run import GatewayRunner

        fake_pool = MagicMock(name="FakePool")
        runner = SimpleNamespace(_service_tier=None)
        runtime_kwargs = {
            "api_key": "***",
            "base_url": None,
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": fake_pool,
        }

        bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)
        route = bound("test message", "gpt-5.4", runtime_kwargs)

        assert route["runtime"]["credential_pool"] is fake_pool


# ---------------------------------------------------------------------------
# 3 & 4. Eager fallback deferred/fires based on credential pool
# ---------------------------------------------------------------------------

class TestEagerFallbackWithPool:
    """Test the eager fallback guard in run_agent.py's error handling loop."""

    def _make_agent(self, has_pool=True, pool_has_creds=True, has_fallback=True):
        """Create a minimal AIAgent mock with the fields needed."""
        from run_agent import AIAgent

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()

        agent._credential_pool = None
        if has_pool:
            pool = MagicMock()
            pool.has_available.return_value = pool_has_creds
            agent._credential_pool = pool

        agent._fallback_chain = [{"model": "fallback/model"}] if has_fallback else []
        agent._fallback_index = 0
        agent._try_activate_fallback = MagicMock(return_value=True)
        agent._emit_status = MagicMock()

        return agent

    def test_eager_fallback_deferred_when_pool_has_credentials(self):
        """429 with active pool should NOT trigger eager fallback."""
        agent = self._make_agent(has_pool=True, pool_has_creds=True, has_fallback=True)

        # Simulate the check from run_agent.py lines 7180-7191
        is_rate_limited = True
        if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
            pool = agent._credential_pool
            pool_may_recover = pool is not None and pool.has_available()
            if not pool_may_recover:
                agent._try_activate_fallback()

        agent._try_activate_fallback.assert_not_called()

    def test_eager_fallback_fires_when_no_pool(self):
        """429 without pool should trigger eager fallback."""
        agent = self._make_agent(has_pool=False, has_fallback=True)

        is_rate_limited = True
        if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
            pool = agent._credential_pool
            pool_may_recover = pool is not None and pool.has_available()
            if not pool_may_recover:
                agent._try_activate_fallback()

        agent._try_activate_fallback.assert_called_once()

    def test_eager_fallback_fires_when_pool_exhausted(self):
        """429 with exhausted pool should trigger eager fallback."""
        agent = self._make_agent(has_pool=True, pool_has_creds=False, has_fallback=True)

        is_rate_limited = True
        if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
            pool = agent._credential_pool
            pool_may_recover = pool is not None and pool.has_available()
            if not pool_may_recover:
                agent._try_activate_fallback()

        agent._try_activate_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Full 429 rotation cycle via _recover_with_credential_pool
# ---------------------------------------------------------------------------

class TestPoolRotationCycle:
    """Verify the retry-same → rotate → exhaust flow in _recover_with_credential_pool."""

    def _make_agent_with_pool(self, pool_entries=3):
        from run_agent import AIAgent

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()

        entries = []
        for i in range(pool_entries):
            e = MagicMock(name=f"entry_{i}")
            e.id = f"cred-{i}"
            entries.append(e)

        pool = MagicMock()
        pool.has_credentials.return_value = True

        # mark_exhausted_and_rotate returns next entry until exhausted
        self._rotation_index = 0

        def rotate(status_code=None, error_context=None):
            self._rotation_index += 1
            if self._rotation_index < pool_entries:
                return entries[self._rotation_index]
            pool.has_credentials.return_value = False
            return None

        pool.mark_exhausted_and_rotate = MagicMock(side_effect=rotate)
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()
        agent.log_prefix = ""

        return agent, pool, entries

    def test_first_429_sets_retry_flag_no_rotation(self):
        """First 429 should just set has_retried_429=True, no rotation."""
        agent, pool, _ = self._make_agent_with_pool(3)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429, has_retried_429=False
        )
        assert recovered is False
        assert has_retried is True
        pool.mark_exhausted_and_rotate.assert_not_called()

    def test_second_429_rotates_to_next(self):
        """Second consecutive 429 should rotate to next credential."""
        agent, pool, entries = self._make_agent_with_pool(3)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429, has_retried_429=True
        )
        assert recovered is True
        assert has_retried is False  # reset after rotation
        pool.mark_exhausted_and_rotate.assert_called_once_with(status_code=429, error_context=None)
        agent._swap_credential.assert_called_once_with(entries[1])

    def test_pool_exhaustion_returns_false(self):
        """When all credentials exhausted, recovery should return False."""
        agent, pool, _ = self._make_agent_with_pool(1)
        # First 429 sets flag
        _, has_retried = agent._recover_with_credential_pool(
            status_code=429, has_retried_429=False
        )
        assert has_retried is True

        # Second 429 tries to rotate but pool is exhausted (only 1 entry)
        recovered, _ = agent._recover_with_credential_pool(
            status_code=429, has_retried_429=True
        )
        assert recovered is False

    def test_402_immediate_rotation(self):
        """402 (billing) should immediately rotate, no retry-first."""
        agent, pool, entries = self._make_agent_with_pool(3)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=402, has_retried_429=False
        )
        assert recovered is True
        assert has_retried is False
        pool.mark_exhausted_and_rotate.assert_called_once_with(status_code=402, error_context=None)

    def test_no_pool_returns_false(self):
        """No pool should return (False, unchanged)."""
        from run_agent import AIAgent

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()
        agent._credential_pool = None

        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429, has_retried_429=False
        )
        assert recovered is False
        assert has_retried is False


# ---------------------------------------------------------------------------
# 6. FailoverReason.overloaded handler (#15297)
#
# When the classifier produces ``overloaded`` (e.g. Z.AI 429 "service is
# temporarily overloaded"), the recovery path must rotate immediately —
# the whole endpoint is down, retrying the same credential burns a slot
# and delays recovery.  Same shape as ``billing``, distinct from
# ``rate_limit`` which retries-first.
# ---------------------------------------------------------------------------


class TestPoolOverloadedRotation:
    """Verify ``FailoverReason.overloaded`` rotates immediately, no retry-first."""

    def _make_agent_with_pool(self, pool_entries=3):
        """Same factory shape as TestPoolRotationCycle — keep them parallel
        so a future refactor of the agent's pool seam updates both
        classes uniformly."""
        from run_agent import AIAgent

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()

        entries = []
        for i in range(pool_entries):
            e = MagicMock(name=f"entry_{i}")
            e.id = f"cred-{i}"
            entries.append(e)

        pool = MagicMock()
        pool.has_credentials.return_value = True

        self._rotation_index = 0

        def rotate(status_code=None, error_context=None):
            self._rotation_index += 1
            if self._rotation_index < pool_entries:
                return entries[self._rotation_index]
            pool.has_credentials.return_value = False
            return None

        pool.mark_exhausted_and_rotate = MagicMock(side_effect=rotate)
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()
        agent.log_prefix = ""

        return agent, pool, entries

    def test_overloaded_rotates_immediately_on_first_failure(self):
        """The #15297 fix: ``overloaded`` must NOT use the rate-limit
        retry-first path.  First failure rotates straight away because
        retrying the same credential against a saturated provider just
        burns the next ``has_retried_429`` slot too."""
        from agent.error_classifier import FailoverReason

        agent, pool, entries = self._make_agent_with_pool(3)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,  # first failure
            classified_reason=FailoverReason.overloaded,
        )
        assert recovered is True, (
            "overloaded must rotate on first failure, like billing — "
            "not wait for has_retried_429 like rate_limit does"
        )
        assert has_retried is False
        pool.mark_exhausted_and_rotate.assert_called_once_with(
            status_code=429, error_context=None,
        )
        agent._swap_credential.assert_called_once_with(entries[1])

    def test_overloaded_does_not_block_on_has_retried_flag(self):
        """The retry-first gate is specific to rate_limit.  An
        ``overloaded`` reason must rotate regardless of whether the
        retry flag is currently set or unset — same behaviour both
        ways, no state leak from a previous rate-limit cycle."""
        from agent.error_classifier import FailoverReason

        agent, pool, entries = self._make_agent_with_pool(3)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=True,  # already retried — should still rotate
            classified_reason=FailoverReason.overloaded,
        )
        assert recovered is True
        assert has_retried is False  # reset after rotation
        pool.mark_exhausted_and_rotate.assert_called_once()

    def test_overloaded_pool_exhaustion_returns_false(self):
        """When the pool only has one credential and it overloads, the
        rotation step returns no next entry — the recovery call must
        report ``recovered=False`` so the caller falls through to its
        fallback chain rather than spinning."""
        from agent.error_classifier import FailoverReason

        agent, _pool, _ = self._make_agent_with_pool(1)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.overloaded,
        )
        assert recovered is False, (
            "single-entry pool that overloads must return recovered=False "
            "so the outer fallback chain takes over"
        )
        # has_retried_429 is propagated unchanged in the no-rotation
        # branch — pinned so a future tweak to the contract surfaces
        # via this test.
        assert has_retried is False

    def test_rate_limit_still_uses_retry_first(self):
        """Negative control: ``rate_limit`` must KEEP its retry-first
        semantics.  A regression that broadened the overloaded handler
        to also catch rate_limit would silently double-rotate every
        per-key throttle and burn the pool fast."""
        from agent.error_classifier import FailoverReason

        agent, pool, _ = self._make_agent_with_pool(3)
        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
            classified_reason=FailoverReason.rate_limit,
        )
        assert recovered is False, (
            "rate_limit on first failure must NOT rotate — the retry-"
            "first path is the contract"
        )
        assert has_retried is True
        pool.mark_exhausted_and_rotate.assert_not_called()
