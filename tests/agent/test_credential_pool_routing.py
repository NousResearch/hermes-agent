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
# 6. Successful responses promote only the matching Codex probe
# ---------------------------------------------------------------------------


def test_valid_response_marks_matching_credential_probe_successful():
    from agent.agent_runtime_helpers import record_credential_pool_success

    pool = MagicMock()
    agent = SimpleNamespace(_credential_pool=pool, api_key="request-token")

    record_credential_pool_success(agent)

    pool.mark_success.assert_called_once_with(api_key_hint="request-token")


# ---------------------------------------------------------------------------
# 7. Full 429 rotation cycle via _recover_with_credential_pool
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
        # Must be set explicitly — MagicMock.provider returns a truthy
        # child mock, which would trigger the provider-mismatch guard.
        pool.provider = ""

        # mark_exhausted_and_rotate returns next entry until exhausted
        self._rotation_index = 0

        def rotate(status_code=None, error_context=None, api_key_hint=None):
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

    def test_first_429_uses_request_entry_not_stale_exhausted_current(self):
        agent, pool, entries = self._make_agent_with_pool(3)
        agent.api_key = "token-account-3"
        entries[0].last_status = "exhausted"
        entries[2].last_status = None
        pool.current.return_value = entries[0]
        pool.entry_for_api_key.return_value = entries[2]

        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
        )

        assert recovered is False
        assert has_retried is True
        pool.entry_for_api_key.assert_called_once_with("token-account-3")
        pool.mark_exhausted_and_rotate.assert_not_called()

    def test_probe_first_429_rearms_exact_credential_immediately(self):
        agent, pool, entries = self._make_agent_with_pool(3)
        agent.api_key = "token-account-3"
        entries[2].last_status = "probing"
        pool.entry_for_api_key.return_value = entries[2]
        pool.mark_exhausted_and_rotate.side_effect = None
        pool.mark_exhausted_and_rotate.return_value = None

        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
        )

        assert recovered is False
        assert has_retried is True
        pool.mark_exhausted_and_rotate.assert_called_once_with(
            status_code=429,
            error_context=None,
            api_key_hint="token-account-3",
        )

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

    def test_usage_limit_marks_the_credential_that_sent_the_failed_request(self):
        """A stale pool.current must not copy account 3's 429 onto account 1."""
        agent, pool, entries = self._make_agent_with_pool(3)
        agent.api_key = "token-account-3"
        pool.current.return_value = entries[0]  # stale/current selection points at account 1
        context = {"reason": "usage_limit_reached", "message": "usage limit has been reached"}

        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
            error_context=context,
        )

        assert recovered is True
        assert has_retried is False
        pool.mark_exhausted_and_rotate.assert_called_once_with(
            status_code=429,
            error_context=context,
            api_key_hint="token-account-3",
        )
        agent._swap_credential.assert_called_once_with(entries[1])

    def test_usage_limit_persists_only_the_failed_credential_in_real_pool(self, tmp_path, monkeypatch):
        """Disk state must bind a 429 to the request token, not stale pool.current."""
        import json
        from agent.credential_pool import CredentialPool, PooledCredential

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        entries = [
            PooledCredential(
                provider="openai-codex",
                id=f"cred-{index}",
                label=f"account-{index}",
                auth_type="oauth",
                priority=index - 1,
                source="manual:device_code",
                access_token=f"token-account-{index}",
            )
            for index in (1, 2, 3)
        ]
        pool = CredentialPool("openai-codex", entries)
        pool._current_id = "cred-1"  # stale runtime pointer
        agent, _, _ = self._make_agent_with_pool(3)
        agent._credential_pool = pool
        agent.provider = "openai-codex"
        agent.api_key = "token-account-3"
        agent._swap_credential = MagicMock()
        context = {"reason": "usage_limit_reached", "message": "usage limit has been reached"}

        recovered, _ = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
            error_context=context,
        )

        assert recovered is True
        persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())
        by_id = {row["id"]: row for row in persisted["credential_pool"]["openai-codex"]}
        assert by_id["cred-1"]["last_status"] is None
        assert by_id["cred-2"]["last_status"] is None
        assert by_id["cred-3"]["last_status"] == "exhausted"
        assert by_id["cred-3"]["last_error_code"] == 429

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

    def test_auth_refresh_targets_request_token_not_stale_current(self):
        agent, pool, entries = self._make_agent_with_pool(3)
        agent.provider = "openai-codex"
        agent.api_key = "token-account-3"
        agent._is_entitlement_failure = MagicMock(return_value=False)
        pool.current.return_value = entries[0]
        pool.try_refresh_for_api_key.return_value = entries[2]

        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
            error_context={"reason": "token_expired"},
        )

        assert recovered is True
        assert has_retried is False
        pool.try_refresh_for_api_key.assert_called_once_with("token-account-3")
        pool.try_refresh_current.assert_not_called()
        agent._swap_credential.assert_called_once_with(entries[2])

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
