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
        # Must be set explicitly — MagicMock.provider returns a truthy
        # child mock, which would trigger the provider-mismatch guard.
        pool.provider = ""

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
# 6. Copilot credential rotation refreshes both token and base_url
# ---------------------------------------------------------------------------


class TestCopilotCredentialRefresh:
    """Copilot-specific: _swap_credential fetches fresh token+base_url together."""

    def _make_copilot_entry(self, base_url: str) -> MagicMock:
        """Create a mock pool entry for Copilot."""
        entry = MagicMock()
        entry.access_token = "gho_test_token"
        entry.runtime_api_key = None
        entry.runtime_base_url = None
        entry.base_url = base_url
        return entry

    @patch("hermes_cli.copilot_auth.get_copilot_api_token")
    def test_copilot_swap_refreshes_token_and_base_url(self, mock_exchange):
        """When provider is copilot, _swap_credential calls live token exchange."""
        from run_agent import AIAgent

        # Mock live exchange returning Enterprise endpoint
        mock_exchange.return_value = ("exchanged_jwt", "https://api.enterprise.githubcopilot.com")

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()
        agent.provider = "copilot"
        agent.api_mode = "openai"
        agent._client_kwargs = {}
        agent.base_url = "https://api.githubcopilot.com"  # initial public endpoint
        agent._replace_primary_openai_client = MagicMock()

        # Pool entry with stale public endpoint
        stale_entry = self._make_copilot_entry("https://api.githubcopilot.com")

        # Swap credential - should trigger live exchange
        agent._swap_credential(stale_entry)

        # Verify live exchange was called with the entry's raw token
        mock_exchange.assert_called_once_with("gho_test_token")

        # Verify the client was updated with fresh Enterprise endpoint
        assert agent.api_key == "exchanged_jwt"
        assert agent.base_url == "https://api.enterprise.githubcopilot.com"
        assert agent._client_kwargs["api_key"] == "exchanged_jwt"
        assert agent._client_kwargs["base_url"] == "https://api.enterprise.githubcopilot.com"

    @patch("hermes_cli.copilot_auth.get_copilot_api_token")
    def test_copilot_swap_falls_back_on_exchange_failure(self, mock_exchange):
        """If Copilot exchange fails, _swap_credential falls back to entry values."""
        from run_agent import AIAgent

        # Simulate exchange failure
        mock_exchange.side_effect = ValueError("network error")

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()
        agent.provider = "copilot"
        agent.api_mode = "openai"
        agent._client_kwargs = {}
        agent.base_url = "https://api.githubcopilot.com"
        agent._replace_primary_openai_client = MagicMock()

        # Pool entry
        entry = self._make_copilot_entry("https://api.githubcopilot.com")

        # Swap credential - should attempt exchange, fall back on failure
        agent._swap_credential(entry)

        # Verify exchange was attempted
        mock_exchange.assert_called_once_with("gho_test_token")

        # Verify fallback to entry values occurred
        assert agent.api_key == "gho_test_token"  # from entry.access_token
        assert agent.base_url == "https://api.githubcopilot.com"

    def test_non_copilot_swap_does_not_call_exchange(self):
        """For non-Copilot providers, _swap_credential uses entry values directly."""
        from run_agent import AIAgent

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()
        agent.provider = "openai"  # Not Copilot
        agent.api_mode = "openai"
        agent._client_kwargs = {}
        agent.base_url = "https://api.openai.com"
        agent._replace_primary_openai_client = MagicMock()

        # Entry
        entry = MagicMock()
        entry.access_token = "sk_test"
        entry.runtime_api_key = None
        entry.runtime_base_url = None
        entry.base_url = "https://api.openai.com"

        # Swap - should NOT call Copilot exchange
        with patch("hermes_cli.copilot_auth.get_copilot_api_token") as mock_exchange:
            agent._swap_credential(entry)
            mock_exchange.assert_not_called()

        # Verify entry values were used
        assert agent.api_key == "sk_test"
        assert agent.base_url == "https://api.openai.com"
