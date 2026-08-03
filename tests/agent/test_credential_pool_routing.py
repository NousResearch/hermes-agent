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

        def rotate(status_code=None, error_context=None, **_kwargs):
            self._rotation_index += 1
            if self._rotation_index < pool_entries:
                return entries[self._rotation_index]
            pool.has_credentials.return_value = False
            return None

        pool.mark_exhausted_and_rotate = MagicMock(side_effect=rotate)
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()
        agent.log_prefix = ""
        agent.api_key = "test-api-key"
        agent.provider = "test-provider"
        pool.provider = "test-provider"

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
        pool.mark_exhausted_and_rotate.assert_called_once_with(status_code=429, error_context=None, api_key_hint="test-api-key")
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
        pool.mark_exhausted_and_rotate.assert_called_once_with(status_code=402, error_context=None, api_key_hint="test-api-key")

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

    def test_api_key_hint_from_pool_current_when_agent_key_missing(self):
        """api_key_hint should fall back to pool.current().runtime_api_key
        when agent.api_key is not set (#43747)."""
        from run_agent import AIAgent

        with patch.object(AIAgent, "__init__", lambda self, **kw: None):
            agent = AIAgent()

        e0 = MagicMock(name="entry_0")
        e0.id = "cred-0"
        e1 = MagicMock(name="entry_1")
        e1.id = "cred-1"

        pool = MagicMock()
        pool.has_credentials.return_value = True
        pool.provider = "test-provider"
        agent.provider = "test-provider"

        # current entry has a runtime_api_key
        cur_entry = MagicMock()
        cur_entry.runtime_api_key = "pool-current-key"
        pool.current.return_value = cur_entry

        pool.mark_exhausted_and_rotate.return_value = e1
        agent._credential_pool = pool
        agent._swap_credential = MagicMock()
        agent.log_prefix = ""
        # No agent.api_key set — should fall back to pool.current().runtime_api_key

        recovered, has_retried = agent._recover_with_credential_pool(
            status_code=402, has_retried_429=False
        )
        assert recovered is True
        pool.mark_exhausted_and_rotate.assert_called_once_with(
            status_code=402, error_context=None, api_key_hint="pool-current-key"
        )


# ---------------------------------------------------------------------------
# 6. Real-pool regression: the hint routes exhaustion to the FAILED entry
# ---------------------------------------------------------------------------

class TestApiKeyHintRealPool:
    """Prove the routing guarantee through the real CredentialPool selector:
    when the failed key differs from the pool's current/first entry, only the
    failed entry is marked exhausted (#43747, wrong-entry marking)."""

    def _seed_pool(self, tmp_path, monkeypatch):
        import json

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "providers": {},
                    "credential_pool": {
                        "openrouter": [
                            {
                                "id": "cred-healthy",
                                "label": "healthy",
                                "auth_type": "api_key",
                                "priority": 0,
                                "source": "manual",
                                "access_token": "sk-or-healthy",
                            },
                            {
                                "id": "cred-failed",
                                "label": "failed",
                                "auth_type": "api_key",
                                "priority": 1,
                                "source": "manual",
                                "access_token": "sk-or-failed",
                            },
                        ]
                    },
                }
            )
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from agent.credential_pool import load_pool

        return load_pool("openrouter")

    def test_hint_marks_failed_entry_not_current(self, tmp_path, monkeypatch):
        pool = self._seed_pool(tmp_path, monkeypatch)
        # Another process/pool instance issued sk-or-failed; THIS pool's
        # current() would resolve to the first (healthy) entry.
        assert pool.select().access_token == "sk-or-healthy"

        next_entry = pool.mark_exhausted_and_rotate(
            status_code=429,
            error_context={"reason": "rate_limit_exceeded"},
            api_key_hint="sk-or-failed",
        )

        statuses = {e.id: e.last_status for e in pool._entries}
        assert statuses["cred-failed"] == "exhausted"
        assert statuses["cred-healthy"] in (None, "ok")
        assert next_entry is not None
        assert next_entry.access_token == "sk-or-healthy"

    def test_without_hint_current_entry_is_marked(self, tmp_path, monkeypatch):
        """Baseline: no hint falls back to current() — the pre-fix behavior."""
        pool = self._seed_pool(tmp_path, monkeypatch)
        assert pool.select().access_token == "sk-or-healthy"

        pool.mark_exhausted_and_rotate(status_code=429, error_context=None)

        statuses = {e.id: e.last_status for e in pool._entries}
        assert statuses["cred-healthy"] == "exhausted"
        assert statuses["cred-failed"] in (None, "ok")
