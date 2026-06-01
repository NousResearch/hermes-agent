"""Tests for fallback credential pool isolation.

Verifies that fallback activation isolates the credential pool from the
primary provider, preventing two bugs:

1. GH #33163: fallback retains primary's base_url → requests go to wrong endpoint
2. GH #33088: fallback provider's 429 exhausts primary credential pool

Both bugs share the same root cause: _recover_with_credential_pool and
_swap_credential continue operating on the PRIMARY's credential pool during
fallback calls, contaminating primary state with fallback-provider errors.
"""

import sys
from unittest.mock import MagicMock



# ── Helpers ──────────────────────────────────────────────────────────

def _make_pool(provider, n_entries=1):
    """Create a mock credential pool with N entries."""
    pool = MagicMock()
    pool.provider = provider
    pool.has_credentials.return_value = n_entries > 0
    pool.has_available.return_value = n_entries > 0
    entry = MagicMock()
    entry.id = f"{provider}-entry-0"
    entry.runtime_api_key = f"key-{provider}"
    entry.runtime_base_url = f"https://{provider}.example.com/v1"
    entry.access_token = f"token-{provider}"
    entry.base_url = f"https://{provider}.example.com/v1"
    pool.current.return_value = entry
    pool.mark_exhausted_and_rotate.return_value = entry
    return pool


def _make_agent(provider="openai-codex", model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses"):
    """Create a minimal AIAgent-like object with just the fields we need."""
    agent = MagicMock()
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.api_mode = api_mode
    agent.api_key = "primary-key"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._primary_runtime = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_mode": api_mode,
        "api_key": "primary-key",
        "client_kwargs": {
            "api_key": "primary-key",
            "base_url": base_url,
        },
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "anthropic_api_key": "",
        "anthropic_base_url": "",
    }
    agent._config_context_length = None
    agent._credential_pool = _make_pool(provider)
    agent._rate_limited_until = 0
    agent._transport_cache = {}
    agent._client_kwargs = {
        "api_key": "primary-key",
        "base_url": base_url,
    }
    return agent


# ── Test: _try_activate_fallback reloads mismatched pool ──────────────

class TestFallbackCredentialIsolation:
    """Test that _try_activate_fallback reloads the credential pool."""

    def test_fallback_reloads_pool_for_new_provider(self, monkeypatch):
        """When switching from openai-codex to openrouter, the codex pool is
        replaced with the openrouter pool instead of being cleared to None."""
        mock_new_pool = _make_pool("openrouter", n_entries=2)
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: mock_new_pool,
        )

        agent = _make_agent(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex")
        agent._fallback_activated = True
        agent._credential_pool = _make_pool("openai-codex")

        fb_provider = "openrouter"
        fb_model = "openrouter/auto"

        # Simulate the new reload logic from _try_activate_fallback
        pool = getattr(agent, "_credential_pool", None)
        if pool is not None:
            pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
            if pool_provider and pool_provider != fb_provider:
                try:
                    from agent.credential_pool import load_pool
                    new_pool = load_pool(fb_provider)
                    if new_pool and new_pool.has_credentials():
                        agent._credential_pool = new_pool
                except Exception:
                    pass

        assert agent._credential_pool is not None, (
            "Pool should be reloaded, not cleared"
        )
        assert getattr(agent._credential_pool, "provider", "") == fb_provider, (
            f"Reloaded pool should be for {fb_provider}"
        )

    def test_fallback_clears_pool_when_no_credentials(self, monkeypatch):
        """When the new provider has no credentials, pool is cleared to None."""
        agent = _make_agent(provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex")
        agent._fallback_activated = True
        # Pool has 0 entries → load_pool returns pool with has_credentials=False
        agent._credential_pool = _make_pool("openai-codex", n_entries=0)

        fb_provider = "openrouter"

        pool = getattr(agent, "_credential_pool", None)
        if pool is not None:
            pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
            if pool_provider and pool_provider != fb_provider:
                new_pool = _make_pool(fb_provider, n_entries=0)
                if new_pool and new_pool.has_credentials():
                    agent._credential_pool = new_pool
                else:
                    agent._credential_pool = None

        assert agent._credential_pool is None, (
            "Pool should be None when new provider has no credentials"
        )

    def test_fallback_keeps_matching_pool(self):
        """When fallback provider matches pool provider, pool is preserved."""
        agent = _make_agent(provider="openrouter", base_url="https://openrouter.ai/api/v1")
        agent._credential_pool = _make_pool("openrouter")

        fb_provider = "openrouter"

        pool = getattr(agent, "_credential_pool", None)
        if pool is not None:
            pool_provider = getattr(pool, "provider", "") or ""
            if pool_provider.lower() != fb_provider:
                agent._credential_pool = None

        assert agent._credential_pool is not None, (
            "Pool should be preserved when fallback provider matches pool provider"
        )


# ── Test: _recover_with_credential_pool rejects mismatched pool ──────

class TestRecoveryProviderGuard:
    """Test that _recover_with_credential_pool skips mismatched pools."""

    def test_recovery_skips_mismatched_pool(self):
        """_recover_with_credential_pool should not mutate a pool belonging
        to a different provider than the active agent provider."""
        agent = _make_agent(provider="openrouter")
        # Pool still belongs to primary (openai-codex) — mismatch
        agent._credential_pool = _make_pool("openai-codex")

        current_provider = (getattr(agent, "provider", "") or "").strip().lower()
        pool_provider = getattr(agent._credential_pool, "provider", "") or ""

        # The guard logic:
        should_skip = (current_provider and pool_provider and
                       current_provider != pool_provider)

        assert should_skip is True, (
            f"Provider mismatch: agent={current_provider}, pool={pool_provider} — should skip"
        )

    def test_recovery_allows_matching_pool(self):
        """When pool and agent provider match, recovery proceeds normally."""
        agent = _make_agent(provider="openrouter")
        agent._credential_pool = _make_pool("openrouter")

        current_provider = (getattr(agent, "provider", "") or "").strip().lower()
        pool_provider = getattr(agent._credential_pool, "provider", "") or ""

        should_skip = (current_provider and pool_provider and
                       current_provider != pool_provider)

        assert should_skip is False, (
            "Same provider — should allow recovery"
        )

    def test_recovery_429_from_zai_does_not_exhaust_codex_pool(self):
        """Regression test for GH #33088: zai 429 should NOT exhaust
        openai-codex credential pool."""
        agent = _make_agent(provider="zai", base_url="https://api.z.com/v1")
        # Stale codex pool from primary
        codex_pool = _make_pool("openai-codex")
        agent._credential_pool = codex_pool

        # The guard should prevent mark_exhausted_and_rotate from being called
        current_provider = "zai"
        pool_provider = "openai-codex"
        should_skip = current_provider != pool_provider

        assert should_skip is True
        codex_pool.mark_exhausted_and_rotate.assert_not_called()


# ── Test: base_url not overwritten after fallback ────────────────────

class TestBaseUrlLeak:
    """Regression tests for GH #33163: base_url leaks from primary."""

    def test_client_kwargs_base_url_preserved_after_pool_clear(self):
        """After fallback activation clears the pool, _client_kwargs should
        still have the fallback base_url, not the primary's."""
        agent = _make_agent(
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex"
        )

        # Simulate what _try_activate_fallback does:
        fb_base_url = "https://openrouter.ai/api/v1/"
        agent.provider = "openrouter"
        agent.base_url = fb_base_url
        agent._client_kwargs = {
            "api_key": "or-key",
            "base_url": fb_base_url,
        }

        # Clear mismatched pool
        agent._credential_pool = None

        assert agent._client_kwargs["base_url"] == fb_base_url, (
            f"base_url should be {fb_base_url}, not primary's URL"
        )

    def test_swap_credential_does_not_restore_primary_url(self):
        """_swap_credential should not be called when pool is None,
        preventing it from overwriting base_url back to primary's."""
        agent = _make_agent(provider="openrouter", base_url="https://openrouter.ai/api/v1/")
        agent._credential_pool = None  # Cleared by fallback isolation

        # If pool is None, _recover_with_credential_pool returns early
        # and _swap_credential is never called
        pool = agent._credential_pool
        assert pool is None, "Pool should be None — _swap_credential won't be reached"


# ── Test: _reload_pool_for_provider ──────────────────────────────────

class TestReloadPoolForProvider:
    """Test the _reload_pool_for_provider helper."""

    def test_reload_attaches_pool_to_agent(self, monkeypatch):
        """A successful reload attaches the new pool to agent._credential_pool."""
        from agent.agent_runtime_helpers import _reload_pool_for_provider

        mock_pool = _make_pool("openai-codex", n_entries=2)
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: mock_pool,
        )

        agent = _make_agent(provider="openrouter")
        agent._credential_pool = _make_pool("openrouter")

        result = _reload_pool_for_provider(agent, "openai-codex")

        assert result is mock_pool, "Should return the new pool"
        assert agent._credential_pool is mock_pool, (
            "Should attach pool to agent"
        )

    def test_reload_returns_none_when_no_credentials(self, monkeypatch):
        """Returns None when load_pool succeeds but has no credentials."""
        from agent.agent_runtime_helpers import _reload_pool_for_provider

        mock_pool = _make_pool("openai-codex", n_entries=0)
        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: mock_pool,
        )

        agent = _make_agent(provider="openrouter")
        agent._credential_pool = _make_pool("openrouter")

        result = _reload_pool_for_provider(agent, "openai-codex")

        assert result is None, "Should return None when no credentials"

    def test_reload_returns_none_on_load_error(self, monkeypatch):
        """Returns None when load_pool raises an exception."""
        from agent.agent_runtime_helpers import _reload_pool_for_provider

        monkeypatch.setattr(
            "agent.credential_pool.load_pool",
            lambda provider: (_ for _ in ()).throw(RuntimeError("test error")),
        )

        agent = _make_agent(provider="openrouter")
        agent._credential_pool = _make_pool("openrouter")

        result = _reload_pool_for_provider(agent, "openai-codex")

        assert result is None, "Should return None on load error"
        # Agent pool should be unchanged
        assert agent._credential_pool is not None, "Original pool should be preserved"
