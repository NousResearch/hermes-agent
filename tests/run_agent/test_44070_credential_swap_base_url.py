"""Tests for GH #44070: credential rotation must not re-point a session
running on a configured base_url override back at the provider's default
endpoint.

Scenario: provider ``xiaomi`` with ``model.base_url`` set to the MiMo
token-plan endpoint (``https://token-plan-cn.xiaomimimo.com/v1``).  The
env-seeded pool entry stores only the registry default
(``https://api.xiaomimimo.com/v1`` — see ``base_url = env_url or
pconfig.inference_base_url`` in agent/credential_pool.py).  Session
resolution layers the config override on top (``pool_url_is_default`` in
hermes_cli/runtime_provider.py), so turn 1 talks to the token-plan host.
The first 401/429 then triggered ``_recover_with_credential_pool`` →
``_swap_credential``, which adopted the raw entry URL and permanently
pinned the session to the default host — where a token-plan-only key
fails every later turn ("HTTP 404: 404 page not found").

Also covers the companion fix: ``switch_model()`` detaches a credential
pool seeded for a different provider (mirror of the #33163 fallback fix),
so switching providers mid-session can't be undone by a later pool swap.
"""

from unittest.mock import MagicMock

from run_agent import AIAgent

XIAOMI_DEFAULT = "https://api.xiaomimimo.com/v1"
TOKEN_PLAN = "https://token-plan-cn.xiaomimimo.com/v1"


def _bare_agent(provider="xiaomi", base_url=TOKEN_PLAN, api_mode="chat_completions"):
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.model = "mimo-v2.5-pro"
    agent.base_url = base_url
    agent.api_mode = api_mode
    agent.api_key = "plan-key"
    agent._client_kwargs = {"api_key": "plan-key", "base_url": base_url}
    return agent


def _entry(base_url=XIAOMI_DEFAULT, key="rotated-key"):
    entry = MagicMock()
    entry.runtime_api_key = key
    entry.access_token = key
    entry.runtime_base_url = base_url
    entry.base_url = base_url
    return entry


class TestPoolEntrySwapBaseUrl:
    def test_registry_default_entry_keeps_configured_override(self):
        """Entry carrying only the provider default must not displace the
        session's configured override endpoint."""
        agent = _bare_agent()
        assert agent._pool_entry_swap_base_url(_entry(XIAOMI_DEFAULT)) == TOKEN_PLAN

    def test_per_credential_endpoint_still_wins(self):
        """Entries with a genuinely credential-specific endpoint (kimi/zai
        region resolution, custom pools) keep overriding the agent URL."""
        agent = _bare_agent()
        regional = "https://token-plan-sgp.xiaomimimo.com/v1"
        assert agent._pool_entry_swap_base_url(_entry(regional)) == regional

    def test_entry_without_url_keeps_current(self):
        agent = _bare_agent()
        entry = _entry(None)
        entry.runtime_base_url = None
        entry.base_url = None
        assert agent._pool_entry_swap_base_url(entry) == TOKEN_PLAN

    def test_agent_on_default_url_adopts_entry_url(self):
        """No override in play: agent already on the default endpoint —
        nothing to preserve, entry URL passes through unchanged."""
        agent = _bare_agent(base_url=XIAOMI_DEFAULT)
        assert agent._pool_entry_swap_base_url(_entry(XIAOMI_DEFAULT)) == XIAOMI_DEFAULT

    def test_unknown_provider_adopts_entry_url(self):
        """Providers outside the registry (e.g. 'custom') have no default to
        compare against; entry URL wins (custom pools track config.yaml)."""
        agent = _bare_agent(provider="custom")
        assert agent._pool_entry_swap_base_url(_entry(XIAOMI_DEFAULT)) == XIAOMI_DEFAULT

    def test_swap_credential_preserves_override(self):
        """End-to-end through _swap_credential: key rotates, URL survives."""
        agent = _bare_agent()
        agent._apply_client_headers_for_base_url = MagicMock()
        agent._replace_primary_openai_client = MagicMock(return_value=True)

        agent._swap_credential(_entry(XIAOMI_DEFAULT, key="fresh-key"))

        assert agent.base_url == TOKEN_PLAN
        assert agent._client_kwargs["base_url"] == TOKEN_PLAN
        assert agent.api_key == "fresh-key"
        assert agent._client_kwargs["api_key"] == "fresh-key"
        agent._replace_primary_openai_client.assert_called_once()


class TestSwitchModelDetachesForeignPool:
    def _switch_ready_agent(self, pool):
        agent = _bare_agent()
        agent._credential_pool = pool
        agent._config_context_length = None
        agent._fallback_activated = True
        agent._fallback_index = 3
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._cached_system_prompt = "cached"
        agent._client_kwargs = {"api_key": "plan-key", "base_url": TOKEN_PLAN}
        agent.context_compressor = None
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._ensure_lmstudio_runtime_loaded = MagicMock()
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._transport_cache = {}
        return agent

    def test_cross_provider_switch_detaches_pool(self):
        pool = MagicMock()
        pool.provider = "xiaomi"
        agent = self._switch_ready_agent(pool)

        agent.switch_model(
            new_model="deepseek-chat",
            new_provider="deepseek",
            api_key="ds-key",
            base_url="https://api.deepseek.com/v1",
            api_mode="chat_completions",
        )

        assert agent._credential_pool is None
        assert agent.base_url == "https://api.deepseek.com/v1"

    def test_same_provider_switch_keeps_pool(self):
        pool = MagicMock()
        pool.provider = "xiaomi"
        agent = self._switch_ready_agent(pool)

        agent.switch_model(
            new_model="mimo-v2.5",
            new_provider="xiaomi",
            api_key="plan-key",
            base_url=TOKEN_PLAN,
            api_mode="chat_completions",
        )

        assert agent._credential_pool is pool
