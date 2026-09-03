"""Finding C (#3): /model transaction boundary — post-client/pre-finalized.

The switch_model transaction has THREE failure windows that must each roll the
host route (model/provider/client/base_url/api_key/api_mode/_client_kwargs/
pre-cap/_primary_runtime) back to its prior coherent state:

  Window A — client rebuild          (already wrapped → _restore_snapshot)
  Window B — LM Studio preload       (already wrapped → _restore_snapshot)
  Window C — engine transition       (already wrapped → _restore_snapshot)

BUT the region BETWEEN Window B and Window C — the prompt-cache policy
re-eval (``_anthropic_prompt_cache_policy``) and context/capability resolution
(``get_model_context_length``) — is NOT wrapped. An exception from either
propagates with the host already committed (Window A) and the engine NOT yet
transitioned (Window C), leaving the agent with a NEW model/provider/client
paired with the OLD engine route — exactly the incoherence the switch_model
rollback exists to prevent.

This test drives the REAL ``agent.agent_runtime_helpers.switch_model`` with:
  * preload succeeds (Window B passes)
  * ``_anthropic_prompt_cache_policy`` raising (prompt-cache policy failure)
  * ``get_model_context_length`` raising (context/capability resolution failure)

and asserts the prior coherent route is FULLY preserved.

RED   — before the fix, these exceptions propagate WITHOUT rollback; the
         agent is left with the NEW host route → assertions FAIL.
GREEN — after the fix (region wrapped in try/except → _restore_snapshot),
         the prior route is restored → assertions PASS.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent_c() -> AIAgent:
    """Agent on openai-compatible with a coherent initial state."""
    agent = AIAgent.__new__(AIAgent)

    # OLD (pre-swap) coherent route
    agent.provider = "custom"
    agent.model = "old-model"
    agent.requested_provider = "custom"
    agent.base_url = "http://localhost:8080/v1"
    agent.api_key = "key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalClient")
    agent._client_kwargs = {
        "api_key": "key-original",
        "base_url": "http://localhost:8080/v1",
    }
    agent._config_context_length = 131072
    agent._pre_cap_context_length = 131072
    agent._primary_runtime = {
        "model": "old-model",
        "provider": "custom",
        "base_url": "http://localhost:8080/v1",
        "api_mode": "chat_completions",
        "compressor_context_length": 131072,
    }

    # Mock context_compressor (engine) — coherent initial state
    cc = MagicMock(name="ContextCompressor")
    cc.model = "old-model"
    cc.provider = "custom"
    cc.base_url = "http://localhost:8080/v1"
    cc.api_key = "key-original"
    cc.api_mode = "chat_completions"
    cc.context_length = 131072
    cc.pre_cap_context_length = 131072
    agent.context_compressor = cc

    # Other fields switch_model touches
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._reasoning_echo_flag = False
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.reasoning_config = None
    agent._custom_providers = None

    return agent


def _assert_prior_route_preserved(agent, original_client, original_primary, original_pre_cap, original_engine):
    """Assert the prior coherent route is fully preserved."""
    assert agent.model == "old-model", f"model must be restored, got {agent.model!r}"
    assert agent.provider == "custom", f"provider must be restored, got {agent.provider!r}"
    assert agent.client is original_client, "client must be restored to the original object"
    assert agent.base_url == "http://localhost:8080/v1", (
        f"base_url must be restored, got {agent.base_url!r}"
    )
    assert agent.api_key == "key-original", (
        f"api_key must be restored, got {agent.api_key!r}"
    )
    assert agent.api_mode == "chat_completions", (
        f"api_mode must be restored, got {agent.api_mode!r}"
    )
    assert agent._pre_cap_context_length == original_pre_cap, (
        f"pre-cap must be restored, got {agent._pre_cap_context_length!r}"
    )
    assert agent._primary_runtime == original_primary, (
        f"_primary_runtime must be restored, got {agent._primary_runtime!r}"
    )
    assert agent.context_compressor is original_engine, (
        "context_compressor must be the same object"
    )
    assert agent.context_compressor.model == "old-model", (
        f"engine model must be old, got {agent.context_compressor.model!r}"
    )
    assert agent.context_compressor.provider == "custom", (
        f"engine provider must be old, got {agent.context_compressor.provider!r}"
    )


class TestPostClientPromptCachePolicyRollback:
    """Finding #3: prompt-cache policy raise after client commit must roll back."""

    def test_prompt_cache_policy_raise_rolls_back_host_route(self):
        """When _anthropic_prompt_cache_policy raises (after preload succeeds),
        the prior coherent route must be fully preserved."""
        agent = _make_agent_c()

        original_client = agent.client
        original_primary = dict(agent._primary_runtime)
        original_pre_cap = agent._pre_cap_context_length
        original_engine = agent.context_compressor

        # Client rebuild SUCCEEDS (Window A passes).
        new_client = MagicMock(name="NewClient")
        agent._create_openai_client = lambda *_a, **_kw: new_client

        # LM Studio preload SUCCEEDS (Window B passes).
        agent._ensure_lmstudio_runtime_loaded = lambda *_a, **_kw: MagicMock(
            context_length=65536, rejected=False, load_attempted=True
        )
        agent._lmstudio_load_was_unverified = lambda *_a, **_kw: False
        agent._effective_lmstudio_context_length = lambda *_a, **_kw: 65536

        # PROMPT-CACHE POLICY RAISES (the unprotected region).
        def boom_policy(*_a, **_kw):
            raise RuntimeError("simulated prompt-cache policy failure")

        agent._anthropic_prompt_cache_policy = boom_policy

        with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
            with pytest.raises(RuntimeError, match="prompt-cache policy failure"):
                agent.switch_model(
                    new_model="new-model",
                    new_provider="custom",
                    api_key="key-new",
                    base_url="http://localhost:8080/v1",
                    api_mode="chat_completions",
                )

        _assert_prior_route_preserved(
            agent, original_client, original_primary, original_pre_cap, original_engine
        )


class TestPostClientContextResolutionRollback:
    """Finding #3: get_model_context_length raise after client commit must roll back."""

    def test_get_model_context_length_raise_rolls_back_host_route(self):
        """When get_model_context_length raises (after preload + prompt-cache
        succeed), the prior coherent route must be fully preserved."""
        agent = _make_agent_c()

        original_client = agent.client
        original_primary = dict(agent._primary_runtime)
        original_pre_cap = agent._pre_cap_context_length
        original_engine = agent.context_compressor

        # Client rebuild SUCCEEDS (Window A passes).
        new_client = MagicMock(name="NewClient")
        agent._create_openai_client = lambda *_a, **_kw: new_client

        # LM Studio preload SUCCEEDS (Window B passes).
        agent._ensure_lmstudio_runtime_loaded = lambda *_a, **_kw: MagicMock(
            context_length=65536, rejected=False, load_attempted=True
        )
        agent._lmstudio_load_was_unverified = lambda *_a, **_kw: False
        agent._effective_lmstudio_context_length = lambda *_a, **_kw: 65536

        # Prompt-cache policy SUCCEEDS (returns valid flags).
        agent._anthropic_prompt_cache_policy = lambda **_kw: (False, False)

        # get_model_context_length RAISES (the unprotected region).
        # We patch it at the module level where it's imported.
        with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None), \
             patch("agent.model_metadata.get_model_context_length",
                   side_effect=RuntimeError("simulated context resolution failure")):
            with pytest.raises(RuntimeError, match="context resolution failure"):
                agent.switch_model(
                    new_model="new-model",
                    new_provider="custom",
                    api_key="key-new",
                    base_url="http://localhost:8080/v1",
                    api_mode="chat_completions",
                )

        _assert_prior_route_preserved(
            agent, original_client, original_primary, original_pre_cap, original_engine
        )
