"""Reproduction test for issue #63425: Provider auto-detection discards credential pools.

When AIAgent is constructed with provider=None and a recognized endpoint URL,
the provider auto-detection works but the credential pool is discarded because
pool validation runs before URL-based provider inference.
"""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest


def _mock_client(api_key="test-key", base_url="https://api.anthropic.com"):
    c = MagicMock()
    c.api_key = api_key
    c.base_url = base_url
    c._default_headers = None
    return c


class TestCredentialPoolPreservedOnAutoDetect:
    """Issue #63425: credential pool should survive provider auto-detection."""

    def test_anthropic_pool_preserved_with_url_auto_detect(self):
        """When provider=None and base_url=api.anthropic.com, the passed
        credential_pool should remain attached after auto-detection."""
        from agent.agent_init import init_agent

        # Build a minimal agent-like object (like tests use object.__new__)
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._base_url = ""
        agent._base_url_lower = ""
        agent._base_url_hostname = ""

        pool = SimpleNamespace(provider="anthropic")

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
             patch("run_agent.get_tool_definitions", return_value=[]), \
             patch('agent.anthropic_adapter.build_anthropic_client', return_value=MagicMock()), \
             patch('agent.anthropic_adapter.resolve_anthropic_token', return_value=''), \
             patch('agent.anthropic_adapter._is_oauth_token', return_value=False), \
             patch('agent.azure_identity_adapter.is_token_provider', return_value=False), \
             patch('hermes_cli.model_normalize.normalize_model_for_provider', return_value='test-model'), \
             patch('agent.credential_pool.load_pool', return_value=MagicMock()), \
             patch('hermes_cli.config.load_config', return_value={}), \
             patch('hermes_cli.config.get_compatible_custom_providers', return_value=[]), \
             patch('agent.iteration_budget.IterationBudget'), \
             patch('hermes_cli.config.cfg_get', return_value=None):

            init_agent(
                agent,
                base_url="https://api.anthropic.com",
                api_key="test-key",
                provider=None,
                model="test-model",
                credential_pool=pool,
                skip_context_files=True,
                skip_memory=True,
                quiet_mode=True,
            )

        print(f"agent.provider = {agent.provider!r}")
        print(f"agent.api_mode = {agent.api_mode!r}")
        print(f"agent._credential_pool is pool = {agent._credential_pool is pool}")

        assert agent.provider == "anthropic", (
            f"Provider should be auto-detected as 'anthropic', got {agent.provider!r}"
        )
        assert agent.api_mode == "anthropic_messages", (
            f"api_mode should be 'anthropic_messages', got {agent.api_mode!r}"
        )
        assert agent._credential_pool is pool, (
            "Credential pool was discarded! agent._credential_pool is NOT the "
            "same object that was passed to AIAgent().\n"
            f"  Expected: {id(pool)}\n"
            f"  Got:      {id(agent._credential_pool)}"
        )

    def test_codex_pool_preserved_with_url_auto_detect(self):
        """When provider=None and base_url=chatgpt.com/backend-api/codex, the
        passed credential_pool should remain attached."""
        from agent.agent_init import init_agent
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._base_url = ""
        agent._base_url_lower = ""
        agent._base_url_hostname = ""

        pool = SimpleNamespace(provider="openai-codex")

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client("key", "https://chatgpt.com/backend-api/codex"), None)), \
             patch("run_agent.get_tool_definitions", return_value=[]), \
             patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch('agent.anthropic_adapter.build_anthropic_client', return_value=MagicMock()), \
             patch('agent.anthropic_adapter.resolve_anthropic_token', return_value=''), \
             patch('agent.anthropic_adapter._is_oauth_token', return_value=False), \
             patch('agent.azure_identity_adapter.is_token_provider', return_value=False), \
             patch('hermes_cli.model_normalize.normalize_model_for_provider', return_value='test-model'), \
             patch('agent.credential_pool.load_pool', return_value=MagicMock()), \
             patch('hermes_cli.config.load_config', return_value={}), \
             patch('hermes_cli.config.get_compatible_custom_providers', return_value=[]), \
             patch('agent.iteration_budget.IterationBudget'), \
             patch('hermes_cli.config.cfg_get', return_value=None):

            init_agent(
                agent,
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="test-key",
                provider=None,
                model="gpt-5.5",
                credential_pool=pool,
                skip_context_files=True,
                skip_memory=True,
                quiet_mode=True,
            )

        print(f"\nagent.provider = {agent.provider!r}")
        print(f"agent.api_mode = {agent.api_mode!r}")
        print(f"agent._credential_pool is pool = {agent._credential_pool is pool}")

        assert agent.provider == "openai-codex", (
            f"Provider should be auto-detected as 'openai-codex', got {agent.provider!r}"
        )
        assert agent._credential_pool is pool, (
            "Credential pool was discarded! agent._credential_pool is NOT the "
            f"same object. Expected: {id(pool)}, Got: {id(agent._credential_pool)}"
        )

    def test_xai_pool_preserved_with_url_auto_detect(self):
        """When provider=None and base_url=api.x.ai, the passed
        credential_pool should remain attached."""
        from agent.agent_init import init_agent
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._base_url = ""
        agent._base_url_lower = ""
        agent._base_url_hostname = ""

        pool = SimpleNamespace(provider="xai")

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client("key", "https://api.x.ai"), None)), \
             patch("run_agent.get_tool_definitions", return_value=[]), \
             patch("run_agent.OpenAI", return_value=MagicMock()), \
             patch('agent.anthropic_adapter.build_anthropic_client', return_value=MagicMock()), \
             patch('agent.anthropic_adapter.resolve_anthropic_token', return_value=''), \
             patch('agent.anthropic_adapter._is_oauth_token', return_value=False), \
             patch('agent.azure_identity_adapter.is_token_provider', return_value=False), \
             patch('hermes_cli.model_normalize.normalize_model_for_provider', return_value='test-model'), \
             patch('agent.credential_pool.load_pool', return_value=MagicMock()), \
             patch('hermes_cli.config.load_config', return_value={}), \
             patch('hermes_cli.config.get_compatible_custom_providers', return_value=[]), \
             patch('agent.iteration_budget.IterationBudget'), \
             patch('hermes_cli.config.cfg_get', return_value=None):

            init_agent(
                agent,
                base_url="https://api.x.ai",
                api_key="test-key",
                provider=None,
                model="grok-5",
                credential_pool=pool,
                skip_context_files=True,
                skip_memory=True,
                quiet_mode=True,
            )

        print(f"\nagent.provider = {agent.provider!r}")
        print(f"agent.api_mode = {agent.api_mode!r}")
        print(f"agent._credential_pool is pool = {agent._credential_pool is pool}")

        assert agent.provider == "xai", (
            f"Provider should be auto-detected as 'xai', got {agent.provider!r}"
        )
        assert agent._credential_pool is pool, (
            "Credential pool was discarded! agent._credential_pool is NOT the "
            f"same object. Expected: {id(pool)}, Got: {id(agent._credential_pool)}"
        )
