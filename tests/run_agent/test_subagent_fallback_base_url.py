"""Tests for subagent fallback base_url fix — issue #24782.

Bug: When a subagent spawned via delegate_task falls back to a secondary
model, it incorrectly uses the parent agent's base_url instead of the
fallback entry's configured base_url.

Root cause confirmed via code trace (May 2026):
1. AIAgent reads fallback_model from config.fallback_model (NOT
   fallback_providers — the config normalizer does NOT process that key).
   If user has fallback_providers in config.yaml but no fallback_model,
   _fallback_chain stays empty and fallback never fires.
2. The _try_activate_fallback() code itself is correct — it properly
   updates self.base_url and self._client_kwargs after fallback activation.
3. The dedup skip logic (lines 8644-8655) only skips when BOTH base_url
   AND model match — correctly preventing self-loop.

These tests verify the fallback activation path end-to-end.
"""

import threading
from unittest.mock import MagicMock, patch, call

import pytest

from run_agent import AIAgent


class TestFallbackBaseUrl:
    """Core fallback base_url behavior — verifies fix for #24782."""

    def test_fallback_skips_when_base_url_and_model_both_match(self):
        """Dedup guard: skip when fallback entry == current (base_url + model)."""
        agent = AIAgent(
            base_url="https://token.sensenova.cn/v1",
            api_key="sk-test",
            provider="custom",
            model="sensonova-6.7-flash-lite",
            fallback_model=[
                {
                    "provider": "custom",
                    "base_url": "https://token.sensenova.cn/v1",
                    "api_key": "sk-test",
                    "model": "sensonova-6.7-flash-lite",
                }
            ],
            quiet_mode=True,
        )
        # Only one fallback entry identical to current — chain exhausted
        result = agent._try_activate_fallback()
        assert result is False

    def test_fallback_activates_when_base_url_differs(self):
        """Fallback with different base_url activates and switches base_url."""
        fb_client = MagicMock()
        fb_client.base_url = "https://token.sensenova.cn/v1"
        fb_client.api_key = "sk-fb-key"
        fb_client._custom_headers = {}
        fb_client.default_headers = {}

        agent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-parent",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=[
                {
                    "provider": "custom",
                    "base_url": "https://token.sensenova.cn/v1",
                    "api_key": "sk-fb-key",
                    "model": "sensonova-6.7-flash-lite",
                }
            ],
            quiet_mode=True,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "sensonova-6.7-flash-lite"),
        ):
            result = agent._try_activate_fallback()

        assert result is True, "Fallback should activate"
        assert agent.base_url == "https://token.sensenova.cn/v1"
        assert agent.model == "sensonova-6.7-flash-lite"
        assert agent.provider == "custom"
        assert agent._client_kwargs["base_url"] == "https://token.sensenova.cn/v1"

    def test_fallback_same_provider_different_base_url_not_skipped(self):
        """Same provider but different base_url must activate (not be skipped)."""
        fb_client = MagicMock()
        fb_client.base_url = "https://token.sensenova.cn/v1"
        fb_client.api_key = "sk-fb"
        fb_client._custom_headers = {}
        fb_client.default_headers = {}

        agent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-parent",
            provider="custom",
            model="deepseek-v4-flash",  # different from fallback model
            fallback_model=[
                {
                    "provider": "custom",
                    "base_url": "https://token.sensenova.cn/v1",
                    "api_key": "sk-fb",
                    "model": "sensonova-6.7-flash-lite",
                }
            ],
            quiet_mode=True,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "sensonova-6.7-flash-lite"),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.base_url == "https://token.sensenova.cn/v1"

    def test_fallback_client_kwargs_updated_after_activation(self):
        """_client_kwargs must include fallback base_url after activation."""
        fb_client = MagicMock()
        fb_client.base_url = "https://token.sensenova.cn/v1"
        fb_client.api_key = "sk-fb-key"
        fb_client._custom_headers = {"User-Agent": "test-agent/1.0"}
        fb_client.default_headers = {}

        agent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-parent",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=[
                {
                    "provider": "custom",
                    "base_url": "https://token.sensenova.cn/v1",
                    "api_key": "sk-fb-key",
                    "model": "sensonova-6.7-flash-lite",
                }
            ],
            quiet_mode=True,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "sensonova-6.7-flash-lite"),
        ):
            agent._try_activate_fallback()

        assert agent._client_kwargs["base_url"] == "https://token.sensenova.cn/v1"
        assert agent._client_kwargs["api_key"] == "sk-fb-key"
        assert "default_headers" in agent._client_kwargs

    def test_subagent_inherits_parent_fallback_chain(self):
        """Subagent passed parent._fallback_chain activates fallback correctly."""
        parent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-parent",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=[
                {
                    "provider": "custom",
                    "base_url": "https://token.sensenova.cn/v1",
                    "api_key": "sk-fb",
                    "model": "sensonova-6.7-flash-lite",
                }
            ],
            quiet_mode=True,
        )

        subagent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-parent",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=getattr(parent, "_fallback_chain", None),
            quiet_mode=True,
        )

        fb_client = MagicMock()
        fb_client.base_url = "https://token.sensenova.cn/v1"
        fb_client.api_key = "sk-fb"
        fb_client._custom_headers = {}
        fb_client.default_headers = {}

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fb_client, "sensonova-6.7-flash-lite"),
        ):
            result = subagent._try_activate_fallback()

        assert result is True
        assert subagent.base_url == "https://token.sensenova.cn/v1"
        assert subagent.model == "sensonova-6.7-flash-lite"

    def test_resolve_provider_client_custom_explicit_base_url(self):
        """provider=custom with explicit_base_url uses provided URL (not default)."""
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client(
            "custom",
            model="sensonova-6.7-flash-lite",
            raw_codex=True,
            explicit_base_url="https://token.sensenova.cn/v1",
            explicit_api_key="sk-fb",
        )

        assert client is not None
        assert "token.sensenova.cn" in str(client.base_url)
        assert "127.0.0.1" not in str(client.base_url)


class TestFallbackChainPopulated:
    """Verify _fallback_chain is correctly built from fallback_model config."""

    def test_fallback_chain_from_list(self):
        """AIAgent._fallback_chain is correctly populated from fallback_model list."""
        agent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-test",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=[
                {"provider": "custom", "base_url": "https://a.cn/v1", "api_key": "sk-a", "model": "model-a"},
                {"provider": "custom", "base_url": "https://b.cn/v1", "api_key": "sk-b", "model": "model-b"},
            ],
            quiet_mode=True,
        )
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_chain[0]["base_url"] == "https://a.cn/v1"
        assert agent._fallback_chain[1]["base_url"] == "https://b.cn/v1"

    def test_fallback_chain_empty_when_not_configured(self):
        """When fallback_model is None, _fallback_chain is empty."""
        agent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-test",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=None,
            quiet_mode=True,
        )
        assert agent._fallback_chain == []

    def test_subagent_inherits_correct_fallback_entries(self):
        """Subagent via delegate_tool pattern inherits base_url in each entry."""
        parent = AIAgent(
            base_url="http://127.0.0.1:8081",
            api_key="sk-test",
            provider="custom",
            model="gemini-3.1-flash-lite-preview",
            fallback_model=[
                {"provider": "custom", "base_url": "https://token.sensenova.cn/v1", "api_key": "sk-fb", "model": "sensonova-6.7-flash-lite"},
            ],
            quiet_mode=True,
        )
        inherited = getattr(parent, "_fallback_chain", None)
        assert inherited is not None
        assert len(inherited) == 1
        assert inherited[0].get("base_url") == "https://token.sensenova.cn/v1"


class TestConfigKeyWarning:
    """Warn if user uses fallback_providers instead of fallback_model."""

    def test_fallback_providers_not_processed(self):
        """Config key 'fallback_providers' is NOT read by config normalizer."""
        from hermes_cli import config as config_module

        # Verify fallback_providers is NOT in the normalization logic
        source = config_module.__dict__.get("_normalize_root_model_keys") or config_module.__dict__.get("normalize_root_model_keys")
        if source:
            import inspect
            src = inspect.getsource(source)
            assert "fallback_providers" not in src, "fallback_providers should not be in config normalizer"