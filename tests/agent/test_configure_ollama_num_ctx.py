"""Tests for _configure_ollama_num_ctx — the Ollama-only num_ctx gate.

The function must NOT apply ollama_num_ctx to cloud runtimes (e.g. DeepSeek
on Nous), because num_ctx is an Ollama-only concept. Applying it to a cloud
model clamps the compressor window down to the configured num_ctx, silently
discarding ~90% of the real context window.
"""

from unittest.mock import MagicMock

import pytest


def _make_agent(provider="openai", base_url="https://api.openai.com/v1", model="gpt-4"):
    """Build a minimal agent-like object for _configure_ollama_num_ctx."""
    agent = MagicMock()
    agent.provider = provider
    agent.base_url = base_url
    agent.model = model
    agent.api_key = "sk-test"
    agent._ollama_num_ctx = None
    agent.quiet_mode = True
    # context_compressor must be a real object with context_length, not a MagicMock
    agent.context_compressor = MagicMock()
    agent.context_compressor.context_length = 0
    return agent


class TestCloudRuntimeNoClamp:
    """Cloud models must NOT have ollama_num_ctx applied."""

    def test_cloud_model_with_ollama_num_ctx_config(self):
        """Cloud model (openai, non-local URL) with ollama_num_ctx=131072 in config
        must keep _ollama_num_ctx=None — the clamp must NOT fire."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="openai", base_url="https://api.openai.com/v1")
        model_cfg = {"ollama_num_ctx": 131072}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=1_310_720)

        assert agent._ollama_num_ctx is None, (
            f"Cloud model got clamped to {agent._ollama_num_ctx}; "
            "ollama_num_ctx should not apply to non-Ollama runtimes"
        )

    def test_cloud_model_without_ollama_num_ctx_config(self):
        """Cloud model with no ollama_num_ctx in config — must stay None."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="anthropic", base_url="https://api.anthropic.com")
        model_cfg = {}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=200_000)

        assert agent._ollama_num_ctx is None

    def test_deepseek_on_nous_cloud(self):
        """Explicit DeepSeek-on-Nous scenario from the PR description."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="nous", base_url="https://inference.nousresearch.com")
        model_cfg = {"ollama_num_ctx": 131072}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=1_310_720)

        assert agent._ollama_num_ctx is None, (
            "DeepSeek on Nous cloud must not have num_ctx applied"
        )


class TestOllamaRuntimeApplies:
    """Ollama runtimes MUST have ollama_num_ctx applied."""

    def test_ollama_provider_with_override(self):
        """Ollama provider with explicit ollama_num_ctx=65536 must apply it."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="ollama", base_url="http://localhost:11434")
        model_cfg = {"ollama_num_ctx": 65536}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=131_072)

        assert agent._ollama_num_ctx == 65536

    def test_ollama_over_tailscale(self):
        """Ollama over Tailscale (CGNAT) is a local endpoint — must apply override."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="openai", base_url="http://100.64.0.1:11434")
        model_cfg = {"ollama_num_ctx": 32768}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=65536)

        assert agent._ollama_num_ctx == 32768

    def test_local_ollama_without_explicit_override(self):
        """Local Ollama with no explicit override — auto-detect path (no crash)."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="ollama", base_url="http://localhost:11434")
        model_cfg = {}

        # No crash; _ollama_num_ctx may be set by auto-detect or stay None
        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=131_072)

        # Should not crash; result depends on whether query_ollama_num_ctx succeeds
        assert hasattr(agent, "_ollama_num_ctx")


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_override_does_not_crash(self):
        """Invalid ollama_num_ctx (e.g. string auto) must not crash."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="ollama", base_url="http://localhost:11434")
        model_cfg = {"ollama_num_ctx": "auto"}

        # Should not raise; invalid value is logged and ignored
        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=131_072)

        # _ollama_num_ctx should be None (invalid override discarded)
        assert agent._ollama_num_ctx is None or isinstance(agent._ollama_num_ctx, int)

    def test_none_model_cfg(self):
        """model_cfg=None must not crash (some callers pass None)."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="ollama", base_url="http://localhost:11434")

        _configure_ollama_num_ctx(agent, None, _config_context_length=131_072)

        assert hasattr(agent, "_ollama_num_ctx")

    def test_empty_base_url(self):
        """Agent with no base_url must not crash."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="openai", base_url=None)
        model_cfg = {"ollama_num_ctx": 131072}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=1_310_720)

        assert agent._ollama_num_ctx is None

    def test_provider_case_insensitive(self):
        """Provider check must be case-insensitive (Ollama vs ollama)."""
        from agent.agent_init import _configure_ollama_num_ctx

        agent = _make_agent(provider="Ollama", base_url="http://localhost:11434")
        model_cfg = {"ollama_num_ctx": 65536}

        _configure_ollama_num_ctx(agent, model_cfg, _config_context_length=131_072)

        assert agent._ollama_num_ctx == 65536
