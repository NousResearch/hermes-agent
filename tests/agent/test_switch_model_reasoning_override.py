"""Tests for per-model reasoning_effort override during /model switch.

Tests that switch_model:
1. Re-resolves reasoning_config when switching to a model with an override
2. Falls back to global when switching to a model without an override
3. Saves reasoning_config into _primary_runtime for fallback recovery
"""

import json

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def configured_agent(tmp_path, monkeypatch):
    """Exercise real config loading, agent construction and switching without a provider call."""
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "model": {"default": "model-a", "provider": "openai"},
        "agent": {"reasoning_effort": None, "reasoning_overrides": {"model-a": "high", "model-b": "low"}},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    agent = AIAgent(
        model="model-a", provider="openai", api_key="test-key", api_mode="chat_completions",
        base_url="http://127.0.0.1:1/v1", enabled_toolsets=[], quiet_mode=True,
        skip_context_files=True, skip_memory=True, skip_background_review=True,
        reasoning_config={"enabled": True, "effort": "high"},
    )
    yield agent
    agent.close()


class TestSwitchModelReasoningOverride:
    """Test switch_model re-resolves reasoning_config on model switch."""

    def _make_fake_agent(self, model="gpt-5", provider="openai"):
        """Create a minimal fake agent for switch_model testing."""
        agent = MagicMock()
        agent.model = model
        agent.provider = provider
        agent.base_url = "https://api.openai.com/v1"
        agent.api_mode = "openai"
        agent.api_key = "test-key"
        agent._client_kwargs = {"api_key": "test-key", "base_url": "https://api.openai.com/v1"}
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._config_context_length = None
        agent._transport_cache = {}
        agent.context_compressor = None
        agent._cached_system_prompt = None
        agent._anthropic_api_key = ""
        agent._anthropic_base_url = None
        agent._is_anthropic_oauth = False
        agent._anthropic_prompt_cache_policy = MagicMock(
            return_value=(False, False)
        )
        agent._ensure_lmstudio_runtime_loaded = MagicMock()
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        return agent

    def test_primary_runtime_includes_reasoning_config(self):
        """After switch_model, _primary_runtime should contain reasoning_config key."""
        from agent.agent_runtime_helpers import switch_model

        agent = self._make_fake_agent()

        fake_cfg = {
            "model": {"default": "claude-opus-4.5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "claude-opus-4.5": "xhigh",
                },
            },
        }

        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            try:
                switch_model(
                    agent,
                    new_model="claude-opus-4.5",
                    new_provider="anthropic",
                    base_url="https://api.anthropic.com",
                    api_mode="anthropic_messages",
                )
            except Exception:
                # Client creation may fail in test env; check _primary_runtime was set
                pass

        assert hasattr(agent, "_primary_runtime")
        assert "reasoning_config" in agent._primary_runtime



    @pytest.mark.parametrize("saved_reasoning", [{"enabled": True, "effort": "xhigh"}, None])
    def test_restore_primary_runtime_restores_reasoning(self, saved_reasoning):
        """restore_primary_runtime should restore reasoning_config from snapshot."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        agent = MagicMock()
        agent._primary_runtime = {
            "model": "claude-opus-4.5",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
            "api_key": "key",
            "client_kwargs": {},
            "use_prompt_caching": True,
            "use_native_cache_layout": False,
            "reasoning_config": saved_reasoning,
            "compressor_model": "claude-opus-4.5",
            "compressor_base_url": "",
            "compressor_api_key": "",
            "compressor_provider": "",
            "compressor_context_length": 0,
            "compressor_api_mode": "",
            "compressor_threshold_tokens": 0,
            "anthropic_api_key": "key",
            "anthropic_base_url": "https://api.anthropic.com",
            "is_anthropic_oauth": False,
        }
        agent._fallback_activated = True
        agent._fallback_index = 0
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._transport_cache = {}
        agent._config_context_length = None
        agent._rate_limited_until = 0
        agent.model = "fallback-model"
        agent.provider = "openai"
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        agent.context_compressor = MagicMock()
        agent.base_url = ""
        # Mock the methods restore_primary_runtime calls
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(True, False))
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._ensure_lmstudio_runtime_loaded = MagicMock()

        result = restore_primary_runtime(agent)
        assert result is True
        assert agent.reasoning_config == saved_reasoning

    @pytest.mark.parametrize("override", [{"enabled": False}, {"enabled": True, "effort": "high"}, None])
    def test_explicit_session_reasoning_override_beats_destination_default(self, configured_agent, override):
        """Explicit values (even matching the source default) and exact None restores win."""
        agent = configured_agent
        for target in ("model-b", "unconfigured-model"):
            agent.switch_model(
                target, "openai", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                api_mode="chat_completions", reasoning_config_override=override,
            )
            assert agent.model == target
            assert agent.reasoning_config == override
            assert agent._primary_runtime["reasoning_config"] == override

    def test_unproven_reasoning_is_re_resolved_for_destination(self, configured_agent):
        """An inherited source-model value must not leak, even when the target has no default."""
        agent = configured_agent
        for target, expected in (("model-b", {"enabled": True, "effort": "low"}), ("unconfigured-model", None)):
            agent.switch_model(
                target, "openai", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                api_mode="chat_completions",
            )
            assert agent.model == target
            assert agent.reasoning_config == expected
