"""Tests that switch_model does not inherit stale context_length overrides."""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.models import LMStudioLoadResult
from run_agent import AIAgent
from agent.agent_init import _normalize_route_base_url
from agent.context_compressor import ContextCompressor


class _StubStartupCompressor:
    def __init__(self, *args, **kwargs):
        self.context_length = kwargs.get("config_context_length") or 272_000
        self.config_context_length = kwargs.get("config_context_length")
        self.threshold_tokens = int(self.context_length * 0.95)
        self.threshold_percent = 0.95

    def get_tool_schemas(self):
        return []

    def on_session_start(self, *args, **kwargs):
        return None


def test_route_url_normalization_preserves_path_slash_before_query():
    """A path slash before a query changes OpenAI SDK URL joining."""
    assert _normalize_route_base_url(
        "https://example.com/v1/?tenant=large"
    ) != _normalize_route_base_url("https://example.com/v1?tenant=large")














def _make_direct_start_agent(
    cfg: dict, *, model: str, provider: str, base_url: str
) -> AIAgent:
    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.agent_init.ContextCompressor", new=_StubStartupCompressor),
    ):
        return AIAgent(
            model=model,
            provider=provider,
            api_key="fake-test-token",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def _make_agent_with_compressor(config_context_length=None) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Store the initial config_context_length override used at agent construction.
    agent._config_context_length = config_context_length

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor

    # For switch_model
    agent._primary_runtime = {}

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_clears_previous_config_context_length(mock_ctx_len):
    """Switching models must not reuse the previous model.context_length override."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    assert agent.context_compressor.model == "primary-model"
    assert agent.context_compressor.context_length == 32_768  # From config override

    # Switch model
    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # Verify the old config override is not passed to the new model.
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") is None

    # Verify compressor was updated from the newly resolved model metadata.
    assert agent.context_compressor.model == "new-model"
    assert agent.context_compressor.context_length == 131_072


def test_switch_model_without_config_context_length():
    """When switching models without config override, config_context_length should be None."""
    agent = _make_agent_with_compressor(config_context_length=None)

    with patch("agent.model_metadata.get_model_context_length", return_value=128_000) as mock_ctx_len:
        # Switch model
        agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

        # Verify get_model_context_length was called with None
        mock_ctx_len.assert_called_once()
        call_kwargs = mock_ctx_len.call_args.kwargs
        assert call_kwargs.get("config_context_length") is None


def test_direct_start_model_override_does_not_inherit_profile_context_length():
    """A CLI ``--model`` startup override must not inherit another model's window."""
    cfg = {
        "model": {
            "default": "kimi-k3",
            "provider": "custom:kimi-coding-1m",
            "base_url": "https://api.kimi.com/coding",
            "context_length": 1_048_576,
        },
        "custom_providers": [
            {
                "name": "kimi-coding-1m",
                "base_url": "https://api.kimi.com/coding",
                "models": {"kimi-k3": {"context_length": 1_048_576}},
            }
        ],
    }
    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length is None
    assert agent.context_compressor.context_length == 272_000


def test_direct_start_preserves_context_for_normalized_default_model_alias():
    """Equivalent vendor-prefixed defaults still own their explicit window."""
    cfg = {
        "model": {
            "default": "openai/gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "context_length": 272_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length == 272_000
    assert agent.context_compressor.context_length == 272_000
































































def test_lmstudio_switch_uses_destination_context_and_verified_runtime(monkeypatch):
    agent = _make_agent_with_compressor(config_context_length=32_768)
    calls = []

    def fake_load_config():
        return {}

    def fake_compatible(_cfg):
        return [{"name": "lmstudio", "base_url": "http://127.0.0.1:1234/v1"}]

    def fake_provider_context(*, model, base_url, custom_providers):
        assert model == "lmstudio/new-model"
        assert base_url == "http://127.0.0.1:1234/v1"
        return 120_000

    def fake_lmstudio_load(self, config_context_length=None):
        calls.append(config_context_length)
        return LMStudioLoadResult(100_000)

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load_config)
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", fake_compatible)
    monkeypatch.setattr("hermes_cli.config.get_custom_provider_context_length", fake_provider_context)
    monkeypatch.setattr(AIAgent, "_ensure_lmstudio_runtime_loaded", fake_lmstudio_load)

    with patch("agent.model_metadata.get_model_context_length", return_value=100_000) as mock_ctx_len:
        agent.switch_model(
            "lmstudio/new-model",
            "lmstudio",
            api_key="",
            base_url="http://127.0.0.1:1234/v1",
        )

    assert calls == [120_000]
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 100_000
    assert agent._config_context_length == 120_000
    assert agent.context_compressor.context_length == 100_000
