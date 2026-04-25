"""Tests that switch_model preserves and re-resolves config context length."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


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

    # Store config_context_length for later use in switch_model
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
def test_switch_model_preserves_config_context_length(mock_ctx_len):
    """When switching models, config_context_length should be passed to get_model_context_length."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    assert agent.context_compressor.model == "primary-model"
    assert agent.context_compressor.context_length == 32_768  # From config override

    # Switch model
    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # Verify get_model_context_length was called with config_context_length
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 32_768

    # Verify compressor was updated
    assert agent.context_compressor.model == "new-model"


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


def test_switch_model_re_resolves_custom_provider_per_model_context_length():
    """Switching between models on the same custom provider must refresh the
    per-model context length instead of reusing the previous model's value."""
    agent = _make_agent_with_compressor(config_context_length=1_048_576)
    agent.provider = "cpa"
    agent.base_url = "http://proxy.example/v1"

    fake_cfg = {
        "model": {
            "provider": "cpa",
            "default": "gemini-3-flash-preview",
            "base_url": "http://proxy.example/v1",
            "context_length": 1_048_576,
        },
        "custom_providers": [
            {
                "name": "cpa",
                "base_url": "http://proxy.example/v1",
                "models": {
                    "gemini-3-flash-preview": {"context_length": 1_048_576},
                    "gemma-4-31b-it": {"context_length": 262_144},
                },
            }
        ],
    }

    with patch("hermes_cli.config.load_config", return_value=fake_cfg), patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=fake_cfg["custom_providers"],
    ), patch("agent.model_metadata.get_model_context_length", return_value=262_144) as mock_ctx_len:
        agent.switch_model(
            "gemma-4-31b-it",
            "cpa",
            api_key="sk-new",
            base_url="http://proxy.example/v1",
        )

    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 262_144
    assert agent._config_context_length == 262_144
