"""Tests that switch_model preserves config_context_length."""

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


# ── Provider-switching regression tests ──────────────────────────────
# Covers the bug where top-level model.context_length (1M) persists
# across /model switches and shadows the new provider's context_length.


@patch("run_agent.load_config")
@patch("hermes_cli.config.get_custom_provider_context_length", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers", return_value=[])
@patch("agent.model_metadata.get_model_context_length", return_value=8_192)
def test_switch_to_local_provider_clears_stale_config_context(
    mock_ctx_len, mock_cps, mock_cp_ctx, mock_load_config
):
    """Switching from cloud (1M ctx) to local (8K ctx) should not shadow
    the local provider's context_length with the stale 1M value."""
    mock_load_config.return_value = {
        "model": {"default": "mimo-pro", "provider": "custom", "context_length": 1_000_000}
    }

    agent = _make_agent_with_compressor(config_context_length=1_000_000)
    assert agent._config_context_length == 1_000_000

    agent.switch_model("local-model", "custom:local", base_url="http://localhost:8080/v1")

    # _config_context_length should be cleared (None) since the new provider
    # is not the default provider
    assert agent._config_context_length is None
    # get_model_context_length should have been called with None
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") is None


@patch("run_agent.load_config")
@patch("hermes_cli.config.get_custom_provider_context_length", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers", return_value=[])
@patch("agent.model_metadata.get_model_context_length", return_value=8_192)
def test_switch_to_local_no_explicit_ctx(
    mock_ctx_len, mock_cps, mock_cp_ctx, mock_load_config
):
    """When config has no explicit context_length and we switch providers,
    _config_context_length should be re-resolved (None if no override)."""
    mock_load_config.return_value = {
        "model": {"default": "mimo-pro", "provider": "custom"}
    }

    agent = _make_agent_with_compressor(config_context_length=None)

    agent.switch_model("local-model", "custom:local", base_url="http://localhost:8080/v1")

    assert agent._config_context_length is None
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") is None


@patch("run_agent.load_config")
@patch("hermes_cli.config.get_custom_provider_context_length", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers", return_value=[])
@patch("agent.model_metadata.get_model_context_length", return_value=1_000_000)
def test_same_provider_keeps_config_context(
    mock_ctx_len, mock_cps, mock_cp_ctx, mock_load_config
):
    """Switching models within the same provider should preserve the
    config context_length."""
    mock_load_config.return_value = {
        "model": {"default": "mimo-pro", "provider": "custom", "context_length": 1_000_000}
    }

    agent = _make_agent_with_compressor(config_context_length=1_000_000)

    # Same provider (custom) — context_length should be preserved
    agent.switch_model("another-model", "custom", base_url="http://localhost:4000/v1")

    assert agent._config_context_length == 1_000_000
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 1_000_000
