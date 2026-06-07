"""Tests that switch_model does not inherit stale context_length overrides."""

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


@patch("hermes_cli.config.load_config", return_value={"model": {"context_length": 65536}})
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_switch_model_re_reads_global_context_length(mock_ctx_len, mock_load_cfg):
    """Global model.context_length in config.yaml must be honored after /model switch."""
    agent = _make_agent_with_compressor(config_context_length=None)

    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # The global model.context_length=65536 must be passed to get_model_context_length.
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 65536


@patch(
    "hermes_cli.config.get_compatible_custom_providers",
    return_value=[{"name": "Private", "base_url": "https://private.example/v1"}],
)
@patch("hermes_cli.config.load_config", return_value={})
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_switch_model_persists_refreshed_custom_providers(
    mock_ctx_len,
    mock_load_cfg,
    mock_get_custom_providers,
):
    """Live custom_providers reload must persist into primary runtime state."""
    agent = _make_agent_with_compressor(config_context_length=None)
    agent._custom_providers = [{"name": "Old", "base_url": "https://old.example/v1"}]

    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    expected = [{"name": "Private", "base_url": "https://private.example/v1"}]
    assert agent._custom_providers == expected
    assert agent._primary_runtime["custom_providers"] == expected
    assert agent._primary_runtime["custom_providers"] is not agent._custom_providers
