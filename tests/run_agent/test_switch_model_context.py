"""Tests that switch_model does not inherit stale context_length overrides."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor
from agent.credential_pool import CredentialPool


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


def test_switch_model_reloads_credential_pool_on_provider_change():
    """When switching providers, _credential_pool must be refreshed."""
    agent = _make_agent_with_compressor()

    # Set up an old pool bound to the original provider
    old_pool = MagicMock(spec=CredentialPool)
    old_pool.provider = "openrouter"
    agent._credential_pool = old_pool

    # Mock load_pool to return a new pool for the target provider
    new_pool = MagicMock(spec=CredentialPool)
    new_pool.provider = "anthropic"

    with patch("agent.credential_pool.load_pool", return_value=new_pool) as mock_load:
        agent.switch_model("claude-sonnet-4", "anthropic", api_key="sk-anthropic")

    # load_pool should be called with the NEW provider
    mock_load.assert_called_once_with("anthropic")
    # Agent's pool should now be the new pool
    assert agent._credential_pool is new_pool
    # Old pool should no longer be referenced
    assert agent._credential_pool is not old_pool


def test_switch_model_does_not_reload_pool_for_same_provider():
    """When switching models on the same provider, pool should not be reloaded."""
    agent = _make_agent_with_compressor()
    existing_pool = MagicMock(spec=CredentialPool)
    existing_pool.provider = "openrouter"
    agent._credential_pool = existing_pool

    with patch("agent.credential_pool.load_pool") as mock_load:
        agent.switch_model("gpt-4o-mini", "openrouter", api_key="sk-other", base_url="https://openrouter.ai/api/v1")

    # load_pool should NOT be called when provider stays the same
    mock_load.assert_not_called()
    # Pool should remain unchanged
    assert agent._credential_pool is existing_pool
