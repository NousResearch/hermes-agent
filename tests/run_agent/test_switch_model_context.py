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


def test_switch_model_request_uses_new_opaque_family():
    agent = _make_agent_with_compressor(config_context_length=None)
    base_url = "https://switch.example/anthropic"
    providers = [
        {
            "base_url": base_url,
            "models": {
                "ep-opus-48": {
                    "anthropic_model_family": "claude-opus-4-8",
                },
            },
        }
    ]
    agent._custom_providers = providers
    agent.tools = []
    agent.max_tokens = 4096
    agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
    agent.request_overrides = {}
    agent._oauth_1m_beta_disabled = False

    with (
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
        patch("agent.credential_pool.load_pool", return_value=None),
        patch("hermes_cli.config.load_config", return_value={}),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=1_000_000,
        ),
    ):
        agent.switch_model(
            "ep-opus-48",
            "custom",
            api_key="switch-key",
            base_url=base_url,
            api_mode="anthropic_messages",
        )

    kwargs = agent._build_api_kwargs(
        [{"role": "user", "content": "hi"}],
    )
    assert kwargs["model"] == "ep-opus-48"
    assert kwargs["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert kwargs["output_config"] == {"effort": "xhigh"}
