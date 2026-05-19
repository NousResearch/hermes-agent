"""Tests that switch_model does not inherit stale context_length overrides."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor
from agent import model_metadata as _mm


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


def test_switch_model_applies_custom_providers_per_model_context():
    """Mid-session /model switch must honor per-model custom_providers overrides."""
    custom = [
        {
            "base_url": "https://proxy.example/v1",
            "models": {
                "glm-4.7": {"context_length": 131_072},
                "minimax-m2.7": {"context_length": 1_048_576},
            },
        }
    ]
    agent = _make_agent_with_compressor(config_context_length=None)
    agent.base_url = "https://proxy.example/v1"
    agent.provider = "custom"
    agent.api_key = "sk-test"
    agent.context_compressor.context_length = 256_000

    probe_patches = [
        patch.object(_mm, "get_cached_context_length", return_value=None),
        patch.object(_mm, "fetch_endpoint_model_metadata", return_value={}),
        patch.object(_mm, "fetch_model_metadata", return_value={}),
        patch.object(_mm, "is_local_endpoint", return_value=False),
        patch.object(_mm, "_is_known_provider_base_url", return_value=False),
    ]
    cfg_patch = patch(
        "hermes_cli.config.get_compatible_custom_providers",
        return_value=custom,
    )
    load_patch = patch("hermes_cli.config.load_config", return_value={"custom_providers": custom})

    for p in probe_patches:
        p.start()
    cfg_patch.start()
    load_patch.start()
    try:
        agent.switch_model(
            "glm-4.7",
            "custom",
            api_key="sk-test",
            base_url="https://proxy.example/v1",
        )
        assert agent.context_compressor.context_length == 131_072
        assert agent._config_context_length == 131_072

        agent.switch_model(
            "minimax-m2.7",
            "custom",
            api_key="sk-test",
            base_url="https://proxy.example/v1",
        )
        assert agent.context_compressor.context_length == 1_048_576
        assert agent._config_context_length == 1_048_576
    finally:
        load_patch.stop()
        cfg_patch.stop()
        for p in probe_patches:
            p.stop()
