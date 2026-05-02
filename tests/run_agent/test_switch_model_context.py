"""Tests that switch_model recomputes runtime config_context_length."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


YUNFEI_BASE_URL = "https://ai.yunfei.best/v1"


def _runtime_cfg():
    return {
        "model": {},
        "custom_providers": [
            {
                "name": "yunfei",
                "base_url": YUNFEI_BASE_URL,
                "models": {
                    "gpt-5.4": {"context_length": 272_000},
                    "gpt-5.3-codex": {"context_length": 200_000},
                },
            }
        ],
    }


def _make_agent_with_compressor(config_context_length=272_000) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    agent.model = "gpt-5.4"
    agent.provider = "yunfei"
    agent.base_url = YUNFEI_BASE_URL
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._transport_cache = {}
    agent._config_context_length = config_context_length
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._cached_system_prompt = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False

    compressor = ContextCompressor(
        model="gpt-5.4",
        threshold_percent=0.50,
        base_url=YUNFEI_BASE_URL,
        api_key="sk-primary",
        provider="yunfei",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor
    agent._primary_runtime = {}
    return agent


@patch("run_agent.get_provider_request_timeout", return_value=None)
@patch.object(AIAgent, "_anthropic_prompt_cache_policy", return_value=(False, False))
@patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", return_value=None)
@patch.object(AIAgent, "_create_openai_client", return_value=MagicMock())
@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_switch_model_recomputes_config_context_length_for_new_model(
    mock_ctx_len,
    mock_load_config,
    mock_get_compatible,
    _mock_create_client,
    _mock_lmstudio,
    _mock_cache_policy,
    _mock_timeout,
):
    """Switching models must recompute config_context_length for the new runtime."""
    cfg = _runtime_cfg()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = _make_agent_with_compressor(config_context_length=272_000)

    agent.switch_model(
        "gpt-5.3-codex",
        "yunfei",
        api_key="sk-new",
        base_url=YUNFEI_BASE_URL,
        api_mode="chat_completions",
    )

    assert agent._config_context_length == 200_000
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 200_000
    assert agent.context_compressor.model == "gpt-5.3-codex"


@patch("run_agent.get_provider_request_timeout", return_value=None)
@patch.object(AIAgent, "_anthropic_prompt_cache_policy", return_value=(False, False))
@patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", return_value=None)
@patch.object(AIAgent, "_create_openai_client", return_value=MagicMock())
@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_switch_model_without_top_level_override_uses_matching_custom_provider_value(
    mock_ctx_len,
    mock_load_config,
    mock_get_compatible,
    _mock_create_client,
    _mock_lmstudio,
    _mock_cache_policy,
    _mock_timeout,
):
    """Per-model custom_provider value should replace the previous runtime value."""
    cfg = _runtime_cfg()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = _make_agent_with_compressor(config_context_length=272_000)

    agent.switch_model(
        "gpt-5.3-codex",
        "yunfei",
        api_key="sk-new",
        base_url=YUNFEI_BASE_URL,
        api_mode="chat_completions",
    )

    assert agent._config_context_length != 272_000
    assert agent._config_context_length == 200_000
    assert mock_ctx_len.call_args.kwargs["config_context_length"] == 200_000
