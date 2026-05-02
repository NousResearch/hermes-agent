"""Tests that _try_activate_fallback updates the context compressor and runtime config context."""

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


def _make_agent_with_compressor() -> AIAgent:
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
    agent._config_context_length = 272_000
    agent._client_kwargs = {}
    agent._primary_runtime = {
        "provider": "yunfei",
        "model": "gpt-5.4",
        "base_url": YUNFEI_BASE_URL,
        "api_mode": "chat_completions",
        "api_key": "sk-primary",
        "client_kwargs": {},
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "compressor_model": "gpt-5.4",
        "compressor_base_url": YUNFEI_BASE_URL,
        "compressor_api_key": "sk-primary",
        "compressor_provider": "yunfei",
        "compressor_context_length": 272_000,
        "compressor_threshold_tokens": 136_000,
    }

    agent._fallback_activated = False
    agent._fallback_model = {
        "provider": "yunfei",
        "model": "gpt-5.3-codex",
    }
    agent._fallback_chain = [agent._fallback_model]
    agent._fallback_index = 0
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._emit_status = lambda msg: None
    agent._is_direct_openai_url = lambda url: False
    agent._provider_model_requires_responses_api = lambda *args, **kwargs: False

    compressor = ContextCompressor(
        model="gpt-5.4",
        threshold_percent=0.50,
        base_url=YUNFEI_BASE_URL,
        api_key="sk-primary",
        provider="yunfei",
        quiet_mode=True,
        config_context_length=272_000,
    )
    agent.context_compressor = compressor

    return agent


@patch("run_agent.get_provider_request_timeout", return_value=None)
@patch.object(AIAgent, "_anthropic_prompt_cache_policy", return_value=(False, False))
@patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
@patch("agent.auxiliary_client.resolve_provider_client")
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_compressor_updated_on_fallback_and_recomputes_config_context_length(
    mock_ctx_len,
    mock_resolve,
    mock_load_config,
    mock_get_compatible,
    _mock_lmstudio,
    _mock_cache_policy,
    _mock_timeout,
):
    """Fallback activation must recompute config_context_length for the fallback runtime."""
    cfg = _runtime_cfg()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = _make_agent_with_compressor()

    fb_client = MagicMock()
    fb_client.base_url = YUNFEI_BASE_URL
    fb_client.api_key = "sk-fallback"
    mock_resolve.return_value = (fb_client, None)

    result = agent._try_activate_fallback()

    assert result is True
    assert agent._fallback_activated is True
    assert agent._config_context_length == 200_000
    assert mock_ctx_len.call_args.kwargs["config_context_length"] == 200_000

    c = agent.context_compressor
    assert c.model == "gpt-5.3-codex"
    assert c.base_url == YUNFEI_BASE_URL
    assert c.api_key == "sk-fallback"
    assert c.provider == "yunfei"
    assert c.context_length == 200_000
    assert c.threshold_tokens == int(200_000 * c.threshold_percent)


@patch("run_agent.get_provider_request_timeout", return_value=None)
@patch.object(AIAgent, "_anthropic_prompt_cache_policy", return_value=(False, False))
@patch.object(AIAgent, "_ensure_lmstudio_runtime_loaded", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
@patch("agent.auxiliary_client.resolve_provider_client")
@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
def test_compressor_not_present_does_not_crash(
    mock_ctx_len,
    mock_resolve,
    mock_load_config,
    mock_get_compatible,
    _mock_lmstudio,
    _mock_cache_policy,
    _mock_timeout,
):
    """If the agent has no compressor, fallback should still succeed."""
    cfg = _runtime_cfg()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = _make_agent_with_compressor()
    agent.context_compressor = None

    fb_client = MagicMock()
    fb_client.base_url = YUNFEI_BASE_URL
    fb_client.api_key = "sk-fallback"
    mock_resolve.return_value = (fb_client, None)

    result = agent._try_activate_fallback()
    assert result is True
    assert agent._config_context_length == 200_000
