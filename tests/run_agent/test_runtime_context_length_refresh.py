"""Focused regression tests for runtime _config_context_length refresh."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


YUNFEI_BASE_URL = "https://ai.yunfei.best/v1"


def _cfg_with_top_level():
    return {
        "model": {"context_length": 200_000},
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


def _cfg_without_top_level():
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


def _make_restore_agent() -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.model = "gpt-5.3-codex"
    agent.provider = "yunfei"
    agent.base_url = YUNFEI_BASE_URL
    agent.api_mode = "chat_completions"
    agent.api_key = "sk-fallback"
    agent._transport_cache = {}
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._rate_limited_until = 0
    agent._config_context_length = 200_000
    agent._client_kwargs = {}
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.client = MagicMock()

    agent.context_compressor = ContextCompressor(
        model="gpt-5.3-codex",
        threshold_percent=0.50,
        base_url=YUNFEI_BASE_URL,
        api_key="sk-fallback",
        provider="yunfei",
        quiet_mode=True,
        config_context_length=200_000,
    )

    agent._primary_runtime = {
        "model": "gpt-5.4",
        "provider": "yunfei",
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
    return agent


@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
def test_helper_prefers_top_level_context_length(
    mock_load_config,
    mock_get_compatible,
):
    """Top-level model.context_length must outrank per-model custom overrides."""
    cfg = _cfg_with_top_level()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = AIAgent.__new__(AIAgent)

    result = agent._resolve_runtime_config_context_length(
        model="gpt-5.4",
        base_url=YUNFEI_BASE_URL,
    )

    assert result == 200_000


@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
def test_helper_uses_matching_custom_provider_when_top_level_absent(
    mock_load_config,
    mock_get_compatible,
):
    """When top-level override is absent, helper must return the matching model override."""
    cfg = _cfg_without_top_level()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = AIAgent.__new__(AIAgent)

    result = agent._resolve_runtime_config_context_length(
        model="gpt-5.4",
        base_url=YUNFEI_BASE_URL,
    )

    assert result == 272_000


@patch("run_agent.get_provider_request_timeout", return_value=None)
@patch.object(AIAgent, "_create_openai_client", return_value=MagicMock())
@patch("hermes_cli.config.get_compatible_custom_providers")
@patch("hermes_cli.config.load_config")
def test_restore_primary_runtime_recomputes_config_context_length(
    mock_load_config,
    mock_get_compatible,
    _mock_create_client,
    _mock_timeout,
):
    """Restoring the primary runtime must replace fallback context_length with the primary one."""
    cfg = _cfg_without_top_level()
    mock_load_config.return_value = cfg
    mock_get_compatible.return_value = cfg["custom_providers"]
    agent = _make_restore_agent()

    restored = agent._restore_primary_runtime()

    assert restored is True
    assert agent.model == "gpt-5.4"
    assert agent._config_context_length == 272_000
    assert agent.context_compressor.context_length == 272_000
    assert agent._fallback_activated is False
    assert agent._fallback_index == 0
