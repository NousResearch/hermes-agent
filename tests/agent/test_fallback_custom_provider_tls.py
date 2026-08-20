"""Regression: fallback activation must re-apply per-provider TLS settings.

``agent_init`` and ``switch_model`` both call
``apply_custom_provider_tls_to_client_kwargs`` when they (re)build
``agent._client_kwargs``, because every request-scoped client is rebuilt from
that dict and ``create_openai_client`` resolves ``ssl_ca_cert`` /
``ssl_verify`` out of it. ``try_activate_fallback`` is the third place that
rebuilds ``_client_kwargs``; without the same re-apply, falling back to a
custom HTTPS endpoint signed by a private CA works for the very first
request (the initial fb_client resolves TLS via the auxiliary path) and then
fails ``APIConnectionError`` on every rebuilt client.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import try_activate_fallback

_FB_BASE_URL = "https://ollama.internal.example/v1"
_CA_PATH = "/etc/ssl/mkcert-root.pem"


def _make_agent():
    agent = MagicMock()
    agent.provider = "openrouter"
    agent.model = "openrouter/auto"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "primary-key"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = [
        {"provider": "my-ollama", "model": "llama4", "base_url": _FB_BASE_URL},
    ]
    agent._primary_runtime = {
        "provider": "openrouter",
        "model": "openrouter/auto",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
        "api_key": "primary-key",
        "client_kwargs": {
            "api_key": "primary-key",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "anthropic_api_key": "",
        "anthropic_base_url": "",
    }
    agent._config_context_length = None
    agent._credential_pool = None
    agent._rate_limited_until = 0
    agent._transport_cache = {}
    agent._client_kwargs = {
        "api_key": "primary-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent._buffer_status = MagicMock()
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    agent._replace_primary_openai_client = MagicMock()
    agent.context_compressor = None
    return agent


def _activate(agent, config):
    fallback_client = SimpleNamespace(
        api_key="fb-key",
        base_url=_FB_BASE_URL,
        _custom_headers={},
    )
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "llama4"),
    ), patch(
        "agent.credential_pool.load_pool",
        return_value=None,
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value=config,
    ):
        assert try_activate_fallback(agent) is True


def test_fallback_applies_per_provider_ssl_ca_cert():
    agent = _make_agent()
    _activate(agent, {
        "custom_providers": [
            {"name": "my-ollama", "base_url": _FB_BASE_URL, "ssl_ca_cert": _CA_PATH},
        ],
    })
    assert agent.base_url == _FB_BASE_URL
    assert agent._client_kwargs.get("ssl_ca_cert") == _CA_PATH


def test_fallback_applies_per_provider_ssl_verify_false():
    agent = _make_agent()
    _activate(agent, {
        "custom_providers": [
            {"name": "my-ollama", "base_url": _FB_BASE_URL, "ssl_verify": False},
        ],
    })
    assert agent._client_kwargs.get("ssl_verify") is False


def test_fallback_without_tls_settings_leaves_kwargs_clean():
    agent = _make_agent()
    _activate(agent, {
        "custom_providers": [
            {"name": "my-ollama", "base_url": _FB_BASE_URL},
        ],
    })
    assert "ssl_ca_cert" not in agent._client_kwargs
    assert "ssl_verify" not in agent._client_kwargs
