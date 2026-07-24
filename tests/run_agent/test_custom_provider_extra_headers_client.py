"""Per-provider ``extra_headers`` applied to the OpenAI client (#3526 salvage).

Custom providers (``providers`` / ``custom_providers`` in config.yaml) can
declare an ``extra_headers`` dict that must land on the OpenAI client's
``default_headers`` at construction and survive header re-application on
credential swaps / rebuilds. Values may carry credentials — the plumbing must
never log them.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_PROXY_URL = "https://llm.internal.example.com/v1"
_PROXY_CONFIG = {
    "custom_providers": [
        {
            "name": "my-proxy",
            "base_url": _PROXY_URL,
            "api_key": "proxy-key",
            "extra_headers": {
                "CF-Access-Client-Id": "xxxx.access",
                "X-Client-Name": "hermes-agent",
            },
        }
    ]
}


@patch("run_agent.OpenAI")
def test_custom_provider_extra_headers_applied_at_construction(mock_openai):
    mock_openai.return_value = MagicMock()
    with patch("hermes_cli.config.load_config", return_value=_PROXY_CONFIG):
        agent = AIAgent(
            api_key="proxy-key",
            base_url=_PROXY_URL,
            model="my-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    headers = agent._client_kwargs["default_headers"]
    assert headers["CF-Access-Client-Id"] == "xxxx.access"
    assert headers["X-Client-Name"] == "hermes-agent"


@patch("run_agent.OpenAI")
def test_extra_headers_not_applied_for_other_base_url(mock_openai):
    mock_openai.return_value = MagicMock()
    with patch("hermes_cli.config.load_config", return_value=_PROXY_CONFIG):
        agent = AIAgent(
            api_key="other-key",
            base_url="http://localhost:8080/v1",
            model="my-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    headers = agent._client_kwargs.get("default_headers") or {}
    assert "CF-Access-Client-Id" not in headers
    assert "X-Client-Name" not in headers


@patch("run_agent.OpenAI")
def test_extra_headers_survive_header_reapplication(mock_openai):
    """_apply_client_headers_for_base_url (credential swaps, rebuilds) must
    re-apply per-provider extra_headers rather than dropping them."""
    mock_openai.return_value = MagicMock()
    with patch("hermes_cli.config.load_config", return_value=_PROXY_CONFIG):
        agent = AIAgent(
            api_key="proxy-key",
            base_url=_PROXY_URL,
            model="my-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._client_kwargs.pop("default_headers", None)
        agent._apply_client_headers_for_base_url(_PROXY_URL)

    headers = agent._client_kwargs["default_headers"]
    assert headers["CF-Access-Client-Id"] == "xxxx.access"


@patch("run_agent.OpenAI")
def test_extra_headers_merge_with_global_default_headers(mock_openai):
    """Per-provider extra_headers win over global model.default_headers on
    key collisions; non-colliding globals are preserved."""
    mock_openai.return_value = MagicMock()
    config = {
        "model": {"default_headers": {"User-Agent": "curl/8.7.1", "X-Global": "1"}},
        "custom_providers": [
            {
                "name": "my-proxy",
                "base_url": _PROXY_URL,
                "api_key": "proxy-key",
                "extra_headers": {"User-Agent": "hermes-proxy", "X-Local": "2"},
            }
        ],
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        agent = AIAgent(
            api_key="proxy-key",
            base_url=_PROXY_URL,
            model="my-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"] == "hermes-proxy"  # per-provider wins
    assert headers["X-Global"] == "1"
    assert headers["X-Local"] == "2"
