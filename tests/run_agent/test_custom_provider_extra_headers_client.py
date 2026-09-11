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


@patch("agent.process_bootstrap.OpenAI")
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






