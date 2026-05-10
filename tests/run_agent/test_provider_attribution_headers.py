"""Attribution default_headers applied per provider via base-URL detection.

Mirrors the OpenRouter pattern for the Vercel AI Gateway so that
referrerUrl / appName / User-Agent flow into gateway analytics.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


@patch("run_agent.OpenAI")
def test_openrouter_base_url_applies_or_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-Title"] == "Hermes Agent"


@patch("run_agent.OpenAI")
def test_ai_gateway_base_url_applies_attribution_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://ai-gateway.vercel.sh/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-Title"] == "Hermes Agent"
    assert headers["User-Agent"].startswith("HermesAgent/")


@patch("run_agent.OpenAI")
def test_routermint_base_url_applies_user_agent_header(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.routermint.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://api.routermint.com/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"].startswith("HermesAgent/")


@patch("run_agent.OpenAI")
def test_gmi_base_url_picks_up_profile_user_agent(mock_openai):
    """GMI declares User-Agent on its ProviderProfile.default_headers.

    The ``_apply_client_headers_for_base_url`` else-branch looks up the
    provider profile and applies its default_headers, so no GMI-specific
    branch is needed in run_agent.
    """
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.gmi-serving.com/v1",
        model="test/model",
        provider="gmi",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://api.gmi-serving.com/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"].startswith("HermesAgent/")


@patch("run_agent.OpenAI")
def test_unknown_base_url_clears_default_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._client_kwargs["default_headers"] = {"X-Stale": "yes"}

    agent._apply_client_headers_for_base_url("https://api.example.com/v1")

    assert "default_headers" not in agent._client_kwargs


@patch("run_agent.OpenAI")
def test_openrouter_headers_include_response_cache_when_enabled(mock_openai):
    """When openrouter.response_cache is True, the cache header is injected."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={
        "openrouter": {"response_cache": True, "response_cache_ttl": 600},
    }):
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-OpenRouter-Cache"] == "true"
    assert headers["X-OpenRouter-Cache-TTL"] == "600"


@patch("run_agent.OpenAI")
def test_openrouter_headers_no_cache_when_disabled(mock_openai):
    """When openrouter.response_cache is False, no cache headers are sent."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={
        "openrouter": {"response_cache": False},
    }):
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert "X-OpenRouter-Cache" not in headers
    assert "X-OpenRouter-Cache-TTL" not in headers
