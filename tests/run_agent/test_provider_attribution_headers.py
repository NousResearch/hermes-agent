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
@patch.dict(
    "os.environ",
    {
        "HERMES_OPENROUTER_TITLE": "DownstreamApp",
        "HERMES_OPENROUTER_REFERER": "https://downstream.example/app",
    },
    clear=False,
)
def test_openrouter_attribution_headers_respect_env_vars(mock_openai):
    # _OR_HEADERS is module-scoped and captures env vars at import time,
    # so we reload the module under the patched environment to prove the
    # override plumbing works end-to-end.
    import importlib

    import agent.auxiliary_client as ac

    importlib.reload(ac)

    mock_openai.return_value = MagicMock()
    agent_inst = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent_inst._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent_inst._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://downstream.example/app"
    assert headers["X-Title"] == "DownstreamApp"

    # Restore the module to avoid leaking the patched _OR_HEADERS into
    # unrelated tests running in the same process.
    importlib.reload(ac)


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
