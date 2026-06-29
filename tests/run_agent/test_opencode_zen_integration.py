"""Tests for OpenCode Zen/Go integration fixes.

Two bugs fixed:
1. WAF User-Agent: OpenCode Zen/Go return a bare 401 (empty body) for requests
   without a recognised User-Agent.  Hermes must inject one for opencode.ai.

2. Gemini native routing: Zen proxies Gemini models via Google's native
   generateContent API, not OpenAI's /chat/completions.  Hermes must use
   GeminiNativeClient for gemini-* models on opencode.ai.
"""
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Bug 1: User-Agent header injected for opencode.ai
# ---------------------------------------------------------------------------

@patch("run_agent.OpenAI")
def test_user_agent_injected_for_zen(mock_openai):
    """Requests to opencode.ai/zen must carry a User-Agent header."""
    mock_openai.return_value = MagicMock()
    AIAgent(
        api_key="sk-test",
        base_url="https://opencode.ai/zen/v1",
        model="kimi-k2.5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    _, kwargs = mock_openai.call_args
    headers = kwargs.get("default_headers", {})
    assert "User-Agent" in headers, "User-Agent header missing for opencode.ai/zen"
    assert headers["User-Agent"].startswith("hermes-agent"), (
        f"Unexpected User-Agent: {headers['User-Agent']!r}"
    )


@patch("run_agent.OpenAI")
def test_user_agent_injected_for_go(mock_openai):
    """Same User-Agent requirement for the opencode.ai/zen/go endpoint."""
    mock_openai.return_value = MagicMock()
    AIAgent(
        api_key="sk-test",
        base_url="https://opencode.ai/zen/go/v1",
        model="glm-5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    _, kwargs = mock_openai.call_args
    headers = kwargs.get("default_headers", {})
    assert "User-Agent" in headers, "User-Agent header missing for opencode.ai/go"
    assert headers["User-Agent"].startswith("hermes-agent")


@patch("run_agent.OpenAI")
def test_user_agent_not_injected_for_other_hosts(mock_openai):
    """The opencode-specific User-Agent must not leak to other hosts."""
    mock_openai.return_value = MagicMock()
    AIAgent(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="some-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    _, kwargs = mock_openai.call_args
    headers = kwargs.get("default_headers", {})
    ua = headers.get("User-Agent", "")
    assert not ua.startswith("hermes-agent"), (
        "hermes-agent User-Agent should not be set for non-opencode hosts"
    )


# ---------------------------------------------------------------------------
# Bug 2: GeminiNativeClient used for gemini-* models on opencode.ai
# ---------------------------------------------------------------------------

def test_gemini_model_uses_native_client():
    """gemini-* models on opencode.ai must use GeminiNativeClient."""
    from agent.gemini_native_adapter import GeminiNativeClient

    # Do NOT patch run_agent.OpenAI — we need the real client creation path
    # to return a GeminiNativeClient, not a MagicMock.
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://opencode.ai/zen/v1",
        model="gemini-3-flash",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert isinstance(agent.client, GeminiNativeClient), (
        f"Expected GeminiNativeClient for gemini-3-flash on opencode.ai, "
        f"got {type(agent.client).__name__}"
    )


@patch("run_agent.OpenAI")
def test_non_gemini_model_uses_openai_client(mock_openai):
    """Non-Gemini models on opencode.ai must NOT use GeminiNativeClient."""
    from agent.gemini_native_adapter import GeminiNativeClient

    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://opencode.ai/zen/v1",
        model="kimi-k2.5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert not isinstance(agent.client, GeminiNativeClient), (
        "kimi-k2.5 on opencode.ai should not use GeminiNativeClient"
    )


@patch("run_agent.OpenAI")
def test_gemini_model_on_other_host_uses_openai_client(mock_openai):
    """gemini-* models on non-opencode hosts must not trigger GeminiNativeClient."""
    from agent.gemini_native_adapter import GeminiNativeClient

    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gemini-3-flash",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert not isinstance(agent.client, GeminiNativeClient), (
        "gemini-3-flash on a non-opencode host should not use GeminiNativeClient"
    )
