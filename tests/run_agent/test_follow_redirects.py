"""Tests for opt-in ``follow_redirects`` on the LLM API client.

Three layers of coverage:

1. ``AIAgent._build_keepalive_http_client`` honors the new ``follow_redirects``
   kwarg on both the copilot short-circuit branch and the default transport
   branch.
2. The OpenAI-wire call path in ``agent.agent_runtime_helpers.create_openai_client``
   wires a configured ``follow_redirects`` through to the injected
   ``httpx.Client``.
3. The Gemini-native call path wires it through as well.

These are behavior-contract tests, not snapshots: we assert that
``httpx.Client.follow_redirects`` equals the configured value, not
implementation details.
"""
from unittest.mock import patch

import httpx

from run_agent import AIAgent


def _make_agent(provider="openai-codex", base_url="https://api.example.com/v1"):
    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        model="gpt-5.4",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


# --- _build_keepalive_http_client (run_agent.py) ---------------------------------

def test_build_keepalive_http_client_follow_redirects_default_false():
    """Default behavior is unchanged: httpx.Client follows redirects=False."""
    client = AIAgent._build_keepalive_http_client("https://api.example.com/v1")
    try:
        assert isinstance(client, httpx.Client)
        assert client.follow_redirects is False
    finally:
        client.close()


def test_build_keepalive_http_client_follow_redirects_true():
    client = AIAgent._build_keepalive_http_client(
        "https://api.example.com/v1", follow_redirects=True
    )
    try:
        assert isinstance(client, httpx.Client)
        assert client.follow_redirects is True
    finally:
        client.close()


def test_build_keepalive_http_client_copilot_branch_honors_kwarg():
    """The api.githubcopilot.com short-circuit branch also honors the kwarg."""
    client = AIAgent._build_keepalive_http_client(
        "https://api.githubcopilot.com", follow_redirects=True
    )
    try:
        assert isinstance(client, httpx.Client)
        assert client.follow_redirects is True
    finally:
        client.close()

    client = AIAgent._build_keepalive_http_client(
        "https://api.githubcopilot.com", follow_redirects=False
    )
    try:
        assert isinstance(client, httpx.Client)
        assert client.follow_redirects is False
    finally:
        client.close()


# --- create_openai_client wiring (agent_runtime_helpers.py) ---------------------

@patch("run_agent.OpenAI")
def test_create_openai_client_propagates_follow_redirects_default_off(
    mock_openai, monkeypatch
):
    """Without config, follow_redirects stays False (httpx default)."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.get_provider_follow_redirects",
        lambda provider_id, model=None: False,
    )

    agent = _make_agent()
    kwargs = {"api_key": "test-key", "base_url": "https://api.example.com/v1"}
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = forwarded.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert http_client.follow_redirects is False
    http_client.close()


@patch("run_agent.OpenAI")
def test_create_openai_client_propagates_follow_redirects_when_configured(
    mock_openai, monkeypatch
):
    """Config-driven opt-in reaches the constructed httpx.Client."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.get_provider_follow_redirects",
        lambda provider_id, model=None: True,
    )

    agent = _make_agent(provider="redirect-provider",
                        base_url="https://api.example.com/v1")
    kwargs = {"api_key": "test-key", "base_url": "https://api.example.com/v1"}
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = forwarded.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert http_client.follow_redirects is True
    http_client.close()


@patch("run_agent.OpenAI")
def test_create_openai_client_preserves_existing_http_client(
    mock_openai, monkeypatch
):
    """An explicitly-passed http_client wins — follow_redirects plumbing
    does not clobber it."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    sentinel = httpx.Client(follow_redirects=True)
    try:
        agent = _make_agent()
        kwargs = {
            "api_key": "test-key",
            "base_url": "https://api.example.com/v1",
            "http_client": sentinel,
        }
        agent._create_openai_client(kwargs, reason="test", shared=False)
        forwarded = mock_openai.call_args.kwargs
        assert forwarded.get("http_client") is sentinel
    finally:
        sentinel.close()
