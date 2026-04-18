"""Guardrail: _create_openai_client must not mutate its input kwargs.

#10933 injected an httpx.Client directly into the caller's ``client_kwargs``.
When the dict was ``self._client_kwargs``, the shared transport was torn down
after the first request_complete close and subsequent request-scoped clients
wrapped a closed transport, raising ``APIConnectionError('Connection error.')``
with cause ``RuntimeError: Cannot send a request, as the client has been closed``
on every retry. That PR has since been reverted, but the underlying issue
(#10324, connections hanging in CLOSE-WAIT) is still open, so another transport
tweak inside this function is likely. This test pins the contract that the
function must treat its input dict as read-only.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


@patch("run_agent.OpenAI")
def test_create_openai_client_does_not_mutate_input_kwargs(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    kwargs = {"api_key": "test-key", "base_url": "https://api.example.com/v1"}
    snapshot = dict(kwargs)

    agent._create_openai_client(kwargs, reason="test", shared=False)

    assert kwargs == snapshot, (
        f"_create_openai_client mutated input kwargs; expected {snapshot}, got {kwargs}"
    )


@patch("run_agent.OpenAI")
def test_create_openai_client_skips_custom_http_client_for_codex_backend(mock_openai):
    client = MagicMock()
    mock_openai.return_value = client
    agent = AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        model="gpt-5.4",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    kwargs = {"api_key": "test-key", "base_url": "https://chatgpt.com/backend-api/codex"}
    result = agent._create_openai_client(kwargs, reason="test", shared=False)

    assert result is client
    assert "http_client" not in mock_openai.call_args.kwargs


@patch("run_agent.OpenAI")
def test_create_openai_client_keeps_custom_http_client_for_non_chatgpt_codex_urls(mock_openai):
    client = MagicMock()
    mock_openai.return_value = client
    agent = AIAgent(
        api_key="test-key",
        base_url="https://proxy.example.com/v1",
        provider="openai-codex",
        model="gpt-5.4",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    kwargs = {"api_key": "test-key", "base_url": "https://proxy.example.com/v1"}
    result = agent._create_openai_client(kwargs, reason="test", shared=False)

    assert result is client
    assert "http_client" in mock_openai.call_args.kwargs
