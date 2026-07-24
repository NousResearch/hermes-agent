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
@patch("agent.gemini_native_adapter.GeminiNativeClient")
def test_create_openai_client_routes_empty_gemini_base_url_to_native_client(mock_gemini, mock_openai):
    from agent.gemini_native_adapter import DEFAULT_GEMINI_BASE_URL

    native_client = MagicMock()
    mock_gemini.return_value = native_client

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "gemini"
    agent.model = "gemini-2.5-flash"
    agent.base_url = ""
    agent._client_kwargs = {}
    agent._build_keepalive_http_client = lambda base_url="", **_kwargs: None

    kwargs = {"api_key": "AIza-test-key", "base_url": ""}
    snapshot = dict(kwargs)

    client = agent._create_openai_client(kwargs, reason="test", shared=False)

    assert client is native_client
    mock_openai.assert_not_called()
    mock_gemini.assert_called_once_with(
        api_key="AIza-test-key",
        base_url=DEFAULT_GEMINI_BASE_URL,
    )
    assert kwargs == snapshot
