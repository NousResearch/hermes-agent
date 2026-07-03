"""Regression: the native-Ollama branch in ``create_openai_client`` must match the
provider *type*, so a named custom instance (``custom:<name>``) is still routed to
the native ``/api/chat`` client instead of silently falling back to ``/v1``.

Multiple Ollama providers are named ``custom`` (bare) and ``custom:<name>`` (see
``hermes_cli/providers.py``). An exact ``== "custom"`` gate would route only the
bare instance native and drop every ``custom:<name>`` instance to ``/v1`` — losing
per-request ``num_ctx`` and correct streaming ``tool_calls`` for delegated
sub-agents. The routing itself stays gated on the per-endpoint Ollama probe, so a
non-Ollama ``custom:<name>`` (vLLM / llama.cpp / LM Studio) is never mis-routed.
"""

from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import build_api_kwargs
from agent.ollama_native_adapter import OllamaNativeClient
from run_agent import AIAgent

_BASE_URL = "http://localhost:11434/v1"


def _make_agent(provider: str) -> AIAgent:
    agent = AIAgent(
        api_key="ollama",
        base_url=_BASE_URL,
        model="gemma4:e2b",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.provider = provider
    return agent


@patch("run_agent.OpenAI")
def test_bare_custom_routes_native(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = _make_agent("custom")
    with patch("agent.ollama_native_adapter.is_native_ollama_base_url", return_value=True):
        client = agent._create_openai_client(
            {"api_key": "ollama", "base_url": _BASE_URL}, reason="test", shared=False
        )
    assert isinstance(client, OllamaNativeClient)


@patch("run_agent.OpenAI")
def test_suffixed_custom_instance_still_routes_native(mock_openai):
    """Client selection: 'custom:ollama-2' must get the native client (previously it
    fell through to /v1). NOTE: this only asserts the client TYPE — that num_ctx
    actually reaches the payload is covered separately by
    test_suffixed_custom_instance_injects_num_ctx."""
    mock_openai.return_value = MagicMock()
    agent = _make_agent("custom:ollama-2")
    with patch("agent.ollama_native_adapter.is_native_ollama_base_url", return_value=True):
        client = agent._create_openai_client(
            {"api_key": "ollama", "base_url": _BASE_URL}, reason="test", shared=False
        )
    assert isinstance(client, OllamaNativeClient)


@patch("run_agent.OpenAI")
def test_suffixed_custom_instance_injects_num_ctx(mock_openai):
    """Routing the native client is not enough — num_ctx must reach extra_body, which
    only happens on the provider-profile path. A named custom instance must resolve to
    the 'custom' profile so ollama_num_ctx is injected; otherwise the model loads at
    Ollama's 4096 default despite using the native client."""
    mock_openai.return_value = MagicMock()
    agent = _make_agent("custom:ollama-2")
    agent._ollama_num_ctx = 64000
    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])
    assert kwargs["extra_body"]["options"]["num_ctx"] == 64000


@patch("run_agent.OpenAI")
def test_suffixed_non_ollama_custom_stays_on_v1(mock_openai):
    """A non-Ollama custom instance (probe says no) is never mis-routed to native."""
    sentinel = MagicMock()
    mock_openai.return_value = sentinel
    agent = _make_agent("custom:vllm")
    with patch("agent.ollama_native_adapter.is_native_ollama_base_url", return_value=False):
        client = agent._create_openai_client(
            {"api_key": "x", "base_url": "http://localhost:8000/v1"}, reason="test", shared=False
        )
    assert not isinstance(client, OllamaNativeClient)
    assert client is sentinel  # fell through to the stock OpenAI client
