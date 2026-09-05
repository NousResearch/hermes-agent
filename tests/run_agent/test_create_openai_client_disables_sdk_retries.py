"""Regression guard: _create_openai_client must disable SDK-level retries.

#26293: the OpenAI SDK default ``max_retries=2`` uses its own 1-2s backoff that
ignores ``Retry-After`` and double-retries *inside* hermes's outer conversation
loop — burning request slots against a rate-limited bucket that won't refill for
minutes. The outer loop already owns rate-limit backoff (honors Retry-After,
adaptive + jittered), so every primary OpenAI/aggregator client must be built
with ``max_retries=0``. This is the OpenAI-path twin of the Anthropic adapter
fix in tests/agent/test_anthropic_adapter.py.
"""
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


@patch("run_agent.OpenAI")
def test_create_openai_client_disables_sdk_retries(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1"},
        reason="test",
        shared=False,
    )

    # Find the construction call that carries our base_url (AIAgent() also
    # builds a client during init; the assertion targets the explicit call).
    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == "https://openrouter.ai/api/v1"
    ]
    assert matching, "OpenAI was never constructed with the expected base_url"
    for call in matching:
        assert call.kwargs.get("max_retries") == 0, (
            "_create_openai_client must set max_retries=0 so the SDK does not "
            "double-retry inside the outer rate-limit loop (#26293); got "
            f"{call.kwargs.get('max_retries')!r}"
        )


@patch("run_agent.OpenAI")
def test_create_openai_client_honors_explicit_max_retries(mock_openai):
    """An explicit max_retries in client_kwargs is respected (setdefault, not
    clobber) — future callers can opt back into SDK retries if needed."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._create_openai_client(
        {
            "api_key": "test-key",
            "base_url": "https://explicit.example.com/v1",
            "max_retries": 5,
        },
        reason="test",
        shared=False,
    )

    matching = [
        c for c in mock_openai.call_args_list
        if c.kwargs.get("base_url") == "https://explicit.example.com/v1"
    ]
    assert matching, "OpenAI was never constructed with the explicit base_url"
    assert matching[-1].kwargs.get("max_retries") == 5
