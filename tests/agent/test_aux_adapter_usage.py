"""Auxiliary wire adapters keep every usage bucket the provider reported.

The Anthropic and Codex aux adapters rebuild ``response.usage`` in chat.completions shape, but
consumers normalize it with the shape their provider/api_mode names (aux accounting passes
``provider="anthropic"``, MoA and the aux hooks pass ``api_mode``). Whatever shape a consumer
picks, the adapter's usage must normalize to the same buckets as the provider's raw usage.
"""

from __future__ import annotations

from dataclasses import astuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.usage_pricing import normalize_usage


def _buckets(usage, **shape):
    cu = normalize_usage(usage, **shape)
    return astuple(cu)[:4]  # input, output, cache_read, cache_write


def test_anthropic_aux_usage_normalizes_like_the_raw_sdk_usage():
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    raw = SimpleNamespace(input_tokens=1200, output_tokens=300,
                          cache_read_input_tokens=40000, cache_creation_input_tokens=2000)
    adapter = _AnthropicCompletionsAdapter(MagicMock(name="anthropic_client"), "claude-haiku-4-5", is_oauth=False)
    normalized = SimpleNamespace(content="ok", tool_calls=None, reasoning=None, finish_reason="stop")
    with patch("agent.anthropic_adapter.create_anthropic_message", return_value=SimpleNamespace(usage=raw)), \
            patch("agent.transports.get_transport") as get_transport:
        get_transport.return_value.normalize_response.return_value = normalized
        usage = adapter.create(model="claude-haiku-4-5", messages=[{"role": "user", "content": "hi"}],
                               max_tokens=64).usage

    expected = _buckets(raw, provider="anthropic")
    assert all(expected)
    assert _buckets(usage, provider="anthropic") == expected
    assert _buckets(usage, api_mode="anthropic_messages") == expected
    assert _buckets(usage, provider="minimax") == expected  # chat shape (Anthropic-wire provider, no api_mode)


@pytest.mark.parametrize("write_field", ["cache_write_tokens", "cache_creation_tokens"])  # current + legacy
def test_codex_aux_usage_normalizes_like_the_raw_responses_usage(write_field):
    from agent.auxiliary_client import _parse_codex_final_response

    raw = SimpleNamespace(input_tokens=50000, output_tokens=700, total_tokens=50700,
                          input_tokens_details=SimpleNamespace(cached_tokens=45000, **{write_field: 1000}))
    _text, _calls, usage = _parse_codex_final_response(SimpleNamespace(output=[], usage=raw))

    expected = _buckets(raw, api_mode="codex_responses")
    assert all(expected)
    assert _buckets(usage, api_mode="codex_responses") == expected
    assert _buckets(usage, provider="openai-codex") == expected  # chat shape (aux accounting)
