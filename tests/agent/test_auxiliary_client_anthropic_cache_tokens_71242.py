"""Anthropic auxiliary usage carries native and OpenAI-shaped token buckets.

normalize_usage() must recover the same canonical input/output/cache buckets
from the adapted response on both the Anthropic and OpenAI-compatible paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_native_response(*, input_tokens=10, output_tokens=20,
                           cache_read=0, cache_creation=0, include_cache_fields=True):
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    if include_cache_fields:
        usage.cache_read_input_tokens = cache_read
        usage.cache_creation_input_tokens = cache_creation
    return SimpleNamespace(
        content=[MagicMock()],
        usage=usage,
        stop_reason="end_turn",
        model="claude-test",
    )


def _patch_transport():
    fake_nr = SimpleNamespace(
        content="hello",
        tool_calls=None,
        reasoning=None,
        finish_reason="stop",
    )
    fake_transport = MagicMock(name="anthropic_transport")
    fake_transport.normalize_response.return_value = fake_nr
    return patch(
        "agent.transports.get_transport",
        return_value=fake_transport,
    )


def _patch_create_anthropic_message(response):
    return patch(
        "agent.anthropic_adapter.create_anthropic_message",
        return_value=response,
    )


def _adapt(native):
    from agent.auxiliary_client import _AnthropicCompletionsAdapter

    adapter = _AnthropicCompletionsAdapter(
        real_client=MagicMock(), model="claude-test", is_oauth=False,
    )
    with _patch_create_anthropic_message(native), _patch_transport():
        return adapter.create(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    ("cache_read", "cache_creation", "include_cache_fields"),
    [(2048, 512, True), (0, 400, True), (0, 0, False)],
)
def test_normalize_usage_anthropic_path_preserves_cache_buckets(
    cache_read, cache_creation, include_cache_fields
):
    """Native Anthropic accounting keeps fresh, read, and write buckets separate."""
    from agent.usage_pricing import normalize_usage

    result = _adapt(_make_native_response(
        input_tokens=100, output_tokens=50,
        cache_read=cache_read, cache_creation=cache_creation,
        include_cache_fields=include_cache_fields,
    ))
    canon = normalize_usage(
        result.usage, provider="anthropic", api_mode="anthropic_messages"
    )
    assert (canon.input_tokens, canon.output_tokens) == (100, 50)
    assert (canon.cache_read_tokens, canon.cache_write_tokens) == (cache_read, cache_creation)


@pytest.mark.parametrize(
    ("cache_read", "cache_creation", "include_cache_fields"),
    [(2048, 512, True), (0, 400, True), (0, 0, False)],
)
def test_normalize_usage_openai_path_preserves_fresh_input(
    cache_read, cache_creation, include_cache_fields
):
    """Inclusive prompt totals prevent the OpenAI branch from clamping fresh input to zero."""
    from agent.usage_pricing import normalize_usage

    result = _adapt(_make_native_response(
        input_tokens=100, output_tokens=50,
        cache_read=cache_read, cache_creation=cache_creation,
        include_cache_fields=include_cache_fields,
    ))
    canon = normalize_usage(result.usage, provider="openrouter")
    assert (canon.input_tokens, canon.output_tokens) == (100, 50)
    assert (canon.cache_read_tokens, canon.cache_write_tokens) == (cache_read, cache_creation)
