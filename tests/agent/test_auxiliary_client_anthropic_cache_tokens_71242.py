"""#71242 dual-shape Anthropic usage shim for auxiliary accounting.

The adapter must expose BOTH:
1. Native Anthropic fields (input_tokens / output_tokens / cache_*).
2. Inclusive OpenAI-compatible prompt_tokens + prompt_tokens_details.

Canonical normalize_usage() is asserted for both provider paths.
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
                           cache_read=0, cache_creation=0):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
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


def test_adapter_exposes_native_and_inclusive_openai_shape():
    result = _adapt(_make_native_response(
        input_tokens=100, output_tokens=50, cache_read=2048, cache_creation=512,
    ))
    u = result.usage
    assert u is not None
    # Native Anthropic
    assert u.input_tokens == 100
    assert u.output_tokens == 50
    assert u.cache_read_input_tokens == 2048
    assert u.cache_creation_input_tokens == 512
    # Inclusive OpenAI shape (fresh + cache buckets)
    assert u.prompt_tokens == 100 + 2048 + 512
    assert u.completion_tokens == 50
    assert u.prompt_tokens_details.cached_tokens == 2048
    assert u.prompt_tokens_details.cache_write_tokens == 512


def test_normalize_usage_anthropic_path_matches_native():
    from agent.usage_pricing import normalize_usage

    result = _adapt(_make_native_response(
        input_tokens=100, output_tokens=50, cache_read=2048, cache_creation=512,
    ))
    canon = normalize_usage(result.usage, provider="anthropic")
    assert canon.input_tokens == 100
    assert canon.output_tokens == 50
    assert canon.cache_read_tokens == 2048
    assert canon.cache_write_tokens == 512


def test_normalize_usage_openai_path_does_not_zero_fresh():
    """OpenAI normalizer subtracts cache from prompt_tokens — inclusive input prevents clamp-to-zero."""
    from agent.usage_pricing import normalize_usage

    result = _adapt(_make_native_response(
        input_tokens=100, output_tokens=50, cache_read=2048, cache_creation=512,
    ))
    # Without provider=anthropic → OpenAI-compatible branch
    canon = normalize_usage(result.usage, provider="openrouter")
    assert canon.cache_read_tokens == 2048
    assert canon.cache_write_tokens == 512
    # Fresh input preserved (not clamped to 0)
    assert canon.input_tokens == 100
    assert canon.output_tokens == 50


def test_cache_write_only_case():
    from agent.usage_pricing import normalize_usage

    result = _adapt(_make_native_response(
        input_tokens=80, output_tokens=10, cache_read=0, cache_creation=400,
    ))
    anth = normalize_usage(result.usage, provider="anthropic")
    oai = normalize_usage(result.usage, provider="openrouter")
    assert anth.cache_write_tokens == 400 and anth.cache_read_tokens == 0
    assert anth.input_tokens == 80
    assert oai.cache_write_tokens == 400 and oai.input_tokens == 80


def test_absent_cache_fields_default_zero():
    native = SimpleNamespace(
        content=[MagicMock()],
        usage=SimpleNamespace(input_tokens=5, output_tokens=7),
        stop_reason="end_turn",
        model="claude-test",
    )
    result = _adapt(native)
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 7
    assert result.usage.prompt_tokens == 5
    assert result.usage.cache_read_input_tokens == 0
    assert result.usage.cache_creation_input_tokens == 0
    assert result.usage.prompt_tokens_details.cached_tokens == 0
