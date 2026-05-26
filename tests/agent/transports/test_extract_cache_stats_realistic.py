"""Realistic-shape regression tests for ProviderTransport.extract_cache_stats().

The existing tests in test_transport.py and test_chat_completions.py use
hand-crafted minimal mocks. These tests use the exact shape that production
API responses carry, including the fields that misbehaving proxies and edge
cases produce.

Goal: lock down current behavior before the PR-1 refactor renames the method
to `extract_usage()` and changes the return type. Any test in this file that
fails after the refactor signals a behavior regression — the field reads must
produce the same numeric values regardless of method name / return type.
"""

import pytest
from types import SimpleNamespace

from agent.transports import get_transport


# ── Anthropic transport ─────────────────────────────────────────────────


@pytest.fixture
def anthropic_transport():
    import agent.transports.anthropic  # noqa: F401 — register on import
    return get_transport("anthropic_messages")


class TestAnthropicRealisticShapes:
    """Anthropic's `Usage` shape: `input_tokens`, `output_tokens`,
    `cache_read_input_tokens`, `cache_creation_input_tokens`."""

    def test_realistic_message_with_cache_hit(self, anthropic_transport):
        """Sustained-session shape: most input came from cache."""
        usage = SimpleNamespace(
            input_tokens=200,                  # net new tokens this turn
            output_tokens=150,
            cache_read_input_tokens=14_500,    # bulk of system + history
            cache_creation_input_tokens=0,
        )
        response = SimpleNamespace(usage=usage)
        result = anthropic_transport.extract_cache_stats(response)
        assert result == {"cached_tokens": 14_500, "creation_tokens": 0}

    def test_realistic_first_turn_creates_cache(self, anthropic_transport):
        """First turn of session: cache is written, no reads yet."""
        usage = SimpleNamespace(
            input_tokens=200,
            output_tokens=300,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=12_000,
        )
        response = SimpleNamespace(usage=usage)
        result = anthropic_transport.extract_cache_stats(response)
        assert result == {"cached_tokens": 0, "creation_tokens": 12_000}

    def test_no_cache_fields_present_returns_none(self, anthropic_transport):
        """Some proxies omit cache fields entirely. Must return None, not raise."""
        usage = SimpleNamespace(input_tokens=500, output_tokens=200)
        response = SimpleNamespace(usage=usage)
        assert anthropic_transport.extract_cache_stats(response) is None

    def test_explicit_none_cache_values_returns_none(self, anthropic_transport):
        """Field exists but is None (some proxies do this). Must coerce to 0
        and treat as no-cache."""
        usage = SimpleNamespace(
            input_tokens=500,
            output_tokens=200,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        )
        response = SimpleNamespace(usage=usage)
        assert anthropic_transport.extract_cache_stats(response) is None


# ── Chat-completions transport ──────────────────────────────────────────


@pytest.fixture
def chat_transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


class TestChatCompletionsRealisticShapes:
    """OpenAI-wire `Usage` shape: `prompt_tokens`, `completion_tokens`,
    `total_tokens`, with cache fields nested in `prompt_tokens_details`."""

    def test_openai_native_cache_hit(self, chat_transport):
        """OpenAI auto-cache: `prompt_tokens_details.cached_tokens` populated."""
        usage = SimpleNamespace(
            prompt_tokens=5_000,
            completion_tokens=200,
            total_tokens=5_200,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3_500),
        )
        response = SimpleNamespace(usage=usage)
        result = chat_transport.extract_cache_stats(response)
        # Note: the cache_write field is `cache_write_tokens` on
        # prompt_tokens_details — OpenAI doesn't populate it (auto-cache
        # has no write event), so it stays 0.
        assert result == {"cached_tokens": 3_500, "creation_tokens": 0}

    def test_openai_no_cache_returns_none(self, chat_transport):
        """Plain OpenAI call with no cache hit and a prompt_tokens_details
        object that has cached_tokens=0."""
        usage = SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=200,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        response = SimpleNamespace(usage=usage)
        assert chat_transport.extract_cache_stats(response) is None

    def test_response_with_no_prompt_tokens_details_returns_none(self, chat_transport):
        """Some proxies don't include prompt_tokens_details at all."""
        usage = SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=200,
            total_tokens=1_200,
        )
        response = SimpleNamespace(usage=usage)
        assert chat_transport.extract_cache_stats(response) is None

    def test_response_with_cache_write_populated(self, chat_transport):
        """Some proxies surface cache_write_tokens on details (mirroring
        Anthropic's cache_creation_input_tokens). Must read both."""
        usage = SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=200,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=600,
                cache_write_tokens=150,
            ),
        )
        response = SimpleNamespace(usage=usage)
        result = chat_transport.extract_cache_stats(response)
        assert result == {"cached_tokens": 600, "creation_tokens": 150}


# ── Codex Responses-API transport ───────────────────────────────────────
#
# The Codex transport currently has NO extract_cache_stats() override —
# it inherits the base class no-op. That's a real coverage gap: every
# Codex Responses session reports $0 cache hits to telemetry.
#
# This class documents the expected behavior. Today every test here will
# return None because the base class no-op fires. PR-1 adds the impl on
# the Codex transport, and these tests will then pass with real values.


@pytest.fixture
def codex_transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


class TestCodexResponsesRealisticShapes:
    """Codex Responses API shape: `input_tokens`, `output_tokens`,
    `input_tokens_details.cached_tokens` + `cache_creation_tokens`.

    These tests are marked xfail until PR-1 adds the Codex extractor.
    """

    @pytest.mark.xfail(
        reason="Codex transport has no extract_cache_stats override yet — PR-1 adds it",
        strict=True,
    )
    def test_codex_cache_hit_with_creation(self, codex_transport):
        usage = SimpleNamespace(
            input_tokens=2_500,                  # gross total per Responses API contract
            output_tokens=300,
            input_tokens_details=SimpleNamespace(
                cached_tokens=1_500,
                cache_creation_tokens=200,
            ),
        )
        response = SimpleNamespace(usage=usage)
        result = codex_transport.extract_cache_stats(response)
        assert result == {"cached_tokens": 1_500, "creation_tokens": 200}

    def test_codex_no_cache_details_returns_none(self, codex_transport):
        """Trivially passes today (base no-op) and after PR-1 (real impl
        returns None for absent details). No xfail needed."""
        usage = SimpleNamespace(input_tokens=1_000, output_tokens=200)
        response = SimpleNamespace(usage=usage)
        assert codex_transport.extract_cache_stats(response) is None

    def test_codex_today_inherits_base_noop(self, codex_transport):
        """Pin the current (broken) behavior: Codex transport returns None
        for cache stats even when the response has them. Documents the bug;
        remove this test once PR-1 lands."""
        usage = SimpleNamespace(
            input_tokens=2_500,
            output_tokens=300,
            input_tokens_details=SimpleNamespace(
                cached_tokens=1_500,
                cache_creation_tokens=200,
            ),
        )
        response = SimpleNamespace(usage=usage)
        # Base class no-op returns None regardless of input.
        assert codex_transport.extract_cache_stats(response) is None
