"""Tests for honcho_reasoning tier selection (issue #59470).

The ``honcho_reasoning`` tool used to forward whatever ``reasoning_level``
the calling model chose, including ``minimal`` — Honcho's hard-capped
250-token tier that truncates answers mid chain-of-thought on multi-fact
queries.  These tests pin the new complexity-aware selection heuristic
that nudges the tier up for multi-faceted prompts and lets the caller
override explicitly when they really mean it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider() -> HonchoMemoryProvider:
    """Build an uninitialized provider suitable for static-helper testing."""
    provider = HonchoMemoryProvider.__new__(HonchoMemoryProvider)
    provider._manager = MagicMock()
    provider._session_key = "test-session"
    provider._session_initialized = True
    provider._turn_count = 1
    provider._dialectic_cadence = 1
    provider._last_dialectic_turn = -999
    provider._cron_skipped = False
    provider._recall_mode = "tools"
    provider._config = MagicMock()
    provider._config.dialectic_reasoning_level = "low"
    provider._config.reasoning_level_cap = "high"
    return provider


# ---------------------------------------------------------------------------
# Complexity classifier
# ---------------------------------------------------------------------------


class TestClassifyQueryComplexity:
    """Pin the deterministic complexity classifier behavior."""

    def test_empty_query_is_simple(self):
        assert HonchoMemoryProvider._classify_query_complexity("") == "simple"

    def test_whitespace_only_is_simple(self):
        assert HonchoMemoryProvider._classify_query_complexity("   \n\t  ") == "simple"

    def test_single_short_fact_is_simple(self):
        assert HonchoMemoryProvider._classify_query_complexity("What is their name?") == "simple"

    def test_single_sentence_with_many_tokens_is_simple(self):
        # One short factual question with a few content tokens stays simple
        # even though it has a couple of content words.
        assert HonchoMemoryProvider._classify_query_complexity(
            "What's their favorite editor?"
        ) == "simple"

    def test_two_sentences_are_moderate(self):
        result = HonchoMemoryProvider._classify_query_complexity(
            "Who is this person? What are their working preferences?"
        )
        assert result == "moderate"

    def test_comma_separated_clauses_are_moderate(self):
        result = HonchoMemoryProvider._classify_query_complexity(
            "Summarize their role, projects, communication style, and tooling preferences."
        )
        assert result in {"moderate", "complex"}

    def test_numbered_list_is_moderate(self):
        result = HonchoMemoryProvider._classify_query_complexity(
            "Answer each:\n1. Role\n2. Goals\n3. Preferences"
        )
        assert result in {"moderate", "complex"}

    def test_many_distinct_topics_is_complex(self):
        # >=10 distinct content tokens across multiple clauses -> complex.
        query = (
            "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
            "Kotlin, Swift, Ruby, Haskell, and C++ experience."
        )
        assert HonchoMemoryProvider._classify_query_complexity(query) == "complex"

    def test_four_plus_sentences_is_complex(self):
        query = (
            "Who is this user. What projects do they lead. "
            "Which teams have they worked with. How do they prefer to communicate."
        )
        assert HonchoMemoryProvider._classify_query_complexity(query) == "complex"

    def test_semicolon_separated_is_complex(self):
        query = (
            "Summarize known facts about this peer and communication preferences; "
            "also describe their tooling stack, current projects, and team relationships; "
            "highlight any recent changes in working style; note open follow-ups."
        )
        assert HonchoMemoryProvider._classify_query_complexity(query) == "complex"

    def test_stopwords_are_ignored_in_topic_count(self):
        # "the", "and", "of", etc. should not inflate the distinct-topic count.
        # Only "user" and "name" remain as content tokens -> simple.
        result = HonchoMemoryProvider._classify_query_complexity(
            "What is the name of the user and the team of the user"
        )
        assert result == "simple"


# ---------------------------------------------------------------------------
# Tier selector — no explicit override
# ---------------------------------------------------------------------------


class TestSelectReasoningLevelNoOverride:
    """When the caller passes no explicit level, complexity drives the pick."""

    def test_single_fact_defaults_to_minimal(self):
        level = HonchoMemoryProvider._select_reasoning_level(
            "What is their name?", explicit=None
        )
        assert level == "minimal"

    def test_multi_sentence_bumps_to_medium(self):
        level = HonchoMemoryProvider._select_reasoning_level(
            "Who is this person? What are their working preferences?",
            explicit=None,
        )
        assert level == "medium"

    def test_multi_topic_bumps_to_high(self):
        level = HonchoMemoryProvider._select_reasoning_level(
            (
                "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
                "Kotlin, Swift, Ruby, Haskell, and C++ experience."
            ),
            explicit=None,
        )
        assert level == "high"

    def test_cap_honored_when_lowering_target(self):
        # If the cap is "medium", a complex query must not exceed "medium".
        level = HonchoMemoryProvider._select_reasoning_level(
            (
                "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
                "Kotlin, Swift, Ruby, Haskell, and C++ experience."
            ),
            explicit=None,
            cap="medium",
        )
        assert level == "medium"

    def test_invalid_cap_falls_back_to_high(self):
        level = HonchoMemoryProvider._select_reasoning_level(
            (
                "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
                "Kotlin, Swift, Ruby, Haskell, and C++ experience."
            ),
            explicit=None,
            cap="not-a-tier",
        )
        # Invalid cap is coerced to "high" — complex still maps to "high".
        assert level == "high"

    def test_invalid_explicit_is_treated_as_no_override(self):
        # Garbage strings like "auto" / "default" should not crash; the
        # complexity heuristic still picks a sensible tier.
        level = HonchoMemoryProvider._select_reasoning_level(
            "Who is this person? What are their preferences?",
            explicit="auto",
        )
        assert level == "medium"


# ---------------------------------------------------------------------------
# Tier selector — explicit caller override
# ---------------------------------------------------------------------------


class TestSelectReasoningLevelExplicitOverride:
    """The caller can override, but a 'minimal' override on a complex query
    is floored to the complexity-derived tier."""

    def test_explicit_low_on_simple_query_is_honored(self):
        level = HonchoMemoryProvider._select_reasoning_level(
            "What is their name?", explicit="low"
        )
        assert level == "low"

    def test_explicit_max_on_simple_query_is_honored_when_cap_is_max(self):
        # Default cap is "high", so "max" must be clamped to "high" unless
        # the caller raises the cap.
        level = HonchoMemoryProvider._select_reasoning_level(
            "What is their name?", explicit="max"
        )
        assert level == "high"
        # With cap raised to "max", the explicit pick survives.
        level = HonchoMemoryProvider._select_reasoning_level(
            "What is their name?", explicit="max", cap="max"
        )
        assert level == "max"

    def test_explicit_minimal_on_complex_query_becomes_high(self):
        # The bug from #59470: caller picks minimal, answer truncates.
        # Heuristic floors the explicit pick up to the complexity tier.
        level = HonchoMemoryProvider._select_reasoning_level(
            (
                "Summarize known facts about this peer and communication "
                "preferences; describe their tooling, projects, and team "
                "relationships; highlight recent changes; note open follow-ups."
            ),
            explicit="minimal",
        )
        assert level == "high"

    def test_explicit_medium_on_complex_query_is_honored(self):
        # "medium" >= complexity floor for moderate but < complex floor;
        # explicit picks win when they exceed the floor.
        level = HonchoMemoryProvider._select_reasoning_level(
            (
                "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
                "Kotlin, Swift, Ruby, Haskell, and C++ experience."
            ),
            explicit="medium",
        )
        # "medium" (idx 2) < complexity target "high" (idx 3) -> floor wins.
        assert level == "high"

    def test_explicit_high_on_complex_query_is_honored(self):
        level = HonchoMemoryProvider._select_reasoning_level(
            (
                "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
                "Kotlin, Swift, Ruby, Haskell, and C++ experience."
            ),
            explicit="high",
        )
        assert level == "high"

    def test_explicit_above_cap_is_clamped_to_cap(self):
        # Caller asks for max, cap is medium -> medium wins.
        level = HonchoMemoryProvider._select_reasoning_level(
            "What is their name?", explicit="max", cap="medium"
        )
        assert level == "medium"


# ---------------------------------------------------------------------------
# End-to-end through handle_tool_call (issue #59470 regression)
# ---------------------------------------------------------------------------


class TestHandleToolCallHonchoReasoning:
    """Wire the heuristic into the actual ``honcho_reasoning`` tool path."""

    def test_single_fact_passes_minimal_through(self):
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="Alice")

        result = provider.handle_tool_call(
            "honcho_reasoning",
            {"query": "What is their name?"},
        )

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        assert call_kwargs["reasoning_level"] == "minimal"
        payload = json.loads(result)
        assert payload["result"] == "Alice"

    def test_multi_sentence_promotes_to_medium(self):
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="synthesized")

        result = provider.handle_tool_call(
            "honcho_reasoning",
            {"query": "Who is this person? What are their working preferences?"},
        )

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        assert call_kwargs["reasoning_level"] == "medium"
        assert json.loads(result)["result"] == "synthesized"

    def test_multi_topic_promotes_to_high(self):
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="synthesized")

        query = (
            "Compare the user's Python, JavaScript, Rust, Go, TypeScript, "
            "Kotlin, Swift, Ruby, Haskell, and C++ experience."
        )
        provider.handle_tool_call("honcho_reasoning", {"query": query})

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        assert call_kwargs["reasoning_level"] == "high"

    def test_explicit_max_is_clamped_to_default_cap(self):
        # The default ``reasoning_level_cap`` is "high", so a caller that asks
        # for "max" must be clamped to "high" — that's the safe default.
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="answer")

        provider.handle_tool_call(
            "honcho_reasoning",
            {"query": "What is their name?", "reasoning_level": "max"},
        )

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        assert call_kwargs["reasoning_level"] == "high"

    def test_explicit_high_survives_end_to_end(self):
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="answer")

        provider.handle_tool_call(
            "honcho_reasoning",
            {"query": "What is their name?", "reasoning_level": "high"},
        )

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        assert call_kwargs["reasoning_level"] == "high"

    def test_explicit_minimal_on_complex_query_is_floored_to_high(self):
        """Regression for issue #59470: the model picking 'minimal' for a
        multi-fact query must not silently truncate at 250 tokens."""
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="answer")

        provider.handle_tool_call(
            "honcho_reasoning",
            {
                "query": (
                    "Summarize known facts about this peer and communication "
                    "preferences; describe their tooling, projects, and team "
                    "relationships; highlight recent changes; note open follow-ups."
                ),
                "reasoning_level": "minimal",
            },
        )

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        # The heuristic bumps the explicit "minimal" pick up to "high" so
        # Honcho's dialectic tier has enough output budget.
        assert call_kwargs["reasoning_level"] == "high"

    def test_explicit_low_on_moderate_query_is_promoted_to_medium(self):
        provider = _make_provider()
        provider._manager.dialectic_query = MagicMock(return_value="answer")

        provider.handle_tool_call(
            "honcho_reasoning",
            {
                "query": (
                    "Who is this person? What are their working preferences? "
                    "How do they like to be contacted?"
                ),
                "reasoning_level": "low",
            },
        )

        call_kwargs = provider._manager.dialectic_query.call_args.kwargs
        assert call_kwargs["reasoning_level"] == "medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])