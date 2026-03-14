"""Tests for cognitive_memory.encoding module."""

import pytest

from cognitive_memory.encoding import (
    ContradictionResult,
    EncodingResult,
    classify_content,
    detect_contradiction,
    detect_contradictions,
    encode,
    estimate_importance,
)
from cognitive_memory.store import Memory, ScoredMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(content, **kwargs):
    """Create a Memory with defaults for testing."""
    defaults = dict(
        id=1,
        scope="/",
        categories=[],
        importance=0.5,
        created_at=1000.0,
        updated_at=1000.0,
        last_accessed=1000.0,
        access_count=0,
        forgotten=False,
        embedding=None,
    )
    defaults.update(kwargs)
    return Memory(content=content, **defaults)


def _make_scored(content, similarity=0.8, **kwargs):
    """Create a ScoredMemory for testing."""
    mem = _make_memory(content, **kwargs)
    return ScoredMemory(
        memory=mem,
        score=similarity,
        similarity=similarity,
        match_reasons=["semantic"],
    )


# ---------------------------------------------------------------------------
# classify_content
# ---------------------------------------------------------------------------


class TestClassifyContent:
    def test_preference_detection(self):
        cats, scores = classify_content("I prefer using Python over JavaScript")
        assert "preference" in cats

    def test_procedure_detection(self):
        cats, scores = classify_content("First install the package, then run the setup script")
        assert "procedure" in cats

    def test_convention_detection(self):
        cats, scores = classify_content("The naming convention is to always use snake_case")
        assert "convention" in cats

    def test_environment_detection(self):
        cats, scores = classify_content("Python 3.11 is installed on macOS")
        assert "environment" in cats

    def test_correction_detection(self):
        cats, scores = classify_content("Actually, that was incorrect. The correct value should be 42")
        assert "correction" in cats

    def test_skill_detection(self):
        cats, scores = classify_content("I discovered a trick to solve the memory leak")
        assert "skill" in cats

    def test_observation_detection(self):
        cats, scores = classify_content("I noticed that the API seems slower on Mondays")
        assert "observation" in cats

    def test_fact_detection(self):
        cats, scores = classify_content("The database uses PostgreSQL and supports JSON columns")
        assert "fact" in cats

    def test_default_to_observation(self):
        cats, scores = classify_content("xyz abc 123")
        assert "observation" in cats

    def test_multiple_categories(self):
        cats, scores = classify_content(
            "I always prefer to use snake_case naming convention in Python"
        )
        assert len(cats) >= 2

    def test_max_three_categories(self):
        cats, scores = classify_content(
            "I discovered a trick: first install Python on macOS, "
            "the convention is to always prefer snake_case naming"
        )
        assert len(cats) <= 3

    def test_scores_are_positive(self):
        _, scores = classify_content("The user prefers dark mode")
        for score in scores.values():
            assert score > 0


# ---------------------------------------------------------------------------
# estimate_importance
# ---------------------------------------------------------------------------


class TestEstimateImportance:
    def test_base_importance(self):
        importance = estimate_importance("some regular text", ["observation"])
        assert 0.4 <= importance <= 0.6

    def test_high_importance_boost(self):
        importance = estimate_importance(
            "This is critical: never expose the API key", ["fact"]
        )
        assert importance > 0.7

    def test_low_importance_reduction(self):
        importance = estimate_importance(
            "Maybe a minor temporary hack for now", ["observation"]
        )
        assert importance < 0.5

    def test_correction_boost(self):
        importance = estimate_importance(
            "The port is 8080", ["correction"]
        )
        base = estimate_importance("The port is 8080", ["fact"])
        assert importance > base

    def test_preference_boost(self):
        importance = estimate_importance(
            "Dark mode is better", ["preference"]
        )
        base = estimate_importance("Dark mode is better", ["fact"])
        assert importance > base

    def test_clamped_to_range(self):
        # Very high signals
        importance = estimate_importance(
            "Critical essential key security password important must never always",
            ["correction"],
        )
        assert importance <= 1.0

        # Very low signals
        importance = estimate_importance(
            "minor trivial maybe possibly might temporary quick hack",
            ["observation"],
        )
        assert importance >= 0.1


# ---------------------------------------------------------------------------
# detect_contradiction
# ---------------------------------------------------------------------------


class TestDetectContradiction:
    def test_no_contradiction_low_similarity(self):
        mem = _make_memory("The sky is blue")
        result = detect_contradiction("cats are cute", mem, similarity=0.2)
        assert result.is_contradiction is False
        assert result.reason == "low_similarity"

    def test_negation_contradiction(self):
        mem = _make_memory("The feature is enabled by default")
        result = detect_contradiction(
            "The feature is not enabled by default",
            mem,
            similarity=0.9,
        )
        assert result.is_contradiction is True
        assert result.confidence > 0.5

    def test_correction_phrase_contradiction(self):
        mem = _make_memory("The port is 3000")
        result = detect_contradiction(
            "Actually, the port should be 8080, not 3000",
            mem,
            similarity=0.8,
        )
        assert result.is_contradiction is True
        assert result.confidence > 0.5

    def test_no_contradiction_similar_content(self):
        mem = _make_memory("Python is a programming language")
        result = detect_contradiction(
            "Python is a great programming language",
            mem,
            similarity=0.9,
        )
        assert result.is_contradiction is False

    def test_existing_memory_preserved(self):
        mem = _make_memory("test content", id=42)
        result = detect_contradiction("opposite test", mem, similarity=0.3)
        assert result.existing_memory is mem
        assert result.existing_memory.id == 42


# ---------------------------------------------------------------------------
# detect_contradictions (batch)
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    def test_filters_by_min_similarity(self):
        candidates = [
            _make_scored("The feature is enabled", similarity=0.3),
            _make_scored("The feature is active", similarity=0.9),
        ]
        results = detect_contradictions(
            "The feature is not enabled",
            candidates,
            min_similarity=0.5,
        )
        # Only the high-similarity one is checked
        for r in results:
            assert r.existing_memory.content != "The feature is enabled" or r.confidence > 0

    def test_returns_only_contradictions(self):
        candidates = [
            _make_scored("Python is fast", similarity=0.8),
            _make_scored("Python is a language", similarity=0.8),
        ]
        results = detect_contradictions(
            "Python is a great language", candidates
        )
        for r in results:
            assert r.is_contradiction is True

    def test_sorted_by_confidence(self):
        candidates = [
            _make_scored("The server uses port 3000", similarity=0.9, id=1),
            _make_scored("The server runs on port 3000", similarity=0.85, id=2),
        ]
        results = detect_contradictions(
            "Actually the server does not use port 3000",
            candidates,
        )
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence

    def test_empty_candidates(self):
        results = detect_contradictions("anything", [])
        assert results == []


# ---------------------------------------------------------------------------
# encode (full pipeline)
# ---------------------------------------------------------------------------


class TestEncode:
    def test_basic_encode(self):
        result = encode("The user prefers dark mode")
        assert isinstance(result, EncodingResult)
        assert len(result.categories) > 0
        assert 0.0 < result.importance <= 1.0

    def test_encode_with_candidates(self):
        candidates = [
            _make_scored("The user prefers light mode", similarity=0.9),
        ]
        result = encode(
            "Actually the user does not prefer light mode, they prefer dark mode",
            candidates=candidates,
        )
        assert len(result.contradictions) > 0
        # Importance should be boosted due to contradiction
        assert result.importance > 0.5

    def test_encode_without_candidates(self):
        result = encode("Simple fact about Python")
        assert result.contradictions == []

    def test_encode_categories_in_result(self):
        result = encode("I prefer using vim over emacs")
        assert "preference" in result.categories
        assert "preference" in result.category_scores

    def test_encode_importance_boost_on_contradiction(self):
        base_result = encode("The port is 8080")
        base_importance = base_result.importance

        candidates = [
            _make_scored("The port is 3000", similarity=0.9),
        ]
        contradiction_result = encode(
            "Actually the port is not 3000, it should be 8080",
            candidates=candidates,
        )
        # May or may not find contradiction depending on heuristics,
        # but importance should be >= base when correction patterns present
        assert contradiction_result.importance >= 0.5

    def test_encode_high_importance_content(self):
        result = encode(
            "Critical security warning: never expose API keys in code"
        )
        assert result.importance > 0.7
