"""Tests for the collaboration learning framework — all 4 layers.

Runs against a mocked graph client so tests are deterministic and
require no external API access.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

from agent.collaboration.graph_client import GraphClient, GraphMemory
from agent.collaboration.observation import (
    detect_correction,
    classify_response_format,
    CORRECTION_REDIRECT,
    CORRECTION_TERSE,
    CORRECTION_SCOPE,
    CORRECTION_REJECTION,
    CORRECTION_LANGUAGE,
)
from agent.collaboration.extraction import (
    CollaborationExtractor,
    compute_confidence,
    PreferenceSignal,
)
from agent.collaboration.feedback import (
    CorrectionRateTracker,
    DriftDetector,
    SessionCorrectionRate,
    validate_preference_impact,
)
from agent.collaboration.adaptation import load_preference_summary, format_preference_entries


# =========================================================================
# Mock graph client
# =========================================================================


class MockGraphClient:
    """In-memory mock of GraphClient for testing."""

    def __init__(self):
        self.memories: List[Dict] = []
        self.decisions: List[Dict] = []
        self._avail = True

    def is_available(self):
        return self._avail

    def create_memory(self, memory: GraphMemory) -> str:
        mid = f"mem_{len(self.memories)}"
        self.memories.append({
            "id": mid,
            "content": memory.content,
            "category": memory.category,
            "tags": memory.tags,
            "confidence": memory.confidence,
            "sessionId": memory.session_id,
        })
        return mid

    def search_memories(self, query: str, category: str = None,
                        tags: List[str] = None, limit: int = 20) -> List[Dict]:
        results = list(self.memories)
        if category:
            results = [r for r in results if r.get("category") == category]
        return results[:limit]

    def delete_memory(self, memory_id: str) -> bool:
        self.memories = [m for m in self.memories if m.get("id") != memory_id]
        return True

    def create_decision(self, description: str, **kwargs) -> str:
        did = f"dec_{len(self.decisions)}"
        self.decisions.append({"id": did, "description": description, **kwargs})
        return did


@pytest.fixture
def mock_client():
    return MockGraphClient()


# =========================================================================
# Correction detection tests
# =========================================================================


class TestCorrectionDetection:
    def test_redirect_detected(self):
        c = detect_correction("not hermes agent! xentropy/ai containers")
        assert c is not None
        assert c.type == CORRECTION_REDIRECT
        assert c.weight == 1.0

    def test_terse_fix_detected(self):
        c = detect_correction("go.")
        assert c is not None
        assert c.type == CORRECTION_TERSE

    def test_scope_shrink_detected(self):
        c = detect_correction("ensure X has Y configured properly")
        assert c is not None
        assert c.type == CORRECTION_SCOPE

    def test_rejection_detected(self):
        for msg in ["no", "wrong.", "stop", "nope"]:
            c = detect_correction(msg)
            assert c is not None, f"'{msg}' not detected as rejection"
            assert c.type == CORRECTION_REJECTION

    def test_language_switch_detected(self):
        c = detect_correction("speak english")
        assert c is not None
        assert c.type == CORRECTION_LANGUAGE

    def test_normal_question_not_correction(self):
        c = detect_correction("Can you deploy the service to GKE?")
        assert c is None

    def test_empty_message_not_correction(self):
        assert detect_correction("") is None
        assert detect_correction(None) is None

    def test_terse_triggers_comprehensive(self):
        triggers = ["go.", "do it.", "now.", "fix.", "just do it.", "move.", "execute."]
        for msg in triggers:
            c = detect_correction(msg)
            assert c is not None, f"'{msg}' not detected as terse fix"


# =========================================================================
# Format classification tests
# =========================================================================


class TestFormatClassification:
    def test_terse_format(self):
        assert classify_response_format("OK", 0) == "terse"
        assert classify_response_format("Done", 0) == "terse"

    def test_prose_format(self):
        text = "# Header\n\nThis is a paragraph of prose text with multiple sentences."
        assert classify_response_format(text, 1) == "prose"

    def test_table_format(self):
        text = "| col1 | col2 |\n|------|------|\n| a | b |"
        assert classify_response_format(text, 0) == "table"

    def test_code_format(self):
        text = "```python\nprint('hello')\n```"
        assert classify_response_format(text, 0) == "code"

    def test_bullet_format(self):
        text = "- item 1\n- item 2\n- item 3"
        assert classify_response_format(text, 0) == "bullet"

    def test_mixed_format(self):
        text = "# Summary\n\n| col | val |\n|-----|-----|\n| a | 1 |\n\n```code```"
        assert classify_response_format(text, 0) == "mixed"


# =========================================================================
# Graph client tests (with mock)
# =========================================================================


class TestGraphClientMock:
    def test_create_and_search(self, mock_client):
        """Create memories and search them."""
        mem = GraphMemory(
            content="test observation",
            category="observation",
            tags=["test", "collab"],
        )
        mid = mock_client.create_memory(mem)
        assert mid is not None

        results = mock_client.search_memories("test")
        assert len(results) == 1
        assert results[0]["category"] == "observation"

    def test_delete_memory(self, mock_client):
        mem = GraphMemory(content="delete me", category="test")
        mid = mock_client.create_memory(mem)
        assert len(mock_client.memories) == 1
        mock_client.delete_memory(mid)
        assert len(mock_client.memories) == 0

    def test_create_decision(self, mock_client):
        did = mock_client.create_decision("test decision", tags=["collab"])
        assert did is not None
        assert len(mock_client.decisions) == 1
        assert mock_client.decisions[0]["description"] == "test decision"


# =========================================================================
# Confidence scoring tests
# =========================================================================


class TestConfidenceScoring:
    def test_below_min_samples(self):
        """Confidence is 0 with fewer than 3 samples."""
        assert compute_confidence(0, 0.8, 0.5, 0.5) == 0.0
        assert compute_confidence(1, 0.8, 0.5, 0.5) == 0.0
        assert compute_confidence(2, 0.8, 0.5, 0.5) == 0.0

    def test_low_consistency(self):
        """Confidence is 0 with consistency below 0.3."""
        assert compute_confidence(10, 0.2, 0.5, 0.5) == 0.0

    def test_high_confidence(self):
        """High sample + consistency yields high confidence."""
        c = compute_confidence(50, 0.9, 0.8, 1.0)
        assert c > 0.7

    def test_medium_confidence(self):
        """Medium evidence yields medium confidence."""
        c = compute_confidence(8, 0.6, 0.5, 0.5)
        assert 0.3 < c < 0.7


# =========================================================================
# Feedback tests
# =========================================================================


class TestCorrectionRateTracker:
    def test_trend_negative_on_improvement(self, mock_client):
        tracker = CorrectionRateTracker(window_sessions=5, client=mock_client)
        tracker.record_session("s1", 5, 10)  # rate=0.5
        tracker.record_session("s2", 3, 10)  # rate=0.3
        tracker.record_session("s3", 1, 10)  # rate=0.1
        # Trend should be negative (corrections decreasing)
        assert tracker.trend() < 0

    def test_trend_positive_on_regression(self, mock_client):
        tracker = CorrectionRateTracker(window_sessions=5, client=mock_client)
        tracker.record_session("s1", 1, 10)  # rate=0.1
        tracker.record_session("s2", 3, 10)  # rate=0.3
        tracker.record_session("s3", 5, 10)  # rate=0.5
        # Trend should be positive (corrections increasing)
        assert tracker.trend() > 0

    def test_trend_zero_with_insufficient_data(self, mock_client):
        tracker = CorrectionRateTracker(window_sessions=5, client=mock_client)
        tracker.record_session("s1", 2, 10)
        assert tracker.trend() == 0.0

    def test_avg_rate(self, mock_client):
        tracker = CorrectionRateTracker(window_sessions=5, client=mock_client)
        tracker.record_session("s1", 2, 10)  # 0.2
        tracker.record_session("s2", 4, 10)  # 0.4
        avg = tracker.avg_rate()
        assert 0.2 <= avg <= 0.4


class TestValidatePreferenceImpact:
    def test_positive_impact(self):
        before = [SessionCorrectionRate("s1", 0.5, 5, 10)]
        after = [SessionCorrectionRate("s2", 0.2, 2, 10)]
        impact = validate_preference_impact(before, after)
        assert impact > 0  # correction rate decreased

    def test_negative_impact(self):
        before = [SessionCorrectionRate("s1", 0.2, 2, 10)]
        after = [SessionCorrectionRate("s2", 0.5, 5, 10)]
        impact = validate_preference_impact(before, after)
        assert impact < 0  # correction rate increased

    def test_no_data(self):
        assert validate_preference_impact([], []) == 0.0
