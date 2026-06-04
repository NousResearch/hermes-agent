"""Tests for content-production intent detection in IntentClassifier.

Covers the Baumbad workflow integration: messages requesting blog posts,
articles, and editorial content should route to the baumbad-content-pipeline-v1
workflow via DELEGATE_SPECIALIST.

Positive triggers, negative tests, edge cases, and language detection are all
validated here.
"""

from __future__ import annotations

import pytest

from agent.modules.intent_classifier import (
    ClassifiedIntent,
    Route,
    _detect_content_production,
    classify_intent,
)
from agent.modules.interpreter import Interpretation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interpretation(text: str) -> Interpretation:
    """Minimal Interpretation for classifier tests."""
    return Interpretation(
        intent="unknown",
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# _detect_content_production — unit tests for the fast-path detector
# ---------------------------------------------------------------------------


class TestDetectContentProductionPositive:
    """Messages that should be recognised as content-production."""

    def test_german_blog_schreibe(self):
        result = _detect_content_production("schreibe einen blog über baumpflege")
        assert result is not None
        assert result["intent"] == "content_production"
        assert result["workflow_ref"] == "baumbad-content-pipeline-v1"

    def test_german_blog_schreib(self):
        result = _detect_content_production("schreib einen blog über gartenarbeit")
        assert result is not None

    def test_german_artikel_ich_brauche(self):
        result = _detect_content_production("ich brauche einen artikel zu baumschnitt")
        assert result is not None
        assert result["intent"] == "content_production"

    def test_german_neuer_content(self):
        result = _detect_content_production("neuer content über herbstpflege")
        assert result is not None

    def test_german_artikel_schreiben(self):
        result = _detect_content_production("schreiben Sie einen Artikel über Rasenpflege")
        assert result is not None

    def test_german_beitrag_erstellen(self):
        result = _detect_content_production("erstelle einen Beitrag über Hecken schneiden")
        assert result is not None

    def test_german_redaktion(self):
        result = _detect_content_production("wir brauchen einen Redaktionsplan")
        assert result is not None

    def test_german_redaktion_lowercase(self):
        result = _detect_content_production("redaktion für den nächsten monat planen")
        assert result is not None

    def test_english_write_article(self):
        result = _detect_content_production("write an article about garden tools")
        assert result is not None
        assert result["intent"] == "content_production"

    def test_english_write_blog(self):
        result = _detect_content_production("write a blog about tree care")
        assert result is not None

    def test_english_blog_about(self):
        result = _detect_content_production("blog about lawn maintenance")
        assert result is not None

    def test_english_article_about(self):
        result = _detect_content_production("article about seasonal pruning")
        assert result is not None

    def test_english_content_production(self):
        result = _detect_content_production("content production for our website")
        assert result is not None

    def test_english_produce_content(self):
        result = _detect_content_production("produce content about shrub care")
        assert result is not None

    def test_english_write_post(self):
        result = _detect_content_production("write a post about planting tips")
        assert result is not None


class TestDetectContentProductionLanguage:
    """Language detection in content-production metadata."""

    def test_german_language_detected(self):
        result = _detect_content_production("schreibe einen blog über baumpflege")
        assert result is not None
        assert result["extracted"]["language"] == "de"

    def test_english_language_detected(self):
        result = _detect_content_production("write an article about garden tools")
        assert result is not None
        assert result["extracted"]["language"] == "en"

    def test_german_with_ich_brauche(self):
        result = _detect_content_production("ich brauche einen artikel über rosen")
        assert result is not None
        assert result["extracted"]["language"] == "de"


class TestDetectContentProductionTopic:
    """Topic extraction from content-production messages."""

    def test_topic_extracted_german_ueber(self):
        result = _detect_content_production("schreibe einen blog über baumpflege")
        assert result is not None
        assert result["extracted"]["topic"] == "baumpflege"

    def test_topic_extracted_english_about(self):
        result = _detect_content_production("write an article about garden tools")
        assert result is not None
        assert result["extracted"]["topic"] == "garden tools"

    def test_topic_none_when_not_present(self):
        # No preposition phrase; topic should be None (not crash)
        result = _detect_content_production("schreibe einen blog")
        assert result is not None
        # topic may be None or an empty-ish value — must not raise
        topic = result["extracted"]["topic"]
        assert topic is None or isinstance(topic, str)

    def test_workflow_ref_constant(self):
        result = _detect_content_production("schreibe einen blog über baumpflege")
        assert result is not None
        assert result["workflow_ref"] == "baumbad-content-pipeline-v1"


class TestDetectContentProductionNegative:
    """Messages that must NOT match content-production."""

    def test_dashboard_request(self):
        result = _detect_content_production("show me the dashboard")
        assert result is None

    def test_task_creation(self):
        result = _detect_content_production("create a task for tomorrow")
        assert result is None

    def test_question(self):
        result = _detect_content_production("what time is it?")
        assert result is None

    def test_memory_lookup(self):
        result = _detect_content_production("find the notes from last week")
        assert result is None

    def test_generic_german(self):
        result = _detect_content_production("was ist die Uhrzeit?")
        assert result is None

    def test_unrelated_write(self):
        # "write" without "article/blog/post/content" should not match
        result = _detect_content_production("write the code for this feature")
        assert result is None


class TestDetectContentProductionEdgeCases:
    """Edge cases: empty/whitespace input, mixed language."""

    def test_empty_string(self):
        result = _detect_content_production("")
        assert result is None

    def test_whitespace_only(self):
        result = _detect_content_production("   \t\n  ")
        assert result is None

    def test_none_like_empty(self):
        # Simulates what happens if raw_text arrives as empty string
        result = _detect_content_production("")
        assert result is None

    def test_mixed_language_german_trigger(self):
        # English preamble, German keyword — should still detect
        result = _detect_content_production("please schreibe einen artikel über Heckenpflege")
        assert result is not None
        assert result["intent"] == "content_production"

    def test_case_insensitive_german(self):
        result = _detect_content_production("SCHREIBE EINEN BLOG ÜBER BAUMSCHNITT")
        assert result is not None

    def test_case_insensitive_english(self):
        result = _detect_content_production("WRITE AN ARTICLE ABOUT pruning")
        assert result is not None


# ---------------------------------------------------------------------------
# classify_intent — integration tests using Interpretation objects
# ---------------------------------------------------------------------------


class TestClassifyIntentContentProduction:
    """classify_intent() end-to-end for content-production messages."""

    def test_german_blog_routes_to_delegate_specialist(self):
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        result = classify_intent(interp)
        assert isinstance(result, ClassifiedIntent)
        assert result.route == Route.DELEGATE_SPECIALIST

    def test_german_blog_confidence(self):
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        result = classify_intent(interp)
        assert result.confidence == pytest.approx(0.9)

    def test_german_blog_metadata_intent(self):
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        result = classify_intent(interp)
        assert result.interpretation.metadata.get("intent") == "content_production"

    def test_german_blog_metadata_language(self):
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        result = classify_intent(interp)
        extracted = result.interpretation.metadata.get("extracted", {})
        assert extracted.get("language") == "de"

    def test_german_blog_metadata_topic(self):
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        result = classify_intent(interp)
        extracted = result.interpretation.metadata.get("extracted", {})
        assert extracted.get("topic") == "baumpflege"

    def test_english_article_routes_to_delegate_specialist(self):
        interp = _make_interpretation("write an article about garden tools")
        result = classify_intent(interp)
        assert result.route == Route.DELEGATE_SPECIALIST

    def test_english_article_language(self):
        interp = _make_interpretation("write an article about garden tools")
        result = classify_intent(interp)
        extracted = result.interpretation.metadata.get("extracted", {})
        assert extracted.get("language") == "en"

    def test_english_article_topic(self):
        interp = _make_interpretation("write an article about garden tools")
        result = classify_intent(interp)
        extracted = result.interpretation.metadata.get("extracted", {})
        assert extracted.get("topic") == "garden tools"

    def test_workflow_ref_in_metadata(self):
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        result = classify_intent(interp)
        assert result.interpretation.metadata.get("workflow_ref") == "baumbad-content-pipeline-v1"

    def test_original_metadata_preserved(self):
        """Existing metadata keys must not be clobbered by enrichment."""
        interp = _make_interpretation("schreibe einen blog über baumpflege")
        # Inject existing metadata before classification
        interp = interp.model_copy(update={"metadata": {"session_ref": "abc123"}})
        result = classify_intent(interp)
        assert result.interpretation.metadata.get("session_ref") == "abc123"
        assert result.interpretation.metadata.get("intent") == "content_production"

    def test_raw_text_preserved(self):
        text = "schreibe einen blog über baumpflege"
        interp = _make_interpretation(text)
        result = classify_intent(interp)
        assert result.interpretation.raw_text == text


class TestClassifyIntentNonContentProduction:
    """classify_intent() for messages that are NOT content-production."""

    def test_dashboard_routes_to_answer_directly(self):
        interp = _make_interpretation("show me the dashboard")
        result = classify_intent(interp)
        assert result.route == Route.ANSWER_DIRECTLY

    def test_general_question_answer_directly(self):
        interp = _make_interpretation("what time is it?")
        result = classify_intent(interp)
        assert result.route == Route.ANSWER_DIRECTLY

    def test_non_content_has_zero_confidence(self):
        interp = _make_interpretation("show me the dashboard")
        result = classify_intent(interp)
        assert result.confidence == pytest.approx(0.0)

    def test_non_content_no_intent_key_in_metadata(self):
        interp = _make_interpretation("show me the dashboard")
        result = classify_intent(interp)
        # No content metadata should be injected
        assert result.interpretation.metadata.get("intent") != "content_production"
        assert result.interpretation.metadata.get("workflow_ref") is None

    def test_empty_input_answer_directly(self):
        interp = _make_interpretation("")
        result = classify_intent(interp)
        assert result.route == Route.ANSWER_DIRECTLY
