"""
Tests for routing module (P2-9).

Covers: keyword scoring, capability scoring, composite routing,
historical signal, confidence computation, and feedback recording.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from hermes_cli.routing import (
    RoutingEngine,
    RouteResult,
    RouteDecision,
    _keyword_score,
    _capability_score,
    _tokenize,
    MIN_CONFIDENCE_AUTO_ROUTE,
    DEFAULT_SPECIALISTS,
)


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------


class TestKeywordScore:
    def test_exact_match(self):
        score = _keyword_score(
            "Fix the cron job that fails every Monday",
            ["cron", "job", "failure", "monday"],
        )
        assert score > 0.5

    def test_no_match(self):
        score = _keyword_score(
            "Fix the cron job",
            ["social media", "twitter", "content"],
        )
        assert score == 0.0

    def test_multi_word_keyword(self):
        score = _keyword_score(
            "The memory leak in the gateway is causing crashes",
            ["memory leak", "crash"],
        )
        assert score > 0.0

    def test_partial_match(self):
        score = _keyword_score(
            "Deploy the new feature to production",
            ["deploy", "code", "pr"],
        )
        assert 0.0 < score < 1.0

    def test_empty_keywords(self):
        score = _keyword_score("any text", [])
        assert score == 0.0

    def test_empty_text(self):
        score = _keyword_score("", ["keyword"])
        assert score == 0.0

    def test_phrase_match_bonus(self):
        # "memory leak" appears as exact phrase — bonus applied
        score1 = _keyword_score(
            "There is a memory leak in production",
            ["memory leak"],
        )
        assert score1 > 0.0

    def test_case_insensitive(self):
        score_lower = _keyword_score("memory leak", ["MEMORY LEAK"])
        score_upper = _keyword_score("MEMORY LEAK", ["memory leak"])
        assert score_lower == score_upper


# ---------------------------------------------------------------------------
# Capability scoring
# ---------------------------------------------------------------------------


class TestCapabilityScore:
    def test_devops_match(self):
        score = _capability_score(
            "The gateway service crashed with OOM error",
            ["devops", "infrastructure"],
            "infrastructure",
        )
        assert score > 0.0

    def test_coding_match(self):
        score = _capability_score(
            "Implement a new React component with TypeScript",
            ["coding", "frontend"],
            "software",
        )
        assert score > 0.0

    def test_no_match(self):
        score = _capability_score(
            "random unrelated text about gardening",
            ["devops", "infrastructure"],
            "infrastructure",
        )
        assert score == 0.0

    def test_multiple_capabilities(self):
        score = _capability_score(
            "Fix the SQL injection in the Python API",
            ["coding", "security", "qa"],
            "software",
        )
        assert score > 0.0

    def test_empty_capabilities(self):
        score = _capability_score("any text", [], "")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Routing engine
# ---------------------------------------------------------------------------


class TestRoutingEngine:
    def test_route_devops_task_to_wesker(self):
        engine = RoutingEngine()
        results = engine.route(
            "Fix cron job timeout on production",
            "The daily backup cron job is timing out after 600s",
        )
        assert len(results) > 0
        assert results[0].specialist == "wesker"

    def test_route_coding_task_to_octacon(self):
        engine = RoutingEngine()
        results = engine.route(
            "Fix TypeError in kanban_db.py",
            "TypeError at line 1234: NoneType has no attribute 'connect'",
        )
        assert len(results) > 0
        # octacon-frontend should be top for code bugs
        assert results[0].specialist.startswith("octacon")

    def test_route_research_task_to_remii(self):
        engine = RoutingEngine()
        results = engine.route(
            "Research: compare React Native vs Flutter for CoachOS",
            "Need a deep dive analysis of tradeoffs",
        )
        assert len(results) > 0
        assert results[0].specialist.startswith("remii")

    def test_route_content_task_to_ceecee(self):
        engine = RoutingEngine()
        results = engine.route(
            "Draft a LinkedIn thread about the KENSEI pipeline",
            "Write a thread about the autonomous feature pipeline we built",
        )
        assert len(results) > 0
        assert results[0].specialist == "ceecee"

    def test_route_admin_task_to_gojo(self):
        engine = RoutingEngine()
        results = engine.route(
            "Book meeting with recruitment agent",
            "Schedule a 30-minute call for Thursday afternoon",
        )
        assert len(results) > 0
        assert results[0].specialist == "gojo"

    def test_route_qa_task_to_quan(self):
        engine = RoutingEngine()
        results = engine.route(
            "QA sign-off for release v2.4.0",
            "Run regression suite and verify all checklists",
        )
        assert len(results) > 0
        assert results[0].specialist.startswith("quan")

    def test_route_documentation_to_light(self):
        engine = RoutingEngine()
        results = engine.route(
            "Document the pipeline architecture decision",
            "Write a runbook for the autonomous feature pipeline",
        )
        assert len(results) > 0
        assert results[0].specialist.startswith("light")

    def test_route_returns_top_n(self):
        engine = RoutingEngine()
        results = engine.route("test", top_n=3)
        assert len(results) <= 3

    def test_route_best_returns_none_on_low_confidence(self):
        engine = RoutingEngine()
        result = engine.route_best(
            "xyzzy random gibberish that matches nothing",
            "blargh flargh",
        )
        assert result is None

    def test_signal_breakdown_present(self):
        engine = RoutingEngine()
        results = engine.route("Fix server memory leak", "OOM killer triggered")
        assert "keyword" in results[0].signals
        assert "capability" in results[0].signals
        assert "historical" in results[0].signals

    def test_confidence_computed(self):
        engine = RoutingEngine()
        results = engine.route("Deploy the new version", "")
        assert 0.0 <= results[0].confidence <= 1.0

    def test_all_specialists_in_results(self):
        engine = RoutingEngine()
        results = engine.route("test task", top_n=50)
        names = [r.specialist for r in results]
        assert len(names) == len(DEFAULT_SPECIALISTS)


# ---------------------------------------------------------------------------
# Historical signal
# ---------------------------------------------------------------------------


class TestHistoricalSignal:
    def test_positive_history_boosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "routing.db")
            engine = RoutingEngine(history_db=db)

            # Record accepted assignments for wesker
            engine.record_feedback("t_1", "wesker", "accepted", score=0.8)
            engine.record_feedback("t_2", "wesker", "accepted", score=0.9)

            # Now route a new ops task — wesker should benefit
            results = engine.route(
                "Fix cron job", "cron failure", task_id="t_new",
            )
            wesk = next(r for r in results if r.specialist == "wesker")
            assert wesk.signals["historical"] > 0.5

    def test_negative_history_penalises(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "routing.db")
            engine = RoutingEngine(history_db=db)

            # Record rejected assignments for wesker
            engine.record_feedback("t_1", "wesker", "rejected", score=0.8)
            engine.record_feedback("t_2", "wesker", "rejected", score=0.7)

            # Route — historical should be below 0.5
            results = engine.route(
                "Fix cron job", "cron failure", task_id="t_new",
            )
            wesk = next(r for r in results if r.specialist == "wesker")
            assert wesk.signals["historical"] < 0.5

    def test_no_history_is_neutral(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = RoutingEngine(history_db=os.path.join(tmp, "routing.db"))
            results = engine.route("test", task_id="unknown")
            # All specialists should have historical=0.5 (neutral)
            hist_values = [r.signals["historical"] for r in results]
            assert all(h == 0.5 for h in hist_values)


# ---------------------------------------------------------------------------
# Feedback integration
# ---------------------------------------------------------------------------


class TestFeedbackIntegration:
    def test_reassignment_updates_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "routing.db")
            engine = RoutingEngine(history_db=db)

            # Task was assigned to gojo but should have gone to wesker
            engine.record_feedback(
                "t_1", "gojo", "reassigned_to",
                reassigned_to="wesker", score=0.3,
            )

            # Check historical signals directly
            # gojo: reassigned_to is neutral (0.0 signal), so avg = 0.0 / sqrt(1) = 0.0, +0.5 = 0.5
            # wesker: no direct records, so 0.5
            gojo_hist = engine._historical_signal("gojo", "t_new")
            wesk_hist = engine._historical_signal("wesker", "t_new")
            # Both should be at or near neutral — reassignment is not a penalty
            assert 0.45 <= gojo_hist <= 0.55
            assert 0.45 <= wesk_hist <= 0.55

    def test_effectiveness_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "routing.db")
            engine = RoutingEngine(history_db=db)

            engine.record_feedback("t_1", "wesker", "accepted", score=0.8)
            engine.record_feedback("t_2", "wesker", "accepted", score=0.9)
            engine.record_feedback("t_3", "ceecee", "rejected", score=0.4)
            engine.record_feedback("t_4", "ceecee", "rejected", score=0.3)

            eff = engine.specialist_effectiveness(days=365)
            assert eff["wesker"] == 1.0
            assert eff["ceecee"] == 0.0
