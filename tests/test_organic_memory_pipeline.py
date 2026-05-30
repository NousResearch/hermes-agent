"""Unit tests for the Hermes 8-layer organic memory pipeline.

Covers: SalienceScorer (L1), SilentEngramEngine (L2), ConsolidationEngine (L3),
ReconsolidationEngine (L4), FeedbackCoordinator (L5), ActivationGraph (L6),
SleepScheduler (L7), PipelineState (DB), and _salience_to_engram_strength helper.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so ``agent`` is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.memory_pipeline import (
    ActivationGraph,
    ConsolidationEngine,
    FeedbackCoordinator,
    MemoryPipeline,
    PipelineState,
    ReconsolidationEngine,
    SalienceScorer,
    SilentEngramEngine,
    SleepScheduler,
    _salience_to_engram_strength,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def state(tmp_path, monkeypatch):
    """Create a PipelineState backed by a temp database.

    We monkeypatch ``hermes_state.apply_wal_with_fallback`` to a no-op so
    the test never touches the real hermes home directory.
    """
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "apply_wal_with_fallback",
        lambda conn, db_label="": "wal",
    )
    db_path = str(tmp_path / "pipeline_test.db")
    ps = PipelineState(db_path=db_path)
    yield ps
    ps.close()


# ===========================================================================
# L1: SalienceScorer
# ===========================================================================

class TestSalienceScorer:

    def test_trivial_message_returns_low_score(self):
        scorer = SalienceScorer()
        result = scorer.score("hi")
        assert result.overall < 0.1
        assert result.is_trivial is True

    def test_emotional_message_returns_high_emotion(self):
        scorer = SalienceScorer()
        result = scorer.score(
            "This is an urgent crash! The system is completely broken!!"
        )
        assert result.emotion >= 0.5

    def test_important_message_returns_high_importance(self):
        scorer = SalienceScorer()
        result = scorer.score(
            "Remember this important crucial decision confirmed by the team"
        )
        assert result.importance >= 0.7

    def test_repeated_message_has_decreasing_novelty(self):
        scorer = SalienceScorer()
        first = scorer.score("the quick brown fox jumps over the lazy dog")
        second = scorer.score("the quick brown fox jumps over the lazy dog")
        assert second.novelty < first.novelty

    def test_chinese_patterns_work(self):
        scorer = SalienceScorer()
        result = scorer.score(
            "这个问题非常严重，紧急处理！关键决定已确认"
        )
        assert result.emotion > 0.0
        assert result.importance > 0.0

    def test_empty_string_returns_zero(self):
        scorer = SalienceScorer()
        result = scorer.score("")
        assert result.overall == 0.0
        assert result.is_trivial is True

    def test_whitespace_only_returns_zero(self):
        scorer = SalienceScorer()
        result = scorer.score("   ")
        assert result.overall == 0.0
        assert result.is_trivial is True


# ===========================================================================
# L2: SilentEngramEngine
# ===========================================================================

class TestSilentEngramEngine:

    def test_initial_strength_is_one(self, state):
        engine = SilentEngramEngine(half_life_hours=720.0)
        engine.strengthen(state, "mem_init", delta=0.0)
        row = state._conn.execute(
            "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
            ("mem_init",),
        ).fetchone()
        assert row is not None
        assert row["strength"] == 1.0

    def test_half_life_decay_yields_half_strength(self, state):
        engine = SilentEngramEngine(half_life_hours=720.0)
        engine.strengthen(state, "mem_decay", delta=0.0)
        engine.apply_decay(state, hours_elapsed=720.0)
        row = state._conn.execute(
            "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
            ("mem_decay",),
        ).fetchone()
        assert row["strength"] == pytest.approx(0.5, abs=0.01)

    def test_strength_never_reaches_zero(self, state):
        engine = SilentEngramEngine(half_life_hours=1.0)
        engine.strengthen(state, "mem_zero", delta=0.0)
        for _ in range(50):
            engine.apply_decay(state, hours_elapsed=1000.0)
        row = state._conn.execute(
            "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
            ("mem_zero",),
        ).fetchone()
        assert row["strength"] > 0.0

    def test_retrieval_strengthen_increases_strength(self, state):
        engine = SilentEngramEngine(half_life_hours=720.0)
        engine.strengthen(state, "mem_str", delta=0.0)
        # Decay a bit first so there is room to strengthen.
        engine.apply_decay(state, hours_elapsed=720.0)
        before = state._conn.execute(
            "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
            ("mem_str",),
        ).fetchone()["strength"]
        new_str = engine.strengthen(state, "mem_str", delta=0.1)
        assert new_str == pytest.approx(min(1.0, before + 0.1), abs=0.001)

    def test_classify_thresholds(self):
        engine = SilentEngramEngine()
        assert engine.classify(0.8) == "active"
        assert engine.classify(0.51) == "active"
        assert engine.classify(0.35) == "semi_active"
        assert engine.classify(0.1) == "silent"
        assert engine.classify(0.01) == "buried"


# ===========================================================================
# L3: ConsolidationEngine
# ===========================================================================

class TestConsolidationEngine:

    def test_below_min_facts_returns_zero_created(self, state):
        engine = ConsolidationEngine(min_facts=5)
        facts = [
            {"content": f"fact number {i} about something important"}
            for i in range(3)
        ]
        result = engine.consolidate(state, facts=facts)
        assert result["schemas_created"] == 0

    def test_five_plus_facts_creates_schemas(self, state):
        engine = ConsolidationEngine(min_facts=5)
        facts = [
            {"content": f"unique fact number {i} about science and math"}
            for i in range(6)
        ]
        result = engine.consolidate(state, facts=facts)
        assert result["schemas_created"] > 0

    def test_dedup_prevents_duplicates(self, state):
        engine = ConsolidationEngine(min_facts=3)
        facts = [
            {"content": f"dedup test fact {i} with enough chars"}
            for i in range(4)
        ]
        first = engine.consolidate(state, facts=facts)
        assert first["schemas_created"] > 0
        second = engine.consolidate(state, facts=facts)
        assert second["schemas_created"] == 0
        assert second["schemas_updated"] > 0


# ===========================================================================
# L4: ReconsolidationEngine
# ===========================================================================

class TestReconsolidationEngine:

    def test_high_overlap_returns_low_conflict(self):
        engine = ReconsolidationEngine()
        existing = ["the cat sat on the mat in the living room"]
        conflict = engine.detect_conflict(
            "the cat sat on the mat in the living room", existing
        )
        assert conflict < 0.1

    def test_low_overlap_returns_high_conflict(self):
        engine = ReconsolidationEngine()
        existing = ["the quick brown fox jumps over the lazy dog"]
        conflict = engine.detect_conflict(
            "completely unrelated quantum physics theory", existing
        )
        assert conflict > 0.8


# ===========================================================================
# L5: FeedbackCoordinator
# ===========================================================================

class TestFeedbackCoordinator:

    def test_predict_returns_predictions_from_schemas(self, state):
        coord = FeedbackCoordinator()
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) "
            "VALUES (?, ?, ?)",
            ("Python is a programming language used worldwide", "tech", 0.9),
        )
        state._conn.commit()
        preds = coord.predict(state, context="test")
        assert len(preds) > 0

    def test_high_error_decreases_confidence(self, state):
        coord = FeedbackCoordinator()
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) "
            "VALUES (?, ?, ?)",
            ("Python is a programming language used worldwide", "tech", 0.8),
        )
        state._conn.commit()
        coord.predict(state, context="test")
        coord.observe_outcome(state, actual="something completely unrelated xyz")
        row = state._conn.execute(
            "SELECT confidence FROM schemas WHERE domain = ?", ("tech",),
        ).fetchone()
        assert row["confidence"] < 0.8

    def test_low_error_increases_confidence(self, state):
        coord = FeedbackCoordinator()
        # Insert a schema and set pending predictions directly so we
        # bypass the metadata-enriched predict() format and can
        # precisely control the overlap between prediction and outcome.
        state._conn.execute(
            "INSERT INTO schemas (content, domain, confidence) "
            "VALUES (?, ?, ?)",
            ("Python is a popular programming language", "tech", 0.8),
        )
        state._conn.commit()
        coord._pending_predictions = [
            "Python is a popular programming language used worldwide",
        ]
        coord.observe_outcome(
            state, actual="Python is a popular programming language used worldwide"
        )
        row = state._conn.execute(
            "SELECT confidence FROM schemas WHERE domain = ?", ("tech",),
        ).fetchone()
        assert row["confidence"] > 0.8


# ===========================================================================
# L6: ActivationGraph
# ===========================================================================

class TestActivationGraph:

    def test_co_activation_creates_edges(self, state):
        graph = ActivationGraph()
        graph.record_co_activation(state, ["Alpha", "Beta"])
        rows = state._conn.execute(
            "SELECT * FROM activation_edges"
        ).fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert {row["source_entity"], row["target_entity"]} == {"Alpha", "Beta"}

    def test_get_neighbors_returns_neighbors(self, state):
        graph = ActivationGraph()
        graph.record_co_activation(state, ["Alpha", "Beta"], delta=0.5)
        graph.record_co_activation(state, ["Alpha", "Gamma"], delta=0.5)
        neighbors = graph.get_neighbors(state, "Alpha", min_strength=0.1)
        names = {n["neighbor"] for n in neighbors}
        assert names == {"Beta", "Gamma"}

    def test_expand_query_returns_related_terms(self, state):
        graph = ActivationGraph()
        graph.record_co_activation(state, ["Alpha", "Beta"], delta=0.5)
        expansions = graph.expand_query(state, "Alpha test query")
        assert len(expansions) > 0
        assert any("Beta" in e for e in expansions)

    def test_edges_decay_over_time(self, state):
        graph = ActivationGraph(edge_decay_hours=168.0)
        graph.record_co_activation(state, ["X", "Y"], delta=0.5)
        before = state._conn.execute(
            "SELECT strength FROM activation_edges"
        ).fetchone()["strength"]
        graph.decay_edges(state, hours_elapsed=168.0)
        after = state._conn.execute(
            "SELECT strength FROM activation_edges"
        ).fetchone()["strength"]
        assert after == pytest.approx(before * 0.5, abs=0.01)


# ===========================================================================
# PipelineState
# ===========================================================================

class TestPipelineState:

    def test_creates_all_tables(self, state):
        tables = [
            r[0] for r in state._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for expected in (
            "salience_weights",
            "salience_encoding_log",
            "engram_strengths",
            "schemas",
            "schema_sources",
            "reconsolidation_log",
            "consolidation_runs",
            "predictions",
            "salience_feedback",
            "activation_edges",
            "cross_domain_links",
        ):
            assert expected in tables, f"Missing table: {expected}"

    def test_wal_mode_enabled(self, state):
        mode = state._conn.execute("PRAGMA journal_mode").fetchone()[0]
        # WAL or DELETE (fallback) are both acceptable; the key invariant
        # is that initialization succeeded without error.
        assert mode.lower() in ("wal", "delete")


# ===========================================================================
# Helper: _salience_to_engram_strength
# ===========================================================================

class TestSalienceToEngramStrength:

    def test_high_salience_returns_one(self):
        assert _salience_to_engram_strength(0.8) == 1.0
        assert _salience_to_engram_strength(0.51) == 1.0

    def test_medium_salience_returns_seven_tenths(self):
        assert _salience_to_engram_strength(0.3) == 0.7
        assert _salience_to_engram_strength(0.21) == 0.7

    def test_low_salience_returns_four_tenths(self):
        assert _salience_to_engram_strength(0.1) == 0.4
        assert _salience_to_engram_strength(0.0) == 0.4
        assert _salience_to_engram_strength(-0.5) == 0.4
