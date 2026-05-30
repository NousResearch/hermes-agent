"""Integration tests for the 9 layer-to-layer connections in MemoryPipeline.

Each test creates a fresh MemoryPipeline with all features enabled and
verifies data flows between adjacent organic memory layers:

    L1 -> L2:  salience scoring creates engrams with appropriate strength
    L2 -> L3:  engram strength determines consolidation priority
    L3 -> L5:  consolidated schemas appear in predictions
    L5 -> L4:  prediction error triggers reconsolidation
    L4 -> L1:  conflict resolution updates salience weights
    L6 -> retrieval: co-activation edges cause query expansion
    L7 -> L3:  episode close triggers consolidation
    L8 -> L3:  dream cycle boosts schema confidence
    L8 -> L5:  dream hypotheses appear in pending predictions
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import time
import threading
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.memory_pipeline import (
    MemoryPipeline,
    PipelineState,
)

# ---------------------------------------------------------------------------
# Plugin availability
# ---------------------------------------------------------------------------

_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins" / "memory" / "holographic"

_DREAMING_AVAILABLE = (_PLUGIN_DIR / "dreaming.py").is_file()
_EPISODIC_AVAILABLE = (_PLUGIN_DIR / "episodic.py").is_file()

_dreaming_mod = None
_episodic_mod = None

if _DREAMING_AVAILABLE:
    try:
        _spec = importlib.util.spec_from_file_location(
            "holographic_dreaming", str(_PLUGIN_DIR / "dreaming.py"))
        _dreaming_mod = importlib.util.module_from_spec(_spec)
        sys.modules["holographic_dreaming"] = _dreaming_mod
        _spec.loader.exec_module(_dreaming_mod)
    except Exception:
        _DREAMING_AVAILABLE = False

if _EPISODIC_AVAILABLE:
    try:
        _spec = importlib.util.spec_from_file_location(
            "holographic_episodic", str(_PLUGIN_DIR / "episodic.py"))
        _episodic_mod = importlib.util.module_from_spec(_spec)
        sys.modules["holographic_episodic"] = _episodic_mod
        _spec.loader.exec_module(_episodic_mod)
    except Exception:
        _EPISODIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_config(db_path: str) -> dict:
    """Config dict that enables every organic layer."""
    return {
        "enabled": True,
        "db_path": db_path,
        "salience": {"enabled": True, "novelty_window": 50},
        "silent_engram": {
            "enabled": True,
            "half_life_hours": 720.0,
            "emotion_modulated_decay_enabled": False,
            "emotion_decay_multiplier": 2.0,
        },
        "consolidation": {"enabled": True, "min_facts_for_consolidation": 5},
        "reconsolidation": {
            "enabled": True,
            "prediction_error_threshold": 0.3,
            "semantic_conflict_enabled": False,
        },
        "feedback": {"enabled": True},
        "activation": {
            "enabled": True,
            "edge_decay_hours": 168.0,
            "pagerank_enabled": False,
        },
        "episodic": {"enabled": True},
        "dreaming": {"enabled": True, "cooldown_hours": 0.0},
        "sleep": {"enabled": False},
    }


def _make_pipeline(tmp_path: Path, *, session_id: str = "integration-test"):
    """Create and initialize a MemoryPipeline with all features.

    Returns (pipeline, conn).  Caller must call pipeline.shutdown().
    """
    db_path = str(tmp_path / "pipeline_state.db")
    config = _full_config(db_path)
    pipeline = MemoryPipeline(config)
    pipeline.initialize(session_id)
    conn = pipeline._state._conn

    # Inject DreamEngine if plugin loaded but was not wired by initialize
    if _DREAMING_AVAILABLE and _dreaming_mod and pipeline._dreaming is None:
        pipeline._dreaming = _dreaming_mod.DreamEngine(
            conn, pipeline._state._lock, cooldown_hours=0.0)
        pipeline._dreaming.init_tables()

    # Inject EpisodicTimeline if plugin loaded but was not wired by initialize
    if _EPISODIC_AVAILABLE and _episodic_mod and pipeline._episodic is None:
        pipeline._episodic = _episodic_mod.EpisodicTimeline(
            conn, pipeline._state._lock)
        pipeline._episodic.init_tables()

    return pipeline, conn


# ---------------------------------------------------------------------------
# L1 -> L2:  High-salience message creates engram with strength > 0.5
# ---------------------------------------------------------------------------

class TestL1ToL2SalienceEngram:
    """pre_sync with a high-salience message creates an engram whose
    strength exceeds 0.5."""

    def test_high_salience_creates_strong_engram(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # This message hits CRITICAL(0.5) + URGENT(0.5) + "!!"(0.6)
            #   + "remember"(0.8) + "important"(0.4)  => importance=0.8
            # Raw = 0.25*0.6 + 0.30*1.0 + 0.30*0.8 + 0.15*1.0 = 0.65
            # Overall = 0.65 > 0.5  =>  engram init_str = 1.0
            msg = (
                "CRITICAL BUG!! Remember the important decision about "
                "the urgent emergency deployment"
            )
            meta = pipeline.pre_sync(user=msg, asst="ok")

            assert meta is not None
            assert meta["salience_overall"] > 0.5, (
                f"Expected salience > 0.5, got {meta['salience_overall']}"
            )

            # Verify engram was created with strength > 0.5
            ref = sha256(msg.encode()).hexdigest()[:16]
            row = conn.execute(
                "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
                (ref,),
            ).fetchone()
            assert row is not None, "Engram row not created"
            assert row["strength"] > 0.5, (
                f"Expected strength > 0.5, got {row['strength']}"
            )
        finally:
            pipeline.shutdown()

    def test_low_salience_creates_weak_engram(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            msg = "hi"
            meta = pipeline.pre_sync(user=msg, asst="hello")

            assert meta is not None
            # Trivial messages produce low salience
            ref = sha256(msg.encode()).hexdigest()[:16]
            row = conn.execute(
                "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
                (ref,),
            ).fetchone()
            # May or may not exist; if it does, strength should be modest
            if row is not None:
                assert row["strength"] <= 0.7
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L2 -> L3:  Weaker engrams are consolidated first
# ---------------------------------------------------------------------------

class TestL2ToL3EngramConsolidation:
    """Facts with weaker engram strengths are consolidated before stronger
    ones, mimicking sleep's preferential replay of fragile memories."""

    def test_weaker_engrams_consolidated_first(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            facts = [
                {"content": f"Alpha domain knowledge fact number {i}",
                 "domain": "alpha"}
                for i in range(6)
            ]

            # Pre-seed engram strengths: fact 0 is strongest, fact 5 weakest
            for i, fact in enumerate(facts):
                ref = sha256(fact["content"].encode()).hexdigest()[:16]
                strength = 1.0 - (i * 0.15)  # 1.0, 0.85, 0.70, ...
                conn.execute(
                    "INSERT INTO engram_strengths "
                    "(memory_ref, provider, strength) VALUES (?, 'test', ?)",
                    (ref, strength),
                )
            conn.commit()

            result = pipeline._consolidation.consolidate(
                pipeline._state, facts=facts)

            # Weakest facts should have been processed first, creating schemas
            # in order of ascending engram strength
            assert result["schemas_created"] > 0, "No schemas were created"

            schemas = conn.execute(
                "SELECT content, domain FROM schemas ORDER BY schema_id"
            ).fetchall()

            # The first schema should come from the weakest engram (fact 5)
            first_schema_content = schemas[0]["content"]
            weakest_fact = facts[-1]["content"]
            assert first_schema_content == weakest_fact, (
                "Weakest engram fact should be consolidated first"
            )
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L3 -> L5:  Schema with confidence > 0.3 appears in predictions
# ---------------------------------------------------------------------------

class TestL3To5SchemaPredictions:
    """Consolidated schemas with sufficient confidence surface in the
    predictive feedback loop."""

    def test_high_confidence_schema_appears_in_predictions(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            conn.execute(
                "INSERT INTO schemas (content, domain, confidence) "
                "VALUES (?, ?, ?)",
                ("User prefers Python for backend services",
                 "preferences", 0.8),
            )
            conn.commit()

            predictions = pipeline._feedback.predict(
                pipeline._state, "what language to use")

            assert len(predictions) > 0, "No predictions generated"

            # The schema content should appear in at least one prediction
            combined = " ".join(predictions)
            assert "Python" in combined, (
                f"Schema content missing from predictions: {combined}"
            )

            # Verify predictions table was populated
            rows = conn.execute("SELECT * FROM predictions").fetchall()
            assert len(rows) > 0, "Predictions table is empty"
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L5 -> L4:  Prediction error > 0.5 triggers reconsolidation_log entry
# ---------------------------------------------------------------------------

class TestL5ToL4PredictionReconsolidation:
    """When the predictive model observes a high-error outcome, the
    reconsolidation engine is invoked and schema confidence is decreased."""

    def test_high_prediction_error_triggers_reconsolidation(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # Seed a schema so predict() has data to work with
            conn.execute(
                "INSERT INTO schemas (content, domain, confidence) "
                "VALUES (?, ?, ?)",
                ("The system prefers Python for backend services",
                 "preferences", 0.8),
            )
            conn.commit()

            # Generate predictions from the schema
            pipeline._feedback.predict(pipeline._state, "test context")
            with pipeline._feedback._lock:
                assert len(pipeline._feedback._pending_predictions) > 0, (
                    "No pending predictions after predict()"
                )

            # Record pre-observation schema confidence
            pre_conf = conn.execute(
                "SELECT confidence FROM schemas WHERE domain = 'preferences'"
            ).fetchone()["confidence"]

            # Observe a completely unrelated outcome -> high error
            error = pipeline._feedback.observe_outcome(
                pipeline._state,
                actual="Quantum entanglement in photosynthesis research",
            )

            assert error > 0.5, (
                f"Expected prediction error > 0.5, got {error}"
            )

            # High error decreases schema confidence (L5 -> L4 effect)
            post_conf = conn.execute(
                "SELECT confidence FROM schemas WHERE domain = 'preferences'"
            ).fetchone()["confidence"]
            assert post_conf < pre_conf, (
                f"Schema confidence should decrease: {pre_conf} -> {post_conf}"
            )

            # Pending predictions cleared after observe_outcome
            with pipeline._feedback._lock:
                assert len(pipeline._feedback._pending_predictions) == 0
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L4 -> L1:  Conflict resolution updates salience_weights table
# ---------------------------------------------------------------------------

class TestL4ToL1SalienceWeights:
    """Conflict resolution events adjust the learned salience signal
    weights in the salience_weights table."""

    def test_conflict_resolution_updates_salience_weights(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # Seed initial weights
            for sig in ("emotion", "novelty", "importance"):
                conn.execute(
                    "INSERT INTO salience_weights "
                    "(signal_type, weight, sample_count, success_count) "
                    "VALUES (?, 0.5, 0, 0)",
                    (sig,),
                )
            conn.commit()

            # High error -> weights nudged DOWN
            pipeline._update_salience_weights(error_score=0.8)

            for sig in ("emotion", "novelty", "importance"):
                row = conn.execute(
                    "SELECT weight FROM salience_weights "
                    "WHERE signal_type = ?",
                    (sig,),
                ).fetchone()
                assert row is not None
                assert row["weight"] < 0.5, (
                    f"{sig}: expected weight < 0.5 after high error, "
                    f"got {row['weight']}"
                )

            # Low error -> weights nudged UP
            pipeline._update_salience_weights(error_score=0.1)

            for sig in ("emotion", "novelty", "importance"):
                row = conn.execute(
                    "SELECT weight FROM salience_weights "
                    "WHERE signal_type = ?",
                    (sig,),
                ).fetchone()
                assert row["weight"] > 0.48, (
                    f"{sig}: expected weight > 0.48 after recovery, "
                    f"got {row['weight']}"
                )
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L6 -> retrieval:  Co-activation edges cause expand_query to return terms
# ---------------------------------------------------------------------------

class TestL6ActivationExpansion:
    """When entities have co-activation edges in the activation graph,
    expand_query returns additional context terms from spreading activation."""

    def test_coactivation_edges_expand_query(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # Manually insert co-activation edges
            edges = [
                ("Alice", "Bob", 0.6),
                ("Alice", "Paris", 0.4),
            ]
            for src, tgt, strength in edges:
                conn.execute(
                    "INSERT INTO activation_edges "
                    "(source_entity, target_entity, strength, "
                    " co_activation_count) "
                    "VALUES (?, ?, ?, 1)",
                    (src, tgt, strength),
                )
            conn.commit()

            expansions = pipeline._activation.expand_query(
                pipeline._state, "Alice went to Paris")

            assert len(expansions) > 0, "No expansions returned"

            combined = " ".join(expansions)
            assert "Alice" in combined, (
                f"'Alice' missing from expansions: {combined}"
            )
            # At least one neighbor should appear
            assert "Bob" in combined or "Paris" in combined, (
                f"Neighbor entities missing from expansions: {combined}"
            )
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L7 -> L3:  Episode close triggers consolidation
# ---------------------------------------------------------------------------

class TestL7ToL3EpisodeConsolidation:
    """Closing an episodic episode triggers a consolidation run that
    processes the episode's facts into schemas."""

    def test_episode_close_triggers_consolidation(self, tmp_path):
        if not _EPISODIC_AVAILABLE:
            pytest.skip("EpisodicTimeline plugin not available")

        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # Seed enough messages for both the main consolidation
            # (needs >= 5 facts with len > 20) and the episodic
            # mini-consolidation (needs >= 2 facts with len > 15)
            messages = [
                {"content": f"Session message {i} about database design "
                            f"and architecture patterns for microservices"}
                for i in range(7)
            ]

            pipeline.post_session_end(messages)

            # Verify at least one consolidation run was logged
            row = conn.execute(
                "SELECT memories_processed, schemas_created "
                "FROM consolidation_runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
            assert row is not None, "No consolidation_runs entry"
            assert row["memories_processed"] > 0
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L8 -> L3:  Dream cycle boosts schema confidence
# ---------------------------------------------------------------------------

class TestL8ToL3DreamSchemaBoost:
    """After a dream cycle, schemas with confidence > 0.5 receive a
    confidence boost via the post-processing step."""

    @pytest.mark.skipif(
        not _DREAMING_AVAILABLE,
        reason="DreamEngine plugin not available",
    )
    def test_dream_cycle_boosts_schema_confidence(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # Seed a schema with confidence just above 0.5
            conn.execute(
                "INSERT INTO schemas (content, domain, confidence) "
                "VALUES (?, ?, ?)",
                ("User authentication requires OAuth2 tokens",
                 "security", 0.55),
            )
            conn.commit()

            # Run dream post-processing directly (avoids daemon thread)
            pipeline._run_dream_postprocessing()

            row = conn.execute(
                "SELECT confidence FROM schemas WHERE domain = 'security'"
            ).fetchone()
            assert row is not None
            assert row["confidence"] > 0.55, (
                f"Expected confidence > 0.55 after dream boost, "
                f"got {row['confidence']}"
            )
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# L8 -> L5:  Dream hypotheses appear in pending_predictions
# ---------------------------------------------------------------------------

class TestL8ToL5DreamPredictions:
    """Dream-generated hypotheses are injected into the FeedbackCoordinator's
    pending_predictions list and persisted in the predictions table."""

    @pytest.mark.skipif(
        not _DREAMING_AVAILABLE,
        reason="DreamEngine plugin not available",
    )
    def test_dream_hypotheses_appear_in_pending_predictions(self, tmp_path):
        pipeline, conn = _make_pipeline(tmp_path)
        try:
            # Clear any predictions from initialization
            conn.execute("DELETE FROM predictions")
            with pipeline._feedback._lock:
                pipeline._feedback._pending_predictions.clear()
            conn.commit()

            # Seed a schema and a dream hypothesis directly
            conn.execute(
                "INSERT INTO schemas (content, domain, confidence) "
                "VALUES (?, ?, ?)",
                ("Performance tuning requires caching strategies",
                 "performance", 0.8),
            )
            conn.execute(
                "INSERT INTO dream_hypotheses "
                "(source_schema_id, content, confidence) "
                "VALUES (?, ?, ?)",
                (1, "Caching may reduce latency by 40%", 0.3),
            )
            conn.commit()

            # Mock dream_cycle to report hypotheses were generated,
            # while keeping the real get_hypotheses that reads the DB.
            fake_result = _dreaming_mod.DreamResult(
                mode="auto", hypotheses=1)
            with patch.object(
                pipeline._dreaming, "dream_cycle",
                return_value=fake_result,
            ):
                pipeline._run_dream_postprocessing()

            # Verify the hypothesis was added to pending predictions
            with pipeline._feedback._lock:
                pending = list(pipeline._feedback._pending_predictions)

            dream_preds = [p for p in pending if p.startswith("Dream:")]
            assert len(dream_preds) > 0, (
                f"No dream predictions in pending list: {pending}"
            )
            assert "Caching may reduce latency" in dream_preds[0], (
                f"Hypothesis content missing from prediction: {dream_preds[0]}"
            )
        finally:
            pipeline.shutdown()
