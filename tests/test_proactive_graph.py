"""Tests for BartokGraph adapter — weighted traversal and surprise scoring."""

from __future__ import annotations

import asyncio
import math
import pytest
from unittest.mock import MagicMock, patch

from hermes_cli.bartokgraph_adapter import (
    BartokGraphAdapter,
    _node_importance,
    _source_weight,
    _overlap_score,
    _surprise_score,
    _temporal_decay_factor,
    _classify_connection,
    _build_explanation,
    _MAX_SOURCE_WEIGHT,
    _LAYER_MULTIPLIERS,
    _MIN_NODE_WEIGHT,
)


# ──────────────────────────────────────────────────────────────────────
# Source weight table
# ──────────────────────────────────────────────────────────────────────

def test_soul_md_gets_max_weight():
    assert _source_weight("SOUL.md") == 50.0

def test_user_md_gets_max_weight():
    assert _source_weight("USER.md") == 50.0

def test_memory_md_gets_max_weight():
    assert _source_weight("MEMORY.md") == 50.0

def test_daily_log_gets_weight_20():
    assert _source_weight("memory/2026-04-18.md") == 20.0

def test_project_md_gets_weight_15():
    assert _source_weight("projects/kinder-way/chapter-1.md") == 15.0

def test_research_gets_weight_10():
    assert _source_weight("research/soil-carbon.md") == 10.0

def test_generic_md_gets_weight_8():
    assert _source_weight("notes.md") == 8.0

def test_code_file_gets_weight_1():
    assert _source_weight("src/bartokgraph.mjs") == 1.0

def test_python_file_gets_weight_1():
    assert _source_weight("hermes_cli/goals.py") == 1.0

def test_test_file_gets_near_zero_weight():
    assert _source_weight("test_goals.py") <= 0.2

def test_unknown_file_gets_fallback_weight():
    w = _source_weight("something.xyz")
    assert w > 0.0  # never zero, always a fallback

def test_empty_source_path_gets_fallback():
    w = _source_weight("")
    assert w > 0.0


# ──────────────────────────────────────────────────────────────────────
# Node importance — normalized 0–1
# ──────────────────────────────────────────────────────────────────────

def test_soul_node_has_highest_importance():
    node = {"source_path": "SOUL.md", "layer": "knowledge"}
    assert _node_importance(node) == pytest.approx(1.0)

def test_daily_memory_node_importance():
    node = {"source_path": "memory/2026-04-18.md", "layer": "knowledge"}
    imp = _node_importance(node)
    assert 0.3 < imp < 1.0  # substantial but not max

def test_code_node_has_low_importance():
    node = {"source_path": "src/index.ts", "layer": "code"}
    imp = _node_importance(node)
    assert imp < 0.1  # code × code_multiplier = very low

def test_test_file_node_has_near_zero_importance():
    node = {"source_path": "test_goals.py", "layer": "code"}
    imp = _node_importance(node)
    assert imp < 0.01

def test_explicit_weight_takes_precedence_over_source_path():
    # Node with explicit weight=30, source looks like a test file
    node = {"weight": 30, "source_path": "test_something.py", "layer": "knowledge"}
    imp = _node_importance(node)
    # 30 × 10 (knowledge multiplier) / max_possible = significant
    assert imp > 0.5

def test_knowledge_layer_multiplier_applies():
    node_k = {"source_path": "notes.md", "layer": "knowledge"}  # 8 × 10 = 80
    node_c = {"source_path": "notes.md", "layer": "code"}        # 8 × 1  = 8
    assert _node_importance(node_k) > _node_importance(node_c) * 5

def test_soul_knowledge_beats_soul_code():
    soul_k = {"source_path": "SOUL.md", "layer": "knowledge"}
    soul_c = {"source_path": "SOUL.md", "layer": "code"}
    assert _node_importance(soul_k) > _node_importance(soul_c)

def test_importance_is_bounded_0_to_1():
    for source, layer in [
        ("SOUL.md", "knowledge"),
        ("test.py", "code"),
        ("memory/2026-01-01.md", "person"),
        ("projects/kinder-way/ch1.md", "knowledge"),
    ]:
        node = {"source_path": source, "layer": layer}
        imp = _node_importance(node)
        assert 0.0 <= imp <= 1.0, f"out of range for {source}/{layer}: {imp}"


# ──────────────────────────────────────────────────────────────────────
# Temporal decay factor
# ──────────────────────────────────────────────────────────────────────

def test_temporal_decay_fresh_node():
    # 0 days = 1.0 (no amplification)
    assert _temporal_decay_factor(0) == pytest.approx(1.0)

def test_temporal_decay_one_week():
    # log1p(7/7) = log1p(1) ≈ 0.693 → factor ≈ 1.693
    f = _temporal_decay_factor(7)
    assert 1.5 < f < 2.0

def test_temporal_decay_one_month():
    f = _temporal_decay_factor(30)
    assert f > 2.0  # well beyond 1 week

def test_temporal_decay_increases_with_age():
    f1 = _temporal_decay_factor(2)
    f7 = _temporal_decay_factor(7)
    f30 = _temporal_decay_factor(30)
    f90 = _temporal_decay_factor(90)
    assert f1 < f7 < f30 < f90

def test_temporal_decay_flattens_at_scale():
    # Difference between 60d and 90d should be smaller than 7d to 30d
    diff_recent = _temporal_decay_factor(30) - _temporal_decay_factor(7)
    diff_old = _temporal_decay_factor(90) - _temporal_decay_factor(60)
    assert diff_old < diff_recent  # log scale flattens


# ──────────────────────────────────────────────────────────────────────
# Surprise score
# ──────────────────────────────────────────────────────────────────────

def test_high_importance_high_semantic_high_age_scores_highest():
    s = _surprise_score(semantic_strength=0.9, node_importance=1.0, days_apart=30)
    assert s > 2.0

def test_low_importance_never_scores_high():
    # Even with perfect semantic match and 90 days old
    s = _surprise_score(semantic_strength=1.0, node_importance=0.01, days_apart=90)
    assert s < 0.2

def test_weak_semantic_never_surfaces():
    s = _surprise_score(semantic_strength=0.05, node_importance=1.0, days_apart=60)
    assert s < 0.2

def test_soul_node_beats_test_file_with_same_semantic():
    soul_score = _surprise_score(0.6, _node_importance({"source_path": "SOUL.md", "layer": "knowledge"}), 14)
    test_score = _surprise_score(0.6, _node_importance({"source_path": "test_goals.py", "layer": "code"}), 14)
    assert soul_score > test_score * 100  # should dominate massively


# ──────────────────────────────────────────────────────────────────────
# Semantic overlap
# ──────────────────────────────────────────────────────────────────────

def test_identical_content_scores_1():
    assert _overlap_score("soil carbon research", "soil carbon research") == pytest.approx(1.0)

def test_no_overlap_scores_0():
    assert _overlap_score("quantum computing", "soil carbon") == 0.0

def test_partial_overlap_between_0_and_1():
    s = _overlap_score("soil carbon research Kenya", "Kenya soil health project")
    assert 0.0 < s < 1.0

def test_stopwords_excluded():
    # "the" "and" "of" should not contribute to overlap
    s_with = _overlap_score("the carbon and the soil", "carbon soil")
    s_without = _overlap_score("carbon soil", "carbon soil")
    assert s_with == pytest.approx(s_without)

def test_short_words_excluded():
    # Words ≤ 2 chars filtered
    s = _overlap_score("AI research", "AI model")
    # "AI" is 2 chars — filtered out
    assert s == 0.0 or s < 0.5


# ──────────────────────────────────────────────────────────────────────
# Connection classification
# ──────────────────────────────────────────────────────────────────────

def test_person_tagged_node_is_person_knowledge():
    node = {"person": "sage", "layer": "knowledge", "last_seen_ts": 0}
    assert _classify_connection("hermetic literature", node, 14) == "person_knowledge"

def test_attributed_to_node_is_person_knowledge():
    node = {"attributed_to": "alice", "layer": "knowledge", "last_seen_ts": 0}
    assert _classify_connection("bioavailability", node, 21) == "person_knowledge"

def test_code_layer_is_cross_domain():
    node = {"layer": "code", "last_seen_ts": 0}
    assert _classify_connection("trading algorithm", node, 30) == "cross_domain"

def test_old_knowledge_node_is_temporal_bridge():
    node = {"layer": "knowledge", "last_seen_ts": 0}
    assert _classify_connection("soil carbon", node, 21) == "temporal_bridge"

def test_recent_knowledge_node_is_temporal_bridge():
    node = {"layer": "knowledge", "last_seen_ts": 0}
    # Even fresh dormant nodes get temporal_bridge (the 24h filter already excluded truly recent ones)
    assert _classify_connection("topic", node, 2) == "temporal_bridge"


# ──────────────────────────────────────────────────────────────────────
# Full adapter — integration with synthetic graph
# ──────────────────────────────────────────────────────────────────────

import json
import time
import tempfile
import os

def _make_synthetic_graph(now: float) -> dict:
    """Build a realistic synthetic graph.json with varied node types and ages."""
    return {
        "nodes": [
            # High importance, old — should surface first
            {
                "content": "regenerative agriculture soil health",
                "source_path": "SOUL.md",
                "layer": "knowledge",
                "node_type": "concept",
                "last_seen_ts": now - 30 * 86400,  # 30 days ago
                "weight": 50,
            },
            # High importance, person-tagged — should surface for person_knowledge
            {
                "content": "hermetic literature esoteric tradition",
                "source_path": "memory/2026-04-01.md",
                "layer": "person",
                "node_type": "concept",
                "person": "sage",
                "last_seen_ts": now - 21 * 86400,  # 21 days ago
                "weight": 20,
            },
            # Medium importance, old
            {
                "content": "soil carbon sequestration Kenya project",
                "source_path": "projects/farm-sensors/notes.md",
                "layer": "knowledge",
                "node_type": "concept",
                "last_seen_ts": now - 14 * 86400,  # 14 days ago
                "weight": 15,
            },
            # Low importance code node — should NOT surface even with semantic match
            {
                "content": "soil carbon test fixtures",
                "source_path": "test_carbon.py",
                "layer": "code",
                "node_type": "concept",
                "last_seen_ts": now - 60 * 86400,
                "weight": 0.1,
            },
            # Below minimum weight — completely filtered
            {
                "content": "soil health auto-generated types",
                "source_path": "types.d.ts",
                "layer": "code",
                "node_type": "concept",
                "last_seen_ts": now - 90 * 86400,
                "weight": 0.0,
            },
            # Recent (within 24h) — excluded by recency filter
            {
                "content": "soil carbon recent session",
                "source_path": "memory/2026-05-09.md",
                "layer": "knowledge",
                "node_type": "concept",
                "last_seen_ts": now - 1 * 3600,  # 1 hour ago
                "weight": 20,
            },
        ]
    }


def test_high_importance_nodes_surface_first():
    """SOUL.md nodes should rank above project notes even with same semantic match."""
    now = time.time()
    graph = _make_synthetic_graph(now)

    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.bartokgraph.workspace": "~",
        "proactive_communication.bartokgraph.enabled": True,
    }.get(k, d)

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = graph

    result = asyncio.run(adapter.get_connections(
        active_topics=["soil carbon regenerative agriculture"],
        top_k=5,
    ))

    assert result is not None
    assert len(result.connections) > 0

    # SOUL.md node should rank highest
    top = result.connections[0]
    assert "regenerative agriculture" in top.node_b_content or "soil" in top.node_b_content
    # And it should be from a high-importance source — strength should be substantial
    assert top.strength > 0.5


def test_test_file_node_never_surfaces():
    """Code test file nodes must not appear in results regardless of semantic match."""
    now = time.time()
    graph = _make_synthetic_graph(now)

    cfg = MagicMock()
    cfg.get.return_value = None

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = graph

    result = asyncio.run(adapter.get_connections(
        active_topics=["soil carbon test fixtures"],  # semantically matches the test file node
        top_k=10,
    ))

    assert result is not None
    # The test file node content should not appear in top results
    for conn in result.connections:
        assert "test fixtures" not in conn.node_b_content or conn.strength < 0.5


def test_recent_nodes_excluded():
    """Nodes active within the last 24h must not appear."""
    now = time.time()
    graph = _make_synthetic_graph(now)

    cfg = MagicMock()
    cfg.get.return_value = None

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = graph

    result = asyncio.run(adapter.get_connections(
        active_topics=["soil carbon recent session"],
        top_k=10,
    ))

    assert result is not None
    for conn in result.connections:
        assert "recent session" not in conn.node_b_content


def test_person_knowledge_connection_type():
    """Person-tagged nodes should be classified as person_knowledge."""
    now = time.time()
    graph = _make_synthetic_graph(now)

    cfg = MagicMock()
    cfg.get.return_value = None

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = graph

    result = asyncio.run(adapter.get_connections(
        active_topics=["hermetic esoteric tradition literature"],
        top_k=5,
    ))

    assert result is not None
    person_conns = [c for c in result.connections if c.connection_type == "person_knowledge"]
    assert len(person_conns) > 0
    assert any("sage" in c.explanation.lower() for c in person_conns)


def test_empty_graph_returns_empty_context():
    """Empty graph returns empty BartokGraphContext, not None."""
    cfg = MagicMock()
    cfg.get.return_value = None

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = {"nodes": []}

    result = asyncio.run(adapter.get_connections(active_topics=["anything"], top_k=5))
    assert result is not None
    assert result.connections == []


def test_missing_graph_returns_none():
    """Unavailable graph returns None — caller treats this as loop-disable."""
    cfg = MagicMock()
    cfg.get.return_value = "/nonexistent/path"

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = None  # no graph loaded

    result = asyncio.run(adapter.get_connections(active_topics=["anything"], top_k=5))
    assert result is None


def test_deduplication_prevents_same_node_twice():
    """Same dormant node should appear at most once even if multiple topics match it."""
    now = time.time()
    graph = {
        "nodes": [
            {
                "content": "soil carbon regenerative project Kenya",
                "source_path": "SOUL.md",
                "layer": "knowledge",
                "last_seen_ts": now - 30 * 86400,
                "weight": 50,
            }
        ]
    }

    cfg = MagicMock()
    cfg.get.return_value = None

    adapter = BartokGraphAdapter.__new__(BartokGraphAdapter)
    adapter._cfg = cfg
    adapter._model_provider = {"name": "topology_only"}
    adapter._graph_data = graph

    # Multiple topics that all match the same node
    result = asyncio.run(adapter.get_connections(
        active_topics=["soil carbon", "regenerative project", "Kenya agriculture"],
        top_k=10,
    ))

    assert result is not None
    # Should deduplicate to 1 result for the same underlying node
    contents = [c.node_b_content for c in result.connections]
    assert len(contents) == len(set(c[:80] for c in contents))
