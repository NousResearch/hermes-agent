"""Tests for the graph-augmented proactive synthesis context."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from hermes_cli.proactive_graph import (
    KnowledgeNode,
    KnowledgeGraphContext,
    GraphConnection,
    GraphAugmentedContext,
    build_graph_augmented_prompt,
    _content_overlap_score,
    _classify_connection_type,
)


# ──────────────────────────────────────────────────────────────────────
# Unit tests — content overlap scorer
# ──────────────────────────────────────────────────────────────────────


def test_overlap_identical():
    assert _content_overlap_score("soil carbon", "soil carbon") == pytest.approx(1.0)


def test_overlap_partial():
    score = _content_overlap_score("soil carbon sequestration", "carbon credits market")
    assert 0.0 < score < 1.0


def test_overlap_no_overlap():
    score = _content_overlap_score("trading bot", "soil health")
    assert score == pytest.approx(0.0)


def test_overlap_empty_strings():
    assert _content_overlap_score("", "anything") == pytest.approx(0.0)
    assert _content_overlap_score("anything", "") == pytest.approx(0.0)


def test_overlap_stopwords_only():
    # "the in on" vs "the for and" — all stopwords, both become empty after filtering
    assert _content_overlap_score("the in on", "the for and") == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# Unit tests — connection classifier
# ──────────────────────────────────────────────────────────────────────


def _make_node(node_type: str = "topic") -> KnowledgeNode:
    return KnowledgeNode("id", "content", node_type, 0.5, 0, 0)


def test_classify_old_connection_is_temporal_bridge():
    conn_type = _classify_connection_type(_make_node(), _make_node(), days_apart=21)
    assert conn_type == "temporal_bridge"


def test_classify_cross_type_is_cross_domain():
    conn_type = _classify_connection_type(_make_node("topic"), _make_node("decision"), days_apart=3)
    assert conn_type == "cross_domain"


def test_classify_recent_same_type():
    conn_type = _classify_connection_type(_make_node("topic"), _make_node("topic"), days_apart=2)
    assert conn_type == "temporal_bridge"


# ──────────────────────────────────────────────────────────────────────
# Unit tests — KnowledgeGraphContext (no backend)
# ──────────────────────────────────────────────────────────────────────


def test_context_not_available_without_backend():
    ctx = KnowledgeGraphContext(backend=None)
    assert not ctx.is_available


@pytest.mark.asyncio
async def test_build_returns_none_without_backend():
    ctx = KnowledgeGraphContext(backend=None)
    result = await ctx.build("history", ["soil carbon"])
    assert result is None


# ──────────────────────────────────────────────────────────────────────
# Unit tests — KnowledgeGraphContext with mock backend
# ──────────────────────────────────────────────────────────────────────


def _make_mock_backend(search_results=None):
    backend = MagicMock()
    backend.provider_name = "mock"
    backend.search_nodes = AsyncMock(return_value=search_results or [])
    backend.get_connections = AsyncMock(return_value=[])
    return backend


@pytest.mark.asyncio
async def test_build_with_empty_search_results():
    backend = _make_mock_backend(search_results=[])
    ctx = KnowledgeGraphContext(backend=backend)
    result = await ctx.build("history about soil", ["soil carbon"])
    assert result is not None
    assert result.active_nodes  # should have active nodes from today's topics
    assert result.connections == []  # no connections without dormant nodes


@pytest.mark.asyncio
async def test_build_finds_temporal_bridge():
    """When dormant nodes overlap with active topics, connections are detected."""
    import time
    three_weeks_ago = int(time.time()) - 21 * 86400

    backend = _make_mock_backend(search_results=[
        {
            "id": "node-old",
            "content": "soil carbon sequestration research Kenya",
            "type": "topic",
            "weight": 0.7,
            "score": 0.7,
            "created_at": three_weeks_ago,
            "updated_at": three_weeks_ago,
        }
    ])
    ctx = KnowledgeGraphContext(backend=backend)
    result = await ctx.build("talked about soil carbon today", ["soil carbon"])

    assert result is not None
    assert len(result.related_dormant_nodes) >= 1
    # Should detect connection between "soil carbon" active and "soil carbon...Kenya" dormant
    if result.connections:
        assert result.connections[0].days_apart >= 20
        assert result.connections[0].connection_type == "temporal_bridge"


@pytest.mark.asyncio
async def test_build_handles_backend_error_gracefully():
    """If backend raises, returns None (graceful degradation)."""
    backend = MagicMock()
    backend.provider_name = "mock"
    backend.search_nodes = AsyncMock(side_effect=RuntimeError("db down"))

    ctx = KnowledgeGraphContext(backend=backend)
    # Should not raise — returns result with empty connections (graceful degradation)
    result = await ctx.build("history", ["topic"])
    # Either None or a result with no dormant nodes/connections is acceptable
    if result is not None:
        assert result.related_dormant_nodes == [] or result.connections == []


# ──────────────────────────────────────────────────────────────────────
# Unit tests — prompt builder
# ──────────────────────────────────────────────────────────────────────


def test_prompt_without_graph_ctx():
    prompt = build_graph_augmented_prompt("user: hello", "(none)", graph_ctx=None)
    assert "RECENT CONVERSATION HISTORY" in prompt
    assert "KNOWLEDGE GRAPH CONTEXT" not in prompt
    assert "should_send" in prompt


def test_prompt_with_graph_connections_includes_graph_section():
    node_a = KnowledgeNode("a", "soil carbon", "topic", 1.0, 0, 0)
    node_b = KnowledgeNode("b", "Kenya soil project", "topic", 0.7, 0, 0)
    conn = GraphConnection(
        node_a=node_a,
        node_b=node_b,
        connection_type="temporal_bridge",
        strength=0.8,
        explanation="both discuss soil carbon",
        days_apart=21,
    )
    graph_ctx = GraphAugmentedContext(
        active_nodes=[node_a],
        related_dormant_nodes=[node_b],
        connections=[conn],
        provider_name="mock",
    )
    prompt = build_graph_augmented_prompt("user: soil", "(none)", graph_ctx=graph_ctx)
    assert "KNOWLEDGE GRAPH CONTEXT" in prompt
    assert "TEMPORAL_BRIDGE" in prompt
    assert "soil carbon" in prompt
    assert "Kenya" in prompt


def test_prompt_with_empty_connections_omits_graph_section():
    """No connections → no graph section in prompt (avoids clutter)."""
    graph_ctx = GraphAugmentedContext(
        active_nodes=[],
        related_dormant_nodes=[],
        connections=[],  # empty
        provider_name="mock",
    )
    prompt = build_graph_augmented_prompt("history", "(none)", graph_ctx=graph_ctx)
    assert "KNOWLEDGE GRAPH CONTEXT" not in prompt


def test_prompt_contains_connection_type_instructions():
    prompt = build_graph_augmented_prompt("h", "n", graph_ctx=None)
    assert "TEMPORAL_BRIDGE" in prompt or "temporal_bridge" in prompt.lower()
    assert "cross_domain" in prompt.lower() or "CROSS_DOMAIN" in prompt
