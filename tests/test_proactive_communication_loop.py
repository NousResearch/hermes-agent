"""Tests for the Proactive Communication Loop."""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hermes_cli.proactive_communication_loop import (
    ProactiveCommunicationLoop,
    SynthesisResult,
    THRESHOLD_SCORES,
    _build_synthesis_prompt,
    _parse_synthesis_response,
    _get_threshold_score,
    register_threshold,
    BartokGraphContext,
    BartokGraphConnection,
)


# ──────────────────────────────────────────────────────────────────────
# Threshold constants
# ──────────────────────────────────────────────────────────────────────


def test_conservative_is_highest_threshold():
    assert THRESHOLD_SCORES["conservative"] > THRESHOLD_SCORES["balanced"] > THRESHOLD_SCORES["eager"]


def test_all_thresholds_between_zero_and_one():
    for name, score in THRESHOLD_SCORES.items():
        assert 0.0 <= score <= 1.0, f"{name} out of range"


# ──────────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────────


def test_parse_valid_json():
    raw = json.dumps({
        "should_send": True, "message": "Hey, found something.",
        "novelty": 0.8, "relevance": 0.9,
        "connection_type": "temporal_bridge",
        "reasoning": "Completed task.", "candidates": [],
    })
    result = _parse_synthesis_response(raw)
    assert result["should_send"] is True
    assert result["novelty"] == pytest.approx(0.8)
    assert result["connection_type"] == "temporal_bridge"


def test_parse_markdown_fence():
    raw = "```json\n{\"should_send\": false, \"message\": null, \"novelty\": 0.1, \"relevance\": 0.2, \"connection_type\": \"none\", \"reasoning\": \"nothing\", \"candidates\": []}\n```"
    result = _parse_synthesis_response(raw)
    assert result["should_send"] is False


def test_parse_malformed_returns_no_send():
    result = _parse_synthesis_response("not json!!!")
    assert result["should_send"] is False
    assert result["message"] is None
    assert "parse failure" in result["reasoning"]


# ──────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────


def test_prompt_always_includes_graph_connections():
    conn = BartokGraphConnection(
        node_a_content="anomaly detection",
        node_b_content="grid monitoring project",
        connection_type="temporal_bridge",
        strength=0.8,
        days_apart=21,
        explanation="both discuss state transitions in time-series",
    )
    graph_ctx = BartokGraphContext(connections=[conn], provider_name="mock")
    prompt = _build_synthesis_prompt("user: anomaly", "(none)", graph_ctx=graph_ctx)
    assert "KNOWLEDGE GRAPH CONNECTIONS" in prompt
    assert "TEMPORAL_BRIDGE" in prompt
    assert "anomaly detection" in prompt
    assert "grid monitoring" in prompt


def test_prompt_instructs_no_mechanism_disclosure():
    conn = BartokGraphConnection(
        node_a_content="a", node_b_content="b",
        connection_type="cross_domain", strength=0.7,
        days_apart=14, explanation="test",
    )
    graph_ctx = BartokGraphContext(connections=[conn], provider_name="mock")
    prompt = _build_synthesis_prompt("history", "(none)", graph_ctx=graph_ctx)
    assert "Never mention the graph" in prompt or "mechanism" in prompt
    assert "should_send" in prompt


def test_prompt_instructs_silence_as_default():
    conn = BartokGraphConnection(
        node_a_content="a", node_b_content="b",
        connection_type="temporal_bridge", strength=0.6,
        days_apart=7, explanation="test",
    )
    graph_ctx = BartokGraphContext(connections=[conn], provider_name="mock")
    prompt = _build_synthesis_prompt("history", "(none)", graph_ctx=graph_ctx)
    assert "Silence is correct" in prompt or "silence" in prompt.lower()


# ──────────────────────────────────────────────────────────────────────
# Custom threshold registration
# ──────────────────────────────────────────────────────────────────────


def test_register_custom_threshold():
    @register_threshold("pcl_test_always")
    class AlwaysSend:
        def should_send(self, result: SynthesisResult) -> bool:
            return True

    from hermes_cli.proactive_communication_loop import _registered_thresholds
    assert "pcl_test_always" in _registered_thresholds
    result = SynthesisResult(False, None, "", 0.0, 0.0, 0.0)
    assert _registered_thresholds["pcl_test_always"].should_send(result) is True


# ──────────────────────────────────────────────────────────────────────
# Core: no BartokGraph = silence
# ──────────────────────────────────────────────────────────────────────


def test_no_bartokgraph_returns_no_send():
    """Without BartokGraph, the loop must stay silent — it IS the feature."""
    db = MagicMock()
    db.get_meta.return_value = None
    cfg = MagicMock()
    cfg.get.return_value = "conservative"

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=None,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    result = asyncio.run(loop.run_synthesis("session-no-graph"))
    assert result.should_send is False
    assert "BartokGraph" in result.reasoning


def test_no_graph_connections_returns_silence():
    """Empty connections from BartokGraph = stay silent, never fall back to recency."""
    import time as _time
    db = MagicMock()
    db.get_messages.return_value = [
        {"role": "user", "content": "working on anomaly detection today", "timestamp": _time.time() - 3600}
    ]
    db.get_meta.return_value = None
    cfg = MagicMock()
    cfg.get.return_value = "conservative"

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(
        return_value=BartokGraphContext(connections=[], provider_name="mock")
    )

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    result = asyncio.run(loop.run_synthesis("session-empty-graph"))
    assert result.should_send is False
    assert "no connections" in result.reasoning


def test_graph_traversal_failure_stays_silent():
    """Graph error = silence, not a fallback to recency."""
    db = MagicMock()
    db.get_messages.return_value = [{"role": "user", "content": "hello", "timestamp": 1778369330.238517}]
    db.get_meta.return_value = None
    cfg = MagicMock()
    cfg.get.return_value = "conservative"

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(side_effect=RuntimeError("graph exploded"))

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    result = asyncio.run(loop.run_synthesis("session-graph-error"))
    assert result.should_send is False


# ──────────────────────────────────────────────────────────────────────
# Core: rate limit
# ──────────────────────────────────────────────────────────────────────


def test_daily_limit_blocks_send():
    db = MagicMock()
    db.get_messages.return_value = [{"role": "user", "content": "hello", "timestamp": 1778369330.238517}]
    db.get_meta.return_value = '[{"summary": "already sent one", "ts": 1778372930}]'
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 1,
    }.get(k, d)

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(return_value=BartokGraphContext(
        connections=[BartokGraphConnection(
            node_a_content="a", node_b_content="b",
            connection_type="temporal_bridge", strength=0.9,
            days_apart=14, explanation="test",
        )],
        provider_name="mock",
    ))

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    result = asyncio.run(loop.run_synthesis("session-limited"))
    assert result.should_send is False
    assert "daily message limit" in result.reasoning


# ──────────────────────────────────────────────────────────────────────
# Core: temporal bridge triggers send
# ──────────────────────────────────────────────────────────────────────


def test_temporal_bridge_high_score_sends():
    """A high-scoring temporal bridge with model agreement → sends."""
    db = MagicMock()
    db.get_messages.return_value = [
        {"role": "user", "content": "working on anomaly detection in time-series today", "timestamp": 1778369330.238528},
    ]
    db.get_meta.return_value = None
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": True,
        "proactive_communication.bartokgraph.workspace": "~",
    }.get(k, d)

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(return_value=BartokGraphContext(
        connections=[BartokGraphConnection(
            node_a_content="anomaly detection",
            node_b_content="grid monitoring project from 3 weeks ago",
            connection_type="temporal_bridge",
            strength=0.88,
            days_apart=21,
            explanation="same concept appeared 21 days ago",
        )],
        provider_name="mock",
    ))

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    bridge_response = json.dumps({
        "should_send": True,
        "message": "Hey — just connected something. You worked on the same problem three weeks ago in a different context. The approach you found then applies directly here.",
        "novelty": 0.9, "relevance": 0.87,
        "connection_type": "temporal_bridge",
        "reasoning": "High-novelty temporal bridge — user unlikely to have made this connection.",
        "candidates": ["grid monitoring project"],
    })

    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=bridge_response)):
        result = asyncio.run(loop.run_synthesis("session-bridge"))

    assert result.should_send is True
    assert result.connection_type == "temporal_bridge"
    assert result.message is not None
    assert result.novelty_score == pytest.approx(0.9)


def test_low_scoring_connection_blocked():
    """Low novelty/relevance → no send even if model wants to."""
    db = MagicMock()
    db.get_messages.return_value = [{"role": "user", "content": "hello", "timestamp": 1778369330.238517}]
    db.get_meta.return_value = None
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 3,
    }.get(k, d)

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(return_value=BartokGraphContext(
        connections=[BartokGraphConnection(
            node_a_content="a", node_b_content="b",
            connection_type="cross_domain", strength=0.3,
            days_apart=5, explanation="weak link",
        )],
        provider_name="mock",
    ))

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    low_response = json.dumps({
        "should_send": True,
        "message": "Weak connection, maybe worth mentioning.",
        "novelty": 0.2, "relevance": 0.3,
        "connection_type": "cross_domain",
        "reasoning": "low novelty",
        "candidates": [],
    })

    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=low_response)):
        result = asyncio.run(loop.run_synthesis("session-low"))

    # combined = 0.6*0.2 + 0.4*0.3 = 0.24 < 0.75 (conservative)
    assert result.should_send is False


def test_model_veto_respected():
    """If model says should_send=false, respect it even with high scores."""
    db = MagicMock()
    db.get_messages.return_value = [{"role": "user", "content": "something", "timestamp": 1778369330.238535}]
    db.get_meta.return_value = None
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "balanced",
        "proactive_communication.max_per_day": 3,
    }.get(k, d)

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(return_value=BartokGraphContext(
        connections=[BartokGraphConnection(
            node_a_content="a", node_b_content="b",
            connection_type="temporal_bridge", strength=0.9,
            days_apart=30, explanation="strong link",
        )],
        provider_name="mock",
    ))

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    veto_response = json.dumps({
        "should_send": False,
        "message": None,
        "novelty": 0.9, "relevance": 0.9,  # high scores but model says no
        "connection_type": "temporal_bridge",
        "reasoning": "User already discussed this recently — would be repetitive.",
        "candidates": [],
    })

    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=veto_response)):
        result = asyncio.run(loop.run_synthesis("session-veto"))

    assert result.should_send is False


# ──────────────────────────────────────────────────────────────────────
# Exception safety
# ──────────────────────────────────────────────────────────────────────


def test_run_synthesis_never_raises():
    """Any exception anywhere → silent no-send, never propagates."""
    db = MagicMock()
    db.get_proactive_sent.side_effect = RuntimeError("db exploded")
    cfg = MagicMock()
    cfg.get.return_value = "conservative"

    mock_graph = MagicMock()

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    result = asyncio.run(loop.run_synthesis("session-explode"))
    assert result.should_send is False
