"""Integration smoke tests for Proactive Communication Loop + BartokGraph context."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from hermes_cli.proactive_communication_loop import (
    BartokGraphConnection,
    BartokGraphContext,
    ProactiveCommunicationLoop,
)

VALID_CONNECTION_TYPES = frozenset({"none", "temporal_bridge", "cross_domain", "person_knowledge"})


def _make_graph_json_five_nodes_30d() -> dict:
    """Synthetic BartokGraph document: 5 nodes spread across ~30 days (all dormant)."""
    now = int(time.time())
    day = 86400
    return {
        "nodes": [
            {
                "content": "alpha research thread",
                "weight": 1.0,
                "last_seen_ts": now - 30 * day,
                "node_type": "topic",
            },
            {
                "content": "beta shipping milestone",
                "weight": 1.0,
                "last_seen_ts": now - 23 * day,
                "node_type": "topic",
            },
            {
                "content": "gamma soil carbon",
                "weight": 0.9,
                "last_seen_ts": now - 16 * day,
                "node_type": "research",
            },
            {
                "content": "delta alice introduced kenya",
                "weight": 0.85,
                "last_seen_ts": now - 9 * day,
                "node_type": "person_link",
            },
            {
                "content": "epsilon hmm regime",
                "weight": 0.8,
                "last_seen_ts": now - 2 * day,
                "node_type": "topic",
            },
        ],
        "edges": [],
    }


class _SmokeSessionDB:
    """Minimal session DB for smoke tests."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.proactive_rows: list[dict] = []

    def get_messages_since(self, session_id: str, cutoff: float) -> list[dict]:
        return [m for m in self.messages if m.get("ts", time.time()) >= cutoff]

    def get_proactive_sent(self, session_id: str, since_hours: int = 24) -> list[dict]:
        return list(self.proactive_rows)

    def record_proactive_sent(self, session_id: str, payload: dict) -> None:
        self.proactive_rows.append({"session_id": session_id, **payload})


def test_smoke_high_score_sends_natural_message_without_branding():
    db = _SmokeSessionDB()
    base = time.time() - 3600
    for i in range(10):
        db.messages.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: discussing soil carbon and regime detection work.",
            "ts": base + i * 60,
        })

    graph_doc = _make_graph_json_five_nodes_30d()
    assert len(graph_doc["nodes"]) == 5
    span = max(n["last_seen_ts"] for n in graph_doc["nodes"]) - min(
        n["last_seen_ts"] for n in graph_doc["nodes"]
    )
    assert span >= 20 * 86400

    mock_graph = MagicMock()
    mock_graph.get_connections = AsyncMock(
        return_value=BartokGraphContext(
            connections=[
                BartokGraphConnection(
                    node_a_content="soil",
                    node_b_content="gamma soil carbon",
                    connection_type="temporal_bridge",
                    strength=0.72,
                    days_apart=18,
                    explanation="Prior soil carbon thread from weeks ago.",
                )
            ],
            provider_name="smoke_mock",
        )
    )

    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": True,
        "proactive_communication.bartokgraph.workspace": "~",
    }.get(k, d)

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=mock_graph,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    high = json.dumps({
        "should_send": True,
        "message": (
            "Hey — tying something together: your soil work lines up with what you "
            "explored a few weeks back on regime shifts."
        ),
        "novelty": 0.9,
        "relevance": 0.88,
        "connection_type": "temporal_bridge",
        "reasoning": "Strong non-obvious link.",
        "candidates": ["soil", "regime"],
    })

    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=high)):
        result = asyncio.run(loop.run_synthesis("smoke-session"))

    assert result.should_send is True
    assert result.message
    assert "BartokGraph" not in result.message
    assert result.connection_type in VALID_CONNECTION_TYPES


def test_smoke_low_score_no_send():
    db = _SmokeSessionDB()
    db.messages.append({"role": "user", "content": "hi", "ts": time.time()})

    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": False,
    }.get(k, d)

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=None,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    low = json.dumps({
        "should_send": True,
        "message": "Low value ping.",
        "novelty": 0.1,
        "relevance": 0.15,
        "connection_type": "none",
        "reasoning": "noise",
        "candidates": [],
    })

    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=low)):
        result = asyncio.run(loop.run_synthesis("smoke-low"))

    assert result.should_send is False
    assert result.message is None
