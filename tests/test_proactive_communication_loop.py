"""Tests for the Proactive Communication Loop."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hermes_cli.bartokgraph_adapter import (
    BartokGraphAdapter,
    _overlap_score,
    _resolve_local_model_provider,
)

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


def test_prompt_without_graph_has_no_bartokgraph_section():
    prompt = _build_synthesis_prompt("user: hello", "(none)", graph_ctx=None)
    assert "RECENT CONVERSATION HISTORY" in prompt
    assert "BARTOKGRAPH CONNECTIONS" not in prompt  # section only appears when connections exist
    assert "should_send" in prompt


def test_prompt_with_graph_connections_includes_bartokgraph_section():
    conn = BartokGraphConnection(
        node_a_content="anomaly detection",
        node_b_content="grid monitoring project",
        connection_type="temporal_bridge",
        strength=0.8,
        days_apart=21,
        explanation="both discuss state transitions in time-series",
    )
    graph_ctx = BartokGraphContext(connections=[conn], provider_name="mock")
    prompt = _build_synthesis_prompt("user: soil", "(none)", graph_ctx=graph_ctx)
    assert "BARTOKGRAPH CONNECTIONS" in prompt
    assert "TEMPORAL_BRIDGE" in prompt
    assert "anomaly detection" in prompt
    assert "grid monitoring" in prompt


def test_prompt_with_empty_connections_has_no_bartokgraph_section():
    graph_ctx = BartokGraphContext(connections=[], provider_name="mock")
    prompt = _build_synthesis_prompt("history", "(none)", graph_ctx=graph_ctx)
    assert "BARTOKGRAPH CONNECTIONS" not in prompt  # section only appears when connections exist


def test_prompt_instructs_natural_message():
    prompt = _build_synthesis_prompt("h", "n", graph_ctx=None)
    assert "natural" in prompt.lower() or "conversation" in prompt.lower()
    assert "Never mention BartokGraph" in prompt
    assert "mechanism" in prompt.lower()


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


def test_custom_threshold_can_block():
    @register_threshold("pcl_test_never")
    class NeverSend:
        def should_send(self, result: SynthesisResult) -> bool:
            return False

    score = _get_threshold_score("pcl_test_never", 0.9, {"novelty": 0.9, "relevance": 0.9, "message": "hi"})
    # Custom "never" threshold should return score > 1.0 (impossible to exceed)
    assert score > 1.0


# ──────────────────────────────────────────────────────────────────────
# ProactiveCommunicationLoop — error handling
# ──────────────────────────────────────────────────────────────────────


def test_run_synthesis_no_send_on_exception():
    """Any error → no-send result, never raises."""
    db = MagicMock()
    db.get_messages_since.side_effect = RuntimeError("db exploded")
    db.get_proactive_sent.return_value = []
    cfg = MagicMock()
    cfg.get.return_value = "conservative"

    with patch("hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph", return_value=None):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)
    result = asyncio.run(loop.run_synthesis("session-1"))

    assert result.should_send is False
    assert result.message is None


def test_run_synthesis_no_send_on_empty_history():
    db = MagicMock()
    db.get_messages_since.return_value = []
    db.get_proactive_sent.return_value = []
    cfg = MagicMock()
    cfg.get.return_value = "conservative"

    with patch("hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph", return_value=None):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)
    result = asyncio.run(loop.run_synthesis("session-empty"))

    assert result.should_send is False
    assert "no conversation history" in result.reasoning


def test_run_synthesis_respects_daily_limit():
    """If daily limit reached, never sends."""
    db = MagicMock()
    db.get_messages_since.return_value = [{"role": "user", "content": "hello"}]
    db.get_proactive_sent.return_value = [{"summary": "sent one"}]  # already sent today
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 1,  # limit = 1
    }.get(k, d)

    with patch("hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph", return_value=None):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)
    result = asyncio.run(loop.run_synthesis("session-limited"))

    assert result.should_send is False
    assert "daily message limit" in result.reasoning


# ──────────────────────────────────────────────────────────────────────
# End-to-end: high-quality synthesis sends, low-quality doesn't
# ──────────────────────────────────────────────────────────────────────


def test_high_score_sends_with_conservative_threshold():
    """High novelty + relevance → sends even with conservative threshold."""
    db = MagicMock()
    db.get_messages_since.return_value = [
        {"role": "user", "content": "can you check the logs for errors?"},
        {"role": "assistant", "content": "Scanning now."},
    ]
    db.get_proactive_sent.return_value = []
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": False,
    }.get(k, d)

    with patch("hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph", return_value=None):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    high_score = json.dumps({
        "should_send": True,
        "message": "Hey — finished the log scan. Errors repeat every 4h at :15. Cron job.",
        "novelty": 0.9, "relevance": 0.88,
        "connection_type": "none",
        "reasoning": "Completed task with clear result.",
        "candidates": ["log result"],
    })
    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=high_score)):
        result = asyncio.run(loop.run_synthesis("session-logs"))

    assert result.should_send is True
    assert result.message is not None
    assert result.novelty_score == pytest.approx(0.9)


def test_low_score_blocked_by_conservative_threshold():
    """Low scores → no send even if model says should_send=True."""
    db = MagicMock()
    db.get_messages_since.return_value = [{"role": "user", "content": "hi"}]
    db.get_proactive_sent.return_value = []
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "conservative",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": False,
    }.get(k, d)

    with patch("hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph", return_value=None):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    low_score = json.dumps({
        "should_send": True,  # model says yes — scores say no
        "message": "Just checking in!",
        "novelty": 0.2, "relevance": 0.3,
        "connection_type": "none",
        "reasoning": "low novelty",
        "candidates": [],
    })
    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=low_score)):
        result = asyncio.run(loop.run_synthesis("session-low"))

    # combined = 0.6*0.2 + 0.4*0.3 = 0.24 < 0.75 (conservative)
    assert result.should_send is False


def test_bartokgraph_temporal_bridge_triggers_send():
    """A BartokGraph temporal bridge with high scores → sends."""
    db = MagicMock()
    db.get_messages_since.return_value = [
        {"role": "user", "content": "working on anomaly detection in time-series today"},
    ]
    db.get_proactive_sent.return_value = []
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
            strength=0.85,
            days_apart=21,
            explanation="same concept appeared 21 days ago",
        )],
        provider_name="mock",
    ))

    with patch("hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph", return_value=mock_graph):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    bridge_msg = json.dumps({
        "should_send": True,
        "message": "You worked on anomaly detection 3 weeks ago in the grid monitoring project. The approach you found then applies here.",
        "novelty": 0.88, "relevance": 0.85,
        "connection_type": "temporal_bridge",
        "reasoning": "BartokGraph temporal bridge — high novelty.",
        "candidates": ["grid monitoring project"],
    })
    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=bridge_msg)):
        result = asyncio.run(loop.run_synthesis("session-bridge"))

    assert result.should_send is True
    assert result.connection_type == "temporal_bridge"
    assert "anomaly detection" in (result.message or "")


# ──────────────────────────────────────────────────────────────────────
# record_sent
# ──────────────────────────────────────────────────────────────────────


def test_record_sent_writes_to_db():
    db = MagicMock()
    cfg = MagicMock()
    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=None,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    result = SynthesisResult(
        should_send=True,
        message="Hello — finished the scan.",
        reasoning="done",
        novelty_score=0.9,
        relevance_score=0.85,
        combined_score=0.88,
        connection_type="temporal_bridge",
    )
    asyncio.run(loop.record_sent("session-record", result))

    db.record_proactive_sent.assert_called_once()
    sid, payload = db.record_proactive_sent.call_args[0]
    assert sid == "session-record"
    assert payload["connection_type"] == "temporal_bridge"
    assert "Hello" in payload["summary"]
    assert payload["score"] == pytest.approx(0.88)
    assert "ts" in payload


# ──────────────────────────────────────────────────────────────────────
# BartokGraphAdapter + graph fixture
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def bartok_graph_fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "bartokgraph_graph.json"


def test_bartokgraph_adapter_loads_fixture_graph_json(bartok_graph_fixture_path, tmp_path):
    bg_dir = tmp_path / ".bartokgraph"
    bg_dir.mkdir()
    shutil.copy(bartok_graph_fixture_path, bg_dir / "graph.json")

    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.bartokgraph.workspace": str(tmp_path),
    }.get(k, d)

    with patch(
        "hermes_cli.bartokgraph_adapter._resolve_local_model_provider",
        return_value={"name": "topology_only"},
    ):
        adapter = BartokGraphAdapter(config=cfg)

    assert adapter.is_available is True

    ctx = asyncio.run(
        adapter.get_connections(
            active_topics=["soil", "carbon", "regime"],
            top_k=10,
            min_strength=0.35,
        )
    )
    assert ctx is not None
    assert len(ctx.connections) >= 1
    assert ctx.connections[0].connection_type in (
        "temporal_bridge",
        "cross_domain",
        "person_knowledge",
    )


# ──────────────────────────────────────────────────────────────────────
# Local model provider / overlap
# ──────────────────────────────────────────────────────────────────────


def test_resolve_local_model_provider_returns_topology_when_no_servers(monkeypatch):
    """When no HTTP LLM endpoints respond, fall back to topology-only traversal."""

    def boom(*_a, **_kw):
        raise OSError("connection refused")

    monkeypatch.setenv("BARTOKGRAPH_API_BASE", "")
    monkeypatch.setenv("BARTOKGRAPH_API_KEY", "")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        boom,
    )
    info = _resolve_local_model_provider()
    assert info["name"] == "topology_only"


def test_overlap_score_stopwords_only_returns_zero():
    assert _overlap_score("the a an", "of and or") == 0.0


def test_overlap_score_unicode_and_long_strings():
    assert _overlap_score("café résumé", "café résumé détail") > 0.0
    long_a = "word " * 500 + "uniqueanchor"
    long_b = "other " * 500 + "uniqueanchor"
    assert 0.0 < _overlap_score(long_a, long_b) <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Custom threshold during synthesis + model should_send flag
# ──────────────────────────────────────────────────────────────────────


def test_custom_threshold_used_in_synthesis_pass():
    """Registered threshold gates send via impossible score when should_send is False."""

    @register_threshold("pcl_integration_strict")
    class StrictNovelty:
        def should_send(self, result: SynthesisResult) -> bool:
            return result.novelty_score >= 0.95

    db = MagicMock()
    db.get_messages_since.return_value = [{"role": "user", "content": "hello"}]
    db.get_proactive_sent.return_value = []
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "pcl_integration_strict",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": False,
    }.get(k, d)

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=None,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    blocked = json.dumps({
        "should_send": True,
        "message": "Hi",
        "novelty": 0.8,
        "relevance": 0.9,
        "connection_type": "none",
        "reasoning": "below strict novelty",
        "candidates": [],
    })
    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=blocked)):
        result = asyncio.run(loop.run_synthesis("s1"))
    assert result.should_send is False

    allowed = json.dumps({
        "should_send": True,
        "message": "Insight!",
        "novelty": 0.96,
        "relevance": 0.9,
        "connection_type": "none",
        "reasoning": "passes strict novelty",
        "candidates": [],
    })
    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=allowed)):
        result2 = asyncio.run(loop.run_synthesis("s2"))
    assert result2.should_send is True


def test_model_should_send_false_blocks_despite_high_scores():
    db = MagicMock()
    db.get_messages_since.return_value = [{"role": "user", "content": "hello"}]
    db.get_proactive_sent.return_value = []
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {
        "proactive_communication.threshold": "eager",
        "proactive_communication.max_per_day": 3,
        "proactive_communication.bartokgraph.enabled": False,
    }.get(k, d)

    with patch(
        "hermes_cli.proactive_communication_loop.ProactiveCommunicationLoop._try_load_bartokgraph",
        return_value=None,
    ):
        loop = ProactiveCommunicationLoop(session_db=db, config=cfg)

    raw = json.dumps({
        "should_send": False,
        "message": "would spam",
        "novelty": 0.99,
        "relevance": 0.99,
        "connection_type": "none",
        "reasoning": "model veto",
        "candidates": [],
    })
    with patch.object(loop, "_call_synthesis_model", new=AsyncMock(return_value=raw)):
        result = asyncio.run(loop.run_synthesis("s-veto"))
    assert result.should_send is False
