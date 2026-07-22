"""Tests for hermes_trader.memory — P3 episodic + strategic memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_trader.config import TraderConfig
from hermes_trader.loop.context import retrieve_context
from hermes_trader.loop.scheduler import run_trading_cycle
from hermes_trader.market_state import build_market_state
from hermes_trader.memory.episodes import EpisodeStore, TradeEpisode, episode_from_cycle
from hermes_trader.memory.migrate import migrate
from hermes_trader.memory.retrieval import retrieve_similar_episodes, sanitize_episode_context
from hermes_trader.memory.strategic_rules import load_strategic_rules
from hermes_trader.memory.summary import build_market_summary, compute_embedding_id, liquidity_band
from hermes_trader.risk.gate import GateDecision, RejectReason, TradeIntent
from tests.hermes_trader.test_loop import McpRecorder


@pytest.fixture
def episode_db(tmp_path):
    return EpisodeStore(tmp_path / "trade_episodes.db")


def _sample_state():
    return build_market_state(
        chain="base",
        captured_at="2026-07-07T12:00:00+00:00",
        portfolio_payload={"tokens": [{"symbol": "USDC", "address": "0x1", "balance": 1000, "balance_usd": 1000}]},
        trending_payload={
            "pools": [
                {
                    "pool_address": "0xpool1",
                    "base_token": {"symbol": "AAA"},
                    "quote_token": {"symbol": "WETH"},
                    "liquidity_usd": 200_000,
                    "volume_24h_usd": 90_000,
                }
            ]
        },
    )


def _sample_cycle_result():
    from hermes_trader.loop.executor import ExecutionResult
    from hermes_trader.loop.scheduler import CycleResult

    state = _sample_state()
    intent = TradeIntent(
        action="buy",
        chain="base",
        token_address="0xpool1",
        size_usd=50.0,
        confidence=0.75,
        reasoning="momentum test",
        strategy_tag="memecoin_momentum",
        pool_liquidity_usd=200_000.0,
    )
    decision = GateDecision(
        approved=False,
        reason_code=RejectReason.PAPER_MODE,
        message="paper",
    )
    return CycleResult(
        market_state=state,
        intent=intent,
        decision=decision,
        execution=ExecutionResult(status="skipped", message="paper"),
        context=[],
    )


def test_migrate_creates_schema(episode_db):
    version = migrate(str(episode_db.db_path))
    assert version >= 2
    assert episode_db.db_path.is_file()


def test_record_and_load_episode(episode_db):
    episode = episode_db.record_cycle(_sample_cycle_result())
    assert episode.episode_id
    assert episode.gate_decision == "REJECT"
    assert episode.gate_reason == "PAPER_MODE"
    loaded = episode_db.get_episode(episode.episode_id)
    assert loaded is not None
    assert loaded.intent["action"] == "buy"
    assert episode_db.count() == 1


def test_episode_from_cycle_sets_embedding_id(episode_db):
    episode = episode_from_cycle(_sample_cycle_result())
    assert episode.embedding_id
    summary = episode.market_summary
    assert compute_embedding_id(summary) == episode.embedding_id


def test_liquidity_bands():
    assert liquidity_band(25_000) == "micro"
    assert liquidity_band(100_000) == "small"
    assert liquidity_band(300_000) == "mid"
    assert liquidity_band(1_000_000) == "large"


def test_retrieve_similar_episodes_ranks_strategy_tag(episode_db):
    state = _sample_state()
    base_cycle = _sample_cycle_result()
    ep1 = episode_db.record_cycle(base_cycle)

    other_intent = TradeIntent(
        action="buy",
        chain="base",
        token_address="0xother",
        size_usd=30.0,
        confidence=0.7,
        reasoning="rebalance idea",
        strategy_tag="rebalance",
        pool_liquidity_usd=80_000.0,
    )
    other_cycle = _sample_cycle_result()
    from hermes_trader.loop.scheduler import CycleResult

    episode_db.record_cycle(
        CycleResult(
            market_state=other_cycle.market_state,
            intent=other_intent,
            decision=other_cycle.decision,
            execution=other_cycle.execution,
            context=[],
        )
    )

    hits = retrieve_similar_episodes(
        state,
        episode_db,
        limit=3,
        strategy_tag="memecoin_momentum",
        liquidity_usd=200_000.0,
        token_address="0xpool1",
    )
    assert hits
    assert hits[0]["episode_id"] == ep1.episode_id
    assert hits[0]["trust"] == "untrusted"


def test_sanitize_episode_context_includes_disclaimer():
    episode = TradeEpisode(
        episode_id="abc",
        timestamp="2026-07-07T12:00:00+00:00",
        chain="base",
        strategy_tag="memecoin_momentum",
        gate_decision="REJECT",
        gate_reason="PAPER_MODE",
        intent={"action": "buy", "reasoning": "test thesis"},
        decision=None,
        execution=None,
        market_summary={},
        liquidity_usd=200_000.0,
        token_address="0x1",
    )
    ctx = sanitize_episode_context(episode, score=4.5)
    assert ctx["trust"] == "untrusted"
    assert "UNTRUSTED" in ctx["disclaimer"]


def test_load_strategic_rules_bundled():
    rules = load_strategic_rules()
    snippets = rules.to_context_snippets()
    kinds = {s["kind"] for s in snippets}
    assert "positive_heuristic" in kinds
    assert "negative_constraint" in kinds


def test_retrieve_context_includes_strategic_and_working(episode_db):
    state = _sample_state()
    ctx = retrieve_context(state, episode_store=episode_db, config=TraderConfig())
    kinds = [c["kind"] for c in ctx]
    assert "working_memory" in kinds
    assert "positive_heuristic" in kinds


def test_cycle_records_episode_and_next_retrieves(tmp_path):
    recorder = McpRecorder()
    store = EpisodeStore(tmp_path / "episodes.db")
    cfg = TraderConfig(mode="paper")

    run_trading_cycle(config=cfg, mcp_call=recorder, episode_store=store)
    assert store.count() == 1

    state = _sample_state()
    ctx = retrieve_context(state, episode_store=store, config=cfg)
    episodic = [c for c in ctx if c.get("kind") == "episodic_memory"]
    assert len(episodic) >= 1


def test_build_market_summary_json_serializable():
    summary = build_market_summary(_sample_state())
    json.dumps(summary)