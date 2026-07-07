"""Tests for hermes_trader.loop — P2 agent cycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hermes_trader.config import TraderConfig
from hermes_trader.loop.audit import CycleAuditLog
from hermes_trader.loop.executor import OrderExecutor
from hermes_trader.loop.intent import hold_intent, parse_trade_intent, validate_trade_intent
from hermes_trader.loop.perceive import perceive_market
from hermes_trader.loop.reason import heuristic_reasoner
from hermes_trader.loop.scheduler import (
    TradingCycleRunner,
    build_cron_job_spec,
    count_write_tool_calls,
    cron_schedule_from_config,
    run_trading_cycle,
)
from hermes_trader.risk.gate import OrderRequest, RejectReason, TradeIntent
from hermes_trader.tools import LIVE_WRITE_TOOLS, PAPER_MODE_READ_TOOLS

TEST_KEY = b"test-mandate-secret-key"
WALLET = "0xabcdef1234567890abcdef1234567890abcdef12"


class McpRecorder:
    def __init__(self, responses: dict[tuple[str, str], Any] | None = None):
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.responses = responses or {}

    def __call__(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((server, tool, dict(args)))
        return self.responses.get(
            (server, tool),
            self._default_response(tool, args),
        )

    @staticmethod
    def _default_response(tool: str, args: dict[str, Any]) -> Any:
        chain = args.get("chain", "base")
        if tool == "get_portfolio_tokens":
            return {
                "tokens": [
                    {
                        "symbol": "USDC",
                        "address": "0xusdc",
                        "balance": 5000,
                        "balance_usd": 5000,
                    }
                ]
            }
        if tool == "get_trending_pools":
            return {
                "pools": [
                    {
                        "pool_address": "0xpool1",
                        "base_token": {"symbol": "DEGEN"},
                        "quote_token": {"symbol": "WETH"},
                        "liquidity_usd": 250_000,
                        "volume_24h_usd": 180_000,
                        "chain": chain,
                    }
                ]
            }
        if tool == "get_new_pools":
            return {"pools": []}
        if tool in LIVE_WRITE_TOOLS:
            return {"tx_hash": "0xdeadbeef"}
        return {}


@pytest.fixture
def paper_config():
    return TraderConfig(
        mode="paper",
        primary_chain="base",
        min_confidence=0.6,
        min_pool_liquidity_usd=100_000.0,
        scan_interval_minutes=15,
    )


def test_parse_trade_intent_from_markdown_fence():
    text = """
    Analysis complete.

    ```json
    {"action": "hold", "chain": "base", "token_address": "", "size_usd": 0, "confidence": 0.9, "reasoning": "wait"}
    ```
    """
    intent = parse_trade_intent(text)
    assert intent.action == "hold"
    assert intent.chain == "base"


def test_parse_trade_intent_from_dict():
    intent = parse_trade_intent(
        {
            "action": "buy",
            "chain": "base",
            "token_address": "0xabc",
            "size_usd": 50,
            "confidence": 0.8,
        }
    )
    assert intent.action == "buy"
    assert intent.size_usd == 50.0


def test_validate_trade_intent_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    _ = jsonschema
    intent = hold_intent("base")
    validate_trade_intent(intent)


def test_perceive_market_builds_state(paper_config):
    recorder = McpRecorder()
    state = perceive_market(paper_config, recorder)
    assert state.chain == "base"
    assert len(state.portfolio_tokens) == 1
    assert len(state.trending_pools) == 1
    tools = {tool for _s, tool, _a in recorder.calls}
    assert tools <= PAPER_MODE_READ_TOOLS


def test_heuristic_reasoner_picks_trending_pool(paper_config):
    recorder = McpRecorder()
    state = perceive_market(paper_config, recorder)
    intent = heuristic_reasoner(state, paper_config)
    assert intent.action == "buy"
    assert intent.pool_liquidity_usd == 250_000


def test_executor_skips_in_paper_mode(paper_config):
    recorder = McpRecorder()
    executor = OrderExecutor(paper_config, recorder)
    order = OrderRequest(
        chain="base",
        token_address="0xabc",
        action="buy",
        size_usd=50,
        max_slippage_bps=100,
        tool="submit_gasless_swap",
        reasoning="test",
    )
    result = executor.execute(order)
    assert result.status == "skipped"
    assert count_write_tool_calls(recorder.calls) == 0


@pytest.fixture
def episode_store(tmp_path):
    from hermes_trader.memory.episodes import EpisodeStore

    return EpisodeStore(tmp_path / "trade_episodes.db")


def test_paper_cycle_rejects_buy_and_never_calls_write_tools(paper_config, episode_store):
    recorder = McpRecorder()
    result = run_trading_cycle(
        config=paper_config, mcp_call=recorder, episode_store=episode_store
    )
    assert result.intent.action == "buy"
    assert not result.decision.approved
    assert result.decision.reason_code == RejectReason.PAPER_MODE
    assert result.execution is not None
    assert result.execution.status == "skipped"
    assert count_write_tool_calls(recorder.calls) == 0


def test_100_paper_cycles_never_invoke_write_tools(paper_config, tmp_path):
    """P2 deliverable: 100 cycles without write tool invocation."""
    from hermes_trader.memory.episodes import EpisodeStore

    log_path = tmp_path / "cycles.jsonl"
    audit = CycleAuditLog(log_path)
    store = EpisodeStore(tmp_path / "trade_episodes.db")
    write_hits = 0

    for _ in range(100):
        recorder = McpRecorder()
        result = run_trading_cycle(
            config=paper_config,
            mcp_call=recorder,
            audit_log=audit,
            episode_store=store,
        )
        write_hits += count_write_tool_calls(recorder.calls)
        assert not result.decision.approved or paper_config.mode == "paper"
        assert count_write_tool_calls(recorder.calls) == 0

    assert write_hits == 0
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100


def test_hold_intent_cycle_logs_rejection(paper_config, tmp_path):
    from hermes_trader.memory.episodes import EpisodeStore

    recorder = McpRecorder()

    def always_hold(state, config, context):
        _ = state, config, context
        return hold_intent("base")

    audit = CycleAuditLog(tmp_path / "cycles.jsonl")
    store = EpisodeStore(tmp_path / "trade_episodes.db")
    result = run_trading_cycle(
        config=paper_config,
        mcp_call=recorder,
        reason_fn=always_hold,
        audit_log=audit,
        episode_store=store,
    )
    assert result.decision.reason_code == RejectReason.HOLD


def test_cron_schedule_from_config():
    cfg = TraderConfig(scan_interval_minutes=15)
    assert cron_schedule_from_config(cfg) == "*/15 * * * *"
    cfg2 = TraderConfig(scan_interval_minutes=120)
    assert cron_schedule_from_config(cfg2) == "0 */2 * * *"


def test_build_cron_job_spec(paper_config):
    spec = build_cron_job_spec(paper_config)
    assert spec["skill"] == "hermes-agentic-trader"
    assert spec["schedule"] == "*/15 * * * *"
    assert "TradeIntent" in spec["prompt"] or "trading cycle" in spec["prompt"].lower()


def test_live_approved_cycle_submits_via_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())
    hh = tmp_path / "hermes-home"
    trader = hh / "trader"
    trader.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hh))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hh)

    from hermes_trader.risk.mandate import save_mandate, sign_mandate

    save_mandate(sign_mandate(WALLET, signing_key=TEST_KEY), trader / "mandate.json")

    cfg = TraderConfig(
        mode="live",
        min_confidence=0.5,
        max_position_pct=10.0,
        min_pool_liquidity_usd=100_000.0,
    )
    recorder = McpRecorder()

    def aggressive_buy(state, config, context):
        _ = state, context
        return TradeIntent(
            action="buy",
            chain="base",
            token_address="0xpool1",
            size_usd=50.0,
            confidence=0.8,
            reasoning="live test",
            pool_liquidity_usd=250_000.0,
            slippage_bps=50,
        )

    result = run_trading_cycle(
        config=cfg,
        mcp_call=recorder,
        reason_fn=aggressive_buy,
    )
    assert result.decision.approved
    assert result.execution is not None
    assert result.execution.status == "submitted"
    assert count_write_tool_calls(recorder.calls) == 1
    assert recorder.calls[-1][1] == "submit_gasless_swap"


def test_intent_schema_file_exists():
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "hermes_trader"
        / "loop"
        / "intent_schema.json"
    )
    assert schema_path.is_file()
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    assert data["title"] == "TradeIntent"