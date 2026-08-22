"""Reasoning helpers — LLM cron uses prompts; tests use heuristics."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, List

from hermes_trader.config import TraderConfig
from hermes_trader.loop.intent import hold_intent
from hermes_trader.market_state import MarketState, PoolSnapshot
from hermes_trader.risk.gate import TradeIntent


def build_cycle_prompt(config: TraderConfig, state: MarketState) -> str:
    """Render the scan prompt template with current config and state summary."""
    template_path = (
        Path(__file__).resolve().parents[2]
        / "optional-skills"
        / "trading"
        / "hermes-agentic-trader"
        / "prompts"
        / "cycle.md"
    )
    if template_path.is_file():
        template = template_path.read_text(encoding="utf-8")
    else:
        template = (
            "Run trading cycle on ${primary_chain}. "
            "Output TradeIntent JSON. Mode: ${mode}."
        )
    return Template(template).safe_substitute(
        primary_chain=config.primary_chain,
        min_pool_liquidity_usd=config.min_pool_liquidity_usd,
        min_confidence=config.min_confidence,
        mode=config.mode,
        trending_count=len(state.trending_pools),
        new_pool_count=len(state.new_pools),
        portfolio_count=len(state.portfolio_tokens),
    )


def _pick_best_pool(pools: List[PoolSnapshot], config: TraderConfig) -> PoolSnapshot | None:
    best: PoolSnapshot | None = None
    best_volume = -1.0
    for pool in pools:
        liq = pool.liquidity_usd
        if liq is None or liq < config.min_pool_liquidity_usd:
            continue
        vol = pool.volume_24h_usd or 0.0
        if vol > best_volume:
            best = pool
            best_volume = vol
    return best


def heuristic_reasoner(state: MarketState, config: TraderConfig) -> TradeIntent:
    """Deterministic reasoner for tests and no-LLM dry runs."""
    candidates = list(state.trending_pools) + list(state.new_pools)
    best = _pick_best_pool(candidates, config)
    if best is None:
        return hold_intent(state.chain, reasoning="No pool meets liquidity threshold")

    size_usd = max(25.0, min(100.0, (config.max_position_pct / 100.0) * 10_000))
    confidence = 0.72 if (best.volume_24h_usd or 0) > 50_000 else 0.55
    return TradeIntent(
        action="buy",
        chain=state.chain,
        token_address=best.pool_address,
        size_usd=size_usd,
        confidence=confidence,
        reasoning=(
            f"Momentum on {best.base_token_symbol}/{best.quote_token_symbol}; "
            f"liquidity ${best.liquidity_usd:,.0f}"
        ),
        pool_liquidity_usd=best.liquidity_usd,
        slippage_bps=min(config.max_slippage_bps, 50),
        strategy_tag="memecoin_momentum",
    )