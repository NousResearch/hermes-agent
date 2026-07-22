"""Tests for DeFiLlama pool-discovery fallback."""

from __future__ import annotations

from hermes_trader.config import TraderConfig
from hermes_trader.loop.perceive import perceive_market
from hermes_trader.market_state import build_market_state
from hermes_trader.sources.defillama import (
    fetch_new_pools_payload,
    fetch_trending_pools_payload,
    normalize_yields_pool,
)


SAMPLE_ROWS = [
    {
        "chain": "Base",
        "project": "uniswap-v3",
        "symbol": "WETH-USDC",
        "tvlUsd": 250000,
        "volumeUsd1d": 180000,
        "count": 30,
        "underlyingTokens": [
            "0x4200000000000000000000000000000000000006",
            "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
        ],
    },
    {
        "chain": "Base",
        "project": "uniswap-v4",
        "symbol": "DEGEN-USDC",
        "tvlUsd": 150000,
        "volumeUsd1d": 95000,
        "apyPct1D": 12.5,
        "count": 2,
        "underlyingTokens": [
            "0x4ed4e862860bed51a9570b96d89af5e1b0efefed",
            "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
        ],
    },
    {
        "chain": "Ethereum",
        "project": "curve",
        "symbol": "ETH-USDC",
        "tvlUsd": 500000,
        "volumeUsd1d": 400000,
        "count": 40,
        "underlyingTokens": ["0x1", "0x2"],
    },
]


def _fake_http_get(_url: str):
    return {"data": SAMPLE_ROWS}


def test_normalize_yields_pool_picks_volatile_token_when_usdc_is_base():
    row = {
        "chain": "Base",
        "symbol": "USDC-CLAUDE",
        "tvlUsd": 338806,
        "volumeUsd1d": 161609307,
        "underlyingTokens": [
            "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
            "0xef34d1ba20131f0e6ea93a8c3e9397a871ab7b07",
        ],
    }
    pool = normalize_yields_pool(row, chain="base")
    assert pool is not None
    assert pool["pool_address"] == "0xef34d1ba20131f0e6ea93a8c3e9397a871ab7b07"


def test_normalize_yields_pool_maps_symbols_and_liquidity():
    row = SAMPLE_ROWS[1]
    pool = normalize_yields_pool(row, chain="base")
    assert pool is not None
    assert pool["base_token"]["symbol"] == "DEGEN"
    assert pool["quote_token"]["symbol"] == "USDC"
    assert pool["liquidity_usd"] == 150000
    assert pool["volume_24h_usd"] == 95000
    assert pool["pool_address"] == "0x4ed4e862860bed51a9570b96d89af5e1b0efefed"


def test_fetch_trending_pools_payload_filters_chain_and_sorts_by_volume():
    payload = fetch_trending_pools_payload(
        "base",
        min_liquidity_usd=100_000,
        http_get=_fake_http_get,
    )
    assert payload["source"] == "defillama"
    assert len(payload["pools"]) == 2
    assert payload["pools"][0]["base_token"]["symbol"] == "WETH"
    assert payload["pools"][0]["volume_24h_usd"] == 180000


def test_fetch_new_pools_payload_prefers_low_history_pools():
    payload = fetch_new_pools_payload(
        "base",
        min_liquidity_usd=10_000,
        http_get=_fake_http_get,
    )
    assert len(payload["pools"]) == 1
    assert payload["pools"][0]["base_token"]["symbol"] == "DEGEN"


def test_build_market_state_accepts_defillama_payload():
    trending = fetch_trending_pools_payload("base", http_get=_fake_http_get)
    new_pools = fetch_new_pools_payload("base", http_get=_fake_http_get)
    state = build_market_state(
        chain="base",
        trending_payload=trending,
        new_pools_payload=new_pools,
    )
    assert len(state.trending_pools) == 2
    assert len(state.new_pools) == 1


class McpErrorRecorder:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, server: str, tool: str, args: dict):
        self.calls.append((server, tool, dict(args)))
        if tool == "get_portfolio_tokens":
            return {"tokens": []}
        return {"error": "HTTP 401: Unauthorized"}


def test_perceive_market_falls_back_to_defillama_on_mcp_pool_errors(monkeypatch):
    cfg = TraderConfig(mode="paper", primary_chain="base", pool_data_fallback="auto")

    def _fake_trending(chain: str, **kwargs):
        return fetch_trending_pools_payload(chain, http_get=_fake_http_get, **kwargs)

    def _fake_new(chain: str, **kwargs):
        return fetch_new_pools_payload(chain, http_get=_fake_http_get, **kwargs)

    monkeypatch.setattr(
        "hermes_trader.sources.defillama.fetch_trending_pools_payload",
        _fake_trending,
    )
    monkeypatch.setattr(
        "hermes_trader.sources.defillama.fetch_new_pools_payload",
        _fake_new,
    )

    recorder = McpErrorRecorder()
    state = perceive_market(cfg, recorder)
    assert state.chain == "base"
    assert len(state.trending_pools) == 2
    assert len(state.new_pools) == 1
    tools = {tool for _s, tool, _a in recorder.calls}
    assert "get_trending_pools" in tools
    assert "get_new_pools" in tools