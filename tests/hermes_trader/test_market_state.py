"""Tests for hermes_trader.market_state normalization and schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_trader.market_state import build_market_state


def _schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "hermes_trader"
        / "schema"
        / "market_state.schema.json"
    )


def test_build_market_state_normalizes_mcp_payloads():
    state = build_market_state(
        chain="base",
        captured_at="2026-07-07T12:00:00+00:00",
        portfolio_payload={
            "tokens": [
                {
                    "symbol": "ETH",
                    "address": "0xabc",
                    "balance": "1.5",
                    "balance_usd": 4500,
                    "price_usd": 3000,
                }
            ]
        },
        trending_payload={
            "pools": [
                {
                    "pool_address": "0xpool1",
                    "base_token": {"symbol": "WETH"},
                    "quote_token": {"symbol": "USDC"},
                    "liquidity_usd": 250000,
                    "volume_24h_usd": 120000,
                    "price_change_24h": -1.2,
                }
            ]
        },
        new_pools_payload=[
            {
                "address": "0xpool2",
                "base_token_symbol": "DEGEN",
                "quote_token_symbol": "WETH",
                "reserve_in_usd": 150000,
            }
        ],
    )

    assert state.chain == "base"
    assert len(state.portfolio_tokens) == 1
    assert state.portfolio_tokens[0].symbol == "ETH"
    assert state.portfolio_tokens[0].balance == 1.5

    assert len(state.trending_pools) == 1
    assert state.trending_pools[0].base_token_symbol == "WETH"
    assert state.trending_pools[0].source == "trending"

    assert len(state.new_pools) == 1
    assert state.new_pools[0].pool_address == "0xpool2"
    assert state.new_pools[0].source == "new"


def test_market_state_to_dict_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")

    state = build_market_state(
        chain="Base",
        captured_at="2026-07-07T12:00:00+00:00",
        portfolio_payload={"portfolio": [{"symbol": "USDC", "address": "0x1", "balance": 100}]},
        trending_payload={"data": {"pools": []}},
        new_pools_payload={"new_pools": []},
    )
    payload = state.to_dict()

    schema_path = _schema_path()
    if not schema_path.is_file():
        pytest.skip("market_state.schema.json not present")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, schema)
    assert payload["chain"] == "base"